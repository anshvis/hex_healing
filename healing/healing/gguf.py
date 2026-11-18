"""A GGUF file's header, read and rewritten without the `gguf` package (healing has no NeMo env).

The header is the magic, the version, the tensor and key/value counts, the key/value pairs, the tensor
infos (name, shape, type, offset into the data), then padding to `general.alignment` (32 by default)
before the tensor data. Offsets are relative to the data's start, so a rewrite that changes one string
value only rebuilds the header and pads it again; the tensor data is copied byte for byte.
"""

from __future__ import annotations

import io
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
STRING, ARRAY = 8, 9
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class GGUFError(ValueError):
    pass


@dataclass
class Header:
    version: int
    kv: dict = field(default_factory=dict)
    tensors: list[str] = field(default_factory=list)  # names, in file order
    spans: dict = field(default_factory=dict)  # key -> (start, end) of its whole key/value bytes
    kv_start: int = 0
    kv_end: int = 0
    tensors_end: int = 0  # where the tensor infos end (padding follows)
    data_start: int = 0

    def string(self, key: str) -> str | None:
        value = self.kv.get(key)
        return value if isinstance(value, str) else None


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def take(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise GGUFError("the header ends early")
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        size = self.unpack("<Q")
        if size > 1 << 24:
            raise GGUFError("a header string is implausibly long")
        return self.take(size).decode("utf-8", errors="replace")

    def value(self, kind: int):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            inner, count = self.unpack("<I"), self.unpack("<Q")
            if count > 1 << 24:
                raise GGUFError("a header array is implausibly long")
            return [self.value(inner) for _ in range(count)]
        if kind not in _SCALARS:
            raise GGUFError(f"unknown value type {kind}")
        return self.unpack(_SCALARS[kind])


def read(path: Path) -> Header:
    """The header of `path`: every key/value (arrays as lists), the tensor names, and where each part
    sits. A file that isn't GGUF, or whose header is cut short, is a `GGUFError`."""
    with open(path, "rb", buffering=1 << 20) as handle:
        reader = _Reader(handle)
        if handle.read(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        header = Header(version=reader.unpack("<I"))
        n_tensors, n_kv = reader.unpack("<Q"), reader.unpack("<Q")
        header.kv_start = handle.tell()
        for _ in range(n_kv):
            start = handle.tell()
            key = reader.string()
            header.kv[key] = reader.value(reader.unpack("<I"))
            header.spans[key] = (start, handle.tell())
        header.kv_end = handle.tell()
        for _ in range(n_tensors):
            header.tensors.append(reader.string())
            dims = reader.unpack("<I")
            reader.take(8 * dims + 4 + 8)  # shape, type, offset
        header.tensors_end = handle.tell()
    alignment = header.kv.get("general.alignment", DEFAULT_ALIGNMENT)
    header.data_start = _aligned(header.tensors_end, alignment)
    return header


def _aligned(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def encode_string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def encode_kv(key: str, value) -> bytes:
    """One key/value pair: str, bool, int (as INT64 unless it fits a UINT32), float or a list of str."""
    if isinstance(value, str):
        return encode_string(key) + struct.pack("<I", STRING) + encode_string(value)
    if isinstance(value, bool):
        return encode_string(key) + struct.pack("<I?", 7, value)
    if isinstance(value, int):
        if 0 <= value < 1 << 32:
            return encode_string(key) + struct.pack("<II", 4, value)
        return encode_string(key) + struct.pack("<Iq", 11, value)
    if isinstance(value, float):
        return encode_string(key) + struct.pack("<If", 6, value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return encode_string(key) + struct.pack("<IIQ", ARRAY, STRING, len(value)) + b"".join(map(encode_string, value))
    raise GGUFError(f"cannot encode {key}")


def write(path: Path, kv: dict, tensors: dict[str, bytes]) -> None:
    """A small GGUF (F32 1-D tensors from raw bytes): what tests use in place of the converter."""
    alignment = kv.get("general.alignment", DEFAULT_ALIGNMENT)
    body = io.BytesIO()
    infos, offset = [], 0
    for name, data in tensors.items():
        infos.append(encode_string(name) + struct.pack("<IQIQ", 1, len(data) // 4, 0, offset))
        body.write(data + b"\0" * (_aligned(len(data), alignment) - len(data)))
        offset = body.tell()
    head = MAGIC + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(encode_kv(k, v) for k, v in kv.items())
    head += b"".join(infos)
    with open(path, "wb") as out:
        out.write(head + b"\0" * (_aligned(len(head), alignment) - len(head)))
        out.write(body.getvalue())


def set_string(source: Path, target: Path, key: str, value: str) -> None:
    """Copy `source` to `target` with the string `key` set to `value` (replaced in place, or appended
    after the other pairs). The tensor data is copied unchanged."""
    header = read(source)
    if key in header.kv and not isinstance(header.kv[key], str):
        raise GGUFError(f"{key} is not a string in {source}")
    alignment = header.kv.get("general.alignment", DEFAULT_ALIGNMENT)
    with open(source, "rb") as handle:
        raw = handle.read(header.tensors_end)
        pair = encode_kv(key, value)
        n_kv = len(header.kv) + (key not in header.kv)
        start, end = header.spans.get(key, (header.kv_end, header.kv_end))
        head = raw[:8] + raw[8:16] + struct.pack("<Q", n_kv) + raw[header.kv_start:start] + pair + raw[end:]
        handle.seek(header.data_start)
        with open(target, "wb") as out:
            out.write(head + b"\0" * (_aligned(len(head), alignment) - len(head)))
            shutil.copyfileobj(handle, out, 1 << 20)
            out.flush()
            os.fsync(out.fileno())
