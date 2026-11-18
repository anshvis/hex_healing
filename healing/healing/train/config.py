"""The training config (`healing/train/configs/lora.yaml`): read, override for `--smoke`, write back.

`healing` has no YAML dependency (D-48), so this reads the small subset the config uses: `key:
value` scalars (int, float, bool, null, plain or quoted strings), flow lists of scalars (`[a, b]`)
and nested mappings by two-space indentation, with `#` comments. Anything else is an error, never a
guess. `dump` writes the same subset back in the same order, so the effective config of a round
(after `--smoke`) is a file the registry hashes (AC-11) and T6.3 reads.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "train" / "configs" / "lora.yaml"
SMOKE = "smoke"
REQUIRED = ("arch", "base", "targets", "rank", "alpha", "dropout", "lr", "epochs", "batch_seconds", "batch_max_clips",
            "max_steps", "checkpoint_every", "validate_every", "validation_share", "early_stopping_patience", "seed",
            "threads", "device", "max_rss_gb")
BASE_KEYS = ("repo", "revision", "file", "sha256")
_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s+(.*))?$")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


class ConfigError(Exception):
    pass


def _strip_comment(line: str) -> str:
    quote, escaped = None, False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif ch == "\\" and quote == '"':
            escaped = True
        elif ch in "\"'" and quote in (None, ch):
            quote = None if quote else ch  # YAML's '' inside single quotes toggles twice: still inside
        elif ch == "#" and quote is None and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def _quoted(text: str, where: str) -> str:
    """One whole quoted scalar, as YAML reads it: '…' with '' for a quote, or "…" with JSON's escapes.
    Anything after the closing quote (`'a' 'b'`, `'a'b`) is an error, never part of the value."""
    quote, i, out = text[0], 1, []
    while i < len(text):
        ch = text[i]
        if quote == "'" and ch == "'":
            if text[i + 1:i + 2] == "'":
                out.append("'")
                i += 2
                continue
            break
        if quote == '"' and ch == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        if quote == '"' and ch == '"':
            break
        out.append(ch)
        i += 1
    else:
        raise ConfigError(f"{where}: unterminated string {text!r}")
    if i != len(text) - 1:
        raise ConfigError(f"{where}: text after the closing quote in {text!r}")
    if quote == "'":
        return "".join(out)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{where}: cannot read {text!r}: {error.msg}") from None


def _scalar(text: str, where: str):
    text = text.strip()
    if text in ("null", "~", ""):
        return None
    if text in ("true", "false"):
        return text == "true"
    if text[0] in "\"'":
        return _quoted(text, where)
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if _NUMBER.match(text):
        return float(text)
    if any(ch in text for ch in "[]{}:,&*!|>%@`"):
        raise ConfigError(f"{where}: {text!r} needs quotes")
    return text


def _value(text: str, where: str):
    if text.startswith("["):
        if not text.endswith("]"):
            raise ConfigError(f"{where}: a list must close on its line")
        inner = text[1:-1].strip()
        if any(ch in inner for ch in "\"'"):
            raise ConfigError(f"{where}: quoted items in a flow list are not supported")
        return [_scalar(part, where) for part in inner.split(",")] if inner else []
    return _scalar(text, where)


def parse(text: str, name: str = "config") -> dict:
    root: dict = {}
    stack: list[list] = [[-1, root, None]]  # [the owning key's indent, the mapping, its keys' indent]
    pending: tuple[int, dict, str] | None = None  # a `key:` with no value opens a mapping
    for number, raw in enumerate(text.splitlines(), 1):
        where = f"{name}:{number}"
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent + 1]:
            raise ConfigError(f"{where}: tabs are not allowed")
        match = _KEY.match(line.strip())
        if not match:
            raise ConfigError(f"{where}: expected `key: value`")
        if pending is not None:
            parent_indent, parent, key = pending
            if indent > parent_indent:
                parent[key] = {}
                stack.append([parent_indent, parent[key], None])
            else:
                parent[key] = None
            pending = None
        while stack[-1][0] >= indent:
            stack.pop()
        level = stack[-1]
        if level[2] is None:
            level[2] = indent
        elif level[2] != indent:
            raise ConfigError(f"{where}: bad indentation")
        target = stack[-1][1]
        key, value = match.group(1), match.group(2)
        if key in target:
            raise ConfigError(f"{where}: {key} is set twice")
        if value is None or not value.strip():
            pending = (indent, target, key)
        else:
            target[key] = _value(value.strip(), where)
    if pending is not None:
        pending[1][pending[2]] = None
    return root


def _format(value) -> str:
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    text = str(value)
    plain = text and text == text.strip() and not _NUMBER.match(text) and text not in ("true", "false", "null", "~") \
        and not any(ch in text for ch in "[]{}:,&*!|>%@`#'\"")
    return text if plain else json.dumps(text)


def dump(config: dict, indent: int = 0) -> str:
    lines = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(" " * indent + f"{key}:")
            lines.append(dump(value, indent + 2).rstrip("\n"))
        else:
            lines.append(" " * indent + f"{key}: {_format(value)}")
    return "\n".join(lines) + "\n"


def check(config: dict, name: str = "config") -> dict:
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise ConfigError(f"{name} lacks {', '.join(missing)}")
    base = config["base"]
    if not isinstance(base, dict) or any(not base.get(k) for k in BASE_KEYS):
        raise ConfigError(f"{name}: base needs {', '.join(BASE_KEYS)}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(base["sha256"])):
        raise ConfigError(f"{name}: base.sha256 must be a SHA-256")
    if not isinstance(config["targets"], list) or not config["targets"]:
        raise ConfigError(f"{name}: targets must be a non-empty list")
    for key in ("rank", "epochs", "batch_max_clips", "checkpoint_every", "validate_every", "threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ConfigError(f"{name}: {key} must be a positive integer")
    for key in ("alpha", "lr", "batch_seconds", "max_rss_gb"):
        if not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise ConfigError(f"{name}: {key} must be positive")
    if not 0 <= config["dropout"] < 1 or not 0 <= config["validation_share"] < 1:
        raise ConfigError(f"{name}: dropout and validation_share must be in [0, 1)")
    return config


def load(path: Path, smoke: bool = False) -> tuple[dict, str]:
    """The effective config (with the `smoke` overrides applied when asked) and its YAML text,
    which is what the round records and hashes. The `smoke` section itself is never part of it."""
    try:
        config = parse(Path(path).read_text(), str(path))
    except OSError as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    overrides = config.pop(SMOKE, None) or {}
    if smoke:
        if not overrides:
            raise ConfigError(f"{path} has no smoke section")
        unknown = [k for k in overrides if k not in config and k != "max_clips"]
        if unknown:
            raise ConfigError(f"{path}: smoke overrides unknown keys {', '.join(unknown)}")
        config.update(overrides)
        config["smoke"] = True
    check(config, str(path))
    text = dump(config)
    return config, text


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
