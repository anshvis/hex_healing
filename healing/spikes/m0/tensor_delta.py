"""How much of a fine-tune's weight change survives Q8_0.

Runs in transcribe.cpp's Parakeet environment (it needs `gguf`):

    python tensor_delta.py base-F32.gguf tuned-F32.gguf base-Q8_0.gguf tuned-Q8_0.gguf out.json

For every tensor the fine-tune changed it reports ‖ΔW‖ (merged − base, F32),
the mean Q8_0 rounding step of the base tensor (block scale = max|x| / 127
over each 32-value block, the step between adjacent codes), and the share of
the change still present after quantization: ‖Q(merged) − Q(base)‖ / ‖ΔW‖,
and the cosine between Q(merged) − Q(base) and ΔW. When ΔW is smaller than a
rounding step the norm ratio exceeds 1 (rounding noise), so the cosine is the
measure of whether the change's direction survived.
It also asserts that both conversions have identical tensor names.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize


def tensors(path: str) -> dict[str, np.ndarray]:
    reader = GGUFReader(path)
    out = {}
    for tensor in reader.tensors:
        if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            out[tensor.name] = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
        else:
            out[tensor.name] = dequantize(tensor.data, tensor.tensor_type).reshape(-1)
    return out


def q8_step(values: np.ndarray) -> float:
    usable = values[: len(values) // 32 * 32].reshape(-1, 32)
    return float((np.abs(usable).max(axis=1) / 127.0).mean())


def main() -> int:
    base_f32, tuned_f32, base_q8, tuned_q8, out = sys.argv[1:6]
    base, tuned = tensors(base_f32), tensors(tuned_f32)
    qbase, qtuned = tensors(base_q8), tensors(tuned_q8)
    names_equal = list(base) == list(tuned) and list(qbase) == list(qtuned) and set(base) == set(qbase)
    rows = []
    for name in base:
        delta = tuned[name] - base[name]
        norm = float(np.linalg.norm(delta))
        if norm == 0.0:
            continue
        quantized = qtuned[name] - qbase[name]
        quantized_delta = float(np.linalg.norm(quantized))
        cosine = float(quantized @ delta / (quantized_delta * norm)) if quantized_delta else 0.0
        rows.append(
            {
                "tensor": name,
                "delta_norm": norm,
                "delta_rms": norm / np.sqrt(delta.size),
                "q8_step_mean": q8_step(base[name]),
                "survival": quantized_delta / norm,
                "cosine": cosine,
                "q8_requantized": qbase[name].shape == base[name].shape
                and not np.array_equal(qbase[name], base[name]),
            }
        )
    survival = [row["survival"] for row in rows if row["q8_requantized"]]
    cosines = [row["cosine"] for row in rows if row["q8_requantized"]]
    summary = {
        "names_equal": names_equal,
        "tensors": len(base),
        "changed_tensors": len(rows),
        "changed_q8_tensors": len(survival),
        "survival_median": float(np.median(survival)) if survival else None,
        "survival_min": float(np.min(survival)) if survival else None,
        "cosine_median": float(np.median(cosines)) if cosines else None,
        "cosine_min": float(np.min(cosines)) if cosines else None,
        "delta_rms_over_step_median": float(
            np.median([row["delta_rms"] / row["q8_step_mean"] for row in rows])
        ) if rows else None,
        "rows": rows,
    }
    with open(out, "w") as file:
        json.dump(summary, file, indent=1)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=1))
    return 0 if names_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
