"""Merge a LoRA delta into the base NeMo checkpoint (T6.3), run in NeMo's own interpreter.

    <nemo python> -I merge.py --base <base.nemo> --delta <lora_delta.safetensors> --rank R --alpha A
                              --transcribe-cpp <checkout> --out <merged.nemo>

The trainer's math (`lora.LoraLinear`, M0's `lora_tiny.merged`): W ← W + (α/r)·B·A for every module
the delta names, on the model loaded exactly as training loaded it (`lora.load_model`). The delta's
own metadata must agree with the config's rank and alpha, every module must be an `nn.Linear` with
A of r×in and B of out×r, and the merged model's parameter names and shapes must equal the base's,
so the converter sees the base's tensors (D-31). Exit 0 merged, 2 refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora  # noqa: E402  (stdlib-only above its torch line)

REFUSED = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("--base", "--delta", "--transcribe-cpp", "--out"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        merged = merge(args.base, args.delta, args.rank, args.alpha, args.transcribe_cpp, args.out)
    except lora.TargetError as error:
        print(f"refused: {error}", file=sys.stderr)
        return REFUSED
    print(f"merged {merged} modules into {args.out}")
    return 0


def merge(base: Path, delta: Path, rank: int, alpha: float, transcribe_cpp: Path, out: Path) -> int:
    import torch
    from safetensors import safe_open
    from torch import nn

    with safe_open(str(delta), framework="pt") as handle:
        problems = lora.metadata_problems(handle.metadata() or {}, rank, alpha)
        if problems:
            raise lora.TargetError("; ".join(problems))
        state = {name: handle.get_tensor(name) for name in handle.keys()}
    modules = sorted({name.rsplit(".", 1)[0] for name in state})
    problems = lora.delta_problems(list(state), modules)
    if problems:
        raise lora.TargetError("; ".join(problems[:5]))
    model = lora.load_model(base, transcribe_cpp)
    model.eval()
    before = {name: tuple(p.shape) for name, p in model.state_dict().items()}
    scale = alpha / rank
    with torch.no_grad():
        for name in modules:
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise lora.TargetError(f"{name} is not a linear layer of the base")
            a, b = state[f"{name}.lora_a"], state[f"{name}.lora_b"]
            if tuple(a.shape) != (rank, module.in_features) or tuple(b.shape) != (module.out_features, rank):
                raise lora.TargetError(f"{name}: A {tuple(a.shape)} and B {tuple(b.shape)} do not fit "
                                       f"{module.out_features}×{module.in_features} at rank {rank}")
            module.weight += (b.to(module.weight.dtype) @ a.to(module.weight.dtype)) * scale
    after = {name: tuple(p.shape) for name, p in model.state_dict().items()}
    if after != before:
        raise lora.TargetError("the merged model's parameter names or shapes differ from the base's")
    tmp = out.with_name(out.stem + ".partial.nemo")
    model.save_to(str(tmp))
    tmp.replace(out)
    return len(modules)


if __name__ == "__main__":
    raise SystemExit(main())
