"""Tiny LoRA fine-tune of a Parakeet NeMo checkpoint, merged back into its weights.

Runs in transcribe.cpp's Parakeet environment (it needs NeMo):

    uv run --project $TRANSCRIBE_CPP/scripts/envs/parakeet \
        spikes/m0/lora_tiny.py --base base.nemo --train train.json \
        --out merged.nemo --adapter lora.pt

LoRA (rank 16, D-14) wraps every encoder attention and feed-forward linear and
the joint's linears. The predictor is an LSTM, which has no linear layers to
wrap, so it stays frozen. After training, each wrapped weight becomes
W + (alpha / rank) · B·A and the wrapper is removed, so the merged checkpoint
has exactly the base's parameter names and shapes, which the converter needs.
Trains on CPU (NeMo's RNN-T loss is unreliable on MPS) with dropout and
SpecAugment off. A seed fixes the data order and LoRA init, but multithreaded
CPU sums make runs differ in the last bits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
from torch import nn

TRANSCRIBE_CPP = Path(os.environ.get("TRANSCRIBE_CPP", Path.home() / "hex_healing_work/transcribe.cpp"))


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    def merged(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight += (self.b @ self.a) * self.scale
        return self.base


def targets(model) -> list[str]:
    """Names of the linears LoRA wraps: encoder attention and feed-forward, joint."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        in_encoder = name.startswith("encoder.layers.") and (
            ".self_attn.linear_" in name or ".feed_forward" in name
        )
        in_joint = name.startswith("joint.")
        if in_encoder or in_joint:
            names.append(name)
    return names


def swap(model, name: str, replacement: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, replacement)


def load(base: Path):
    sys.path.insert(0, str(TRANSCRIBE_CPP / "scripts"))
    from dump_reference_parakeet_nemo import _patch_conformer_for_offline

    _patch_conformer_for_offline(force_regular_att_style=True)
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

    try:
        return ASRModel.restore_from(str(base), map_location="cpu")
    except TypeError as error:
        if "abstract class" not in str(error):
            raise
        return EncDecRNNTBPEModel.restore_from(str(base), map_location="cpu")


def batches(clips, root: Path, tokenizer, batch_size: int, rng: random.Random):
    order = list(range(len(clips)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            picked = [clips[index] for index in order[start : start + batch_size]]
            audio = [soundfile.read(root / clip["audio"], dtype="float32")[0] for clip in picked]
            tokens = [tokenizer.text_to_ids(clip["expected"]) for clip in picked]
            signal = np.zeros((len(audio), max(map(len, audio))), dtype=np.float32)
            for row, samples in enumerate(audio):
                signal[row, : len(samples)] = samples
            transcript = np.zeros((len(tokens), max(map(len, tokens))), dtype=np.int64)
            for row, ids in enumerate(tokens):
                transcript[row, : len(ids)] = ids
            yield (
                torch.from_numpy(signal),
                torch.tensor([len(samples) for samples in audio]),
                torch.from_numpy(transcript),
                torch.tensor([len(ids) for ids in tokens]),
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True, help="benchmark manifest")
    parser.add_argument("--out", type=Path, required=True, help="merged .nemo")
    parser.add_argument("--adapter", type=Path, required=True, help="LoRA weights (.pt)")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model = load(args.base)
    model.eval()  # no dropout, no SpecAugment
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    names = targets(model)
    wrappers = {}
    for name in names:
        wrapper = LoraLinear(model.get_submodule(name), args.rank, args.alpha)
        swap(model, name, wrapper)
        wrappers[name] = wrapper
    trainable = [p for w in wrappers.values() for p in (w.a, w.b)]
    print(f"lora: {len(names)} linears, {sum(p.numel() for p in trainable):,} parameters")

    manifest = json.loads(args.train.read_text())
    data = batches(manifest["clips"], args.train.parent, model.tokenizer, args.batch_size,
                   random.Random(args.seed))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        signal, signal_len, transcript, transcript_len = next(data)
        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        decoder, target_len, _ = model.decoder(targets=transcript, target_length=transcript_len)
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss = model.loss(log_probs=joint, targets=transcript, input_lengths=encoded_len,
                          target_lengths=target_len)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:4} loss {loss.item():.4f} ({time.monotonic() - started:.0f}s)",
                  flush=True)

    args.adapter.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: {"a": w.a.detach(), "b": w.b.detach(), "scale": w.scale}
                for name, w in wrappers.items()}, args.adapter)
    for name, wrapper in wrappers.items():
        swap(model, name, wrapper.merged())
    assert all(not isinstance(m, LoraLinear) for m in model.modules())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(args.out))
    print(f"merged: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
