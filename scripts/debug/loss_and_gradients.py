"""Analyse loss sensitivity to output scaling and gradient flow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from scripts.config import load_config
from scripts.core.model import create_model


def _synthesize_batch(batch_size: int, in_channels: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randn(batch_size, in_channels, height, width)
    targets = torch.randn(batch_size)
    return inputs, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Loss and gradient diagnostics")
    parser.add_argument("config", help="Config file")
    parser.add_argument("--checkpoint", help="Optional checkpoint to load")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scales", type=float, nargs="*", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    config = load_config(args.config)
    model = create_model(config["model_config"])
    if args.checkpoint:
        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
    model.train()

    in_channels = config["model_config"].get("in_channels", 6)
    criterion = nn.HuberLoss(delta=args.delta)
    inputs, targets = _synthesize_batch(args.batch_size, in_channels, args.height, args.width)

    for scale in args.scales:
        model.zero_grad(set_to_none=True)
        outputs = model(inputs) * scale
        loss = criterion(outputs, targets)
        loss.backward()
        grad_norm = torch.sqrt(
            sum((p.grad.detach().float() ** 2).sum() for p in model.parameters() if p.grad is not None)
        )
        print(f"Scale={scale:.2f} -> loss={loss.item():.6f}, grad_norm={grad_norm.item():.6f}")


if __name__ == "__main__":
    main()
