"""Utility to probe model output scales and bias initialisation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.config import load_config
from scripts.core.model import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model output range")
    parser.add_argument("config", help="Config file (YAML/JSON)")
    parser.add_argument("--checkpoint", help="Optional checkpoint to load")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-shape", type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    config = load_config(args.config)
    model = create_model(config["model_config"])
    if args.checkpoint:
        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
    model.eval()

    h, w = args.input_shape
    outputs = []
    with torch.no_grad():
        for _ in range(args.batches):
            dummy = torch.randn(args.batch_size, config["model_config"].get("in_channels", 6), h, w)
            out = model(dummy)
            outputs.append(out)
    concatenated = torch.cat(outputs)
    print("Output statistics (unnormalised):")
    print(f"  mean: {concatenated.mean().item():.6f}")
    print(f"  std: {concatenated.std().item():.6f}")
    print(f"  min: {concatenated.min().item():.6f}")
    print(f"  max: {concatenated.max().item():.6f}")


if __name__ == "__main__":
    main()
