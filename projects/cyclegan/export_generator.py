"""Extract a deployable photo-to-art generator from a training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export generator_B from a trusted CycleGAN checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Training checkpoint (.pt)")
    parser.add_argument("output", type=Path, help="Output generator state dict (.pt)")
    parser.add_argument(
        "--generator",
        default="generator_B",
        choices=("generator_A", "generator_B"),
        help="Generator to extract; generator_B is photo-to-art in this project.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # weights_only=False is needed for the notebook's plots and optimizer data.
    # Only run this utility on checkpoints you created or otherwise trust.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model_state = checkpoint.get("model_state_dict", checkpoint)
    prefix = f"{args.generator}."
    generator_state = {
        key.removeprefix(prefix): value.detach().cpu()
        for key, value in model_state.items()
        if key.startswith(prefix)
    }

    if not generator_state:
        raise ValueError(f"No parameters beginning with {prefix!r} were found")

    # Validate that the exported state exactly matches the deployed architecture.
    Generator().load_state_dict(generator_state, strict=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generator_state, args.output)
    print(f"Saved {args.generator} to {args.output}")


if __name__ == "__main__":
    main()
