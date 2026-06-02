#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mnist_realtime import MnistGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test realtime MNIST C bindings.")
    parser.add_argument("--model", type=Path, default=Path("Qwen3-2M-MNIST-GRPO.raw"))
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("realtime_10_samples.png"))
    return parser.parse_args()


def save_grid(images: list[np.ndarray], labels: list[int], out: Path) -> None:
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.1, 1.4))
    if cols == 1:
        axes = [axes]
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(str(label), fontsize=9)
        ax.axis("off")
    plt.tight_layout(pad=0.15)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labels = [i % 10 for i in range(args.calls)]

    load_t0 = time.perf_counter()
    with MnistGenerator(
        args.model,
        threads=args.threads,
        temperature=args.temperature,
        seed=args.seed,
    ) as gen:
        load_seconds = time.perf_counter() - load_t0
        t0 = time.perf_counter()
        images = gen.generate_many(labels)
        gen_seconds = time.perf_counter() - t0

    pixels_per_image = 28 * 28
    total_tokens = args.calls * pixels_per_image
    print(f"model_load_seconds={load_seconds:.6f}")
    print(f"calls={args.calls}")
    print(f"generation_seconds={gen_seconds:.6f}")
    print(f"seconds_per_image={gen_seconds / args.calls:.6f}")
    print(f"images_per_second={args.calls / gen_seconds:.3f}")
    print(f"tokens_per_second={total_tokens / gen_seconds:.3f}")

    save_grid(images, labels, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
