"""Render an example simulation to an animated GIF (headless).

Useful for sharing the result where an interactive pygame window isn't
available (CI, web, docs). Works with any demo that exposes an
``iter_frames(frames, seed)`` generator yielding a rendered pygame surface per
frame (currently ``integrated_sim`` and ``loc_track_control_sim``).

Usage:
    python examples/make_gif.py [output.gif] [--demo NAME] [--frames N] [--scale S]

Examples:
    python examples/make_gif.py integrated.gif --frames 90 --scale 0.6
    python examples/make_gif.py loctrack.gif --demo loc_track_control_sim --frames 300
"""

import argparse
import importlib
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from PIL import Image

DEMOS = ["integrated_sim", "loc_track_control_sim"]


def render_frames(demo: str, frames: int, seed: int) -> list[Image.Image]:
    """Drive ``demo.iter_frames`` and capture each surface as a PIL image."""
    mod = importlib.import_module(f"examples.{demo}")
    images = []
    for surface in mod.iter_frames(frames, seed):
        # pygame surface (W, H, 3) -> PIL image (H, W, 3).
        arr = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        images.append(Image.fromarray(arr))
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="demo.gif")
    parser.add_argument("--demo", choices=DEMOS, default="integrated_sim")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--colors", type=int, default=64)
    args = parser.parse_args()

    images = render_frames(args.demo, args.frames, args.seed)
    if args.scale != 1.0:
        w, h = images[0].size
        size = (int(w * args.scale), int(h * args.scale))
        images = [im.resize(size, Image.BILINEAR) for im in images]
    # Adaptive palette keeps the GIF small.
    images = [im.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for im in images]

    out = args.output
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True,
    )
    size_kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({len(images)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
