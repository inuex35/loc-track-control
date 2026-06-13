"""Render an example simulation to an animated GIF (headless).

Useful for sharing the result where an interactive pygame window isn't
available (CI, web, docs). Renders the integrated track-and-avoid demo by
default.

Usage:
    python examples/make_gif.py [output.gif] [--frames N] [--scale S]

Example:
    python examples/make_gif.py media/integrated.gif --frames 90 --scale 0.6
"""

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from PIL import Image

import examples.bicycle_sim as bsim
import examples.integrated_sim as sim


def render_frames(frames: int, seed: int) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    mpc = sim.make_controller()
    model = mpc.model

    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((bsim.WIDTH, bsim.HEIGHT))
    font = pygame.font.SysFont("monospace", 16)

    ego = np.array([-22.0, 0.0, 0.0, 0.0])
    goal_xy = np.array([22.0, 0.0])
    obstacles = [
        sim.Obstacle((-2.0, -12.0), (0.3, 2.2)),
        sim.Obstacle((6.0, 12.0), (-0.4, -2.0)),
    ]

    images = []
    for _ in range(frames):
        for obs in obstacles:
            obs.step(sim.DT, rng)
        predictions = [
            sim.predict_horizon(sim.estimate_state(o.history, sim.DT), sim.HORIZON, sim.DT)
            for o in obstacles if len(o.history) > 0
        ]
        heading = np.arctan2(goal_xy[1] - ego[1], goal_xy[0] - ego[0])
        xref = np.array([goal_xy[0], goal_xy[1], heading, 0.0])
        plan = mpc.solve(ego, xref, obstacles=predictions)
        ego = model.step(ego, plan.u0)

        sim.draw_scene(surface, font, ego, goal_xy, obstacles, predictions, plan)
        # pygame surface (W, H, 3) -> PIL image (H, W, 3).
        arr = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        images.append(Image.fromarray(arr))

    pygame.quit()
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="integrated.gif")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    images = render_frames(args.frames, args.seed)
    if args.scale != 1.0:
        w = int(bsim.WIDTH * args.scale)
        h = int(bsim.HEIGHT * args.scale)
        images = [im.resize((w, h), Image.BILINEAR) for im in images]
    # Adaptive palette keeps the GIF small.
    images = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in images]

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
