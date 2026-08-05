"""
Disclaimer: This script was written by code prompting with the Claude Code Sonnet & Opus 5.


Static scenes for screenshots.

Places 3 blue + 3 yellow robots and the ball in hand-picked formations that make
cooperative play readable (open pass lane, overlapping run, wide build-up shape,
2v1 break). Nothing is simulated: the frame is written straight into the env and
only `render()` is called, so the picture never moves.

Uses SSLDynamicRobots (our env), so robots are drawn exactly as in training:
team-colored bodies and 4-dot id tags from `myenvs`, plus its enlarged ball.

Blue attacks the right goal (+x). Poses are authored on a 1.5 x 1.3 reference
pitch and scaled to the chosen field, so layouts survive any --field-type.
Screen y grows downward, i.e. +y is the lower half.

Usage:
    python scene.py                       # window, LEFT/RIGHT switch scene, S saves, ESC quits
    python scene.py --scene overlap
    python scene.py --save-all            # headless, writes every scene to --out-dir
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import pygame
import tyro

import myenvs  # noqa: F401  # registers the env and patches in the colored robot bodies
import rsoccer_gym.Render.ball as render_ball
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render import SSLRenderField, SSLRobot

# Pose tables below are written in these units; _to_field() rescales them.
REF_LENGTH = 1.5
REF_WIDTH = 1.3

# (x, y, facing target) per robot; the ball is placed at the carrier's dribbler.
# Order is blue 0,1,2 then yellow 0,1,2.
Pose = Tuple[float, float, Tuple[float, float]]
GOAL_BLUE = (0.75, 0.0)  # goal blue attacks
GOAL_YELLOW = (-0.75, 0.0)

SCENES: Dict[str, Dict] = {
    "pass": {
        # Both defenders commit to the ball side; blue 1 is free on the far side
        # with an open lane. The pass is the obvious move.
        "doc": "open pass lane: press on the ball carrier, receiver free wide",
        "carrier": 0,
        "blue": [
            (-0.02, -0.22, (0.42, 0.10)),  # carrier, already turned to the receiver
            (0.42, 0.10, GOAL_BLUE),  # receiver, facing goal
            (-0.30, -0.05, (0.05, -0.20)),  # trailing support
        ],
        "yellow": [
            (0.16, -0.28, (-0.02, -0.22)),  # presses the carrier
            (0.34, -0.25, (0.05, -0.20)),  # doubles up on the same side
            (0.65, 0.00, (0.05, -0.20)),  # keeper
        ],
    },
    "overlap": {
        # Carrier is blocked inside; team-mate runs outside past them, third
        # robot waits central for the cutback.
        "doc": "overlapping run down the wing plus a cutback option",
        "carrier": 0,
        "blue": [
            (0.10, 0.35, GOAL_BLUE),  # carrier on the wing
            (0.30, 0.52, GOAL_BLUE),  # overlapping outside
            (0.20, 0.00, (0.16, 0.36)),  # central, waiting for the cutback
        ],
        "yellow": [
            (0.30, 0.26, (0.10, 0.35)),  # blocks the inside
            (0.45, 0.15, (0.16, 0.36)),  # covers the centre
            (0.65, 0.05, (0.16, 0.36)),  # keeper
        ],
    },
    "spread": {
        # Defence is compressed centrally, attackers hold maximum width: the
        # shape itself is the cooperation.
        "doc": "wide build-up triangle against a compact defence",
        "carrier": 2,
        "blue": [
            (0.05, -0.45, (-0.40, 0.00)),  # wide left, showing for the pass
            (0.05, 0.45, (-0.40, 0.00)),  # wide right, showing for the pass
            (-0.40, 0.00, GOAL_BLUE),  # carrier, deep
        ],
        "yellow": [
            (-0.10, 0.00, (-0.40, 0.00)),  # steps to the ball
            (0.20, -0.15, (-0.33, 0.00)),  # narrow cover
            (0.65, 0.00, (-0.40, 0.00)),  # keeper
        ],
    },
    "break": {
        # Two attackers against one defender who cannot cover both.
        "doc": "2v1 counter-attack, defender caught between carrier and runner",
        "carrier": 0,
        "blue": [
            (0.30, -0.10, GOAL_BLUE),  # carrier driving at the defender
            (0.38, 0.25, GOAL_BLUE),  # runner in support
            (-0.20, 0.05, GOAL_BLUE),  # trailing
        ],
        "yellow": [
            (0.52, 0.02, (0.30, -0.10)),  # lone defender
            (0.05, -0.30, GOAL_BLUE),  # recovering from behind
            (0.65, -0.03, (0.30, -0.10)),  # keeper
        ],
    },
}


def _theta(x: float, y: float, target: Tuple[float, float]) -> float:
    """Heading in degrees so the robot faces `target`.

    World -> screen is a pure scale, and the renderer draws the heading with
    (cos, sin) in screen space, so atan2 on world deltas is already correct.
    """
    return math.degrees(math.atan2(target[1] - y, target[0] - x))


def enlarge_robots(factor: float) -> None:
    """Draw robots and the ball `factor` times bigger, for legible screenshots.

    SSLRobot's body radius is `size` and its team/id tag radii are derived from
    `scale`, so both have to grow for the whole marker to scale. Call once.
    """
    if factor == 1.0:
        return
    base_init = SSLRobot.__init__

    def scaled_init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self.size *= factor
        self.scale *= factor

    SSLRobot.__init__ = scaled_init
    render_ball.Ball.radius *= factor


def build_frame(name: str, field, robot_scale: float = 1.0) -> Frame:
    scene = SCENES[name]

    def to_field(x, y):
        return x * field.length / REF_LENGTH, y * field.width / REF_WIDTH

    frame = Frame()
    for team, yellow in (("blue", False), ("yellow", True)):
        robots = frame.robots_yellow if yellow else frame.robots_blue
        for i, (rx, ry, target) in enumerate(scene[team]):
            x, y = to_field(rx, ry)
            robots[i] = Robot(
                yellow=yellow, id=i, x=x, y=y, theta=_theta(x, y, to_field(*target))
            )

    # Ball sits against the carrier's front, so possession reads at any field size.
    # Uses the *drawn* radii, which enlarge_robots may have inflated.
    carrier = frame.robots_blue[scene["carrier"]]
    reach = robot_scale * field.rbt_radius + render_ball.Ball.radius
    frame.ball = Ball(
        x=carrier.x + reach * math.cos(math.radians(carrier.theta)),
        y=carrier.y + reach * math.sin(math.radians(carrier.theta)),
    )
    _validate(name, frame, field, robot_scale)
    return frame


def _validate(name: str, frame: Frame, field, robot_scale: float) -> None:
    """Poses must be on the pitch and not stacked on top of each other."""
    bodies = list(frame.robots_blue.values()) + list(frame.robots_yellow.values())
    points = [(b.x, b.y) for b in bodies]
    r = robot_scale * field.rbt_radius
    for x, y in points:
        assert abs(x) < field.length / 2 - r, f"{name}: x={x:.2f} off pitch"
        assert abs(y) < field.width / 2 - r, f"{name}: y={y:.2f} off pitch"
    for i, (x0, y0) in enumerate(points):
        for x1, y1 in points[i + 1 :]:
            d = math.hypot(x1 - x0, y1 - y0)
            # 2.2r not 2r: a visible gap, so bodies don't read as one blob
            assert d > 2.2 * r, f"{name}: robots too close ({d:.3f} m apart)"


@dataclass
class SceneArgs:
    scene: str = "pass"
    """which scene to show (LEFT/RIGHT switch at runtime)"""
    save_all: bool = False
    """render every scene to --out-dir without opening a window"""
    out_dir: str = "screenshots"
    """where S / --save-all write PNGs"""
    field_type: int = 2
    """SSL field size: 0=(12x9), 1=(9x6), 2=(6x4) -- same meaning as in SSLDynamicRobots"""
    robot_scale: float = 1.25
    """draw robots and ball this many times bigger than life size (rendering only)"""
    scale: int = 300
    """output resolution in pixels per metre (rSoccer's own default is 100)"""
    supersample: int = 2
    """render this many times above --scale, then downsample; 1 disables antialiasing"""
    fullscreen: bool = False
    """open the window fullscreen instead of resizable"""


def _make_env(render_mode: str, field_type: int, robot_scale: float, scale: int, supersample: int):
    # Everything (field, robots, ball) is drawn from this one px/m factor, so
    # raising it is a straight resolution increase, not a zoom.
    SSLRenderField._scale = scale * supersample
    env = gym.make(
        "SSLDynamicRobots-v0",
        render_mode=render_mode,
        field_type=field_type,
        n_robots_blue=3,
        n_robots_yellow=3,
    ).unwrapped
    # after gym.make: it lazily imports myenvs.DynamicRobots, which sets the
    # ball radius itself and would otherwise overwrite our scaling
    enlarge_robots(robot_scale)
    return env


def _show(env, name: str, robot_scale: float) -> None:
    env.frame = build_frame(name, env.field, robot_scale)
    env.render()


def open_window(env, fullscreen: bool, caption: str = "SSL scene") -> None:
    """Draw into an offscreen surface at full render resolution and show a
    downscaled copy, so the window fits the desktop and gets antialiased edges.

    The env only calls set_mode itself while window_surface is None, so handing
    it the offscreen surface keeps it out of the display entirely.
    """
    pygame.init()
    pygame.display.init()
    pygame.display.set_caption(caption)
    w, h = env.window_size
    desktop = pygame.display.Info()
    fit = min(1.0, 0.9 * desktop.current_w / w, 0.9 * desktop.current_h / h)
    flags = pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE
    pygame.display.set_mode((int(w * fit), int(h * fit)), flags)
    env.window_surface = pygame.Surface((w, h))


def present(env) -> None:
    """Blit the offscreen render onto the window, fitted and aspect-preserving."""
    screen = pygame.display.get_surface()
    sw, sh = screen.get_size()
    w, h = env.window_surface.get_size()
    fit = min(sw / w, sh / h)
    size = (int(w * fit), int(h * fit))
    screen.fill((0, 0, 0))
    screen.blit(
        pygame.transform.smoothscale(env.window_surface, size),
        ((sw - size[0]) // 2, (sh - size[1]) // 2),
    )
    pygame.display.flip()


def _save(env, name: str, out_dir: str, supersample: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"scene_{name}.png")
    surface = env.window_surface
    if supersample > 1:
        # pygame draws without antialiasing, so downsampling an oversized render
        # is what smooths the robot/ball edges
        w, h = surface.get_size()
        surface = pygame.transform.smoothscale(surface, (w // supersample, h // supersample))
    pygame.image.save(surface, path)
    return path


def main(args: SceneArgs) -> None:
    names = list(SCENES)
    assert args.scene in SCENES, f"unknown scene {args.scene!r}, pick from {names}"

    if args.save_all:
        env = _make_env(
            "rgb_array", args.field_type, args.robot_scale, args.scale, args.supersample
        )
        for name in names:
            _show(env, name, args.robot_scale)
            print(_save(env, name, args.out_dir, args.supersample))
        env.close()
        return

    env = _make_env("human", args.field_type, args.robot_scale, args.scale, args.supersample)
    open_window(env, args.fullscreen)

    idx = names.index(args.scene)
    print(f"{names[idx]}: {SCENES[names[idx]]['doc']}")
    running = True
    while running:
        moved: Optional[int] = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_RIGHT:
                    moved = 1
                elif event.key == pygame.K_LEFT:
                    moved = -1
                elif event.key == pygame.K_s:
                    print(_save(env, names[idx], args.out_dir, args.supersample))
        if moved is not None:
            idx = (idx + moved) % len(names)
            print(f"{names[idx]}: {SCENES[names[idx]]['doc']}")
        _show(env, names[idx], args.robot_scale)
        present(env)
    env.close()


if __name__ == "__main__":
    main(tyro.cli(SceneArgs))
