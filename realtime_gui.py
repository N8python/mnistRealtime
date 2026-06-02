#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mnist_realtime import DEFAULT_LIB, DEFAULT_MODEL, MnistGenerator


PIXELS_PER_IMAGE = 28 * 28
GRID_SIZE = 10
GRID_SLOTS = GRID_SIZE * GRID_SIZE
TILE_SCALE = 2
TILE_SIZE = 28 * TILE_SCALE
CANVAS_SIZE = GRID_SIZE * TILE_SIZE


@dataclass(frozen=True)
class GeneratedFrame:
    image: np.ndarray
    label: int
    seconds: float
    index: int
    created_at: float


class LabelState:
    def __init__(self, label: int):
        self._label = int(label)
        self._lock = threading.Lock()

    def get(self) -> int:
        with self._lock:
            return self._label

    def set(self, label: int) -> None:
        if not 0 <= int(label) <= 9:
            return
        with self._lock:
            self._label = int(label)


class ImageGrid:
    def __init__(self, pygame, label: int):
        self._pygame = pygame
        self.label = int(label)
        self.surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self.next_slot = 0
        self.filled_slots = 0
        self.clear(self.label)

    def clear(self, label: int) -> None:
        self.label = int(label)
        self.next_slot = 0
        self.filled_slots = 0
        self.surface.fill((0, 0, 0))

    def add(self, image: np.ndarray) -> None:
        row = self.next_slot // GRID_SIZE
        col = self.next_slot % GRID_SIZE
        rect = self._pygame.Rect(
            col * TILE_SIZE,
            row * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        )
        self.surface.blit(image_to_surface(self._pygame, image), rect.topleft)
        self.next_slot = (self.next_slot + 1) % GRID_SLOTS
        self.filled_slots = min(self.filled_slots + 1, GRID_SLOTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Qwen3 MNIST generator GUI.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--max-fps", type=int, default=120)
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="Render this many frames with SDL's dummy video driver, then exit.",
    )
    args = parser.parse_args()
    if not 0 <= args.label <= 9:
        parser.error("--label must be in 0..9")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.max_fps < 30:
        parser.error("--max-fps must be at least 30")
    return args


def generator_worker(
    args: argparse.Namespace,
    label_state: LabelState,
    frames: queue.Queue[GeneratedFrame],
    ready: threading.Event,
    stop: threading.Event,
    errors: queue.Queue[BaseException],
) -> None:
    try:
        with MnistGenerator(
            args.model,
            args.lib,
            threads=args.threads,
            temperature=args.temperature,
            seed=args.seed,
        ) as gen:
            ready.set()
            frame_index = 0
            while not stop.is_set():
                label = label_state.get()
                started = time.perf_counter()
                image = gen.generate(label)
                seconds = time.perf_counter() - started
                frame_index += 1
                frame = GeneratedFrame(
                    image=image,
                    label=label,
                    seconds=seconds,
                    index=frame_index,
                    created_at=time.perf_counter(),
                )
                try:
                    frames.put(frame, timeout=0.05)
                except queue.Full:
                    try:
                        frames.get_nowait()
                    except queue.Empty:
                        pass
                    frames.put_nowait(frame)
    except BaseException as exc:
        try:
            errors.put_nowait(exc)
        except queue.Full:
            pass
        ready.set()
        stop.set()


def image_to_surface(pygame, image: np.ndarray):
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return pygame.transform.scale(surface, (TILE_SIZE, TILE_SIZE))


def render_text(pygame, screen, font, text: str, pos, color) -> None:
    screen.blit(font.render(text, True, color), pos)


def draw_button(pygame, screen, font, rect, label: str, selected: bool) -> None:
    fill = (238, 239, 242) if selected else (42, 45, 50)
    border = (255, 255, 255) if selected else (80, 86, 94)
    text = (10, 12, 14) if selected else (228, 231, 235)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=6)
    glyph = font.render(label, True, text)
    screen.blit(glyph, glyph.get_rect(center=rect.center))


def build_button_rects(pygame, panel_x: int, panel_y: int) -> dict[int, object]:
    rects = {}
    button_w = 52
    button_h = 42
    gap = 10
    grid_y = panel_y + 82
    for digit in range(10):
        row = digit // 2
        col = digit % 2
        rects[digit] = pygame.Rect(
            panel_x + col * (button_w + gap),
            grid_y + row * (button_h + gap),
            button_w,
            button_h,
        )
    return rects


def draw_app(
    pygame,
    screen,
    fonts: dict[str, object],
    last_frame: GeneratedFrame | None,
    image_grid: ImageGrid,
    label_state: LabelState,
    button_rects: dict[int, object],
    stats: dict[str, float],
    canvas_rect,
    panel_x: int,
    args: argparse.Namespace,
) -> None:
    screen.fill((13, 14, 16))
    selected_label = label_state.get()

    render_text(
        pygame,
        screen,
        fonts["title"],
        "Qwen3 MNIST realtime",
        (24, 22),
        (244, 245, 247),
    )

    pygame.draw.rect(screen, (0, 0, 0), canvas_rect, border_radius=8)
    screen.blit(image_grid.surface, canvas_rect.topleft)
    pygame.draw.rect(screen, (82, 88, 96), canvas_rect, width=1, border_radius=8)

    render_text(
        pygame,
        screen,
        fonts["label"],
        f"label {selected_label}",
        (panel_x, 84),
        (244, 245, 247),
    )

    for digit, rect in button_rects.items():
        draw_button(pygame, screen, fonts["button"], rect, str(digit), digit == selected_label)

    metric_y = button_rects[8].bottom + 34
    if last_frame is None:
        lines = [
            "warming up",
            f"temp {args.temperature:.2f}",
            f"threads {args.threads}",
        ]
    else:
        gen_ms = last_frame.seconds * 1000.0
        tok_s = PIXELS_PER_IMAGE / max(last_frame.seconds, 1e-9)
        fps = stats.get("fps_ema", 0.0)
        lines = [
            f"{fps:5.1f} img/s",
            f"{gen_ms:5.1f} ms/gen",
            f"{tok_s / 1000.0:5.1f}k tok/s",
            f"frame {int(stats.get('frames', 0))}",
            f"grid {image_grid.filled_slots:3d}/{GRID_SLOTS}",
        ]
    for i, line in enumerate(lines):
        color = (220, 224, 229) if i == 0 else (166, 172, 181)
        render_text(pygame, screen, fonts["metric"], line, (panel_x, metric_y + i * 28), color)


def run() -> None:
    args = parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.headless_frames > 0:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        import pygame

    pygame.init()
    pygame.font.init()

    canvas_size = CANVAS_SIZE
    width = canvas_size + 312
    height = max(canvas_size + 104, 560)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Qwen3 MNIST realtime")

    fonts = {
        "title": pygame.font.SysFont("Menlo", 24, bold=True),
        "label": pygame.font.SysFont("Menlo", 22, bold=True),
        "button": pygame.font.SysFont("Menlo", 20, bold=True),
        "metric": pygame.font.SysFont("Menlo", 18),
    }
    canvas_rect = pygame.Rect(24, 80, canvas_size, canvas_size)
    panel_x = canvas_rect.right + 32
    button_rects = build_button_rects(pygame, panel_x, canvas_rect.y)

    label_state = LabelState(args.label)
    frames: queue.Queue[GeneratedFrame] = queue.Queue(maxsize=64)
    errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
    ready = threading.Event()
    stop = threading.Event()
    worker = threading.Thread(
        target=generator_worker,
        args=(args, label_state, frames, ready, stop, errors),
        name="mnist-generator",
        daemon=True,
    )
    worker.start()

    ready.wait()
    if not errors.empty():
        pygame.quit()
        raise errors.get()

    clock = pygame.time.Clock()
    last_frame: GeneratedFrame | None = None
    image_grid = ImageGrid(pygame, args.label)
    stats: dict[str, float] = {"frames": 0.0, "fps_ema": 0.0}
    first_display_at: float | None = None
    previous_display_at: float | None = None

    try:
        while not stop.is_set():
            selected_before_events = label_state.get()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop.set()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        stop.set()
                    elif pygame.K_0 <= event.key <= pygame.K_9:
                        label_state.set(event.key - pygame.K_0)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for digit, rect in button_rects.items():
                        if rect.collidepoint(event.pos):
                            label_state.set(digit)
                            break

            selected_label = label_state.get()
            if selected_label != selected_before_events:
                image_grid.clear(selected_label)
                last_frame = None
                while True:
                    try:
                        frames.get_nowait()
                    except queue.Empty:
                        break

            if not errors.empty():
                raise errors.get()

            try:
                next_frame = frames.get_nowait()
            except queue.Empty:
                pass
            else:
                if next_frame.label == label_state.get():
                    last_frame = next_frame
                    now = time.perf_counter()
                    image_grid.add(last_frame.image)
                    if first_display_at is None:
                        first_display_at = now
                    if previous_display_at is not None:
                        dt = max(now - previous_display_at, 1e-9)
                        instant_fps = 1.0 / dt
                        current = stats["fps_ema"]
                        stats["fps_ema"] = instant_fps if current == 0.0 else current * 0.85 + instant_fps * 0.15
                    previous_display_at = now
                    stats["frames"] += 1.0

            draw_app(
                pygame,
                screen,
                fonts,
                last_frame,
                image_grid,
                label_state,
                button_rects,
                stats,
                canvas_rect,
                panel_x,
                args,
            )
            pygame.display.flip()

            if args.headless_frames > 0 and stats["frames"] >= args.headless_frames:
                stop.set()

            clock.tick(args.max_fps)
    finally:
        stop.set()
        worker.join(timeout=2.0)
        pygame.quit()

    if args.headless_frames > 0:
        if first_display_at is None or previous_display_at is None:
            elapsed = 0.0
        else:
            elapsed = max(previous_display_at - first_display_at, 1e-9)
        frames_rendered = int(stats["frames"])
        interval_count = max(frames_rendered - 1, 0)
        fps = interval_count / elapsed if elapsed > 0 else 0.0
        last_ms = 0.0 if last_frame is None else last_frame.seconds * 1000.0
        last_tok_s = 0.0 if last_frame is None else PIXELS_PER_IMAGE / max(last_frame.seconds, 1e-9)
        print(f"frames_rendered={frames_rendered}")
        print(f"display_elapsed_seconds={elapsed:.6f}")
        print(f"display_fps={fps:.3f}")
        print(f"last_generation_ms={last_ms:.3f}")
        print(f"last_tokens_per_second={last_tok_s:.3f}")


if __name__ == "__main__":
    run()
