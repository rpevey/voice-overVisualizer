#!/usr/bin/env python3
"""
classic_visualizer.py
A "classic media player" style visualizer: bright geometric shapes and
psychedelic pops that burst on beats, plus a slowly rotating, bass-reactive
"mandala" ring for continuous motion between hits. Renders to a transparent
PNG sequence and/or an alpha video for DaVinci Resolve.

Example:
  python classic_visualizer.py --input song.wav --width 1080 --height 1920 \
      --shape-set mixed --symmetry 6 --trails --output-format both
"""
import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_analysis import load_audio, compute_spectrum_frames, compute_onset_envelope
from palette import parse_color, sample_palette
from render_utils import ensure_dir, save_png_sequence_frame, encode_video_from_pngs
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


SHAPE_SIDES = {
    "circle": 0,       # 0 = special-cased ellipse
    "triangle": 3,
    "square": 4,
    "pentagon": 5,
    "hexagon": 6,
    "star5": -5,
    "star6": -6,
}


def draw_shape(draw, kind, cx, cy, size, rotation, color):
    sides = SHAPE_SIDES.get(kind, 0)
    if kind == "circle" or sides == 0:
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        return
    if sides > 0:
        pts = []
        for i in range(sides):
            a = rotation + i * (2 * math.pi / sides)
            pts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
        draw.polygon(pts, fill=color)
        return
    # star
    n_points = -sides
    inner = size * 0.45
    pts = []
    for i in range(n_points * 2):
        r = size if i % 2 == 0 else inner
        a = rotation + i * (math.pi / n_points)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    draw.polygon(pts, fill=color)


class Particle:
    __slots__ = ("kind", "cx", "cy", "birth", "lifetime", "max_size", "rot_speed",
                 "base_rot", "rgb")

    def __init__(self, kind, cx, cy, birth, lifetime, max_size, rot_speed, base_rot, rgb):
        self.kind = kind
        self.cx = cx
        self.cy = cy
        self.birth = birth
        self.lifetime = lifetime
        self.max_size = max_size
        self.rot_speed = rot_speed
        self.base_rot = base_rot
        self.rgb = rgb  # (r, g, b) resolved once at spawn time from the palette

    def state(self, frame):
        """Return (alive, size, alpha, rotation, color) for this frame."""
        age = frame - self.birth
        t = age / self.lifetime
        if t < 0 or t > 1:
            return False, 0, 0, 0, (0, 0, 0, 0)
        # ease-out grow, hold, ease-in fade (classic "pop")
        if t < 0.2:
            size_t = t / 0.2
            size = self.max_size * (size_t ** 0.5)
            alpha = 255 * (t / 0.2)
        elif t < 0.55:
            size = self.max_size
            alpha = 255
        else:
            fade_t = (t - 0.55) / 0.45
            size = self.max_size * (1 + 0.15 * fade_t)  # slight continued expansion while fading
            alpha = 255 * (1 - fade_t)
        rotation = self.base_rot + self.rot_speed * age
        r, g, b = self.rgb
        color = (r, g, b, int(alpha))
        return True, size, alpha, rotation, color


def spawn_burst(particles, frame, W, H, cx, cy, symmetry, strength, shape_set, color_t,
                 gradient_spread, palette, min_r, max_r, size_min, size_max,
                 lifetime_frames, rng):
    """
    color_t: base palette position (0-1) for this burst, derived from beat strength.
    gradient_spread: how far each individual shape's color drifts from color_t based
                      on its position in the burst (0 = all shapes identical color).
    """
    kinds = shape_set
    base_angle = rng.uniform(0, 2 * math.pi)
    for k in range(symmetry):
        angle = base_angle + k * (2 * math.pi / symmetry) + rng.uniform(-0.12, 0.12)
        r = rng.uniform(min_r, max_r) * (0.4 + 0.6 * strength)
        px, py = cx + r * math.cos(angle), cy + r * math.sin(angle)
        size = (size_min + (size_max - size_min) * strength) * rng.uniform(0.7, 1.15)
        kind = rng.choice(kinds)
        # spread this shape's color around the burst's base position, by its
        # order in the ring (k / symmetry), not randomly -- gives a coherent
        # gradient fan across the burst rather than noisy per-shape jitter.
        offset = (k / max(1, symmetry - 1) - 0.5) * gradient_spread if symmetry > 1 else 0.0
        t = float(np.clip(color_t + offset, 0.0, 1.0))
        r_, g_, b_, _ = sample_palette(t, palette=palette)
        rot_speed = rng.uniform(-0.05, 0.05)
        base_rot = rng.uniform(0, 2 * math.pi)
        lifetime = int(lifetime_frames * rng.uniform(0.85, 1.2))
        particles.append(Particle(kind, px, py, frame, lifetime, size, rot_speed, base_rot,
                                   (r_, g_, b_)))


def draw_mandala(draw, frame, fps, W, H, bass, mid, points, color_t, gradient_spread,
                  palette, rotation_speed):
    cx, cy = W / 2, H / 2
    base_r = min(W, H) * (0.12 + 0.05 * bass)
    t = frame / fps
    for i in range(points):
        angle = t * rotation_speed + i * (2 * math.pi / points)
        r = base_r
        px, py = cx + r * math.cos(angle), cy + r * math.sin(angle)
        size = min(W, H) * (0.012 + 0.02 * mid)
        offset = (i / max(1, points - 1) - 0.5) * gradient_spread
        pt = float(np.clip(color_t + offset, 0.0, 1.0))
        color = sample_palette(pt, palette=palette, alpha=160)
        draw.ellipse([px - size, py - size, px + size, py + size], fill=color)

def render_frame_no_trails(f, shapes, bass, mid, *, args, W, H, SS, bg_color, png_dir):
    sharp_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sharp_layer)

    if args.mandala:
        draw_mandala(draw, f, args.fps, W, H, bass, mid, args.mandala_points, bass,
                     args.gradient_spread, args.palette, rotation_speed=0.6)
    for kind, cx, cy, size, rotation, color in shapes:
        draw_shape(draw, kind, cx, cy, size, rotation, color)

    out_img = Image.new("RGBA", (W, H), bg_color)
    if args.glow_strength > 0:
        glow_layer = sharp_layer.filter(ImageFilter.GaussianBlur(args.glow_strength * SS))
        out_img = Image.alpha_composite(out_img, glow_layer)
    out_img = Image.alpha_composite(out_img, sharp_layer)

    if SS > 1:
        out_img = out_img.resize((args.width, args.height), Image.LANCZOS)
    save_png_sequence_frame(out_img, png_dir, f, prefix=args.name)
    
def render_frame_sharp_and_glow(f, shapes, bass, mid, *, args, W, H, SS, scratch_dir):
    sharp_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sharp_layer)

    if args.mandala:
        draw_mandala(draw, f, args.fps, W, H, bass, mid, args.mandala_points, bass,
                     args.gradient_spread, args.palette, rotation_speed=0.6)
    for kind, cx, cy, size, rotation, color in shapes:
        draw_shape(draw, kind, cx, cy, size, rotation, color)

    glow_layer = sharp_layer.filter(ImageFilter.GaussianBlur(args.glow_strength * SS))
    sharp_layer.save(os.path.join(scratch_dir, f"sharp_{f:06d}.png"))
    glow_layer.save(os.path.join(scratch_dir, f"glow_{f:06d}.png"))

def composite_trail_frame(f, *, args, W, H, SS, bg_color, scratch_dir, png_dir, lookback):
    sharp_layer = Image.open(os.path.join(scratch_dir, f"sharp_{f:06d}.png")).convert("RGBA")

    trail_arr = np.zeros((H, W, 4), dtype=np.float32)
    for k in range(lookback + 1):
        src_f = f - k
        if src_f < 0:
            break
        decay = args.trail_decay ** k
        glow_path = os.path.join(scratch_dir, f"glow_{src_f:06d}.png")
        glow_arr = np.asarray(Image.open(glow_path).convert("RGBA"), dtype=np.float32)
        trail_arr = np.maximum(trail_arr, glow_arr * decay)

    trail_img = Image.fromarray(np.clip(trail_arr, 0, 255).astype(np.uint8), "RGBA")
    out_img = Image.new("RGBA", (W, H), bg_color)
    out_img = Image.alpha_composite(out_img, trail_img)
    out_img = Image.alpha_composite(out_img, sharp_layer)

    if SS > 1:
        out_img = out_img.resize((args.width, args.height), Image.LANCZOS)
    save_png_sequence_frame(out_img, png_dir, f, prefix=args.name)

def main():
    p = argparse.ArgumentParser(description="Classic media-player style geometric/psychedelic visualizer")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="./output")
    p.add_argument("--name", default=None, help="Base output name. Defaults to '<input filename>_classic'")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--bg-color", default="#00000000",
                    help="'#RRGGBB[AA]', 'viridis:T[:AA]', or 'grey:V[:AA]'. Default transparent.")
    p.add_argument("--palette", default="viridis", choices=["viridis", "greyscale"],
                    help="Colormap used to color every shape (default viridis)")
    p.add_argument("--shape-set", default="mixed",
                    choices=["circles", "polygons", "stars", "mixed"])
    p.add_argument("--symmetry", type=int, default=5, help="Shapes spawned per beat, arranged radially")
    p.add_argument("--sensitivity", type=float, default=0.35, help="Onset threshold 0-1 (lower = more pops)")
    p.add_argument("--refractory-ms", type=int, default=90, help="Min gap between beat triggers")
    p.add_argument("--pop-lifetime-ms", type=int, default=650)
    p.add_argument("--size-min", type=float, default=None, help="Min shape radius px (default: 3% of min dim)")
    p.add_argument("--size-max", type=float, default=None, help="Max shape radius px (default: 12% of min dim)")
    p.add_argument("--gradient-spread", type=float, default=0.18,
                    help="0-1: how far each shape's color drifts across the palette within one burst")
    p.add_argument("--trails", action="store_true", help="Enable soft glowing motion trails")
    p.add_argument("--trail-decay", type=float, default=0.85, help="0-1, higher = longer trails")
    p.add_argument("--glow-strength", type=float, default=10.0)
    p.add_argument("--mandala", action="store_true", help="Add a continuous rotating bass-reactive ring")
    p.add_argument("--mandala-points", type=int, default=8)
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--output-format", choices=["png", "video", "both"], default="both")
    p.add_argument("--codec", choices=["prores4444", "qtrle", "vp9"], default="prores4444")
    p.add_argument("--include-audio", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="Parallel render processes (default: all CPU cores)")
    args = p.parse_args()
    
    if args.name is None:
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        args.name = f"{input_stem}_classic"

    rng = random.Random(args.seed)
    bg_color = parse_color(args.bg_color)

    shape_sets = {
        "circles": ["circle"],
        "polygons": ["triangle", "square", "pentagon", "hexagon"],
        "stars": ["star5", "star6"],
        "mixed": ["circle", "triangle", "square", "pentagon", "hexagon", "star5", "star6"],
    }
    kinds = shape_sets[args.shape_set]

    print(f"[1/4] Loading audio: {args.input}")
    y, sr = load_audio(args.input)
    duration = len(y) / sr
    print(f"      sample rate={sr}  duration={duration:.2f}s")

    print("[2/4] Analyzing audio (onsets + band energy)...")
    onset = compute_onset_envelope(y, sr, args.fps)
    bands = compute_spectrum_frames(y, sr, args.fps, n_bands=24, fmin=30, fmax=16000,
                                     attack=0.5, release=0.2)
    n_bands = bands.shape[1]
    bass_e = bands[:, :max(1, n_bands // 6)].mean(axis=1)
    mid_e = bands[:, n_bands // 6: n_bands // 2].mean(axis=1)
    treble_e = bands[:, n_bands // 2:].mean(axis=1)

    num_frames = min(len(onset), bands.shape[0])
    if args.max_frames:
        num_frames = min(num_frames, args.max_frames)

    SS = max(1, args.supersample)
    W, H = args.width * SS, args.height * SS
    cx, cy = W / 2, H / 2

    size_min = args.size_min * SS if args.size_min else min(W, H) * 0.03
    size_max = args.size_max * SS if args.size_max else min(W, H) * 0.12
    min_r, max_r = min(W, H) * 0.05, min(W, H) * 0.42
    lifetime_frames = max(2, int(args.pop_lifetime_ms / 1000 * args.fps))
    refractory_frames = max(1, int(args.refractory_ms / 1000 * args.fps))

    png_dir = ensure_dir(os.path.join(args.outdir, f"{args.name}_png"))
    print(f"[3/4] Rendering {num_frames} frames at {args.width}x{args.height} (supersample x{SS})...")

    particles = []
    refractory_left = 0
    for f in range(num_frames):
        strength = float(onset[f]) if f < len(onset) else 0.0
        if refractory_left <= 0 and strength > args.sensitivity:
            color_t = np.clip((strength - args.sensitivity) / max(1e-6, 1 - args.sensitivity), 0, 1)
            spawn_burst(particles, f, W, H, cx, cy, args.symmetry, strength, kinds,
                        color_t, args.gradient_spread, args.palette,
                        min_r, max_r, size_min, size_max, lifetime_frames, rng)
            refractory_left = refractory_frames
        else:
            refractory_left -= 1
            
    frame_particle_draws = defaultdict(list)
    for particle in particles:
        end = min(particle.birth + particle.lifetime, num_frames)
        for f in range(particle.birth, end):
            alive, size, alpha, rotation, color = particle.state(f)
            if alive:
                frame_particle_draws[f].append((particle.kind, particle.cx, particle.cy, size, rotation, color))
                
    png_dir = ensure_dir(os.path.join(args.outdir, f"{args.name}_png"))
    print(f"[3/4] Rendering {num_frames} frames at {args.width}x{args.height} (supersample x{SS})...")

    if not args.trails:
        worker = partial(render_frame_no_trails, args=args, W=W, H=H, SS=SS,
                          bg_color=bg_color, png_dir=png_dir)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, f, frame_particle_draws[f], bass_e[f], mid_e[f])
                       for f in range(num_frames)]
            completed = 0
            for _ in as_completed(futures):
                completed += 1
                if completed % 30 == 0 or completed == num_frames:
                    print(f"      frame {completed}/{num_frames}", end="\r")
    else:
        scratch_dir = ensure_dir(os.path.join(args.outdir, f"{args.name}_scratch"))
        lookback = int(math.log(1.0 / 255.0) / math.log(args.trail_decay)) + 1

        stage1 = partial(render_frame_sharp_and_glow, args=args, W=W, H=H, SS=SS,
                          scratch_dir=scratch_dir)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(stage1, f, frame_particle_draws[f], bass_e[f], mid_e[f])
                       for f in range(num_frames)]
            for i, _ in enumerate(as_completed(futures)):
                if (i + 1) % 30 == 0 or (i + 1) == num_frames:
                    print(f"      pass 1/2: {i + 1}/{num_frames}", end="\r")
        print()

        stage2 = partial(composite_trail_frame, args=args, W=W, H=H, SS=SS,
                          bg_color=bg_color, scratch_dir=scratch_dir, png_dir=png_dir,
                          lookback=lookback)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(stage2, f) for f in range(num_frames)]
            for i, _ in enumerate(as_completed(futures)):
                if (i + 1) % 30 == 0 or (i + 1) == num_frames:
                    print(f"      pass 2/2: {i + 1}/{num_frames}", end="\r")

        import shutil
        shutil.rmtree(scratch_dir)
    print()

    if args.output_format in ("video", "both"):
        print(f"[4/4] Encoding video ({args.codec})...")
        audio_for_mux = args.input if args.include_audio else None
        out_path = encode_video_from_pngs(png_dir, args.outdir, args.name, args.fps,
                                           codec=args.codec, prefix=args.name,
                                           audio_path=audio_for_mux)
        print(f"      -> {out_path}")
    else:
        print("[4/4] Skipping video encode (--output-format png)")

    if args.output_format == "video":
        import shutil
        shutil.rmtree(png_dir)
    else:
        print(f"      PNG sequence -> {png_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
