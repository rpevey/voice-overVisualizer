#!/usr/bin/env python3
"""
spectrum_visualizer.py
A configurable audio spectrum / waveform visualizer, inspired by HansiMcKlaus's
AudioSpectrumVisualizer (linear or circular bar layouts). Renders straight to a
transparent PNG sequence and/or an alpha-channel video, ready to drop into a
DaVinci Resolve timeline over other footage.

Modes:
  bars_linear     - classic left-to-right (or mirrored) frequency bars
  bars_circular   - bars radiating outward from a center circle
  waveform_linear - oscilloscope-style line across the frame
  waveform_circular - waveform warped around a ring

Examples:
  python spectrum_visualizer.py --input song.wav --mode bars_circular \
      --width 1080 --height 1920 --bar-color "#00E5FFFF" --bg-color "#00000000" \
      --output-format both

  python spectrum_visualizer.py --input song.mp3 --mode bars_linear --mirror \
      --width 1920 --height 1080 --codec prores4444
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_analysis import load_audio, compute_spectrum_frames, compute_waveform_frames
from palette import parse_color
from render_utils import ensure_dir, save_png_sequence_frame, encode_video_from_pngs


def draw_glow(base_layer, blur_radius, intensity=0.9):
    """Return a blurred, slightly boosted copy of base_layer for an additive glow look."""
    glow = base_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    if intensity != 1.0:
        r, g, b, a = glow.split()
        a = a.point(lambda v: min(255, int(v * intensity)))
        glow = Image.merge("RGBA", (r, g, b, a))
    return glow


def render_bars_linear(draw, values, W, H, bar_color, mirror, gap_ratio, rounded):
    n = len(values)
    total_w = W
    bar_slot = total_w / n
    bar_w = bar_slot * (1 - gap_ratio)
    baseline = H if not mirror else H / 2
    max_h = H if not mirror else H / 2
    for i, v in enumerate(values):
        x0 = i * bar_slot + (bar_slot - bar_w) / 2
        x1 = x0 + bar_w
        h = max(2.0, v * max_h)
        if mirror:
            y0, y1 = baseline - h, baseline + h
        else:
            y0, y1 = baseline - h, baseline
        radius = bar_w / 2 if rounded else 0
        if radius > 0:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bar_color)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=bar_color)


def render_bars_circular(draw, values, W, H, bar_color, inner_radius_ratio,
                          outer_radius_ratio, rounded):
    n = len(values)
    cx, cy = W / 2, H / 2
    base_r = min(W, H)
    inner_r = base_r * inner_radius_ratio
    max_len = base_r * (outer_radius_ratio - inner_radius_ratio)
    for i, v in enumerate(values):
        angle = (i / n) * 2 * math.pi - math.pi / 2
        length = max(3.0, v * max_len)
        r0 = inner_r
        r1 = inner_r + length
        x0, y0 = cx + r0 * math.cos(angle), cy + r0 * math.sin(angle)
        x1, y1 = cx + r1 * math.cos(angle), cy + r1 * math.sin(angle)
        width = max(2, int((2 * math.pi * inner_r / n) * (1 - 0.25)))
        draw.line([x0, y0, x1, y1], fill=bar_color, width=width, joint="curve")
        if rounded:
            draw.ellipse([x1 - width / 2, y1 - width / 2, x1 + width / 2, y1 + width / 2], fill=bar_color)


def render_waveform_linear(draw, samples, W, H, line_color, thickness, filled):
    n = len(samples)
    xs = np.linspace(0, W, n)
    ys = H / 2 - samples * (H / 2 * 0.9)
    pts = list(zip(xs.tolist(), ys.tolist()))
    if filled:
        poly = pts + [(W, H / 2), (0, H / 2)]
        draw.polygon(poly, fill=line_color)
    else:
        draw.line(pts, fill=line_color, width=thickness, joint="curve")


def render_waveform_circular(draw, samples, W, H, line_color, thickness, base_radius_ratio, amplitude_ratio):
    n = len(samples)
    cx, cy = W / 2, H / 2
    base_r = min(W, H) * base_radius_ratio
    amp = min(W, H) * amplitude_ratio
    pts = []
    for i, s in enumerate(samples):
        angle = (i / n) * 2 * math.pi - math.pi / 2
        r = base_r + s * amp
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pts.append(pts[0])
    draw.line(pts, fill=line_color, width=thickness, joint="curve")

def render_frame_worker(f, frame_row, *, args, W, H, SS, bg_color, bar_color, png_dir):
    img = Image.new("RGBA", (W, H), bg_color)
    draw_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    if args.mode == "bars_linear":
        render_bars_linear(draw, frame_row, W, H, bar_color,
                            args.mirror, args.gap_ratio, args.rounded)
    elif args.mode == "bars_circular":
        render_bars_circular(draw, frame_row, W, H, bar_color,
                              args.inner_radius_ratio, args.outer_radius_ratio,
                              args.rounded)
    elif args.mode == "waveform_linear":
        render_waveform_linear(draw, frame_row, W, H, bar_color,
                                args.line_thickness * SS, args.filled)
    elif args.mode == "waveform_circular":
        render_waveform_circular(draw, frame_row, W, H, bar_color,
                                  args.line_thickness * SS, args.base_radius_ratio,
                                  args.amplitude_ratio)

    if args.glow:
        glow = draw_glow(draw_layer, args.glow_strength * SS, intensity=0.8)
        img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, draw_layer)

    if SS > 1:
        img = img.resize((args.width, args.height), Image.LANCZOS)

    save_png_sequence_frame(img, png_dir, f, prefix=args.name)


def main():
    p = argparse.ArgumentParser(description="Spectrum / waveform audio visualizer")
    p.add_argument("--input", required=True, help="Path to audio (or any ffmpeg-readable media) file")
    p.add_argument("--outdir", default="./output")
    p.add_argument("--name", default=None, help="Base output name. Defaults to '<input filename>_spectrum'")
    p.add_argument("--mode", choices=["bars_linear", "bars_circular", "waveform_linear", "waveform_circular"],
                    default="bars_linear")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--bars", type=int, default=48, help="Number of frequency bands (bar modes)")
    p.add_argument("--bg-color", default="#00000000",
                    help="'#RRGGBB[AA]', 'viridis:T[:AA]', or 'grey:V[:AA]'. Default transparent.")
    p.add_argument("--bar-color", default="viridis:0.55",
                    help="Single flat color for bars/waveform: hex RGBA, 'viridis:T', or 'grey:V'")
    p.add_argument("--mirror", action="store_true", help="Mirror bars up/down from center (linear mode)")
    p.add_argument("--rounded", action="store_true", help="Rounded bar caps")
    p.add_argument("--gap-ratio", type=float, default=0.3, help="Gap between bars as fraction of slot width")
    p.add_argument("--inner-radius-ratio", type=float, default=0.18, help="Circular mode: inner ring radius")
    p.add_argument("--outer-radius-ratio", type=float, default=0.48, help="Circular mode: max bar reach")
    p.add_argument("--base-radius-ratio", type=float, default=0.3, help="Circular waveform: resting radius")
    p.add_argument("--amplitude-ratio", type=float, default=0.15, help="Circular waveform: max displacement")
    p.add_argument("--line-thickness", type=int, default=4, help="Waveform line thickness (px, at render res)")
    p.add_argument("--filled", action="store_true", help="Fill under linear waveform")
    p.add_argument("--glow", action="store_true", help="Add soft additive glow")
    p.add_argument("--glow-strength", type=float, default=12.0)
    p.add_argument("--supersample", type=int, default=2, help="Render at NxN then downsample for AA quality")
    p.add_argument("--workers", type=int, default=None, help="Parallel render processes (default: all CPU cores)")
    p.add_argument("--fmin", type=float, default=30)
    p.add_argument("--fmax", type=float, default=16000)
    p.add_argument("--output-format", choices=["png", "video", "both"], default="both")
    p.add_argument("--codec", choices=["prores4444", "qtrle", "vp9"], default="prores4444")
    p.add_argument("--include-audio", action="store_true", help="Mux original audio into the output video")
    p.add_argument("--max-frames", type=int, default=None, help="Debug: limit number of rendered frames")
    args = p.parse_args()
    
    if args.name is None:
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        args.name = f"{input_stem}_spectrum"

    bg_color = parse_color(args.bg_color)
    bar_color = parse_color(args.bar_color)

    print(f"[1/4] Loading audio: {args.input}")
    y, sr = load_audio(args.input)
    duration = len(y) / sr
    print(f"      sample rate={sr}  duration={duration:.2f}s")

    is_waveform = args.mode.startswith("waveform")
    print("[2/4] Analyzing audio...")
    if is_waveform:
        win_seconds = 1.0 / args.fps  # one visual cycle per frame worth of samples
        frames_data = compute_waveform_frames(y, sr, args.fps, window_seconds=win_seconds)
        # normalize samples into [-1, 1] robustly
        peak = max(1e-6, np.max(np.abs(y)))
        frames_data = [f / peak for f in frames_data]
    else:
        frames_data = compute_spectrum_frames(y, sr, args.fps, n_bands=args.bars,
                                               fmin=args.fmin, fmax=args.fmax)

    num_frames = len(frames_data) if args.max_frames is None else min(args.max_frames, len(frames_data))

    SS = max(1, args.supersample)
    W, H = args.width * SS, args.height * SS

    png_dir = ensure_dir(os.path.join(args.outdir, f"{args.name}_png"))
    print(f"[3/4] Rendering {num_frames} frames at {args.width}x{args.height} (supersample x{SS})...")

    # for f in range(num_frames):
        # img = Image.new("RGBA", (W, H), bg_color)
        # draw_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        # draw = ImageDraw.Draw(draw_layer)

        # if args.mode == "bars_linear":
            # render_bars_linear(draw, frames_data[f], W, H, bar_color,
                                # args.mirror, args.gap_ratio, args.rounded)
        # elif args.mode == "bars_circular":
            # render_bars_circular(draw, frames_data[f], W, H, bar_color,
                                  # args.inner_radius_ratio, args.outer_radius_ratio,
                                  # args.rounded)
        # elif args.mode == "waveform_linear":
            # render_waveform_linear(draw, frames_data[f], W, H, bar_color,
                                    # args.line_thickness * SS, args.filled)
        # elif args.mode == "waveform_circular":
            # render_waveform_circular(draw, frames_data[f], W, H, bar_color,
                                      # args.line_thickness * SS, args.base_radius_ratio,
                                      # args.amplitude_ratio)

        # if args.glow:
            # glow = draw_glow(draw_layer, args.glow_strength * SS, intensity=0.8)
            # img = Image.alpha_composite(img, glow)
        # img = Image.alpha_composite(img, draw_layer)

        # if SS > 1:
            # img = img.resize((args.width, args.height), Image.LANCZOS)

        # save_png_sequence_frame(img, png_dir, f, prefix=args.name)
        # if f % 30 == 0 or f == num_frames - 1:
            # print(f"      frame {f + 1}/{num_frames}", end="\r")
    # print()
    
    worker = partial(render_frame_worker, args=args, W=W, H=H, SS=SS,
                      bg_color=bg_color, bar_color=bar_color, png_dir=png_dir)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, f, frames_data[f]) for f in range(num_frames)]
        completed = 0
        for _ in as_completed(futures):
            completed += 1
            if completed % 30 == 0 or completed == num_frames:
                print(f"      frame {completed}/{num_frames}", end="\r")
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
