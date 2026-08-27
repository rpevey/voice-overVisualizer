"""
render_utils.py
Shared helpers for writing PNG sequences and encoding them into alpha-capable
video files (ProRes 4444 .mov, WebM VP9 .webm, or QuickTime Animation .mov)
for import into DaVinci Resolve.
"""
import os
import subprocess


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_png_sequence_frame(img, png_dir, frame_index, prefix="frame"):
    """img: PIL.Image in RGBA mode."""
    fname = os.path.join(png_dir, f"{prefix}_{frame_index:06d}.png")
    img.save(fname)
    return fname


# Codec presets. All preserve alpha.
CODEC_PRESETS = {
    "prores4444": {
        # Best general-purpose choice for DaVinci Resolve on any platform.
        "ext": "mov",
        "args": ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"],
    },
    "qtrle": {
        # Lossless, alpha-capable, huge files but universally compatible / simple.
        "ext": "mov",
        "args": ["-c:v", "qtrle", "-pix_fmt", "argb"],
    },
    "vp9": {
        # Smaller files, good alpha support in recent Resolve versions.
        # -auto-alt-ref 0 is required or libvpx silently drops the alpha plane.
        "ext": "webm",
        "args": ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "18",
                 "-auto-alt-ref", "0"],
    },
}


def encode_video_from_pngs(png_dir, output_dir, base_name, fps, codec="prores4444",
                            prefix="frame", audio_path=None):
    """
    Encode a PNG sequence (RGBA) in png_dir into an alpha-preserving video.
    If audio_path is given, mux the original audio in alongside.
    Returns the output file path.
    """
    if codec not in CODEC_PRESETS:
        raise ValueError(f"Unknown codec '{codec}'. Choose from {list(CODEC_PRESETS)}")
    preset = CODEC_PRESETS[codec]
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, f"{base_name}.{preset['ext']}")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(png_dir, f"{prefix}_%06d.png"),
    ]
    if audio_path:
        cmd += ["-i", audio_path]
    cmd += preset["args"]
    if audio_path:
        audio_codec = "libopus" if preset["ext"] == "webm" else "aac"
        cmd += ["-c:a", audio_codec, "-shortest"]
    cmd += [out_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{result.stderr[-3000:]}")
    return out_path
