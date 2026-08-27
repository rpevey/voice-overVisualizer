# Voice-over Spectrum Visualizer

A command-line audio visualizer built in Python, inspired by
HansiMcKlaus's AudioSpectrumVisualizer
(https://github.com/HansiMcKlaus/AudioSpectrumVisualizer).
Designed for voice/podcast audio visualization and produces transparent
PNG sequences or alpha-channel video files ready for import into DaVinci Resolve.

## Intended use

This tool is optimized for voice audio visualization -- podcast recordings,
voice-over tracks, narration. The bar and waveform modes read cleanly at the
lower dynamic range of speech and produce clean, simple visuals that work well
composited over talking-head or screen-capture footage.

It's designed to work on windows OS, because that's what I use. If you need it for
a different platform, it might be worth a major rewrite. I'm not sure this would work
on linux or MacOS.

While there is a music visualization mode for this package, it is wildly inefficient,
so if you're interested in that I recommend using the C++ OpenGL visualizer package
that I made as a sister tool to this instead
(see https://github.com/rpevey/musicVisualizer).

---

## Setup (Windows)

Prerequisites:
- Python 3.10+
- ffmpeg on PATH (confirm with: ffmpeg -version)

Install dependencies:
  pip install -r requirements.txt

Dependencies: numpy, scipy, Pillow. No GPU required.

---

## Usage

### Spectrum visualizer (spectrum_visualizer.py)

Frequency bar graph or oscilloscope waveform, linear or circular layout.

  python spectrum_visualizer.py --input voiceover.wav --mode bars_linear
      --width 1920 --height 1080 --bar-color "viridis:0.6"
      --bg-color "#00000000" --output-format both

Modes:
  bars_linear       left-to-right frequency bars (default)
  bars_circular     bars radiating outward from center
  waveform_linear   oscilloscope line across the frame
  waveform_circular waveform wrapped around a ring

Key flags:
  --input PATH          Any ffmpeg-readable audio (wav, mp3, m4a, etc.)
  --mode MODE           bars_linear|bars_circular|waveform_linear|waveform_circular
  --width / --height    Output resolution
  --fps N               Default 30
  --bars N              Frequency bands for bar modes (default 48)
  --bar-color SPEC      Color: "#RRGGBBAA", "viridis:T", or "grey:V"
  --bg-color SPEC       Background color (default "#00000000" transparent)
  --mirror              Mirror bars up/down from center (linear mode)
  --rounded             Rounded bar caps
  --glow                Soft additive bloom effect
  --filled              Fill under waveform curve (linear waveform mode)
  --output-format       png | video | both
  --codec               prores4444 | qtrle | vp9 (default prores4444)
  --include-audio       Mux input audio into the output video
  --supersample N       Render at NxN then downsample for anti-aliasing (default 2)
  --workers N           Parallel render processes (default: all CPU cores)

Color spec syntax:
  "#RRGGBB"      opaque hex color
  "#RRGGBBAA"    hex with alpha (e.g. "#00000000" for fully transparent)
  "viridis:T"    sample viridis colormap at position T (0=dark purple, 1=bright yellow)
  "grey:V"       greyscale at luminance V (0=black, 1=white)

### Classic visualizer (classic_visualizer.py)

*WARNING* This mode is very computationally inefficient so if you're interested in
this type of usage than you should use my C++ OpenGL based tool at Github.
Beat-triggered geometric shape bursts with viridis/greyscale color palette.
Less suited for voice audio than the spectrum visualizer.

  python classic_visualizer.py --input audio.wav --width 1920 --height 1080
      --shape-set mixed --symmetry 5 --palette viridis --trails --mandala
      --output-format both

Key flags:
  --shape-set           circles | polygons | stars | mixed
  --symmetry N          Shapes per beat burst (default 5)
  --palette NAME        viridis | greyscale
  --gradient-spread F   Color spread across each burst (default 0.18)
  --sensitivity F       Beat detection threshold 0-1 (lower = more pops)
  --trails              Enable glowing motion trails
  --trail-decay F       Trail persistence 0-1 (default 0.85)
  --mandala             Add rotating bass-reactive center ring
  --workers N           Parallel render processes

---

## Output and DaVinci Resolve import

ProRes 4444 (.mov) -- default codec, best for Resolve. Drop directly onto
a timeline track above other footage. Transparency works immediately.

PNG sequence -- universally compatible, lossless. Import by selecting the
first frame in Resolve's Media Pool with "Image sequence" enabled.

VP9 (.webm) -- smaller files, experimental Resolve support.

Output files are named automatically from the input filename and mode:
  voiceover_bars_linear.mov, voiceover_waveform_circular.mov, etc.

---

## Performance notes

- Rendering is CPU-bound. Use --supersample 1 and omit --glow/--trails
  for fast preview renders, then re-render at full quality.
- --workers parallelizes frame rendering across CPU cores.
  Expect ~4-6x speedup on a 12-core machine.
- The --trails option in classic_visualizer.py uses a two-pass render
  that writes intermediate frames to a scratch folder, auto-deleted after encode.
- Use --max-frames N to render a short clip for settings iteration.

---

## File structure

  spectrum_visualizer.py   Visualizer 1: bars and waveform modes
  classic_visualizer.py    Visualizer 2: geometric shape bursts
  audio_analysis.py        Audio decoding, FFT spectrum, onset detection
  palette.py               Viridis and greyscale colormaps + color spec parser
  render_utils.py          PNG sequence writer and ffmpeg video encoder
  requirements.txt
