# ytvault — how much data survives YouTube, and how to store files there

`ytvault` answers one question experimentally: **how densely can bytes be painted into a
video so that they still come back bit-exact after YouTube re-encodes it?** It then uses
the answer to pack arbitrary files into videos with enough error correction to repair what
the transcoder damages.

It is a single Python package (numpy + ffmpeg, yt-dlp for downloads) with no build step.

```
pip install numpy yt-dlp            # ffmpeg + ffprobe must be on PATH
cd ytvault
python -m ytvault selftest          # codec + frame-layout tests, no video needed
```

## The experiment, end to end

```
# 1. make the calibration video: ~36 encodings, sparse first, denser every 2 s
python -m ytvault calib-gen -o calib-1080p.mp4 --res 1920x1080 --fps 30

# 2. (optional) see what a YouTube-like transcoder does to it before uploading
python -m ytvault simulate calib-1080p.mp4 --preset yt1080-h264 --out sim-report/

# 3. upload calib-1080p.mp4 to YouTube (unlisted).  Wait until the 1080p rendition exists.
#    (or: python -m ytvault upload calib-1080p.mp4 --title "ytvault calibration"  -- needs an OAuth client)

# 4. download every rendition at the design height and measure what survived
python -m ytvault analyze https://www.youtube.com/watch?v=XXXX --formats native --out report/

# 5. store a real file with the profile the report recommends, verify locally, upload, verify from YouTube
python -m ytvault pack backup.tar.zst -o backup.mp4 --from-report report/report.json
python -m ytvault unpack https://www.youtube.com/watch?v=YYYY -o recovered/
```

`analyze` prints a table per downloaded rendition and writes `report/report.md` plus a full
`report.json` (per-frame timeline included). Re-run it on the same URL months later to check
whether YouTube's renditions have changed; the profile is only "ideal" for the renditions it
was measured on.

## What the calibration video contains

Every frame is a YUV 4:2:0 image with three bands:

| band | content | purpose |
|---|---|---|
| header (top, 8 px cells, black/white) | 80 bytes: magic, frame index, stream id, profile, RS setting, payload length and CRC32, outer-code position; RS(80,48) protected | lets the decoder learn the body geometry even when the body is destroyed |
| calibration strip (8 px) | 16-step luma ramp, 16-step U ramp, 16-step V ramp | measures how the transcoder moved the grey levels; the decoder maps nominal levels through it, then refines on the body itself |
| body | the data cells | the payload |

A **profile** `c{cell}-y{ybits}-c{cbits}-r{repeat}` fixes the body:

- `cell` — luma cell edge in pixels (2 … 8). Chroma cells are `2·cell` px because YouTube
  always delivers 4:2:0, so a chroma cell is `cell × cell` in the half-resolution plane.
- `ybits` — bits per luma cell: 1 = black/white, 2 = four greys, 3 = eight, 4 = sixteen.
  Levels are spread over studio range 16–235 and Gray-coded so a one-level slip costs one bit.
- `cbits` — bits per chroma cell per plane (0 = luma only). Chroma levels use 40–216.
- `repeat` — how many video frames each data frame is shown for. The decoder averages the
  copies; the second copy of a static frame is usually cleaner because the encoder spends its
  residual bits refining it.

The body bit stream is `ncw` Reed-Solomon codewords of 255 bytes (`nsym` parity each),
byte-interleaved across the frame, XOR-whitened with a per-frame keystream (so a file of
zeros still looks like noise and gets the bitrate the calibration measured), then dealt to
cells in a fixed pseudo-random cell order. The permutation spreads any spatial burst (a
crushed macroblock, a smeared row) across every codeword and makes luma and chroma share
codewords, which is what makes the per-codeword error counts well-behaved enough to model.

The default sweep (`--sweep default`, 36 profiles) at 1080p30 runs from 0.34 GiB per hour of
video up to 16.6 GiB per hour, gross. `python -m ytvault profiles` lists it for any resolution.

Calibration payloads are a deterministic keystream of the (stream id, frame index), so the
analyzer regenerates the exact bytes that were transmitted and measures ground-truth errors
rather than relying on the decoder's own opinion.

## What `analyze` measures per profile

- **SER-Y / SER-C** — fraction of luma / chroma cells that decoded to the wrong level.
- **BER** — bit errors in the body byte stream, before RS.
- **cw max / p99 / mean** — wrong bytes per 255-byte codeword. RS with `nsym` parity bytes
  repairs up to `nsym/2` wrong bytes per codeword, so these numbers decide the parity.
- **crc ok** — frames fully recovered at the transmitted `nsym` (RS succeeded and the
  payload CRC32 matched).
- **eye** — for each pair of adjacent levels, the gap between their observed means divided
  by the sum of their standard deviations. It is the safety margin that still exists when
  zero errors were observed. Below 3 the profile is rejected outright; below 4.5 it is
  forced to at least 32 parity bytes.
- **missing frames** — frame indices that never arrived (YouTube dropped or merged frames).
  Any missing frame disqualifies the profile; the outer code exists for storage, not for
  routine loss.
- Per-copy error rates for `repeat > 1` profiles are kept in `report.json`.

### How the recommendation is made

For each profile the observed byte error rate is inflated by a safety factor (default 2×)
and treated as a binomial over the 255 bytes of a codeword. The recommended `nsym` is the
smallest even parity count such that the modelled probability of *any* codeword in a frame
exceeding capacity is below the target (default 10⁻⁶ per frame), and never less than twice
the worst codeword actually observed. Throughput is then computed net of inner parity and of
the outer frame parity (default 20 data frames + 4 parity frames per group), and the profile
with the highest net throughput wins. A **conservative** pick (zero observed errors, eye ≥ 6)
is printed alongside.

When several renditions were analysed (`--formats native` downloads every codec YouTube
offers at the design height, typically AVC1 and VP9, sometimes AV1), the overall pick is the
best profile that is feasible in *all* of them, with the largest `nsym` any of them needed,
because you do not control which rendition a future download will get.

The model is deliberately pessimistic in two ways: the inflation factor, and the fact that
a 2-second segment cannot observe tail events of a multi-hour upload. The outer code is the
backstop for what the sample could not see.

## Storing a file: `pack` / `unpack`

`pack` writes, in order: 3 manifest frames, then groups of `K` data frames followed by `M`
parity frames, then 3 more manifest frames. The manifest (file name, size, SHA-256, profile,
RS and outer settings, chunk count) is always encoded at the safest profile (`c8-y1-c0-r2`,
64 parity bytes) so it is readable even when the data profile was too optimistic. Parity
frames are Reed-Solomon over the frame columns: byte *i* of the `M` parity frames protects
byte *i* of the `K` data frames, so any `M` lost or unrecoverable frames in a group can be
rebuilt. Every frame's header carries the payload CRC32, so a frame that RS "corrected" to the
wrong thing (possible when the error count is past capacity) is treated as lost rather than
trusted.

`unpack` reports exactly what it had to do: inner corrections, worst codeword, chunks rebuilt
by the outer code, chunks lost, and whether the SHA-256 matched. `verify` does the same without
writing the file. `--dry-run` works on URLs too.

The decoder finds the design resolution by itself (it tries the stream's native size first,
then the common ladder sizes with rescaling), tolerates a global pixel shift of up to 2 px,
merges `repeat` copies by frame index, and treats duplicate or dropped frames as the stream
says, not as the container says.

## What to expect, and the physics behind it

YouTube keeps no original. Every rendition is a fresh lossy encode (AVC1 at every size; VP9
and sometimes AV1 at 1080p and above) at a bitrate ceiling that depends on resolution and
frame rate, in 4:2:0, with the encoder free to blur anything it considers noise. For data
frames that means:

- **Cell size is the main lever.** Data cells are noise to the encoder; at 1080p the AVC
  rendition gets roughly 3–4 Mbit/s and VP9 about 2–3 Mbit/s, which is about 0.05–0.1 bits
  per pixel. A 4×4 black/white cell costs 1 bit per 16 pixels and lands inside the budget
  with room for the deblocking filter to blur edges; 2×2 cells are at the limit and 1-pixel
  cells never survive.
- **Grey levels are cheaper than smaller cells, up to a point.** Four levels in a 4×4 cell
  double the density for a modest eye-margin cost; sixteen levels need the eye to stay above
  ~3 σ after quantisation, which usually only large cells manage.
- **Chroma is worth about a quarter of luma.** Because of 4:2:0 a chroma cell is four times
  the area of a luma cell, and the chroma planes get coarser quantisation. One bit per chroma
  plane in 8×8 px cells is the usual sweet spot; two bits rarely pay.
- **Higher resolution scales almost linearly.** A 1440p or 2160p upload gets a VP9 rendition
  with 3–5× the bitrate of 1080p and 1.8–4× the pixels, so the same cell size survives and the
  per-hour yield grows. Generate one calibration video per candidate resolution
  (`--res 2560x1440`, `--res 3840x2160`); the ladder step you upload at decides which codecs
  are offered.
- **60 fps gets a higher bitrate ceiling** than 30 fps but each frame gets fewer bits; it is
  worth a calibration run of its own (`--fps 60`), not an assumption.
- **Static frames help.** `repeat 2` halves throughput but the averaged pair is markedly
  cleaner, which can let a denser cell survive; the sweep includes a few `r2` profiles so the
  report can show whether that trade pays at your resolution.

Prior art ("Infinite Storage Glitch", "YouTubeDrive", fvid) settled on 2×2 or 4×4
black/white cells at 720p–1080p without measurement. `ytvault`'s point is that you measure
the actual renditions, pick the densest profile with margin, and keep repair parity for the
rest.

### Caveats you should actually weigh

- Storing arbitrary binaries as video is against the spirit of YouTube's terms; videos and
  channels doing it have been removed. Treat this as an experiment and keep a real backup.
- YouTube can re-transcode at any time (new codecs, new ladders). `analyze` on the stored
  video's URL is cheap; run it periodically and re-upload with more parity if margins shrink.
- Audio is omitted; some uploads with no audio track process slower, never worse.
- The upload file is x264 at CRF 12 (visually lossless for block content). `calib-gen` and
  `pack` decode their own output before you upload to prove it is clean; if that check fails,
  lower `--crf`.
- Floating-point cell means plus integer thresholds are deterministic across machines; the
  keystream and the cell permutation are a splitmix64 counter generator that does not depend
  on numpy's RNG, so a video packed today is decodable by any future version.

## Command reference

| command | what it does |
|---|---|
| `profiles [--res] [--fps] [--sweep quick\|default\|wide] [--profiles list]` | list the sweep with payload rates |
| `calib-gen -o out.mp4 [--res] [--fps] [--sweep] [--seconds 2] [--nsym 32] [--crf 12]` | write the calibration video and self-check it |
| `simulate video.mp4 [--preset yt1080-h264\|yt1080-vp9\|yt1440-vp9\|yt2160-vp9\|yt1080-av1\|harsh-h264] [--bitrate] [--two-pass] [--out dir]` | local YouTube-like round trip and analysis |
| `analyze <file\|url> [--formats native\|all\|best\|ids] [--out dir] [--outer K/M] [--safety] [--target] [--min-eye]` | measure and recommend |
| `download <url> [--formats] -o dir` | fetch renditions with yt-dlp |
| `upload video.mp4 --title ... [--privacy unlisted]` | YouTube Data API upload (optional dependencies) |
| `pack <file> -o out.mp4 [--profile c4-y1-c0-r1 \| --from-report report.json] [--nsym] [--outer 20/4] [--res] [--fps]` | encode a file |
| `unpack <file\|url> -o dir [--format id] [--dry-run]` | recover a file, with repair statistics |
| `verify video.mp4` | integrity report without writing |
| `selftest` | in-memory codec and frame tests |

On Windows use `py -m ytvault ...`; everything else is identical.

## Layout

```
ytvault/
  ytvault/gf256.py      GF(2^8) tables, batch Reed-Solomon encode/decode (errors + erasures)
  ytvault/frame.py      header, strip, body layout; render and sample; level calibration
  ytvault/profiles.py   the density sweep and rate arithmetic
  ytvault/video.py      ffmpeg pipes (write yuv420p -> x264, read any video -> yuv420p), simulate presets
  ytvault/decoder.py    video frames -> logical frames (repeat merge, offset/resolution detection)
  ytvault/calibrate.py  calibration generation, ground-truth analysis, recommendation, reports
  ytvault/store.py      pack / unpack with manifest and outer frame parity
  ytvault/youtube.py    yt-dlp format listing/download, Data API upload
  ytvault/cli.py        command line
  tests/                pytest: codec, frame, end-to-end video round trips
```
