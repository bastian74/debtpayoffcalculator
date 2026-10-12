"""ffmpeg / ffprobe wrappers: write yuv420p frames to a file, read them back, re-encode.

Everything goes through raw yuv420p pipes so the only colour conversion that ever
happens is the one YouTube applies.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


class ToolMissing(RuntimeError):
    pass


def _tool(name: str) -> str:
    env = os.environ.get(name.upper().replace("-", "_") + "_BIN")
    path = env or shutil.which(name)
    if not path:
        raise ToolMissing(f"{name} not found on PATH (install ffmpeg, or set {name.upper()}_BIN)")
    return path


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    nb_frames: int
    codec: str
    bit_rate: int
    duration: float
    pix_fmt: str
    size_bytes: int

    def to_dict(self):
        return self.__dict__.copy()


def probe(path: str) -> VideoInfo:
    cmd = [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,codec_name,bit_rate,pix_fmt,duration",
           "-show_entries", "format=duration,bit_rate,size", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    f = j.get("format", {})
    fps = float(Fraction(s.get("avg_frame_rate") or s.get("r_frame_rate") or "30/1")) if s.get("avg_frame_rate") not in (None, "0/0") else float(Fraction(s.get("r_frame_rate", "30/1")))
    nb = int(s.get("nb_frames") or s.get("nb_read_packets") or 0)
    dur = float(s.get("duration") or f.get("duration") or 0.0)
    if nb == 0 and dur:
        nb = int(round(dur * fps))
    br = int(s.get("bit_rate") or f.get("bit_rate") or 0)
    return VideoInfo(int(s["width"]), int(s["height"]), fps, nb, s.get("codec_name", "?"), br, dur,
                     s.get("pix_fmt", "?"), int(f.get("size") or os.path.getsize(path)))


class FrameWriter:
    """Pipe yuv420p frames into ffmpeg -> near-lossless H.264 (the file you upload)."""

    def __init__(self, path: str, width: int, height: int, fps: int = 30, crf: int = 12,
                 preset: str = "medium", codec: str = "libx264", extra: list[str] | None = None,
                 quiet: bool = True):
        self.path, self.W, self.H = path, width, height
        self.frames = 0
        gop = max(1, fps // 2)
        cmd = [_tool("ffmpeg"), "-y", "-v", "error" if quiet else "info",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0",
               "-an", "-c:v", codec, "-pix_fmt", "yuv420p",
               "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-g", str(gop), "-bf", "2", "-movflags", "+faststart"]
        if codec == "libx264":
            cmd += ["-preset", preset, "-crf", str(crf), "-profile:v", "high", "-level", "5.1",
                    "-x264-params", f"keyint={gop}:min-keyint={gop}:scenecut=0"]
        elif codec in ("libx265",):
            cmd += ["-preset", preset, "-crf", str(crf), "-tag:v", "hvc1"]
        cmd += extra or []
        cmd.append(path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, Y: np.ndarray, U: np.ndarray, V: np.ndarray):
        assert Y.shape == (self.H, self.W) and U.shape == (self.H // 2, self.W // 2)
        try:
            self.proc.stdin.write(np.ascontiguousarray(Y, dtype=np.uint8).tobytes())
            self.proc.stdin.write(np.ascontiguousarray(U, dtype=np.uint8).tobytes())
            self.proc.stdin.write(np.ascontiguousarray(V, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg died while encoding: {err}")
        self.frames += 1

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}: {err}")
        return self.frames

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def iter_frames(path: str, width: int | None = None, height: int | None = None, start: int = 0,
                limit: int | None = None):
    """Yield (Y, U, V) uint8 planes.  If width/height differ from the file, ffmpeg rescales."""
    info = probe(path)
    W, H = width or info.width, height or info.height
    cmd = [_tool("ffmpeg"), "-v", "error", "-i", path, "-an", "-sn"]
    vf = []
    if (W, H) != (info.width, info.height):
        vf.append(f"scale={W}:{H}:flags=area")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    n = W * H * 3 // 2
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=n * 4)
    idx = 0
    try:
        while True:
            buf = bytearray(n)
            view = memoryview(buf)
            got = 0
            while got < n:
                chunk = proc.stdout.readinto(view[got:])
                if not chunk:
                    break
                got += chunk
            if got < n:
                break
            if idx >= start:
                arr = np.frombuffer(buf, dtype=np.uint8)
                Y = arr[: W * H].reshape(H, W)
                U = arr[W * H: W * H + (W // 2) * (H // 2)].reshape(H // 2, W // 2)
                V = arr[W * H + (W // 2) * (H // 2):].reshape(H // 2, W // 2)
                yield Y, U, V
                if limit is not None and idx - start + 1 >= limit:
                    break
            idx += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# YouTube-like re-encode presets (approximate the bitrates YouTube's ladder lands on
# for noisy 30 fps content; the real answer comes from an actual upload round trip).
SIM_PRESETS = {
    "yt720-h264":  dict(codec="libx264",   bitrate="1800k"),
    "yt1080-h264": dict(codec="libx264",   bitrate="4000k"),
    "yt1080-vp9":  dict(codec="libvpx-vp9", bitrate="2700k"),
    "yt1440-vp9":  dict(codec="libvpx-vp9", bitrate="8500k"),
    "yt2160-vp9":  dict(codec="libvpx-vp9", bitrate="17000k"),
    "yt1080-av1":  dict(codec="libsvtav1", bitrate="2000k"),
    "harsh-h264":  dict(codec="libx264",   bitrate="1500k"),
}


def reencode(src: str, dst: str, codec: str = "libx264", bitrate: str = "4000k", fps: int | None = None,
             scale: tuple[int, int] | None = None, two_pass: bool = False, quiet: bool = True,
             extra: list[str] | None = None) -> None:
    """Re-encode a video the way a hostile transcoder would: constrained ABR, 4:2:0."""
    ff = _tool("ffmpeg")
    br = bitrate
    num = int(br[:-1]) * (1000 if br.endswith("k") else 1_000_000) if br[-1] in "kM" else int(br)
    base = [ff, "-y", "-v", "error" if quiet else "info", "-i", src, "-an", "-pix_fmt", "yuv420p"]
    vf = []
    if scale:
        vf.append(f"scale={scale[0]}:{scale[1]}")
    if fps:
        base += ["-r", str(fps)]
    if vf:
        base += ["-vf", ",".join(vf)]
    if codec == "libx264":
        enc = ["-c:v", "libx264", "-preset", "medium", "-b:v", br, "-maxrate", str(int(num * 1.25)),
               "-bufsize", str(num * 2), "-g", "60", "-bf", "2", "-profile:v", "high"]
    elif codec == "libvpx-vp9":
        enc = ["-c:v", "libvpx-vp9", "-b:v", br, "-maxrate", str(int(num * 1.45)), "-minrate", str(int(num * 0.5)),
               "-deadline", "good", "-cpu-used", "2", "-row-mt", "1", "-tile-columns", "2", "-g", "240"]
    elif codec in ("libsvtav1", "libaom-av1"):
        enc = ["-c:v", codec, "-b:v", br, "-g", "240"]
        if codec == "libsvtav1":
            enc += ["-preset", "8"]
        else:
            enc += ["-cpu-used", "6", "-row-mt", "1"]
    else:
        enc = ["-c:v", codec, "-b:v", br]
    enc += extra or []
    if two_pass and codec in ("libx264", "libvpx-vp9"):
        log = dst + ".2pass"
        null = "NUL" if sys.platform == "win32" else "/dev/null"
        subprocess.run(base + enc + ["-pass", "1", "-passlogfile", log, "-f", "null", null], check=True)
        subprocess.run(base + enc + ["-pass", "2", "-passlogfile", log, dst], check=True)
        for ext in ("-0.log", "-0.log.mbtree"):
            try:
                os.remove(log + ext)
            except OSError:
                pass
    else:
        subprocess.run(base + enc + [dst], check=True)
