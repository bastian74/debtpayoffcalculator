"""Shared decode loop: video frames -> logical frames (repeat copies merged) -> payloads."""
from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass, field

import numpy as np

from . import frame as F
from . import video

CANDIDATE_DESIGNS = [(1920, 1080), (1280, 720), (2560, 1440), (3840, 2160), (1920, 1088), (1440, 1080),
                     (960, 540), (640, 360)]
OFFSETS = [(0, 0)] + [(dx, dy) for dy in (-1, 1, -2, 2, 0) for dx in (-1, 1, -2, 2, 0) if (dx, dy) != (0, 0)]


@dataclass
class LogicalFrame:
    header: F.Header
    first_video_idx: int
    copies: int = 0
    header_fixed: int = 0
    header_sep: float = 0.0
    vy: np.ndarray | None = None       # running sum of luma cell means
    vu: np.ndarray | None = None
    vv: np.ndarray | None = None
    ramps: dict | None = None
    copy_vals: list = field(default_factory=list)  # per-copy (vy, vu, vv) when requested

    def mean_values(self):
        n = max(self.copies, 1)
        vu = self.vu / n if self.vu is not None and self.vu.size else self.vu
        vv = self.vv / n if self.vv is not None and self.vv.size else self.vv
        return self.vy / n, vu, vv

    def mean_ramps(self):
        n = max(self.copies, 1)
        return {k: v / n for k, v in self.ramps.items()}


@dataclass
class StreamStats:
    video_frames: int = 0
    unreadable_headers: int = 0
    offset_used: tuple = (0, 0)
    design: tuple = (0, 0)
    native: tuple = (0, 0)
    rescaled: bool = False
    info: dict = field(default_factory=dict)


def detect_design(path: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return (design (W,H), native (W,H), pixel offset) by reading the first readable frame."""
    info = video.probe(path)
    native = (info.width, info.height)
    tries = [native] + [c for c in CANDIDATE_DESIGNS if c != native]
    for W, H in tries:
        if W % 16 or H % 8:
            continue
        for Y, U, V in video.iter_frames(path, W, H, limit=3):
            for dx, dy in OFFSETS[:9]:
                hdr, nerr, sep = F.read_header(Y, dx, dy)
                if hdr is not None:
                    if (hdr.width, hdr.height) != (W, H) and hdr.width % 16 == 0 and hdr.height % 8 == 0:
                        return (hdr.width, hdr.height), native, (dx, dy)
                    return (W, H), native, (dx, dy)
    raise RuntimeError("could not find a readable frame header at any candidate resolution")


def iter_logical_frames(path: str, design: tuple[int, int] | None = None, keep_copies: bool = False,
                        progress: bool = True, limit: int | None = None, stats: StreamStats | None = None):
    """Yield LogicalFrame objects; consecutive video frames with the same (stream_id, frame_idx) merge."""
    stats = stats if stats is not None else StreamStats()
    info = video.probe(path)
    stats.info = info.to_dict()
    if design is None:
        design, native, off = detect_design(path)
    else:
        native = (info.width, info.height)
        off = (0, 0)
    stats.design, stats.native, stats.offset_used = design, native, off
    stats.rescaled = design != native
    W, H = design
    cur: LogicalFrame | None = None
    total = info.nb_frames or 0
    for vi, (Y, U, V) in enumerate(video.iter_frames(path, W, H)):
        stats.video_frames += 1
        if progress and (vi % 30 == 0):
            pct = f" {100 * vi / total:5.1f}%" if total else ""
            print(f"\r  decoding frame {vi}{pct}   ", end="", file=sys.stderr, flush=True)
        hdr, nerr, sep = F.read_header(Y, *off)
        if hdr is None:
            # try other pixel offsets once; keep the one that works
            for dx, dy in OFFSETS:
                if (dx, dy) == off:
                    continue
                hdr, nerr, sep = F.read_header(Y, dx, dy)
                if hdr is not None:
                    off = (dx, dy)
                    stats.offset_used = off
                    break
        if hdr is None:
            stats.unreadable_headers += 1
            continue
        if cur is not None and (hdr.stream_id, hdr.frame_idx) != (cur.header.stream_id, cur.header.frame_idx):
            if cur.copies:
                yield cur
            cur = None
            if limit is not None and limit <= 0:
                break
        if cur is None:
            cur = LogicalFrame(hdr, vi)
            if limit is not None:
                limit -= 1
        try:
            g = F.geometry(W, H, hdr.profile, hdr.nsym)
        except ValueError:
            stats.unreadable_headers += 1
            continue
        vy, vu, vv = F.sample_body(Y, U, V, g, *off)
        ramps = F.read_ramps(Y, U, V)
        if cur.vy is None:
            cur.vy, cur.vu, cur.vv = vy.astype(np.float64), vu.astype(np.float64), vv.astype(np.float64)
            cur.ramps = {k: v.copy() for k, v in ramps.items()}
        else:
            cur.vy += vy
            if vu.size:
                cur.vu += vu
                cur.vv += vv
            for k in ramps:
                cur.ramps[k] += ramps[k]
        cur.copies += 1
        cur.header_fixed = max(cur.header_fixed, nerr)
        cur.header_sep = max(cur.header_sep, sep)
        if keep_copies:
            cur.copy_vals.append((vy, vu, vv))
    if cur is not None and cur.copies:
        yield cur
    if progress:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)


def centers_for(lf: LogicalFrame, g: F.Geometry):
    """Level centres for Y, U, V calibrated from the strip then refined on the body."""
    vy, vu, vv = lf.mean_values()
    ramps = lf.mean_ramps()
    p = g.profile
    cy = F.refine_centers(vy, F.calibrated_centers(p.ybits, F.Y_LO, F.Y_HI, ramps["y"]))
    if p.cbits:
        cu = F.refine_centers(vu, F.calibrated_centers(p.cbits, F.C_LO, F.C_HI, ramps["u"]))
        cv = F.refine_centers(vv, F.calibrated_centers(p.cbits, F.C_LO, F.C_HI, ramps["v"]))
    else:
        cu = cv = None
    return cy, cu, cv


def decide_symbols(vy, vu, vv, g: F.Geometry, centers):
    cy, cu, cv = centers
    sy = F.decide(vy, cy)
    if g.profile.cbits:
        su, sv = F.decide(vu, cu), F.decide(vv, cv)
    else:
        su = sv = np.zeros((0, 0), dtype=np.uint8)
    return sy, su, sv


@dataclass
class Decoded:
    payload: np.ndarray
    rs_ok: bool
    crc_ok: bool
    nerr_cw: np.ndarray
    stream: np.ndarray
    symbols: tuple
    centers: tuple


def decode_logical(lf: LogicalFrame, g: F.Geometry) -> Decoded:
    centers = centers_for(lf, g)
    vy, vu, vv = lf.mean_values()
    sy, su, sv = decide_symbols(vy, vu, vv, g, centers)
    stream = F.symbols_to_stream(sy, su, sv, g)
    payload, rs_ok, nerr = F.body_to_bytes(stream, g, lf.header.stream_id, lf.header.frame_idx)
    crc_ok = bool(rs_ok and zlib.crc32(payload.tobytes()) == lf.header.payload_crc
                  and len(payload) == lf.header.payload_len)
    return Decoded(payload, rs_ok, crc_ok, nerr, stream, (sy, su, sv), centers)
