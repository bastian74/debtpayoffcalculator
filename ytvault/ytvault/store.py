"""Pack a file into a video (and back) at a chosen profile, with inner RS + outer frame parity."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import secrets
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from . import frame as F
from . import gf256
from . import video
from .decoder import iter_logical_frames, StreamStats, decode_logical
from .frame import Profile, Header, geometry

MANIFEST_COPIES = 3
MANIFEST_MAGIC = b"YTVM"
MANIFEST_FMT = "<4sBQ32sBBBBBBBHHBIIIIB"   # 74 bytes, then the file name
MANIFEST_FIXED = struct.calcsize(MANIFEST_FMT)


def pack_manifest(m: dict, capacity: int) -> np.ndarray:
    """Binary manifest (fits the 191-byte safe-profile payload of a 640x360 frame)."""
    if capacity < MANIFEST_FIXED + 1:
        raise ValueError("frame too small to hold a manifest; use a larger resolution")
    name = m["name"].encode("utf-8")[: min(255, capacity - MANIFEST_FIXED)]
    name = name.decode("utf-8", "ignore").encode("utf-8")  # never cut a multibyte character
    p = Profile.parse(m["profile"])
    raw = struct.pack(MANIFEST_FMT, MANIFEST_MAGIC, 1, m["size"], bytes.fromhex(m["sha256"]),
                      p.cell, p.ybits, p.cbits, p.repeat, m["nsym"], m["k_outer"], m["m_outer"],
                      m["width"], m["height"], m["fps"], m["chunk_len"], m["nchunks"], m["stream_id"],
                      m["created_unix"], len(name)) + name
    out = np.zeros(capacity, dtype=np.uint8)
    out[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def parse_manifest(payload: np.ndarray) -> dict | None:
    raw = payload.tobytes()
    if len(raw) < MANIFEST_FIXED:
        return None
    (magic, ver, size, sha, cell, yb, cb, rep, nsym, ko, mo, w, h, fps, chunk_len, nchunks, sid, created,
     nlen) = struct.unpack(MANIFEST_FMT, raw[:MANIFEST_FIXED])
    if magic != MANIFEST_MAGIC:
        return None
    name = raw[MANIFEST_FIXED:MANIFEST_FIXED + nlen].decode("utf-8", "replace")
    return dict(format=f"ytvault/{ver}", name=name, size=size, sha256=sha.hex(),
                profile=Profile(cell, yb, cb, rep).label, nsym=nsym, k_outer=ko, m_outer=mo, width=w, height=h,
                fps=fps, chunk_len=chunk_len, nchunks=nchunks, stream_id=sid, created_unix=created,
                created=_dt.datetime.fromtimestamp(created, _dt.timezone.utc).isoformat(timespec="seconds"))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_frame(w: video.FrameWriter, h: Header, payload: np.ndarray, g: F.Geometry):
    Y, U, V = F.render_frame(h, payload, g)
    for _ in range(g.profile.repeat):
        w.write(Y, U, V)


def pack(src: str, out_path: str, profile: Profile, nsym: int = 32, k_outer: int = 20, m_outer: int = 4,
         width: int = 1920, height: int = 1080, fps: int = 30, crf: int = 12, stream_id: int | None = None,
         progress: bool = True) -> dict:
    """Encode ``src`` into ``out_path``.  Returns the manifest dict."""
    if k_outer + m_outer > 255 or (m_outer and k_outer < 1):
        raise ValueError("outer code needs 1 <= K and K + M <= 255")
    stream_id = stream_id if stream_id is not None else secrets.randbits(32)
    g = geometry(width, height, profile, nsym)
    gm = geometry(width, height, F.SAFE_PROFILE, F.SAFE_NSYM)
    size = os.path.getsize(src)
    chunk_len = g.payload_len
    nchunks = max(1, math.ceil(size / chunk_len))
    ngroups = math.ceil(nchunks / k_outer) if m_outer else nchunks
    now = int(time.time())
    manifest = dict(format="ytvault/1", name=os.path.basename(src), size=size, sha256=_sha256(src),
                    profile=profile.label, nsym=nsym, k_outer=k_outer, m_outer=m_outer, width=width,
                    height=height, fps=fps, chunk_len=chunk_len, nchunks=nchunks, stream_id=stream_id,
                    created_unix=now,
                    created=_dt.datetime.fromtimestamp(now, _dt.timezone.utc).isoformat(timespec="seconds"))
    mpayload = pack_manifest(manifest, gm.payload_len)
    mcrc = zlib.crc32(mpayload.tobytes())
    total_logical = 2 * MANIFEST_COPIES + nchunks + (ngroups * m_outer if m_outer else 0)
    frame_idx = 0

    def manifest_frames(w):
        nonlocal frame_idx
        for _ in range(MANIFEST_COPIES):
            h = Header(F.KIND_MANIFEST, stream_id, frame_idx, F.SAFE_PROFILE, F.SAFE_NSYM, 0, 0, width, height,
                       gm.payload_len, mcrc, 0, nchunks)
            _write_frame(w, h, mpayload, gm)
            frame_idx += 1

    with open(src, "rb") as fh, video.FrameWriter(out_path, width, height, fps, crf=crf) as w:
        manifest_frames(w)
        for gi in range(ngroups):
            first = gi * k_outer if m_outer else gi
            kk = min(k_outer, nchunks - first) if m_outer else 1
            chunks = []
            for ci in range(first, first + kk):
                data = fh.read(chunk_len)
                buf = np.zeros(chunk_len, dtype=np.uint8)
                buf[:len(data)] = np.frombuffer(data, dtype=np.uint8)
                chunks.append(buf)
                h = Header(F.KIND_DATA, stream_id, frame_idx, profile, nsym, kk if m_outer else 0, m_outer,
                           width, height, chunk_len, zlib.crc32(buf.tobytes()), ci, nchunks)
                _write_frame(w, h, buf, g)
                frame_idx += 1
                if progress and frame_idx % 5 == 0:
                    print(f"\r  frame {frame_idx}/{total_logical}  chunk {ci + 1}/{nchunks}   ",
                          end="", file=sys.stderr, flush=True)
            if m_outer:
                mat = np.stack(chunks, axis=1)                       # (chunk_len, kk): one column per frame
                cws = gf256.encode(mat, m_outer)                     # (chunk_len, kk + m)
                for j in range(m_outer):
                    par = np.ascontiguousarray(cws[:, kk + j])
                    h = Header(F.KIND_PARITY, stream_id, frame_idx, profile, nsym, kk, m_outer, width, height,
                               chunk_len, zlib.crc32(par.tobytes()), gi * m_outer + j, nchunks)
                    _write_frame(w, h, par, g)
                    frame_idx += 1
        manifest_frames(w)
    if progress:
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    manifest["logical_frames"] = frame_idx
    manifest["video_frames"] = w.frames
    manifest["duration_s"] = w.frames / fps
    manifest["video_bytes"] = os.path.getsize(out_path)
    with open(out_path + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


@dataclass
class UnpackResult:
    manifest: dict | None
    ok: bool
    sha_ok: bool
    out_path: str | None
    logical_frames: int = 0
    data_frames_ok: int = 0
    data_frames_bad: int = 0
    parity_frames_ok: int = 0
    parity_frames_bad: int = 0
    repaired_chunks: int = 0
    lost_chunks: int = 0
    missing_chunks: list = field(default_factory=list)
    inner_corrections: int = 0
    worst_cw_err: int = 0
    stats: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__.copy()


def unpack(video_path: str, out_dir: str | None, design: tuple[int, int] | None = None, progress: bool = True,
           dry_run: bool = False) -> UnpackResult:
    stats = StreamStats()
    manifest = None
    chunks: dict[int, np.ndarray] = {}
    parity: dict[tuple[int, int], np.ndarray] = {}
    group_k: dict[int, int] = {}
    res = UnpackResult(None, False, False, None)
    nchunks_hdr = None
    m_outer = 0
    k_outer = 0
    for lf in iter_logical_frames(video_path, design=design, progress=progress, stats=stats):
        h = lf.header
        res.logical_frames += 1
        W, H = stats.design
        try:
            g = geometry(W, H, h.profile, h.nsym)
        except ValueError:
            continue
        dec = decode_logical(lf, g)
        if dec.rs_ok:
            res.inner_corrections += int(dec.nerr_cw.sum())
            res.worst_cw_err = max(res.worst_cw_err, int(dec.nerr_cw.max()))
        if h.kind == F.KIND_MANIFEST:
            if manifest is None and dec.crc_ok:
                manifest = parse_manifest(dec.payload)
                if manifest is None:
                    res.notes.append("manifest frame decoded but did not parse")
            continue
        if h.kind == F.KIND_DATA:
            nchunks_hdr = h.total
            m_outer, k_outer = h.m_outer, max(k_outer, h.k_outer)
            if dec.crc_ok:
                if h.seq not in chunks:
                    chunks[h.seq] = dec.payload
                res.data_frames_ok += 1
            else:
                res.data_frames_bad += 1
        elif h.kind == F.KIND_PARITY:
            m_outer = h.m_outer
            gi, j = divmod(h.seq, max(h.m_outer, 1))
            group_k[gi] = h.k_outer
            if dec.crc_ok:
                parity.setdefault((gi, j), dec.payload)
                res.parity_frames_ok += 1
            else:
                res.parity_frames_bad += 1
    res.stats = dict(video_frames=stats.video_frames, unreadable_headers=stats.unreadable_headers,
                     design=stats.design, native=stats.native, rescaled=stats.rescaled, offset=stats.offset_used)
    if manifest is None:
        res.notes.append("no manifest frame could be decoded; falling back to header fields")
        if nchunks_hdr is None:
            res.notes.append("no data frames decoded at all")
            return res
        manifest = dict(name="recovered.bin", size=None, sha256=None, nchunks=nchunks_hdr,
                        k_outer=k_outer, m_outer=m_outer)
    res.manifest = manifest
    nchunks = manifest["nchunks"]
    K, M = manifest.get("k_outer", k_outer), manifest.get("m_outer", m_outer)
    chunk_len = manifest.get("chunk_len") or (next(iter(chunks.values())).size if chunks else None)
    if chunk_len is None:
        res.notes.append("could not determine chunk length")
        return res
    # outer repair, group by group
    if M:
        ngroups = math.ceil(nchunks / K)
        for gi in range(ngroups):
            first = gi * K
            kk = min(K, nchunks - first)
            missing = [ci for ci in range(first, first + kk) if ci not in chunks]
            if not missing:
                continue
            have_par = [j for j in range(M) if (gi, j) in parity]
            if len(missing) > len(have_par):
                res.lost_chunks += len(missing)
                res.missing_chunks += missing
                res.notes.append(f"group {gi}: {len(missing)} chunks lost, only {len(have_par)} parity frames usable")
                continue
            cols = [chunks.get(ci, np.zeros(chunk_len, dtype=np.uint8)) for ci in range(first, first + kk)]
            cols += [parity.get((gi, j), np.zeros(chunk_len, dtype=np.uint8)) for j in range(M)]
            cw = np.stack(cols, axis=1)                                     # (chunk_len, kk+M)
            erase = [ci - first for ci in missing] + [kk + j for j in range(M) if j not in have_par]
            fixed, ok, nerr = gf256.decode(cw, M, erase_pos=erase)
            if not ok.all():
                res.lost_chunks += len(missing)
                res.missing_chunks += missing
                res.notes.append(f"group {gi}: outer decode failed on {int((~ok).sum())} columns")
                continue
            for ci in missing:
                chunks[ci] = np.ascontiguousarray(fixed[:, ci - first])
                res.repaired_chunks += 1
    else:
        missing = [ci for ci in range(nchunks) if ci not in chunks]
        res.lost_chunks = len(missing)
        res.missing_chunks = missing
    # assemble
    size = manifest.get("size")
    out = bytearray()
    for ci in range(nchunks):
        out += (chunks[ci].tobytes() if ci in chunks else bytes(chunk_len))
    if size is not None:
        out = out[:size]
    digest = hashlib.sha256(out).hexdigest()
    res.sha_ok = (manifest.get("sha256") == digest)
    res.ok = res.lost_chunks == 0 and (res.sha_ok or manifest.get("sha256") is None)
    if not res.ok and res.lost_chunks == 0 and manifest.get("sha256"):
        res.notes.append("all chunks present but SHA-256 mismatch: an undetected corruption slipped through CRC32")
    if out_dir is not None and not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.basename(manifest.get("name") or "recovered.bin") or "recovered.bin"
        path = os.path.join(out_dir, name if res.ok else name + ".partial")
        with open(path, "wb") as fh:
            fh.write(out)
        res.out_path = path
    return res
