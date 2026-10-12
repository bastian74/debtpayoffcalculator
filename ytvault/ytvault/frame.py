"""Frame layout: header band, calibration strip, body symbol grid.

Every video frame is a YUV 4:2:0 image laid out as::

    +------------------------------------------------------------+
    | header band   : 80-byte RS(80,48) header, 8 px binary cells |  hdr_rows * 8 px
    | calib strip   : Y ramp | U ramp | V ramp (16 steps each)    |  8 px
    | body          : cell x cell luma symbols (2^ybits levels)    |
    |                 2cell x 2cell chroma symbols (2^cbits levels)|
    +------------------------------------------------------------+

The header is always written at the most conservative setting so a decoder can
read it first and learn the body geometry.  The strip lets the decoder measure
how the transcoder moved the grey levels before deciding symbols.

The body bit stream is: ``ncw`` Reed-Solomon codewords of 255 bytes,
byte-interleaved (codeword j byte i sits at stream byte i*ncw + j), XOR-whitened
with a per-frame keystream, then dealt to cells in a fixed pseudo-random cell
order so any spatial burst spreads across every codeword and the luma / chroma
channels share codewords.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

from . import gf256

MAGIC = b"YTVD"
VERSION = 1
HC = 8                 # header cell size in pixels
HEADER_DATA = 48       # header struct size
HEADER_NSYM = 32       # header RS parity bytes -> corrects 16 of 80 bytes
HEADER_BYTES = HEADER_DATA + HEADER_NSYM
HEADER_BITS = HEADER_BYTES * 8
RAMP_STEPS = 16
Y_LO, Y_HI = 16, 235   # studio-range luma
C_LO, C_HI = 40, 216   # conservative chroma range (avoids out-of-gamut clipping)

KIND_CALIB = 0
KIND_DATA = 1
KIND_PARITY = 2
KIND_MANIFEST = 3
KIND_NAMES = {0: "calib", 1: "data", 2: "parity", 3: "manifest"}

HDR_FMT = "<4sBBIIBBBBBBBHHIIII3sI"
assert struct.calcsize(HDR_FMT) == HEADER_DATA

# ------------------------------------------------------------------ PRNG ----
# A counter-based splitmix64 keystream: independent of numpy's RNG so encoded
# videos stay decodable by any future version on any platform.

def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _seed64(*parts: int) -> np.uint64:
    h = np.uint64(0x243F6A8885A308D3)
    for p in parts:
        h = _splitmix64(np.array([h ^ np.uint64(p & 0xFFFFFFFFFFFFFFFF)], dtype=np.uint64))[0]
    return h


def prng_u64(seed_parts: tuple[int, ...], n: int) -> np.ndarray:
    seed = _seed64(*seed_parts)
    with np.errstate(over="ignore"):
        ctr = np.arange(n, dtype=np.uint64) * np.uint64(0xD1B54A32D192ED03) + seed
    return _splitmix64(ctr)


def prng_bytes(seed_parts: tuple[int, ...], n: int) -> np.ndarray:
    words = prng_u64(seed_parts, (n + 7) // 8)
    return words.view(np.uint8)[:n].copy()


TAG_PAYLOAD, TAG_WHITEN, TAG_PERM, TAG_FILL = 0xA1, 0xB2, 0xC3, 0xD4


# ------------------------------------------------------------- Gray code ----

def gray_encode(v: np.ndarray) -> np.ndarray:
    return v ^ (v >> 1)


def gray_decode(g: np.ndarray) -> np.ndarray:
    v = g.copy()
    shift = 1
    while shift < 8:
        v ^= v >> shift
        shift <<= 1
    return v


# --------------------------------------------------------------- profile ----

@dataclass(frozen=True, order=True)
class Profile:
    cell: int = 4      # luma cell edge in px (chroma cells are 2*cell px)
    ybits: int = 1     # bits per luma cell (1..4)
    cbits: int = 0     # bits per chroma cell per plane (0..2)
    repeat: int = 1    # video frames per data frame

    @property
    def label(self) -> str:
        return f"c{self.cell}-y{self.ybits}-c{self.cbits}-r{self.repeat}"

    @classmethod
    def parse(cls, s: str) -> "Profile":
        parts = dict(cell=4, ybits=1, cbits=0, repeat=1)
        keys = {"c": "cell", "y": "ybits", "r": "repeat"}
        toks = s.split("-")
        # format c4-y1-c0-r1 : second 'c' is chroma bits
        seen_c = 0
        for t in toks:
            k, v = t[0], int(t[1:])
            if k == "c":
                parts["cell" if seen_c == 0 else "cbits"] = v
                seen_c += 1
            else:
                parts[keys[k]] = v
        return cls(**parts)

    def validate(self):
        if not (2 <= self.cell <= 64):
            raise ValueError("cell must be 2..64")
        if not (1 <= self.ybits <= 4):
            raise ValueError("ybits must be 1..4")
        if not (0 <= self.cbits <= 3):
            raise ValueError("cbits must be 0..3")
        if not (1 <= self.repeat <= 8):
            raise ValueError("repeat must be 1..8")


SAFE_PROFILE = Profile(cell=8, ybits=1, cbits=0, repeat=2)
SAFE_NSYM = 64


def level_values(nbits: int, lo: int, hi: int) -> np.ndarray:
    L = 1 << nbits
    return np.rint(lo + np.arange(L) * (hi - lo) / (L - 1)).astype(np.uint8)


# -------------------------------------------------------------- geometry ----

@dataclass
class Geometry:
    W: int
    H: int
    profile: Profile
    nsym: int
    hdr_cols: int
    hdr_rows: int
    body_y0: int
    rows: int
    cols: int
    crows: int
    ccols: int
    ncells: int
    capacity_bits: int
    ncw: int
    payload_len: int
    perm: np.ndarray          # stream index -> canonical cell index
    offsets: np.ndarray       # bit offset of each stream cell
    stream_bits: np.ndarray   # bits per stream cell
    is_luma_stream: np.ndarray

    @property
    def nluma(self) -> int:
        return self.rows * self.cols

    @property
    def nchroma(self) -> int:
        return self.crows * self.ccols

    @property
    def body_bytes(self) -> int:
        return self.ncw * 255

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray)}
        d["profile"] = self.profile.label
        return d


def header_geometry(W: int) -> tuple[int, int, int]:
    hdr_cols = W // HC
    hdr_rows = -(-HEADER_BITS // hdr_cols)
    body_y0 = (hdr_rows + 1) * HC  # + one strip row
    return hdr_cols, hdr_rows, body_y0


@lru_cache(maxsize=64)
def geometry(W: int, H: int, profile: Profile, nsym: int) -> Geometry:
    profile.validate()
    if W % 16 or H % 8:
        raise ValueError("width must be a multiple of 16 and height a multiple of 8")
    if not (2 <= nsym <= 128) or nsym % 2:
        raise ValueError("nsym must be even and 2..128")
    c = profile.cell
    hdr_cols, hdr_rows, body_y0 = header_geometry(W)
    rows = (H - body_y0) // c
    cols = W // c
    if profile.cbits:
        crows = ((H - body_y0) // 2) // c
        ccols = (W // 2) // c
    else:
        crows = ccols = 0
    nl, nc = rows * cols, crows * ccols
    ncells = nl + 2 * nc
    cell_bits = np.concatenate([
        np.full(nl, profile.ybits, dtype=np.int64),
        np.full(2 * nc, profile.cbits, dtype=np.int64),
    ])
    keys = prng_u64((TAG_PERM, W, H, c, profile.ybits, profile.cbits), ncells)
    perm = np.argsort(keys, kind="stable")
    stream_bits = cell_bits[perm]
    offsets = np.cumsum(stream_bits) - stream_bits
    capacity = int(stream_bits.sum())
    ncw = capacity // (255 * 8)
    if ncw < 1:
        raise ValueError("frame too small for even one codeword")
    return Geometry(W, H, profile, nsym, hdr_cols, hdr_rows, body_y0, rows, cols, crows, ccols,
                    ncells, capacity, ncw, ncw * (255 - nsym), perm, offsets, stream_bits,
                    perm < nl)


# ---------------------------------------------------------------- header ----

@dataclass
class Header:
    kind: int
    stream_id: int
    frame_idx: int
    profile: Profile
    nsym: int
    k_outer: int
    m_outer: int
    width: int
    height: int
    payload_len: int
    payload_crc: int
    seq: int
    total: int

    def pack(self) -> bytes:
        p = self.profile
        body = struct.pack(HDR_FMT[:-1], MAGIC, VERSION, self.kind, self.stream_id, self.frame_idx,
                           p.cell, p.ybits, p.cbits, p.repeat, self.nsym, self.k_outer, self.m_outer,
                           self.width, self.height, self.payload_len, self.payload_crc,
                           self.seq, self.total, b"\0\0\0")
        return body + struct.pack("<I", zlib.crc32(body))

    @classmethod
    def unpack(cls, raw: bytes) -> "Header | None":
        if len(raw) != HEADER_DATA:
            return None
        (magic, ver, kind, sid, fidx, cell, yb, cb, rep, nsym, ko, mo, w, h, plen, pcrc, seq, total,
         _res, crc) = struct.unpack(HDR_FMT, raw)
        if magic != MAGIC or ver != VERSION or crc != zlib.crc32(raw[:-4]):
            return None
        try:
            prof = Profile(cell, yb, cb, rep)
            prof.validate()
        except ValueError:
            return None
        return cls(kind, sid, fidx, prof, nsym, ko, mo, w, h, plen, pcrc, seq, total)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["profile"] = self.profile.label
        d["kind_name"] = KIND_NAMES.get(self.kind, "?")
        return d


def encode_header_bits(h: Header) -> np.ndarray:
    raw = np.frombuffer(h.pack(), dtype=np.uint8)
    cw = gf256.encode(raw[None, :], HEADER_NSYM)[0]
    return np.unpackbits(cw)


def decode_header_bits(bits: np.ndarray) -> tuple[Header | None, int]:
    """bits: HEADER_BITS 0/1 -> (Header or None, number of corrected bytes or -1)."""
    cw = np.packbits(bits[:HEADER_BITS].astype(np.uint8))
    fixed, ok, nerr = gf256.decode(cw[None, :], HEADER_NSYM)
    if not ok[0]:
        return None, -1
    return Header.unpack(fixed[0, :HEADER_DATA].tobytes()), int(nerr[0])


# ------------------------------------------------------------ rendering ----

def _fill_cells(plane: np.ndarray, y0: int, grid: np.ndarray, c: int):
    rows, cols = grid.shape
    plane[y0:y0 + rows * c, :cols * c] = np.repeat(np.repeat(grid, c, axis=0), c, axis=1)


def render_header_band(Y: np.ndarray, U: np.ndarray, V: np.ndarray, h: Header):
    W = Y.shape[1]
    hdr_cols, hdr_rows, body_y0 = header_geometry(W)
    bits = encode_header_bits(h)
    cells = np.zeros(hdr_rows * hdr_cols, dtype=np.uint8)
    cells[:HEADER_BITS] = bits
    # unused header cells: alternating pattern (keeps the band looking like data)
    if hdr_rows * hdr_cols > HEADER_BITS:
        cells[HEADER_BITS:] = np.arange(hdr_rows * hdr_cols - HEADER_BITS) & 1
    lv = np.array([Y_LO, Y_HI], dtype=np.uint8)
    _fill_cells(Y, 0, lv[cells.reshape(hdr_rows, hdr_cols)], HC)
    U[: body_y0 // 2, :] = 128
    V[: body_y0 // 2, :] = 128
    # calibration strip
    y0 = hdr_rows * HC
    strip_y = Y[y0:y0 + HC, :]
    strip_y[:] = 128
    half, quarter = W // 2, W // 4
    yramp = level_values(4, Y_LO, Y_HI)  # 16 steps
    cramp = level_values(4, C_LO, C_HI)
    xs = (np.arange(half) * RAMP_STEPS) // half
    strip_y[:, :half] = yramp[xs][None, :]
    cx = (np.arange(quarter // 2) * RAMP_STEPS) // (quarter // 2)
    cy0 = y0 // 2
    U[cy0:cy0 + HC // 2, half // 2: half // 2 + quarter // 2] = cramp[cx][None, :]
    V[cy0:cy0 + HC // 2, (half + quarter) // 2: (half + quarter) // 2 + quarter // 2] = cramp[cx][None, :]


def bytes_to_body(payload: np.ndarray, g: Geometry, stream_id: int, frame_idx: int) -> np.ndarray:
    """payload (payload_len bytes) -> whitened, interleaved body byte stream (ncw*255)."""
    if len(payload) != g.payload_len:
        raise ValueError(f"payload must be exactly {g.payload_len} bytes, got {len(payload)}")
    msgs = np.asarray(payload, dtype=np.uint8).reshape(g.ncw, 255 - g.nsym)
    cws = gf256.encode(msgs, g.nsym)                      # (ncw, 255)
    stream = cws.T.reshape(-1)                            # interleave
    return stream ^ prng_bytes((TAG_WHITEN, stream_id, frame_idx), stream.size)


def body_to_bytes(stream: np.ndarray, g: Geometry, stream_id: int, frame_idx: int):
    """Inverse of bytes_to_body: returns (payload, ok, per-codeword corrected counts)."""
    stream = stream ^ prng_bytes((TAG_WHITEN, stream_id, frame_idx), stream.size)
    cws = stream.reshape(255, g.ncw).T.copy()
    fixed, ok, nerr = gf256.decode(cws, g.nsym)
    payload = fixed[:, :255 - g.nsym].reshape(-1)
    return payload, bool(ok.all()), nerr


def stream_to_symbols(stream: np.ndarray, g: Geometry, stream_id: int, frame_idx: int):
    """Body byte stream -> (sym_y (rows,cols), sym_u (crows,ccols), sym_v) level indices."""
    bits = np.unpackbits(stream)
    if bits.size < g.capacity_bits:
        fill = np.unpackbits(prng_bytes((TAG_FILL, stream_id, frame_idx), (g.capacity_bits - bits.size + 7) // 8))
        bits = np.concatenate([bits, fill[: g.capacity_bits - bits.size]])
    bits = bits.astype(np.uint8)
    canon = np.zeros(g.ncells, dtype=np.uint8)
    for mask, nb in ((g.is_luma_stream, g.profile.ybits), (~g.is_luma_stream, g.profile.cbits)):
        if nb == 0 or not mask.any():
            continue
        off = g.offsets[mask]
        val = np.zeros(off.size, dtype=np.uint8)
        for j in range(nb):
            val = (val << 1) | bits[off + j]
        canon[g.perm[mask]] = gray_encode(val)
    nl, nc = g.nluma, g.nchroma
    sy = canon[:nl].reshape(g.rows, g.cols)
    su = canon[nl:nl + nc].reshape(g.crows, g.ccols) if nc else np.zeros((0, 0), np.uint8)
    sv = canon[nl + nc:].reshape(g.crows, g.ccols) if nc else np.zeros((0, 0), np.uint8)
    return sy, su, sv


def symbols_to_stream(sy: np.ndarray, su: np.ndarray, sv: np.ndarray, g: Geometry) -> np.ndarray:
    """Level indices -> body byte stream (ncw*255 bytes)."""
    canon = np.concatenate([sy.reshape(-1), su.reshape(-1), sv.reshape(-1)]).astype(np.uint8)
    bits = np.zeros(g.capacity_bits, dtype=np.uint8)
    for mask, nb in ((g.is_luma_stream, g.profile.ybits), (~g.is_luma_stream, g.profile.cbits)):
        if nb == 0 or not mask.any():
            continue
        off = g.offsets[mask]
        val = gray_decode(canon[g.perm[mask]])
        for j in range(nb):
            bits[off + j] = (val >> (nb - 1 - j)) & 1
    nbytes = g.body_bytes
    return np.packbits(bits[: nbytes * 8])


def render_frame(h: Header, payload: np.ndarray, g: Geometry):
    """Build Y (H,W), U, V (H/2,W/2) planes for one data frame."""
    W, H = g.W, g.H
    Y = np.full((H, W), 128, dtype=np.uint8)
    U = np.full((H // 2, W // 2), 128, dtype=np.uint8)
    V = np.full((H // 2, W // 2), 128, dtype=np.uint8)
    render_header_band(Y, U, V, h)
    stream = bytes_to_body(payload, g, h.stream_id, h.frame_idx)
    sy, su, sv = stream_to_symbols(stream, g, h.stream_id, h.frame_idx)
    ylv = level_values(g.profile.ybits, Y_LO, Y_HI)
    _fill_cells(Y, g.body_y0, ylv[sy], g.profile.cell)
    if g.profile.cbits:
        clv = level_values(g.profile.cbits, C_LO, C_HI)
        _fill_cells(U, g.body_y0 // 2, clv[su], g.profile.cell)
        _fill_cells(V, g.body_y0 // 2, clv[sv], g.profile.cell)
    return Y, U, V


# -------------------------------------------------------------- sampling ----

def cell_means(plane: np.ndarray, y0: int, rows: int, cols: int, c: int, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Mean of the inner part of each c x c cell -> float32 (rows, cols)."""
    m = c // 4
    ys, xs = y0 + dy, dx
    blk = plane[ys:ys + rows * c, xs:xs + cols * c]
    if blk.shape != (rows * c, cols * c):
        # shifted sampling ran off the frame: pad with edge values
        blk = np.pad(plane, ((max(0, -ys), max(0, ys + rows * c - plane.shape[0])),
                             (max(0, -xs), max(0, xs + cols * c - plane.shape[1]))), mode="edge")
        blk = blk[max(ys, 0):max(ys, 0) + rows * c, max(xs, 0):max(xs, 0) + cols * c]
    blk = blk.reshape(rows, c, cols, c)[:, m:c - m, :, m:c - m]
    return blk.mean(axis=(1, 3), dtype=np.float32)


def read_header(Y: np.ndarray, dx: int = 0, dy: int = 0) -> tuple[Header | None, int, float]:
    """Sample the header band -> (Header|None, corrected bytes, binary separation)."""
    W = Y.shape[1]
    hdr_cols, hdr_rows, _ = header_geometry(W)
    v = cell_means(Y, 0, hdr_rows, hdr_cols, HC, dx, dy).reshape(-1)[:HEADER_BITS]
    thr = 0.5 * (Y_LO + Y_HI)
    for _ in range(3):  # 2-means refinement
        lo, hi = v[v < thr], v[v >= thr]
        if lo.size == 0 or hi.size == 0:
            break
        thr = 0.5 * (lo.mean() + hi.mean())
    bits = (v >= thr).astype(np.uint8)
    hdr, nerr = decode_header_bits(bits)
    lo, hi = v[v < thr], v[v >= thr]
    sep = float((hi.mean() - lo.mean()) / (lo.std() + hi.std() + 1e-6)) if lo.size and hi.size else 0.0
    return hdr, nerr, sep


def read_ramps(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> dict:
    """Observed mean of each of the 16 strip steps for Y, U, V."""
    W = Y.shape[1]
    hdr_cols, hdr_rows, _ = header_geometry(W)
    y0 = hdr_rows * HC
    half, quarter = W // 2, W // 4
    out = {}
    step = half // RAMP_STEPS
    ys = Y[y0 + 2:y0 + HC - 2, :half]
    out["y"] = np.array([ys[:, i * step + step // 4: (i + 1) * step - step // 4].mean() for i in range(RAMP_STEPS)])
    cstep = (quarter // 2) // RAMP_STEPS
    cy0 = y0 // 2
    us = U[cy0 + 1:cy0 + HC // 2 - 1, half // 2: half // 2 + quarter // 2]
    vs = V[cy0 + 1:cy0 + HC // 2 - 1, (half + quarter) // 2:(half + quarter) // 2 + quarter // 2]
    out["u"] = np.array([us[:, i * cstep + cstep // 4:(i + 1) * cstep - cstep // 4].mean() for i in range(RAMP_STEPS)])
    out["v"] = np.array([vs[:, i * cstep + cstep // 4:(i + 1) * cstep - cstep // 4].mean() for i in range(RAMP_STEPS)])
    return out


def sample_body(Y, U, V, g: Geometry, dx: int = 0, dy: int = 0):
    vy = cell_means(Y, g.body_y0, g.rows, g.cols, g.profile.cell, dx, dy)
    if g.profile.cbits:
        vu = cell_means(U, g.body_y0 // 2, g.crows, g.ccols, g.profile.cell, dx // 2, dy // 2)
        vv = cell_means(V, g.body_y0 // 2, g.crows, g.ccols, g.profile.cell, dx // 2, dy // 2)
    else:
        vu = vv = np.zeros((0, 0), dtype=np.float32)
    return vy, vu, vv


def calibrated_centers(nbits: int, lo: int, hi: int, ramp_obs: np.ndarray | None) -> np.ndarray:
    """Map the nominal level values through the observed ramp (nominal->observed)."""
    nominal = level_values(nbits, lo, hi).astype(np.float64)
    if ramp_obs is None:
        return nominal
    ramp_nom = level_values(4, lo, hi).astype(np.float64)
    obs = np.asarray(ramp_obs, dtype=np.float64)
    if not np.all(np.diff(obs) > 0):
        obs = np.maximum.accumulate(obs)  # keep monotone; a broken ramp degrades to nominal-ish
        if obs[-1] - obs[0] < 8:
            return nominal
    return np.interp(nominal, ramp_nom, obs)


def refine_centers(values: np.ndarray, centers: np.ndarray, iters: int = 2) -> np.ndarray:
    """A couple of Lloyd iterations: pull the level centres to where the data actually sit."""
    c = centers.astype(np.float64).copy()
    v = values.reshape(-1)
    if v.size == 0:
        return c
    for _ in range(iters):
        mid = 0.5 * (c[1:] + c[:-1])
        idx = np.searchsorted(mid, v)
        sums = np.bincount(idx, weights=v, minlength=c.size)
        cnt = np.bincount(idx, minlength=c.size)
        upd = cnt > max(8, v.size // (c.size * 50))
        c[upd] = sums[upd] / cnt[upd]
        c = np.sort(c)
    return c


def decide(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    mid = 0.5 * (centers[1:] + centers[:-1])
    return np.searchsorted(mid, values.reshape(-1)).astype(np.uint8).reshape(values.shape)


def level_stats(values: np.ndarray, expected: np.ndarray, nlevels: int):
    """Per-level (count, sum, sumsq) of observed values grouped by the expected level."""
    e = expected.reshape(-1).astype(np.int64)
    v = values.reshape(-1).astype(np.float64)
    cnt = np.bincount(e, minlength=nlevels).astype(np.float64)
    s1 = np.bincount(e, weights=v, minlength=nlevels)
    s2 = np.bincount(e, weights=v * v, minlength=nlevels)
    return np.stack([cnt, s1, s2])


def eye_margin(stats: np.ndarray) -> float:
    """Min over adjacent levels of gap / (sigma_a + sigma_b).  inf if noiseless."""
    cnt, s1, s2 = stats
    ok = cnt > 1
    if ok.sum() < 2:
        return float("inf")
    mu = np.where(ok, s1 / np.maximum(cnt, 1), np.nan)
    var = np.where(ok, s2 / np.maximum(cnt, 1) - mu ** 2, np.nan)
    sd = np.sqrt(np.maximum(var, 0))
    mus, sds = mu[ok], sd[ok]
    gaps = np.diff(mus)
    denom = sds[1:] + sds[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(denom > 1e-9, gaps / denom, np.inf)
    return float(np.min(m)) if m.size else float("inf")
