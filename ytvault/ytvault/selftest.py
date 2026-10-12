"""In-memory self tests (no ffmpeg needed): RS codec, frame roundtrip, noisy frame decode."""
from __future__ import annotations

import time
import zlib

import numpy as np

from . import frame as F
from . import gf256 as G
from .frame import Profile, Header, geometry


def test_rs() -> list[str]:
    log = []
    rng = np.random.default_rng(1)
    for nsym in (8, 16, 32, 64):
        k = 255 - nsym
        msg = rng.integers(0, 256, size=(100, k), dtype=np.uint8)
        cw = G.encode(msg, nsym)
        assert not np.any(G.syndromes(cw, nsym)), "clean codewords must have zero syndromes"
        noisy = cw.copy()
        n_true = rng.integers(0, nsym // 2 + 1, size=100)
        for r in range(100):
            pos = rng.choice(255, size=n_true[r], replace=False)
            noisy[r, pos] ^= rng.integers(1, 256, size=len(pos)).astype(np.uint8)
        t = time.time()
        out, ok, nerr = G.decode(noisy, nsym)
        assert ok.all() and np.array_equal(out, cw) and np.array_equal(nerr, n_true), f"RS errors nsym={nsym}"
        log.append(f"RS nsym={nsym}: 100 codewords, up to {nsym // 2} random errors each, all corrected "
                   f"({(time.time() - t) * 1000:.0f} ms)")
        beyond = cw.copy()
        for r in range(100):
            pos = rng.choice(255, size=nsym // 2 + 2, replace=False)
            beyond[r, pos] ^= rng.integers(1, 256, size=len(pos)).astype(np.uint8)
        out2, ok2, _ = G.decode(beyond, nsym)
        wrong = int((ok2 & np.any(out2 != cw, axis=1)).sum())
        log.append(f"RS nsym={nsym}: beyond capacity -> {int((~ok2).sum())}/100 flagged, {wrong} miscorrected "
                   f"(CRC32 catches these)")
    for nsym in (4, 8, 32):
        k = 255 - nsym
        msg = rng.integers(0, 256, size=(2000, k), dtype=np.uint8)
        cw = G.encode(msg, nsym)
        erase = sorted(rng.choice(255, size=nsym, replace=False).tolist())
        noisy = cw.copy()
        noisy[:, erase] = 0
        out, ok, nerr = G.decode(noisy, nsym, erase_pos=erase)
        assert ok.all() and np.array_equal(out, cw), f"erasures nsym={nsym}"
        e = nsym // 2
        tmax = (nsym - e) // 2
        erase = sorted(rng.choice(255, size=e, replace=False).tolist())
        others = [p for p in range(255) if p not in erase]
        noisy = cw[:200].copy()
        noisy[:, erase] = 0
        for r in range(200):
            pos = rng.choice(others, size=rng.integers(0, tmax + 1), replace=False)
            noisy[r, pos] ^= rng.integers(1, 256, size=len(pos)).astype(np.uint8)
        out, ok, nerr = G.decode(noisy, nsym, erase_pos=erase)
        assert ok.all() and np.array_equal(out, cw[:200]), f"mixed nsym={nsym}"
        log.append(f"RS nsym={nsym}: {nsym} shared erasures and mixed erasures+errors repaired")
    msg = rng.integers(0, 256, size=(5000, 20), dtype=np.uint8)
    cw = G.encode(msg, 4)
    noisy = cw.copy()
    noisy[:, [1, 9, 20, 23]] = 7
    out, ok, _ = G.decode(noisy, 4, erase_pos=[1, 9, 20, 23])
    assert ok.all() and np.array_equal(out, cw), "outer code"
    log.append("outer code n=24 k=20: 4 lost frames rebuilt across 5000 columns")
    return log


def test_frames() -> list[str]:
    log = []
    W, H = 1280, 720
    rng = np.random.default_rng(0)
    for prof in [Profile(8, 1, 0, 1), Profile(4, 1, 0, 1), Profile(4, 2, 1, 1), Profile(2, 1, 0, 1), Profile(3, 3, 2, 1)]:
        g = geometry(W, H, prof, 32)
        payload = F.prng_bytes((F.TAG_PAYLOAD, 7, 3), g.payload_len)
        h = Header(F.KIND_CALIB, 7, 3, prof, 32, 0, 0, W, H, g.payload_len, zlib.crc32(payload.tobytes()), 1, 10)
        Y, U, V = F.render_frame(h, payload, g)
        hd, nerr, sep = F.read_header(Y)
        assert hd is not None and hd.pack() == h.pack(), "header roundtrip"
        ramps = F.read_ramps(Y, U, V)
        vy, vu, vv = F.sample_body(Y, U, V, g)
        sy = F.decide(vy, F.refine_centers(vy, F.calibrated_centers(prof.ybits, F.Y_LO, F.Y_HI, ramps["y"])))
        if prof.cbits:
            su = F.decide(vu, F.refine_centers(vu, F.calibrated_centers(prof.cbits, F.C_LO, F.C_HI, ramps["u"])))
            sv = F.decide(vv, F.refine_centers(vv, F.calibrated_centers(prof.cbits, F.C_LO, F.C_HI, ramps["v"])))
        else:
            su = sv = np.zeros((0, 0), np.uint8)
        out, ok, _ = F.body_to_bytes(F.symbols_to_stream(sy, su, sv, g), g, 7, 3)
        assert ok and np.array_equal(out, payload), f"clean roundtrip {prof.label}"

        def noisy(P, sigma):
            Q = P.astype(np.float32)
            if prof.cell >= 4:
                Q = (Q + np.roll(Q, 1, 1) + np.roll(Q, -1, 1) + np.roll(Q, 1, 0) + np.roll(Q, -1, 0)) / 5
            return np.clip(Q + rng.normal(0, sigma, Q.shape), 0, 255).astype(np.uint8)

        sigma = 12 if prof.ybits <= 2 else 4
        Yn, Un, Vn = noisy(Y, sigma), noisy(U, sigma), noisy(V, sigma)
        hd2, _, _ = F.read_header(Yn)
        assert hd2 is not None and hd2.pack() == h.pack(), "noisy header"
        ramps = F.read_ramps(Yn, Un, Vn)
        vy, vu, vv = F.sample_body(Yn, Un, Vn, g)
        sy = F.decide(vy, F.refine_centers(vy, F.calibrated_centers(prof.ybits, F.Y_LO, F.Y_HI, ramps["y"])))
        if prof.cbits:
            su = F.decide(vu, F.refine_centers(vu, F.calibrated_centers(prof.cbits, F.C_LO, F.C_HI, ramps["u"])))
            sv = F.decide(vv, F.refine_centers(vv, F.calibrated_centers(prof.cbits, F.C_LO, F.C_HI, ramps["v"])))
        out2, ok2, nerr_cw = F.body_to_bytes(F.symbols_to_stream(sy, su, sv, g), g, 7, 3)
        esy, _, _ = F.stream_to_symbols(F.bytes_to_body(payload, g, 7, 3), g, 7, 3)
        eye = F.eye_margin(F.level_stats(vy, esy, 1 << prof.ybits))
        assert ok2 and np.array_equal(out2, payload), f"noisy roundtrip {prof.label}"
        log.append(f"frame {prof.label}: payload {g.payload_len} B, clean + noisy (sigma {sigma}) decode OK, "
                   f"worst codeword {int(nerr_cw.max())} bytes, eye {eye:.1f}")
    return log


def run(verbose: bool = True) -> bool:
    ok = True
    for name, fn in (("Reed-Solomon", test_rs), ("frame layout", test_frames)):
        try:
            for line in fn():
                if verbose:
                    print("  " + line)
            print(f"PASS {name}")
        except AssertionError as e:
            ok = False
            print(f"FAIL {name}: {e}")
    return ok
