"""pytest suite: in-memory codec/frame tests plus a small end-to-end video round trip (needs ffmpeg)."""
import hashlib
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ytvault import selftest, calibrate as C, store, video, profiles as P  # noqa: E402
from ytvault.frame import Profile  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def test_reed_solomon():
    assert selftest.test_rs()


def test_frame_layout():
    assert selftest.test_frames()


def test_profile_parse_roundtrip():
    for p in P.sweep("wide"):
        assert Profile.parse(p.label) == p


def test_binomial_tail():
    assert C.binom_tail(255, 0.0, 3) == 0.0
    assert abs(C.binom_tail(10, 0.5, 4) - 0.623046875) < 1e-9
    assert C.binom_tail(255, 0.01, 16) < C.binom_tail(255, 0.01, 8)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_end_to_end_video(tmp_path):
    W, H, fps = 640, 360, 30
    cal = str(tmp_path / "calib.mp4")
    profs = [Profile(8, 1, 0, 1), Profile(4, 1, 0, 1), Profile(4, 2, 1, 1), Profile(2, 2, 0, 1)]
    plan = C.generate(cal, W, H, fps, profs, seconds_per_profile=0.2, nsym=32, min_frames=6, stream_id=99,
                      progress=False)
    assert plan["logical_frames"] == 4 * 6
    a = C.analyze(cal, label="local", progress=False)
    assert a.unreadable_headers == 0
    assert len(a.profiles) == 4
    for pr in a.profiles.values():
        assert pr.frames == 6
        assert pr.frames_crc_ok == 6, pr.profile
        assert pr.max_cw_err == 0, pr.profile
    rec = C.recommend(a)
    assert rec["best"] is not None
    # a harsh re-encode must degrade the dense profiles more than the sparse ones
    sim = str(tmp_path / "sim.mp4")
    video.reencode(cal, sim, codec="libx264", bitrate="600k")
    a2 = C.analyze(sim, label="sim", progress=False)
    by = {p.profile: p for p in a2.profiles.values()}
    assert by["c8-y1-c0-r1"].ber <= by["c2-y2-c0-r1"].ber
    C.save_reports([a, a2], str(tmp_path / "rep"))
    assert (tmp_path / "rep" / "report.md").exists()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_pack_unpack_with_losses(tmp_path):
    W, H = 640, 360
    src = tmp_path / "blob.bin"
    data = np.random.default_rng(3).integers(0, 256, size=60_000, dtype=np.uint8).tobytes() + b"\0" * 5000
    src.write_bytes(data)
    out = str(tmp_path / "packed.mp4")
    man = store.pack(str(src), out, Profile(4, 1, 0, 1), nsym=16, k_outer=4, m_outer=2, width=W, height=H,
                     progress=False)
    assert man["nchunks"] >= 8
    r = store.unpack(out, str(tmp_path / "rec"), progress=False)
    assert r.ok and r.sha_ok, r.notes
    assert r.manifest["name"] == "blob.bin" and r.manifest["size"] == len(data)
    assert hashlib.sha256(open(r.out_path, "rb").read()).hexdigest() == man["sha256"]
    # moderate re-encode: must still come back bit-exact (inner RS and/or outer repair)
    sim = str(tmp_path / "packed-sim.mp4")
    video.reencode(out, sim, codec="libx264", bitrate="900k")
    r2 = store.unpack(sim, str(tmp_path / "rec2"), progress=False)
    assert r2.ok and r2.sha_ok, r2.notes
