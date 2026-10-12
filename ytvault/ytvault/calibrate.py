"""Calibration video generation, integrity analysis and encoding recommendation."""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import secrets
import sys
import zlib
from dataclasses import dataclass, field, asdict

import numpy as np

from . import frame as F
from . import video
from .decoder import iter_logical_frames, StreamStats, decode_logical, decide_symbols
from .frame import Profile, Header, geometry
from .profiles import ordered, gross_rate

NSYM_CHOICES = (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128)


# ------------------------------------------------------------ generation ----

def generate(out_path: str, width: int, height: int, fps: int, profiles: list[Profile],
             seconds_per_profile: float = 2.0, nsym: int = 32, crf: int = 12, stream_id: int | None = None,
             min_frames: int = 12, progress: bool = True) -> dict:
    """Write the progressively-denser calibration video and a sidecar plan JSON."""
    stream_id = stream_id if stream_id is not None else secrets.randbits(32)
    profiles = ordered(profiles, width, height, fps, nsym)
    plan = dict(kind="calibration", stream_id=stream_id, width=width, height=height, fps=fps, nsym=nsym,
                crf=crf, created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), segments=[])
    frame_idx = 0
    vframe = 0
    with video.FrameWriter(out_path, width, height, fps, crf=crf) as w:
        for seq, p in enumerate(profiles):
            g = geometry(width, height, p, nsym)
            n = max(min_frames, int(round(seconds_per_profile * fps / p.repeat)))
            seg = dict(seq=seq, profile=p.label, logical_frames=n, first_frame_idx=frame_idx,
                       first_video_frame=vframe, payload_len=g.payload_len, ncw=g.ncw,
                       gross_bytes_per_s=gross_rate(width, height, fps, p, nsym))
            for i in range(n):
                payload = F.prng_bytes((F.TAG_PAYLOAD, stream_id, frame_idx), g.payload_len)
                h = Header(F.KIND_CALIB, stream_id, frame_idx, p, nsym, 0, 0, width, height, g.payload_len,
                           zlib.crc32(payload.tobytes()), seq, len(profiles))
                Y, U, V = F.render_frame(h, payload, g)
                for _ in range(p.repeat):
                    w.write(Y, U, V)
                    vframe += 1
                frame_idx += 1
                if progress and (frame_idx % 10 == 0):
                    print(f"\r  [{seq + 1}/{len(profiles)}] {p.label:14s} frame {frame_idx}  "
                          f"({vframe / fps:6.1f}s of video)   ", end="", file=sys.stderr, flush=True)
            plan["segments"].append(seg)
    if progress:
        print("\r" + " " * 70 + "\r", end="", file=sys.stderr)
    plan["logical_frames"] = frame_idx
    plan["video_frames"] = vframe
    plan["duration_s"] = vframe / fps
    with open(out_path + ".plan.json", "w") as fh:
        json.dump(plan, fh, indent=1)
    return plan


# -------------------------------------------------------------- analysis ----

@dataclass
class FrameResult:
    frame_idx: int
    seq: int
    profile: str
    copies: int
    video_idx: int
    header_fixed: int
    sym_err_y: int
    cells_y: int
    sym_err_c: int
    cells_c: int
    bit_err: int
    bits: int
    cw_err: list          # per-codeword wrong bytes (vs expected), or RS-corrected counts when expected unknown
    rs_ok: bool
    crc_ok: bool
    eye_y: float
    eye_u: float
    eye_v: float
    copy_sym_err_y: list  # per-copy luma symbol error rate


@dataclass
class ProfileResult:
    seq: int
    profile: str
    cell: int
    ybits: int
    cbits: int
    repeat: int
    nsym_tx: int
    payload_len: int
    ncw: int
    frames: int = 0
    copies: int = 0
    missing_frames: int = 0
    header_fixed_max: int = 0
    sym_err_y: int = 0
    cells_y: int = 0
    sym_err_c: int = 0
    cells_c: int = 0
    bit_err: int = 0
    bits: int = 0
    cw_hist: list = field(default_factory=lambda: [0] * 256)
    frames_rs_ok: int = 0
    frames_crc_ok: int = 0
    level_stats_y: np.ndarray | None = None
    level_stats_u: np.ndarray | None = None
    level_stats_v: np.ndarray | None = None
    eye_min_frame: float = float("inf")
    copy_sym_err_y: list = field(default_factory=list)  # summed per copy index
    copy_cells_y: list = field(default_factory=list)
    gross_bytes_per_s: float = 0.0
    # filled by _finish_profile / recommend()
    eye_y: float = float("inf")
    eye_u: float = float("inf")
    eye_v: float = float("inf")
    max_cw_err: int = 0
    mean_cw_err: float = 0.0
    p99_cw_err: int = 0
    byte_err_rate: float = 0.0
    ber: float = 0.0
    ser_y: float = 0.0
    ser_c: float = 0.0
    nsym_rec: int | None = None
    feasible: bool = False
    reason: str = ""
    net_bytes_per_s: float = 0.0
    frame_fail_prob: float = 0.0

    def to_dict(self):
        d = {k: v for k, v in asdict(self).items() if not k.startswith("level_stats")}
        d["cw_hist"] = {str(i): c for i, c in enumerate(self.cw_hist) if c}
        for k in ("eye_y", "eye_u", "eye_v", "eye_min_frame"):
            if math.isinf(d[k]):
                d[k] = None
        return d


@dataclass
class Analysis:
    source: str
    design: tuple
    native: tuple
    rescaled: bool
    offset: tuple
    video_info: dict
    video_frames: int
    unreadable_headers: int
    logical_frames: int
    stream_id: int | None
    nsym_tx: int | None
    fps: float
    expected_known: bool
    profiles: dict[str, ProfileResult]
    frames: list[FrameResult]
    created: str
    label: str = ""

    def to_dict(self, include_frames: bool = True):
        d = dict(source=self.source, label=self.label, design=self.design, native=self.native,
                 rescaled=self.rescaled, offset=self.offset, video_info=self.video_info,
                 video_frames=self.video_frames, unreadable_headers=self.unreadable_headers,
                 logical_frames=self.logical_frames, stream_id=self.stream_id, nsym_tx=self.nsym_tx,
                 fps=self.fps, expected_known=self.expected_known, created=self.created,
                 profiles=[p.to_dict() for p in sorted(self.profiles.values(), key=lambda p: p.gross_bytes_per_s)])
        if include_frames:
            frames = []
            for f in self.frames:
                fd = asdict(f)
                for k in ("eye_y", "eye_u", "eye_v"):
                    if math.isinf(fd[k]):
                        fd[k] = None
                frames.append(fd)
            d["frames"] = frames
        return d


def _popcount8(a: np.ndarray) -> int:
    return int(np.unpackbits(a.astype(np.uint8)).sum())


def analyze(path: str, label: str = "", design: tuple[int, int] | None = None, progress: bool = True,
            keep_frames: bool = True, limit: int | None = None) -> Analysis:
    """Decode every frame, compare with what was transmitted, aggregate per profile."""
    stats = StreamStats()
    profiles: dict[str, ProfileResult] = {}
    frames: list[FrameResult] = []
    expected_known = True
    stream_id = None
    nsym_tx = None
    last_idx_by_seq: dict[int, int] = {}
    fps = 30.0
    W = H = 0
    nlogical = 0
    for lf in iter_logical_frames(path, design=design, keep_copies=True, progress=progress, limit=limit,
                                  stats=stats):
        h = lf.header
        nlogical += 1
        W, H = stats.design
        fps = stats.info.get("fps", 30.0) or 30.0
        if stream_id is None:
            stream_id, nsym_tx = h.stream_id, h.nsym
        g = geometry(W, H, h.profile, h.nsym)
        key = h.profile.label if h.kind != F.KIND_CALIB else f"{h.seq}:{h.profile.label}"
        pr = profiles.get(key)
        if pr is None:
            pr = ProfileResult(h.seq, h.profile.label, h.profile.cell, h.profile.ybits, h.profile.cbits,
                               h.profile.repeat, h.nsym, g.payload_len, g.ncw,
                               gross_bytes_per_s=gross_rate(W, H, fps, h.profile, h.nsym))
            L = 1 << h.profile.ybits
            Lc = 1 << h.profile.cbits
            pr.level_stats_y = np.zeros((3, L))
            pr.level_stats_u = np.zeros((3, Lc))
            pr.level_stats_v = np.zeros((3, Lc))
            profiles[key] = pr
        # missing-frame accounting (frame_idx is contiguous within a segment)
        prev = last_idx_by_seq.get(h.seq)
        if prev is not None and h.frame_idx > prev + 1:
            pr.missing_frames += h.frame_idx - prev - 1
        last_idx_by_seq[h.seq] = h.frame_idx
        pr.frames += 1
        pr.copies += lf.copies
        pr.header_fixed_max = max(pr.header_fixed_max, lf.header_fixed)

        dec = decode_logical(lf, g)
        sy, su, sv = dec.symbols
        if h.kind == F.KIND_CALIB:
            exp_payload = F.prng_bytes((F.TAG_PAYLOAD, h.stream_id, h.frame_idx), g.payload_len)
            exp_stream = F.bytes_to_body(exp_payload, g, h.stream_id, h.frame_idx)
            esy, esu, esv = F.stream_to_symbols(exp_stream, g, h.stream_id, h.frame_idx)
            sym_err_y = int(np.count_nonzero(sy != esy))
            sym_err_c = int(np.count_nonzero(su != esu) + np.count_nonzero(sv != esv))
            diff = dec.stream ^ exp_stream
            bit_err = _popcount8(diff)
            cw_err = np.count_nonzero(diff.reshape(255, g.ncw).T, axis=1)
            vy, vu, vv = lf.mean_values()
            ls_y = F.level_stats(vy, esy, 1 << h.profile.ybits)
            pr.level_stats_y += ls_y
            eye_y = F.eye_margin(ls_y)
            if h.profile.cbits:
                ls_u = F.level_stats(vu, esu, 1 << h.profile.cbits)
                ls_v = F.level_stats(vv, esv, 1 << h.profile.cbits)
                pr.level_stats_u += ls_u
                pr.level_stats_v += ls_v
                eye_u, eye_v = F.eye_margin(ls_u), F.eye_margin(ls_v)
            else:
                eye_u = eye_v = float("inf")
            copy_err = []
            for ci, (cvy, cvu, cvv) in enumerate(lf.copy_vals):
                csy, _, _ = decide_symbols(cvy, cvu, cvv, g, dec.centers)
                e = int(np.count_nonzero(csy != esy))
                copy_err.append(e / max(esy.size, 1))
                if len(pr.copy_sym_err_y) <= ci:
                    pr.copy_sym_err_y.append(0)
                    pr.copy_cells_y.append(0)
                pr.copy_sym_err_y[ci] += e
                pr.copy_cells_y[ci] += esy.size
        else:
            expected_known = False
            sym_err_y = sym_err_c = bit_err = 0
            cw_err = np.where(dec.nerr_cw < 0, 255, dec.nerr_cw)
            eye_y = eye_u = eye_v = float("inf")
            copy_err = []
        cells_y, cells_c = sy.size, su.size + sv.size
        pr.sym_err_y += sym_err_y
        pr.cells_y += cells_y
        pr.sym_err_c += sym_err_c
        pr.cells_c += cells_c
        pr.bit_err += bit_err
        pr.bits += g.body_bytes * 8
        hist = np.bincount(np.minimum(cw_err, 255), minlength=256)
        pr.cw_hist = (np.asarray(pr.cw_hist) + hist).tolist()
        pr.frames_rs_ok += int(dec.rs_ok)
        pr.frames_crc_ok += int(dec.crc_ok)
        pr.eye_min_frame = min(pr.eye_min_frame, eye_y, eye_u, eye_v)
        if keep_frames:
            frames.append(FrameResult(h.frame_idx, h.seq, h.profile.label, lf.copies, lf.first_video_idx,
                                      lf.header_fixed, sym_err_y, cells_y, sym_err_c, cells_c, bit_err,
                                      g.body_bytes * 8, [int(x) for x in cw_err], dec.rs_ok, dec.crc_ok,
                                      eye_y, eye_u, eye_v, copy_err))
        lf.copy_vals = []  # free memory
    for pr in profiles.values():
        _finish_profile(pr)
    return Analysis(path, stats.design, stats.native, stats.rescaled, stats.offset_used, stats.info,
                    stats.video_frames, stats.unreadable_headers, nlogical, stream_id, nsym_tx, fps,
                    expected_known, profiles, frames,
                    _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), label)


def _finish_profile(pr: ProfileResult):
    hist = np.asarray(pr.cw_hist)
    total_cw = int(hist.sum())
    if total_cw:
        nz = np.nonzero(hist)[0]
        pr.max_cw_err = int(nz.max())
        pr.mean_cw_err = float((hist * np.arange(256)).sum() / total_cw)
        cum = np.cumsum(hist) / total_cw
        pr.p99_cw_err = int(np.searchsorted(cum, 0.99))
        pr.byte_err_rate = pr.mean_cw_err / 255
    pr.ber = pr.bit_err / pr.bits if pr.bits else 0.0
    pr.ser_y = pr.sym_err_y / pr.cells_y if pr.cells_y else 0.0
    pr.ser_c = pr.sym_err_c / pr.cells_c if pr.cells_c else 0.0
    if pr.level_stats_y is not None:
        pr.eye_y = F.eye_margin(pr.level_stats_y)
        pr.eye_u = F.eye_margin(pr.level_stats_u) if pr.cbits else float("inf")
        pr.eye_v = F.eye_margin(pr.level_stats_v) if pr.cbits else float("inf")


# --------------------------------------------------------- recommendation ----

def binom_tail(n: int, p: float, t: int) -> float:
    """P[Binomial(n, p) > t]."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0 if t < n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    s = 0.0
    lg_n = math.lgamma(n + 1)
    for i in range(t + 1, n + 1):
        s += math.exp(lg_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(s, 1.0)


@dataclass
class RecommendOptions:
    safety: float = 2.0              # inflate the observed byte error rate by this factor
    target_frame_fail: float = 1e-6  # acceptable probability that a frame exceeds inner RS capacity
    min_eye: float = 3.0             # reject profiles whose level separation is thinner than this
    cautious_eye: float = 4.5        # below this, force nsym >= 32 even with no observed errors
    min_nsym: int = 16
    max_nsym: int = 128
    outer_k: int = 20
    outer_m: int = 4
    min_frames: int = 5


def recommend_nsym(pr: ProfileResult, opt: RecommendOptions) -> tuple[int | None, str, float]:
    """Pick the inner RS parity for a profile from its observed error statistics."""
    if pr.frames < opt.min_frames:
        return None, f"only {pr.frames} frames observed", 1.0
    if pr.missing_frames:
        return None, f"{pr.missing_frames} frames went missing in transit", 1.0
    eye = min(pr.eye_y, pr.eye_u, pr.eye_v)
    if eye < opt.min_eye:
        return None, f"level separation too thin (eye {eye:.2f} < {opt.min_eye})", 1.0
    p = pr.byte_err_rate * opt.safety
    needed_t = 0
    if p > 0:
        for t in range(0, 128):
            if binom_tail(255, p, t) * pr.ncw <= opt.target_frame_fail:
                needed_t = t
                break
        else:
            return None, f"byte error rate {pr.byte_err_rate:.4f} too high for any RS setting", 1.0
    needed_t = max(needed_t, 2 * pr.max_cw_err)
    nsym = max(opt.min_nsym, 2 * needed_t)
    if eye < opt.cautious_eye:
        nsym = max(nsym, 32)
    nsym = next((c for c in NSYM_CHOICES if c >= nsym), None)
    if nsym is None or nsym > opt.max_nsym:
        return None, f"needs more than {opt.max_nsym} parity bytes per codeword", 1.0
    fail = binom_tail(255, p, nsym // 2) * pr.ncw if p > 0 else 0.0
    why = []
    if pr.max_cw_err:
        why.append(f"worst codeword had {pr.max_cw_err} bad bytes")
    if p > 0:
        why.append(f"byte error rate {pr.byte_err_rate:.2e}")
    if eye < opt.cautious_eye:
        why.append(f"eye {eye:.2f} is cautious-range")
    return nsym, "; ".join(why) or "no errors observed", fail


def recommend(analysis: Analysis, opt: RecommendOptions | None = None) -> dict:
    opt = opt or RecommendOptions()
    W, H = analysis.design
    fps = analysis.fps
    outer = opt.outer_k / (opt.outer_k + opt.outer_m)
    for pr in analysis.profiles.values():
        nsym, reason, fail = recommend_nsym(pr, opt)
        pr.nsym_rec, pr.reason, pr.frame_fail_prob = nsym, reason, fail
        pr.feasible = nsym is not None
        if nsym is not None:
            p = Profile(pr.cell, pr.ybits, pr.cbits, pr.repeat)
            pr.net_bytes_per_s = gross_rate(W, H, fps, p, nsym) * outer
    feas = [p for p in analysis.profiles.values() if p.feasible]
    best = max(feas, key=lambda p: p.net_bytes_per_s) if feas else None
    cons = [p for p in feas if p.max_cw_err == 0 and min(p.eye_y, p.eye_u, p.eye_v) >= 6.0
            and p.frames_crc_ok == p.frames]
    conservative = max(cons, key=lambda p: p.net_bytes_per_s) if cons else None
    return dict(options=asdict(opt), best=best.to_dict() if best else None,
                conservative=conservative.to_dict() if conservative else None,
                outer=f"{opt.outer_k}/{opt.outer_m}")


def combine(analyses: list[Analysis], opt: RecommendOptions | None = None) -> dict:
    """Across several downloaded formats: a profile must be feasible in every one."""
    opt = opt or RecommendOptions()
    if not analyses:
        return {}
    per = {a.label or a.source: recommend(a, opt) for a in analyses}
    W, H = analyses[0].design
    fps = analyses[0].fps
    outer = opt.outer_k / (opt.outer_k + opt.outer_m)
    rows = {}
    for a in analyses:
        for key, pr in a.profiles.items():
            r = rows.setdefault(pr.profile, dict(profile=pr.profile, cell=pr.cell, ybits=pr.ybits, cbits=pr.cbits,
                                                 repeat=pr.repeat, feasible=True, nsym_rec=0, reasons=[],
                                                 formats=0))
            r["formats"] += 1
            if not pr.feasible:
                r["feasible"] = False
                r["reasons"].append(f"{a.label}: {pr.reason}")
            else:
                r["nsym_rec"] = max(r["nsym_rec"], pr.nsym_rec)
    for r in rows.values():
        if r["formats"] != len(analyses):
            r["feasible"] = False
            r["reasons"].append("not present in every format")
        if r["feasible"]:
            p = Profile(r["cell"], r["ybits"], r["cbits"], r["repeat"])
            r["net_bytes_per_s"] = gross_rate(W, H, fps, p, r["nsym_rec"]) * outer
        else:
            r["net_bytes_per_s"] = 0.0
    feas = [r for r in rows.values() if r["feasible"]]
    best = max(feas, key=lambda r: r["net_bytes_per_s"]) if feas else None
    return dict(per_format=per, overall_best=best, table=sorted(rows.values(), key=lambda r: -r["net_bytes_per_s"]))


# ---------------------------------------------------------------- report ----

def _fmt_rate(bps: float) -> str:
    return f"{bps / 1024:8.1f} KiB/s {bps * 3600 / 2 ** 30:6.2f} GiB/h"


def text_report(analysis: Analysis, rec: dict | None = None) -> str:
    lines = []
    vi = analysis.video_info
    lines.append(f"== {analysis.label or analysis.source}")
    lines.append(f"   stream {vi.get('codec')} {analysis.native[0]}x{analysis.native[1]} @ {analysis.fps:.3g} fps, "
                 f"{(vi.get('bit_rate') or 0) / 1000:.0f} kbps, {analysis.video_frames} frames"
                 + (f"  (RESCALED to design {analysis.design[0]}x{analysis.design[1]})" if analysis.rescaled else "")
                 + (f"  pixel offset {analysis.offset}" if tuple(analysis.offset) != (0, 0) else ""))
    lines.append(f"   {analysis.logical_frames} logical frames, {analysis.unreadable_headers} unreadable headers")
    if not analysis.expected_known:
        lines.append("   (not a calibration stream: error counts are RS-corrected bytes, not ground truth)")
    hdr = (f"{'profile':14s} {'frames':>6s} {'miss':>4s} {'SER-Y':>9s} {'SER-C':>9s} {'BER':>9s} "
           f"{'cw max/p99/mean':>16s} {'crc ok':>7s} {'eye Y/U/V':>16s} {'nsym':>5s} {'net rate':>26s}  verdict")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for pr in sorted(analysis.profiles.values(), key=lambda p: p.gross_bytes_per_s):
        eye = "/".join("inf" if math.isinf(x) else f"{x:.1f}" for x in (pr.eye_y, pr.eye_u, pr.eye_v))
        nsym = str(pr.nsym_rec) if pr.nsym_rec else "-"
        rate = _fmt_rate(pr.net_bytes_per_s) if pr.feasible else " " * 26
        verdict = ("OK " + pr.reason) if pr.feasible else ("NO " + pr.reason)
        lines.append(f"{pr.profile:14s} {pr.frames:6d} {pr.missing_frames:4d} {pr.ser_y:9.2e} {pr.ser_c:9.2e} "
                     f"{pr.ber:9.2e} {pr.max_cw_err:5d}/{pr.p99_cw_err:3d}/{pr.mean_cw_err:5.2f} "
                     f"{pr.frames_crc_ok:3d}/{pr.frames:<3d} {eye:>16s} {nsym:>5s} {rate}  {verdict}")
    if rec:
        b, c = rec.get("best"), rec.get("conservative")
        lines.append("")
        if b:
            lines.append(f"RECOMMENDED : {b['profile']} with nsym={b['nsym_rec']} and outer {rec['outer']}"
                         f"  -> {_fmt_rate(b['net_bytes_per_s'])}")
            lines.append(f"              ({b['reason']}; modelled per-frame RS overflow p={b['frame_fail_prob']:.1e})")
        else:
            lines.append("RECOMMENDED : nothing survived -- check the download, or calibrate at a lower resolution")
        if c and (not b or c["profile"] != b["profile"]):
            lines.append(f"CONSERVATIVE: {c['profile']} with nsym={c['nsym_rec']} and outer {rec['outer']}"
                         f"  -> {_fmt_rate(c['net_bytes_per_s'])}")
    return "\n".join(lines)


def markdown_report(analyses: list[Analysis], comb: dict) -> str:
    out = ["# ytvault calibration report", "",
           f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}", ""]
    ob = comb.get("overall_best")
    if ob:
        out.append(f"**Overall recommendation (survives every analysed format):** `{ob['profile']}` "
                   f"with `nsym={ob['nsym_rec']}` -> {_fmt_rate(ob['net_bytes_per_s']).strip()} after outer parity.")
    else:
        out.append("**Overall recommendation:** no profile was feasible in every analysed format.")
    out.append("")
    out.append("| profile | feasible in all | nsym | net rate | notes |")
    out.append("|---|---|---|---|---|")
    for r in comb.get("table", []):
        out.append(f"| `{r['profile']}` | {'yes' if r['feasible'] else 'no'} | {r['nsym_rec'] or '-'} | "
                   f"{_fmt_rate(r['net_bytes_per_s']).strip() if r['feasible'] else '-'} | "
                   f"{'; '.join(r['reasons'])[:200]} |")
    out.append("")
    for a in analyses:
        rec = comb["per_format"].get(a.label or a.source)
        out.append(f"## {a.label or a.source}")
        out.append("")
        out.append("```")
        out.append(text_report(a, rec))
        out.append("```")
        out.append("")
    out.append("### How to read this")
    out.append("")
    out.append("- **SER-Y / SER-C**: fraction of luma / chroma cells decoded to the wrong level "
               "(before error correction).")
    out.append("- **BER**: bit error rate of the body byte stream before Reed-Solomon.")
    out.append("- **cw max/p99/mean**: wrong bytes per 255-byte codeword. RS with `nsym` parity bytes repairs "
               "up to `nsym/2` per codeword.")
    out.append("- **eye**: gap between adjacent grey levels divided by the sum of their spreads "
               "(higher = safer; below 3 is rejected even at zero observed errors).")
    out.append("- **nsym**: parity bytes per codeword giving a modelled per-frame overflow probability below "
               "the target, with the observed byte error rate inflated by the safety factor and at least "
               "twice the worst observed codeword.")
    out.append("- **net rate**: payload throughput after inner RS and the outer K/M frame parity.")
    return "\n".join(out)


def save_reports(analyses: list[Analysis], out_dir: str, opt: RecommendOptions | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    comb = combine(analyses, opt)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(dict(analyses=[a.to_dict() for a in analyses], combined=comb), fh, indent=1,
                  default=_json_default)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(markdown_report(analyses, comb))
    return comb


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(o) or math.isinf(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return None
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(str(type(o)))
