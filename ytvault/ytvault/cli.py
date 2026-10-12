"""ytvault command line."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import calibrate as C
from . import profiles as P
from . import store
from . import video
from . import youtube as YT
from .frame import Profile


def _res(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _outer(s: str) -> tuple[int, int]:
    if s in ("0", "none", "off"):
        return 1, 0
    k, m = s.split("/")
    return int(k), int(m)


def _profiles_from_arg(args) -> list[Profile]:
    if getattr(args, "profiles", None):
        return [Profile.parse(x) for x in args.profiles.split(",")]
    return P.sweep(args.sweep)


def _rec_options(args) -> C.RecommendOptions:
    k, m = _outer(args.outer)
    return C.RecommendOptions(safety=args.safety, target_frame_fail=args.target, min_eye=args.min_eye,
                              outer_k=k, outer_m=m)


def _add_rec_args(p):
    p.add_argument("--outer", default="20/4", help="outer frame parity K/M assumed for net-rate figures (default 20/4)")
    p.add_argument("--safety", type=float, default=2.0, help="inflate observed byte error rate by this factor")
    p.add_argument("--target", type=float, default=1e-6, help="target probability a frame overflows inner RS")
    p.add_argument("--min-eye", type=float, default=3.0, help="reject profiles with level separation below this")


# ----------------------------------------------------------------- commands --

def cmd_profiles(args):
    W, H = _res(args.res)
    ps = P.ordered(_profiles_from_arg(args), W, H, args.fps, args.nsym)
    print(f"{'profile':14s} {'payload/frame':>13s} {'codewords':>9s} {'gross rate':>26s}")
    for p in ps:
        r = P.gross_rate(W, H, args.fps, p, args.nsym)
        print(f"{p.label:14s} {P.payload_per_frame(W, H, p, args.nsym):13d} {P.payload_per_frame(W, H, p, args.nsym) // (255 - args.nsym):9d} "
              f"{r / 1024:10.1f} KiB/s {r * 3600 / 2 ** 30:6.2f} GiB/h")
    print(f"\n{len(ps)} profiles at {W}x{H}@{args.fps} with nsym={args.nsym}; before outer parity.")


def cmd_calib_gen(args):
    W, H = _res(args.res)
    ps = _profiles_from_arg(args)
    print(f"Generating {len(ps)} profiles x {args.seconds}s at {W}x{H}@{args.fps}, nsym={args.nsym}, crf={args.crf}",
          file=sys.stderr)
    plan = C.generate(args.output, W, H, args.fps, ps, seconds_per_profile=args.seconds, nsym=args.nsym,
                      crf=args.crf, stream_id=args.stream_id)
    size = os.path.getsize(args.output)
    print(f"wrote {args.output}: {plan['video_frames']} frames, {plan['duration_s']:.1f}s, {size / 1e6:.1f} MB "
          f"({size * 8 / plan['duration_s'] / 1e6:.1f} Mbps); plan in {args.output}.plan.json")
    if not args.no_verify:
        print("verifying the file decodes cleanly before you upload it ...", file=sys.stderr)
        a = C.analyze(args.output, label="local (pre-upload)")
        rec = C.recommend(a, _rec_options(args))
        print(C.text_report(a, rec))
        bad = [p.profile for p in a.profiles.values() if p.frames_crc_ok != p.frames or p.max_cw_err]
        if bad:
            print(f"\nWARNING: these profiles already show errors in the local file (raise --crf quality or drop them): "
                  f"{', '.join(bad)}")
        else:
            print("\nLocal file is clean. Upload it to YouTube (unlisted is fine), wait until the HD/4K rendition "
                  "is processed, then run:\n  ytvault analyze <youtube-url> --out report/")


def cmd_simulate(args):
    preset = video.SIM_PRESETS.get(args.preset, {}) if args.preset else {}
    codec = args.codec or preset.get("codec", "libx264")
    bitrate = args.bitrate or preset.get("bitrate", "4000k")
    out = args.output or os.path.splitext(args.video)[0] + f".sim-{codec.replace('lib', '')}-{bitrate}.mp4"
    print(f"re-encoding with {codec} @ {bitrate} ({'2-pass' if args.two_pass else '1-pass'}) -> {out}", file=sys.stderr)
    video.reencode(args.video, out, codec=codec, bitrate=bitrate, two_pass=args.two_pass)
    info = video.probe(out)
    print(f"wrote {out}: {info.codec} {info.width}x{info.height} {info.bit_rate / 1000:.0f} kbps, {info.nb_frames} frames")
    if args.analyze:
        a = C.analyze(out, label=f"sim {codec} {bitrate}")
        rec = C.recommend(a, _rec_options(args))
        print(C.text_report(a, rec))
        if args.out:
            C.save_reports([a], args.out, _rec_options(args))
            print(f"\nreports in {args.out}/report.md and report.json")


def _download_set(url: str, formats: str, out_dir: str, design_height: int | None):
    fmts = YT.list_formats(url)
    if not fmts:
        raise SystemExit("no video formats reported by yt-dlp")
    picked = YT.pick_formats(fmts, formats, design_height)
    print("formats available:", file=sys.stderr)
    for f in fmts:
        mark = "*" if f in picked else " "
        print(f"  {mark} {YT.format_label(f):40s} {f['ext']} {(f['filesize'] or 0) / 1e6:7.1f} MB", file=sys.stderr)
    paths = []
    for f in picked:
        print(f"downloading {YT.format_label(f)} ...", file=sys.stderr)
        paths.append((YT.download(url, f["format_id"], out_dir), YT.format_label(f)))
    return paths


def cmd_download(args):
    paths = _download_set(args.url, args.formats, args.output, args.design_height)
    for p, label in paths:
        print(f"{label}: {p}")


def cmd_analyze(args):
    opt = _rec_options(args)
    design = _res(args.design) if args.design else None
    sources = []
    if YT.is_url(args.source):
        dl_dir = os.path.join(args.out or "ytvault-report", "downloads")
        sources = _download_set(args.source, args.formats, dl_dir, design[1] if design else None)
    else:
        sources = [(args.source, args.label or os.path.basename(args.source))]
    analyses = []
    for path, label in sources:
        print(f"analyzing {label} ...", file=sys.stderr)
        a = C.analyze(path, label=label, design=design, limit=args.limit)
        analyses.append(a)
        print(C.text_report(a, C.recommend(a, opt)))
        print()
    out_dir = args.out or "ytvault-report"
    comb = C.save_reports(analyses, out_dir, opt)
    if len(analyses) > 1:
        ob = comb.get("overall_best")
        if ob:
            print(f"OVERALL (feasible in every format): {ob['profile']} nsym={ob['nsym_rec']} -> "
                  f"{ob['net_bytes_per_s'] / 1024:.1f} KiB/s = {ob['net_bytes_per_s'] * 3600 / 2 ** 30:.2f} GiB/hour")
        else:
            print("OVERALL: no profile was feasible in every analysed format")
    print(f"\nreports: {out_dir}/report.md, {out_dir}/report.json")
    if comb.get("overall_best"):
        ob = comb["overall_best"]
        print(f"pack with:  ytvault pack <file> -o out.mp4 --profile {ob['profile']} --nsym {ob['nsym_rec']} "
              f"--outer {args.outer} --res {analyses[0].design[0]}x{analyses[0].design[1]}")


def cmd_upload(args):
    vid = YT.upload(args.video, args.title, args.description, args.privacy, args.client_secrets, args.token)
    print(f"uploaded: https://www.youtube.com/watch?v={vid}")
    print("YouTube needs a few minutes (more for 1440p/4K) before the HD renditions exist; analyze after that.")


def _profile_from_report(path: str) -> tuple[Profile, int]:
    with open(path) as fh:
        rep = json.load(fh)
    comb = rep.get("combined", {})
    ob = comb.get("overall_best")
    if not ob:
        for r in comb.get("per_format", {}).values():
            if r.get("best"):
                ob = r["best"]
                break
    if not ob:
        raise SystemExit("report has no feasible profile to pack with")
    return Profile.parse(ob["profile"]), int(ob["nsym_rec"])


def cmd_pack(args):
    W, H = _res(args.res)
    if args.from_report:
        prof, nsym = _profile_from_report(args.from_report)
        if args.nsym:
            nsym = args.nsym
    else:
        prof, nsym = Profile.parse(args.profile), args.nsym or 32
    k, m = _outer(args.outer)
    print(f"packing {args.file} at {W}x{H}@{args.fps} profile {prof.label} nsym={nsym} outer {k}/{m}", file=sys.stderr)
    man = store.pack(args.file, args.output, prof, nsym=nsym, k_outer=k, m_outer=m, width=W, height=H,
                     fps=args.fps, crf=args.crf)
    eff = man["size"] / man["duration_s"] if man["duration_s"] else 0
    print(f"wrote {args.output}: {man['video_frames']} frames, {man['duration_s']:.1f}s, "
          f"{man['video_bytes'] / 1e6:.1f} MB upload for {man['size'] / 1e6:.2f} MB of data "
          f"({eff / 1024:.1f} KiB of payload per second of video); manifest in {args.output}.manifest.json")
    if not args.no_verify:
        r = store.unpack(args.output, None, dry_run=True)
        print(f"local verify: {'OK' if r.ok else 'FAILED'} (sha {r.sha_ok}, {r.data_frames_ok} data frames, "
              f"{r.inner_corrections} inner corrections)")
        for n in r.notes:
            print("  note:", n)


def cmd_unpack(args):
    design = _res(args.design) if args.design else None
    if YT.is_url(args.source):
        dl_dir = os.path.join(args.output, "downloads")
        fmts = YT.list_formats(args.source)
        picked = YT.pick_formats(fmts, args.format or "best", design[1] if design else None)
        if not picked:
            raise SystemExit("no matching format")
        path = YT.download(args.source, picked[0]["format_id"], dl_dir)
        print(f"downloaded {YT.format_label(picked[0])} -> {path}", file=sys.stderr)
    else:
        path = args.source
    r = store.unpack(path, args.output, design=design, dry_run=args.dry_run)
    _print_unpack(r)
    sys.exit(0 if r.ok else 2)


def _print_unpack(r: store.UnpackResult):
    print(f"result: {'OK' if r.ok else 'INCOMPLETE'}  sha256 {'matches' if r.sha_ok else 'MISMATCH/unknown'}")
    if r.manifest:
        m = r.manifest
        print(f"  file: {m.get('name')} ({m.get('size')} bytes) profile {m.get('profile')} nsym={m.get('nsym')} "
              f"outer {m.get('k_outer')}/{m.get('m_outer')}")
    print(f"  logical frames {r.logical_frames}; data frames ok/bad {r.data_frames_ok}/{r.data_frames_bad}; "
          f"parity ok/bad {r.parity_frames_ok}/{r.parity_frames_bad}")
    print(f"  inner RS corrected {r.inner_corrections} bytes (worst codeword {r.worst_cw_err}); "
          f"outer code rebuilt {r.repaired_chunks} chunks; {r.lost_chunks} chunks lost")
    st = r.stats
    if st:
        print(f"  stream {st.get('native')} design {st.get('design')} rescaled={st.get('rescaled')} "
              f"unreadable headers {st.get('unreadable_headers')}")
    for n in r.notes:
        print("  note:", n)
    if r.out_path:
        print(f"  written: {r.out_path}")


def cmd_verify(args):
    design = _res(args.design) if args.design else None
    r = store.unpack(args.video, None, design=design, dry_run=True)
    if r.manifest is None and r.logical_frames == 0:
        print("no storage frames found; trying as a calibration stream")
        a = C.analyze(args.video, label=os.path.basename(args.video), design=design)
        print(C.text_report(a, C.recommend(a)))
        return
    _print_unpack(r)
    sys.exit(0 if r.ok else 2)


def cmd_selftest(args):
    from . import selftest
    ok = selftest.run(verbose=not args.quiet)
    sys.exit(0 if ok else 1)


# -------------------------------------------------------------------- main --

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ytvault",
                                 description="Measure how densely data survives YouTube, then store files there.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profiles", help="list the density sweep with payload rates")
    p.add_argument("--res", default="1920x1080")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--nsym", type=int, default=32)
    p.add_argument("--sweep", default="default", choices=["quick", "default", "wide"])
    p.add_argument("--profiles", help="comma list like c4-y1-c0-r1,c2-y2-c1-r1 (overrides --sweep)")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("calib-gen", help="generate the progressively denser calibration video")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--res", default="1920x1080", help="design resolution WxH (multiple of 16 x multiple of 8)")
    p.add_argument("--fps", type=int, default=30, choices=[24, 25, 30, 50, 60])
    p.add_argument("--sweep", default="default", choices=["quick", "default", "wide"])
    p.add_argument("--profiles", help="comma list of profiles (overrides --sweep)")
    p.add_argument("--seconds", type=float, default=2.0, help="seconds of video per profile")
    p.add_argument("--nsym", type=int, default=32, help="inner RS parity bytes per 255-byte codeword")
    p.add_argument("--crf", type=int, default=12, help="x264 quality of the upload file (lower = larger, cleaner)")
    p.add_argument("--stream-id", type=int, default=None)
    p.add_argument("--no-verify", action="store_true", help="skip the local decode check after writing")
    _add_rec_args(p)
    p.set_defaults(func=cmd_calib_gen)

    p = sub.add_parser("simulate", help="re-encode a video the way a YouTube-like transcoder would, then analyze")
    p.add_argument("video")
    p.add_argument("-o", "--output")
    p.add_argument("--preset", choices=sorted(video.SIM_PRESETS), default="yt1080-h264")
    p.add_argument("--codec", help="override codec: libx264, libvpx-vp9, libsvtav1, libaom-av1")
    p.add_argument("--bitrate", help="override target bitrate, e.g. 3000k")
    p.add_argument("--two-pass", action="store_true")
    p.add_argument("--no-analyze", dest="analyze", action="store_false")
    p.add_argument("--out", help="report directory")
    _add_rec_args(p)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("download", help="download YouTube renditions with yt-dlp")
    p.add_argument("url")
    p.add_argument("-o", "--output", default="downloads")
    p.add_argument("--formats", default="native", help="native | all | best | comma list of format ids")
    p.add_argument("--design-height", type=int)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("analyze", help="decode a calibration video (file or YouTube URL) and recommend a profile")
    p.add_argument("source", help="local video file or YouTube URL")
    p.add_argument("--formats", default="native", help="for URLs: native | all | best | format ids")
    p.add_argument("--out", help="report directory (default ytvault-report)")
    p.add_argument("--label")
    p.add_argument("--design", help="force design resolution WxH if auto-detection fails")
    p.add_argument("--limit", type=int, help="stop after this many logical frames (quick look)")
    _add_rec_args(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("upload", help="upload a video with the YouTube Data API (optional dependencies)")
    p.add_argument("video")
    p.add_argument("--title", required=True)
    p.add_argument("--description", default="ytvault data video")
    p.add_argument("--privacy", default="unlisted", choices=["unlisted", "private", "public"])
    p.add_argument("--client-secrets", default="client_secrets.json")
    p.add_argument("--token", default="youtube_token.json")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("pack", help="encode a file into a video")
    p.add_argument("file")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--profile", default="c4-y1-c0-r1")
    p.add_argument("--from-report", help="take profile and nsym from an analyze report.json")
    p.add_argument("--nsym", type=int, help="inner RS parity bytes (default 32, or from report)")
    p.add_argument("--outer", default="20/4", help="outer frame parity K/M; 'none' to disable")
    p.add_argument("--res", default="1920x1080")
    p.add_argument("--fps", type=int, default=30, choices=[24, 25, 30, 50, 60])
    p.add_argument("--crf", type=int, default=12)
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("unpack", help="recover a file from a video (file or YouTube URL)")
    p.add_argument("source")
    p.add_argument("-o", "--output", default="recovered")
    p.add_argument("--format", help="for URLs: yt-dlp format id, or native/best (default best)")
    p.add_argument("--design", help="force design resolution WxH")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("verify", help="check a video's integrity without writing the file")
    p.add_argument("video")
    p.add_argument("--design")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("selftest", help="run in-memory codec and frame tests (no ffmpeg needed)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except video.ToolMissing as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
