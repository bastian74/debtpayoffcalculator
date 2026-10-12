"""YouTube transport: list / download formats with yt-dlp, optional upload via the Data API."""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://")) or "youtu.be/" in s or "youtube.com/" in s


def _ytdlp_cmd() -> list[str]:
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        return [exe]
    raise RuntimeError("yt-dlp not found: pip install yt-dlp")


def list_formats(url: str) -> list[dict]:
    """Video-only (or video+audio) formats, as yt-dlp reports them."""
    out = subprocess.run(_ytdlp_cmd() + ["-J", "--no-playlist", url], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    fmts = []
    for f in info.get("formats", []):
        if f.get("vcodec") in (None, "none"):
            continue
        fmts.append(dict(format_id=f["format_id"], ext=f.get("ext"), vcodec=f.get("vcodec"), width=f.get("width"),
                         height=f.get("height"), fps=f.get("fps"), tbr=f.get("tbr"),
                         filesize=f.get("filesize") or f.get("filesize_approx"), note=f.get("format_note"),
                         audio=f.get("acodec") not in (None, "none"), protocol=f.get("protocol")))
    fmts.sort(key=lambda f: ((f["height"] or 0), (f["tbr"] or 0)))
    return fmts


def codec_family(vcodec: str | None) -> str:
    v = (vcodec or "").lower()
    if v.startswith("avc") or v.startswith("h264"):
        return "h264"
    if v.startswith("vp09") or v.startswith("vp9"):
        return "vp9"
    if v.startswith("av01"):
        return "av1"
    return v or "?"


def pick_formats(fmts: list[dict], mode: str = "native", design_height: int | None = None) -> list[dict]:
    """mode: 'native' (every codec at the design height), 'all' (every height, one per codec+height), or ids."""
    fmts = [f for f in fmts if not f["audio"] and (f.get("protocol") or "https").startswith("http")] or fmts
    if mode == "native":
        h = design_height or max((f["height"] or 0) for f in fmts)
        cand = [f for f in fmts if f["height"] == h]
        if not cand:  # fall back to the tallest available
            h = max((f["height"] or 0) for f in fmts)
            cand = [f for f in fmts if f["height"] == h]
    elif mode == "all":
        cand = fmts
    elif mode == "best":
        h = max((f["height"] or 0) for f in fmts)
        cand = [f for f in fmts if f["height"] == h]
        cand = [max(cand, key=lambda f: f["tbr"] or 0)]
    else:
        ids = set(mode.split(","))
        cand = [f for f in fmts if f["format_id"] in ids]
    # one per (codec family, height): the highest bitrate variant
    best: dict[tuple, dict] = {}
    for f in cand:
        k = (codec_family(f["vcodec"]), f["height"])
        if k not in best or (f["tbr"] or 0) > (best[k]["tbr"] or 0):
            best[k] = f
    return sorted(best.values(), key=lambda f: ((f["height"] or 0), codec_family(f["vcodec"])))


def download(url: str, format_id: str, out_dir: str, quiet: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(os.path.join(out_dir, f"f{format_id}.*"))
    existing = [p for p in existing if not p.endswith((".part", ".ytdl"))]
    if existing:
        return existing[0]
    tmpl = os.path.join(out_dir, f"f{format_id}.%(ext)s")
    cmd = _ytdlp_cmd() + ["-f", format_id, "--no-playlist", "--no-part", "-o", tmpl, url]
    if quiet:
        cmd.insert(1 if cmd[0].endswith("yt-dlp") else 3, "-q")
    subprocess.run(cmd, check=True)
    files = [p for p in glob.glob(os.path.join(out_dir, f"f{format_id}.*")) if not p.endswith((".part", ".ytdl"))]
    if not files:
        raise RuntimeError("yt-dlp finished but no file appeared")
    return files[0]


def format_label(f: dict) -> str:
    return f"f{f['format_id']} {codec_family(f['vcodec'])} {f['height']}p{int(f['fps'] or 0)} {int(f['tbr'] or 0)}kbps"


def upload(path: str, title: str, description: str = "", privacy: str = "unlisted",
           client_secrets: str = "client_secrets.json", token_file: str = "youtube_token.json") -> str:
    """Resumable upload through the YouTube Data API v3.  Needs google-api-python-client + google-auth-oauthlib
    and an OAuth client (Desktop app) JSON from Google Cloud Console with the YouTube Data API enabled."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install google-api-python-client google-auth-oauthlib") from e
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets, scopes)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w") as fh:
            fh.write(creds.to_json())
    yt = build("youtube", "v3", credentials=creds)
    body = dict(snippet=dict(title=title[:100], description=description[:5000], categoryId="28"),
                status=dict(privacyStatus=privacy, selfDeclaredMadeForKids=False))
    media = MediaFileUpload(path, chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"\r  uploaded {status.progress() * 100:5.1f}%", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return resp["id"]
