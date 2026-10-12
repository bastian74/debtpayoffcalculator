"""The density sweep: which (cell, bits, chroma, repeat) combinations to try, in what order."""
from __future__ import annotations

from .frame import Profile, geometry


def payload_per_frame(W: int, H: int, p: Profile, nsym: int) -> int:
    return geometry(W, H, p, nsym).payload_len


def gross_rate(W: int, H: int, fps: float, p: Profile, nsym: int) -> float:
    """Payload bytes per second of video for this profile (before any outer code)."""
    return payload_per_frame(W, H, p, nsym) * fps / p.repeat


def sweep(kind: str = "default") -> list[Profile]:
    """Profiles to put in a calibration video.  'quick' ~ 12, 'default' ~ 30, 'wide' ~ 60."""
    cells_default = (8, 6, 5, 4, 3, 2)
    out: set[Profile] = set()
    if kind == "quick":
        for c in (8, 4, 3, 2):
            out.add(Profile(c, 1, 0, 1))
        for c in (4, 2):
            out.add(Profile(c, 2, 0, 1))
            out.add(Profile(c, 1, 1, 1))
            out.add(Profile(c, 2, 1, 1))
        out.add(Profile(4, 3, 0, 1))
        out.add(Profile(4, 4, 1, 1))
    elif kind == "wide":
        for c in cells_default:
            for yb in (1, 2, 3, 4):
                for cb in (0, 1, 2):
                    if cb and c < 3 and yb >= 3:
                        continue
                    out.add(Profile(c, yb, cb, 1))
        for c in (4, 3, 2):
            out.add(Profile(c, 1, 0, 2))
            out.add(Profile(c, 2, 0, 2))
    else:  # default
        for c in cells_default:
            out.add(Profile(c, 1, 0, 1))
            out.add(Profile(c, 2, 0, 1))
        for c in (6, 4, 3, 2):
            out.add(Profile(c, 1, 1, 1))
            out.add(Profile(c, 2, 1, 1))
            out.add(Profile(c, 3, 0, 1))
        for c in (8, 6, 4):
            out.add(Profile(c, 4, 0, 1))
            out.add(Profile(c, 3, 2, 1))
            out.add(Profile(c, 2, 2, 1))
        out.add(Profile(4, 1, 0, 2))
        out.add(Profile(2, 1, 0, 2))
        out.add(Profile(3, 2, 0, 2))
    return sorted(out, key=lambda p: (gross_rate(1920, 1080, 30, p, 32), p))


def ordered(profiles, W: int, H: int, fps: float, nsym: int) -> list[Profile]:
    """Lowest density first, so the calibration video gets progressively denser."""
    return sorted(profiles, key=lambda p: (gross_rate(W, H, fps, p, nsym), p))
