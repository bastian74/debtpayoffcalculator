"""GF(2^8) arithmetic and a batch Reed-Solomon codec (errors + erasures).

Field: GF(256) with primitive polynomial 0x11d, generator alpha = 2, first
consecutive root fcr = 0 -- the same conventions as QR codes and the
``reedsolo`` package, so results can be cross-checked against it.

Codewords are 255 symbols long at most, laid out highest-degree-first (symbol 0
is the coefficient of x^(n-1)), systematic: ``k`` message bytes followed by
``nsym`` parity bytes.

Two usage patterns, both vectorised with numpy over a *batch* of codewords:

* inner code: every 255-byte codeword of one video frame is a row of the batch;
* outer code: every byte *column* across the K+M frames of a group is a row of
  the batch, so the erasure positions (lost frames) are shared by all rows.

The no-error path (syndromes all zero) costs one gather per batch.  Rows that
carry only the shared erasures are repaired in a single vectorised Forney
pass.  Rows with unknown error positions fall back to a per-row
Berlekamp-Massey + Chien search in plain Python (tables make it ~1 ms a row).
"""
from __future__ import annotations

import numpy as np

PRIM = 0x11D
GENERATOR = 2
FCR = 0
FIELD = 255  # multiplicative group order

# ---------------------------------------------------------------- tables ----
EXP = np.zeros(512, dtype=np.int64)
LOG = np.zeros(256, dtype=np.int64)
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIM
EXP[255:510] = EXP[0:255]
EXP[510:512] = EXP[0:2]

# Full multiplication table (64 KiB): MUL[a, b] = a * b in GF(256).
_a = np.arange(256)
MUL = np.where(
    (_a[:, None] == 0) | (_a[None, :] == 0),
    0,
    EXP[LOG[_a][:, None] + LOG[_a][None, :]],
).astype(np.uint8)
# ALPHA_POW[i, d] = alpha^(i*d)  (evaluation table for Chien search / syndromes)
_ij = (np.arange(256)[:, None] * np.arange(256)[None, :]) % 255
ALPHA_POW = EXP[_ij].astype(np.uint8)
INV = np.zeros(256, dtype=np.uint8)
INV[1:] = EXP[(255 - LOG[1:]) % 255]

_MUL_L = MUL.tolist()  # python lists are faster than numpy for scalar loops
_EXP_L = EXP.tolist()
_LOG_L = LOG.tolist()
_INV_L = INV.tolist()


def gf_mul(a: int, b: int) -> int:
    return _MUL_L[a][b]


def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if a == 0:
        return 0
    return _EXP_L[(_LOG_L[a] - _LOG_L[b]) % 255]


def gf_pow(a: int, p: int) -> int:
    if a == 0:
        return 0 if p > 0 else 1
    return _EXP_L[(_LOG_L[a] * p) % 255]


def gf_inverse(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF(256) inverse of zero")
    return _INV_L[a]


# ------------------------------------------------------ scalar polynomials ----
# Polynomials are python lists, highest degree first.

def poly_scale(p, x):
    m = _MUL_L[x]
    return [m[c] for c in p]


def poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        r[i + len(r) - len(p)] = c
    for i, c in enumerate(q):
        r[i + len(r) - len(q)] ^= c
    return r


def poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for j, qc in enumerate(q):
        if qc == 0:
            continue
        m = _MUL_L[qc]
        for i, pc in enumerate(p):
            r[i + j] ^= m[pc]
    return r


def poly_eval(p, x):
    y = p[0]
    m = _MUL_L
    for c in p[1:]:
        y = m[y][x] ^ c
    return y


def poly_div(dividend, divisor):
    out = list(dividend)
    normalizer = divisor[0]
    for i in range(len(dividend) - (len(divisor) - 1)):
        out[i] = gf_div(out[i], normalizer)
        coef = out[i]
        if coef != 0:
            m = _MUL_L[coef]
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    out[i + j] ^= m[divisor[j]]
    sep = -(len(divisor) - 1)
    return out[:sep], out[sep:]


_GEN_CACHE: dict[int, list[int]] = {}


def generator_poly(nsym: int) -> list[int]:
    g = _GEN_CACHE.get(nsym)
    if g is None:
        g = [1]
        for i in range(nsym):
            g = poly_mul(g, [1, gf_pow(GENERATOR, i + FCR)])
        _GEN_CACHE[nsym] = g
    return g


# -------------------------------------------------------------- encoding ----

def encode(msg: np.ndarray, nsym: int) -> np.ndarray:
    """Systematic RS encode a batch.  msg: (B, k) uint8 -> (B, k+nsym) uint8."""
    msg = np.asarray(msg, dtype=np.uint8)
    if msg.ndim == 1:
        msg = msg[None, :]
    B, k = msg.shape
    if k + nsym > 255:
        raise ValueError(f"k + nsym = {k + nsym} exceeds 255")
    gen = np.asarray(generator_poly(nsym)[1:], dtype=np.uint8)  # drop leading 1
    rem = np.zeros((B, nsym), dtype=np.uint8)
    for i in range(k):
        coef = msg[:, i] ^ rem[:, 0]
        rem[:, :-1] = rem[:, 1:]
        rem[:, -1] = 0
        rem ^= MUL[coef[:, None], gen[None, :]]
    return np.concatenate([msg, rem], axis=1)


# -------------------------------------------------------------- decoding ----

def syndromes(cw: np.ndarray, nsym: int) -> np.ndarray:
    """Batch syndromes S_j = cw(alpha^(j+fcr)), j = 0..nsym-1.  (B, n) -> (B, nsym)."""
    B, n = cw.shape
    degs = np.arange(n - 1, -1, -1)  # degree of each position
    js = np.arange(nsym) + FCR
    P = EXP[(js[:, None] * degs[None, :]) % 255].astype(np.uint8)  # alpha^(j*deg)
    prod = MUL[cw[:, None, :], P[None, :, :]]  # (B, nsym, n)
    return np.bitwise_xor.reduce(prod, axis=2)


def _erasure_locator(coef_pos: list[int]) -> list[int]:
    loc = [1]
    for p in coef_pos:
        loc = poly_mul(loc, poly_add([1], [gf_pow(GENERATOR, p), 0]))
    return loc


def _forney_syndromes(synd_list: list[int], erase_pos: list[int], n: int) -> list[int]:
    # synd_list: S_0..S_{nsym-1} (no leading zero padding)
    fsynd = list(synd_list)
    for p in erase_pos:
        x = gf_pow(GENERATOR, n - 1 - p)
        for j in range(len(fsynd) - 1):
            fsynd[j] = _MUL_L[fsynd[j]][x] ^ fsynd[j + 1]
    return fsynd


def _find_error_locator(synd: list[int], nsym: int, erase_count: int) -> list[int]:
    """Berlekamp-Massey on (Forney) syndromes; returns locator highest-degree-first."""
    err_loc = [1]
    old_loc = [1]
    synd_shift = len(synd) - nsym
    for i in range(nsym - erase_count):
        K = i + synd_shift
        delta = synd[K]
        for j in range(1, len(err_loc)):
            delta ^= _MUL_L[err_loc[-(j + 1)]][synd[K - j]]
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = poly_scale(old_loc, delta)
                old_loc = poly_scale(err_loc, gf_inverse(delta))
                err_loc = new_loc
            err_loc = poly_add(err_loc, poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    errs = len(err_loc) - 1
    if (errs - erase_count) * 2 + erase_count > nsym:
        raise ValueError("too many errors to correct")
    return err_loc


def _find_errors(err_loc: list[int], n: int) -> list[int]:
    """Chien search (vectorised): positions (0-based from the start) of the roots."""
    errs = len(err_loc) - 1
    if errs == 0:
        return []
    coeffs = np.asarray(err_loc, dtype=np.uint8)  # highest degree first
    degs = np.arange(errs, -1, -1)
    i = np.arange(n)
    # evaluate at alpha^i for i in 0..n-1
    P = EXP[(i[:, None] * degs[None, :]) % 255].astype(np.uint8)
    vals = np.bitwise_xor.reduce(MUL[coeffs[None, :], P], axis=1)
    roots = np.nonzero(vals == 0)[0]
    pos = [int(n - 1 - r) for r in roots]
    if len(pos) != errs:
        raise ValueError("could not locate all errors")
    return pos


def _correct_errata(cw: list[int], synd: list[int], err_pos: list[int]) -> list[int]:
    """Forney algorithm.  ``synd`` is S_0..S_{nsym-1}; ``err_pos`` are 0-based positions."""
    n = len(cw)
    coef_pos = [n - 1 - p for p in err_pos]
    err_loc = _erasure_locator(coef_pos)                    # highest degree first
    e = len(err_loc) - 1
    # Omega' = (x * S(x) * Lambda(x)) mod x^(e+1), highest degree first.
    xs_rev = synd[::-1] + [0]                               # [S_{nsym-1} .. S_0, 0]
    _, omega = poly_div(poly_mul(xs_rev, err_loc), [1] + [0] * (e + 1))
    X = [gf_pow(GENERATOR, -(255 - cp)) for cp in coef_pos]
    E = [0] * n
    for i, Xi in enumerate(X):
        Xi_inv = gf_inverse(Xi)
        err_loc_prime = 1
        for j, Xj in enumerate(X):
            if j != i:
                err_loc_prime = _MUL_L[err_loc_prime][1 ^ _MUL_L[Xi_inv][Xj]]
        if err_loc_prime == 0:
            raise ValueError("could not find error magnitude")
        y = poly_eval(omega, Xi_inv)
        y = _MUL_L[gf_pow(Xi, 1 - FCR)][y]
        E[err_pos[i]] = gf_div(y, err_loc_prime)
    return [c ^ v for c, v in zip(cw, E)]


def correct_row(cw: list[int], nsym: int, synd: list[int], erase_pos: list[int]) -> list[int]:
    """Decode one codeword given its syndromes.  Raises ValueError if uncorrectable."""
    n = len(cw)
    if len(erase_pos) > nsym:
        raise ValueError("too many erasures")
    fsynd = _forney_syndromes(synd, erase_pos, n)
    err_loc = _find_error_locator(fsynd, nsym, len(erase_pos))
    err_pos = _find_errors(err_loc[::-1], n) if len(err_loc) > 1 else []
    out = _correct_errata(cw, synd, list(erase_pos) + err_pos)
    return out


def decode(cw: np.ndarray, nsym: int, erase_pos: list[int] | None = None):
    """Batch decode.  cw: (B, n) uint8.  Returns (corrected (B, n), ok (B,) bool, nerr (B,) int).

    ``erase_pos`` are positions (0-based from the start of the codeword) known to
    be wrong in *every* row (the outer code's lost frames).  ``nerr`` counts the
    symbols actually changed by the decoder (erasures included) and is -1 for
    rows that could not be corrected (those rows are returned unchanged).
    """
    cw = np.ascontiguousarray(cw, dtype=np.uint8)
    if cw.ndim == 1:
        cw = cw[None, :]
    B, n = cw.shape
    erase_pos = sorted(set(int(p) for p in (erase_pos or [])))
    out = cw.copy()
    ok = np.ones(B, dtype=bool)
    nerr = np.zeros(B, dtype=np.int64)
    if n <= nsym or n > 255:
        raise ValueError("bad codeword length")
    S = syndromes(cw, nsym)
    dirty = np.any(S != 0, axis=1)
    if erase_pos:
        dirty[:] = True  # erased symbols are unknown even if syndromes happen to vanish
    if not dirty.any():
        return out, ok, nerr
    if erase_pos and len(erase_pos) > nsym:
        ok[:] = False
        nerr[:] = -1
        return out, ok, nerr

    rows = np.nonzero(dirty)[0]
    if erase_pos:
        # Vectorised erasure-only pass for rows whose Forney syndromes vanish.
        fixed = _erasures_only_batch(cw[rows], S[rows], nsym, erase_pos)
        if fixed is not None:
            cand, good = fixed
            out[rows[good]] = cand[good]
            nerr[rows[good]] = len(erase_pos)
            rows = rows[~good]
    S_list = S.tolist()
    for r in rows:
        try:
            fixed_row = correct_row(cw[r].tolist(), nsym, S_list[r], erase_pos)
            fixed_arr = np.asarray(fixed_row, dtype=np.uint8)
            if np.any(syndromes(fixed_arr[None, :], nsym) != 0):
                raise ValueError("residual syndromes")
            nerr[r] = int(np.count_nonzero(fixed_arr != cw[r]))
            out[r] = fixed_arr
        except (ValueError, ZeroDivisionError):
            ok[r] = False
            nerr[r] = -1
    return out, ok, nerr


def _erasures_only_batch(cw: np.ndarray, S: np.ndarray, nsym: int, erase_pos: list[int]):
    """Repair rows that have *only* the shared erasures.  Returns (candidate, good_mask)."""
    B, n = cw.shape
    e = len(erase_pos)
    if e == 0:
        return None
    coef_pos = [n - 1 - p for p in erase_pos]
    loc = _erasure_locator(coef_pos)  # highest degree first, len e+1
    loc_arr = np.asarray(loc, dtype=np.uint8)
    # Forney syndromes (shared transform): detect rows with extra errors.
    # Compute via polynomial product S(x)*loc(x) and inspect the top nsym-e coefficients:
    # Equivalent to reedsolo's rs_forney_syndromes; rows where they all vanish have only erasures.
    S_rev = S[:, ::-1]  # (B, nsym) highest-degree-first  [S_{nsym-1} ... S_0]
    # product with loc: (B, nsym + e)
    prod = np.zeros((B, nsym + e), dtype=np.uint8)
    for j, lc in enumerate(loc):
        if lc:
            prod[:, j:j + nsym] ^= MUL[lc, S_rev]
    # Omega = prod mod x^(e+1) -> last e+1 coefficients... but Forney syndromes are the
    # coefficients of degree >= e of (S*loc) truncated to nsym terms: prod[:, e:nsym] after
    # aligning; reedsolo's fsynd[j] for j < nsym-e equal prod coefficients of x^(nsym-1-j).
    fs = prod[:, e:nsym]  # (B, nsym-e): high part == forney syndromes
    good = ~np.any(fs != 0, axis=1) if fs.shape[1] > 0 else np.ones(B, dtype=bool)
    if not good.any():
        return cw.copy(), good
    # Omega' = (x*S*Lambda) mod x^(e+1): the last e coefficients of S*Lambda, then a 0.
    omega = np.concatenate([prod[:, -e:], np.zeros((B, 1), dtype=np.uint8)], axis=1)  # (B, e+1)
    X = [gf_pow(GENERATOR, -(255 - cp)) for cp in coef_pos]
    cand = cw.copy()
    degs = np.arange(e, -1, -1)
    for i, Xi in enumerate(X):
        xi_inv = gf_inverse(Xi)
        lp = 1
        for j, Xj in enumerate(X):
            if j != i:
                lp = _MUL_L[lp][1 ^ _MUL_L[xi_inv][Xj]]
        if lp == 0:
            return None
        pw = EXP[(_LOG_L[xi_inv] * degs) % 255].astype(np.uint8)
        y = np.bitwise_xor.reduce(MUL[omega, pw[None, :]], axis=1)  # Omega'(Xi^-1) per row
        y = MUL[gf_pow(Xi, 1 - FCR), y]
        mag = MUL[y, gf_inverse(lp)]
        cand[:, erase_pos[i]] ^= mag
    # verify the candidates for the good rows
    chk = syndromes(cand[good], nsym)
    good_idx = np.nonzero(good)[0]
    bad_after = np.any(chk != 0, axis=1)
    good[good_idx[bad_after]] = False
    return cand, good
