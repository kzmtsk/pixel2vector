"""Stage 3: geometry fitting on each chain.

Per chain:
  1. corner detection on the lattice polyline (two-scale turning-angle test, so curves
     of small radius are not mistaken for corners)
  2. split at corners, Gaussian-smooth each piece with endpoints pinned
  3. closed loops without corners: whole-shape test (circle, ellipse)
  4. recursive fit with a primitive ladder: line -> circular arc -> cubic Bezier
     (Schneider least-squares with Newton reparametrisation), splitting at max error

Segment tuples (working-pixel coordinates, y down):
  ("L", p0, p1)
  ("A", p0, p1, c, R, a0, sweep)        angles in radians, image coords, sweep signed
  ("C", p0, c1, c2, p1)
  ("CIRCLE", c, R, direction)           full closed loop; direction +1 / -1 (angle increasing / decreasing)
  ("ELLIPSE", c, a, b, theta, direction)
"""
import math
import numpy as np
import cv2


# ---------------------------------------------------------------- helpers
def _unit(v):
    n = math.hypot(v[0], v[1])
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def smooth_open(P, sigma):
    n = len(P)
    if n < 4 or sigma <= 0:
        return P.copy()
    sig = max(0.4, min(sigma, (n - 1) / 3.0))
    k, r = _gauss_kernel(sig)
    if r > n - 1:
        r = n - 1
        x = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / sig) ** 2)
        k /= k.sum()
    # odd reflection about endpoints keeps tangent direction and pins the endpoint
    left = 2 * P[0] - P[1:r + 1][::-1]
    right = 2 * P[-1] - P[-r - 1:-1][::-1]
    Q = np.vstack([left, P, right])
    out = np.empty_like(P)
    for c in range(2):
        out[:, c] = np.convolve(Q[:, c], k, mode="valid")
    out[0], out[-1] = P[0], P[-1]
    return out


def smooth_closed(P, sigma):
    n = len(P)
    if n < 6 or sigma <= 0:
        return P.copy()
    sig = max(0.4, min(sigma, (n // 2 - 1) / 3.0))
    k, r = _gauss_kernel(sig)
    r = min(r, n // 2 - 1)
    if (len(k) - 1) // 2 != r:
        x = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / sig) ** 2)
        k /= k.sum()
    Q = np.vstack([P[-r:], P, P[:r]])
    out = np.empty_like(P)
    for c in range(2):
        out[:, c] = np.convolve(Q[:, c], k, mode="valid")
    return out


def _turn(P, i, k, closed):
    n = len(P)
    if closed:
        a, b = P[(i - k) % n], P[(i + k) % n]
    else:
        if i - k < 0 or i + k >= n:
            return 0.0
        a, b = P[i - k], P[i + k]
    v1 = P[i] - a
    v2 = b - P[i]
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = (v1 @ v2) / (n1 * n2)
    return math.acos(max(-1.0, min(1.0, c)))


def prune_corners(P, corners, closed, s, min_len_px=None, keep_deg=110.0):
    """Drop corners that only bound a tiny piece (pixel stairs, JPEG noise): every corner costs
    anchors, so a corner must earn its place with legs of real length, unless it is very sharp."""
    if len(corners) < 2 and not (not closed and corners):
        return corners
    n = len(P)
    k = max(3, int(round(3 * s)))
    Lmin = min_len_px if min_len_px is not None else 2.5 * s
    seg_len = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    total = seg_len[-1] + (math.hypot(*(P[0] - P[-1])) if closed else 0)
    cs = list(corners)
    changed = True
    while changed and cs:
        changed = False
        ang = {c: _turn(P, c, k, closed) for c in cs}
        bounds = cs if closed else [0] + cs + [n - 1]
        for idx in range(len(bounds) - (0 if closed else 1)):
            a, b = bounds[idx], bounds[(idx + 1) % len(bounds)]
            L = (seg_len[b] - seg_len[a]) if b > a else (total - seg_len[a] + seg_len[b])
            if L >= Lmin:
                continue
            cand = [c for c in (a, b) if c in ang and math.degrees(ang[c]) < keep_deg]
            if not cand:
                continue
            drop = min(cand, key=lambda c: ang[c])
            cs.remove(drop)
            changed = True
            break
    return cs


def detect_corners(P, closed, s, angle_deg=50.0):
    n = len(P)
    k = max(3, int(round(3 * s)))
    k2 = max(2, k // 2)
    if (closed and n < 2 * k + 2) or (not closed and n < 2 * k + 1):
        return []
    thr = math.radians(angle_deg)
    rng = range(n) if closed else range(k, n - k)
    t = np.zeros(n)
    for i in rng:
        a = _turn(P, i, k, closed)
        if a < thr:
            continue
        a2 = _turn(P, i, k2, closed)
        # sharpness test: a true corner keeps its angle at half window, a curve halves it
        if a2 >= 0.62 * a:
            t[i] = a
    cand = np.nonzero(t)[0]
    corners = []
    for i in cand:
        lo, hi = i - k, i + k
        if closed:
            win = [(j % n) for j in range(lo, hi + 1)]
        else:
            win = list(range(max(0, lo), min(n, hi + 1)))
        if t[i] >= t[win].max() - 1e-12:
            if not corners or (i - corners[-1]) > k // 2:
                corners.append(int(i))
    if closed and len(corners) > 1 and (corners[0] + n - corners[-1]) <= k // 2:
        corners.pop()
    return corners


# ---------------------------------------------------------------- primitive fits
def fit_line(P, tol):
    p0, p1 = P[0], P[-1]
    d = p1 - p0
    L = math.hypot(*d)
    if L < 1e-9:
        return None
    nrm = np.array([-d[1], d[0]]) / L
    err = np.abs((P - p0) @ nrm).max()
    # also reject back-tracking (projection must stay within the chord)
    proj = (P - p0) @ (d / L)
    if proj.min() < -tol or proj.max() > L + tol:
        return None
    return ("L", p0.copy(), p1.copy()) if err <= tol else None


def fit_arc(P, tol, min_sweep=0.2, max_r_factor=40.0):
    n = len(P)
    if n < 6:
        return None
    p0, p1 = P[0], P[-1]
    d = p1 - p0
    L = math.hypot(*d)
    if L < 1e-9:
        return None
    m = (p0 + p1) / 2
    nrm = np.array([-d[1], d[0]]) / L
    D = P - p0
    a = (P ** 2).sum(1) - (p0 ** 2).sum() - 2 * (D @ m)
    b = 2 * (D @ nrm)
    bb = (b * b).sum()
    if bb < 1e-12:
        return None
    t = (a * b).sum() / bb
    c = m + t * nrm
    R = math.hypot(*(p0 - c))
    if R > max_r_factor * L:
        return None
    err = np.abs(np.hypot(*(P - c).T) - R).max()
    if err > tol:
        return None
    ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    da = np.diff(ang)
    sweep = ang[-1] - ang[0]
    if abs(sweep) < min_sweep:
        return None
    # monotonic angular progress
    if sweep > 0 and (da < -0.05).any():
        return None
    if sweep < 0 and (da > 0.05).any():
        return None
    return ("A", p0.copy(), p1.copy(), c, R, float(ang[0]), float(sweep))


def _bez(ctrl, u):
    p0, p1, p2, p3 = ctrl
    mu = 1 - u
    return (mu ** 3)[:, None] * p0 + (3 * mu * mu * u)[:, None] * p1 + (3 * mu * u * u)[:, None] * p2 + (u ** 3)[:, None] * p3


def _bez_d1(ctrl, u):
    p0, p1, p2, p3 = ctrl
    mu = 1 - u
    return (3 * mu * mu)[:, None] * (p1 - p0) + (6 * mu * u)[:, None] * (p2 - p1) + (3 * u * u)[:, None] * (p3 - p2)


def _bez_d2(ctrl, u):
    p0, p1, p2, p3 = ctrl
    return (6 * (1 - u))[:, None] * (p2 - 2 * p1 + p0) + (6 * u)[:, None] * (p3 - 2 * p2 + p1)


def _chord_param(P):
    d = np.hypot(*np.diff(P, axis=0).T)
    u = np.r_[0, np.cumsum(d)]
    return u / u[-1] if u[-1] > 0 else np.linspace(0, 1, len(P))


def _gen_bezier(P, u, t0, t1):
    p0, p3 = P[0], P[-1]
    mu = 1 - u
    b0, b1, b2, b3 = mu ** 3, 3 * mu * mu * u, 3 * mu * u * u, u ** 3
    A1 = b1[:, None] * t0
    A2 = b2[:, None] * t1
    C00 = (A1 * A1).sum()
    C01 = (A1 * A2).sum()
    C11 = (A2 * A2).sum()
    tmp = P - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3
    X0 = (tmp * A1).sum()
    X1 = (tmp * A2).sum()
    det = C00 * C11 - C01 * C01
    seg = math.hypot(*(p3 - p0))
    a1 = a2 = 0.0
    if abs(det) > 1e-12:
        a1 = (X0 * C11 - X1 * C01) / det
        a2 = (C00 * X1 - C01 * X0) / det
    eps = 1e-6 * max(seg, 1.0)
    if a1 < eps or a2 < eps or a1 > 2.5 * seg or a2 > 2.5 * seg:
        a1 = a2 = seg / 3.0
    return [p0, p0 + a1 * t0, p3 + a2 * t1, p3]


def _max_err(P, ctrl, u):
    Q = _bez(ctrl, u)
    d = np.hypot(*(Q - P).T)
    i = int(d.argmax())
    return float(d[i]), i


def _reparam(P, ctrl, u):
    Q = _bez(ctrl, u)
    d1 = _bez_d1(ctrl, u)
    d2 = _bez_d2(ctrl, u)
    diff = Q - P
    num = (diff * d1).sum(1)
    den = (d1 * d1).sum(1) + (diff * d2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(np.abs(den) > 1e-12, num / den, 0.0)
    nu = np.clip(u - step, 0, 1)
    nu[0], nu[-1] = 0.0, 1.0
    return np.maximum.accumulate(nu)


def fit_cubic(P, t0, t1, tol):
    u = _chord_param(P)
    ctrl = _gen_bezier(P, u, t0, t1)
    err, idx = _max_err(P, ctrl, u)
    if err <= tol:
        return ctrl, err, idx
    if err <= 6 * tol:
        for _ in range(6):
            u = _reparam(P, ctrl, u)
            ctrl = _gen_bezier(P, u, t0, t1)
            err, idx = _max_err(P, ctrl, u)
            if err <= tol:
                break
    return ctrl, err, idx


def _tangent_start(P, k=4):
    j = min(len(P) - 1, k)
    return _unit(P[j] - P[0])


def _tangent_end(P, k=4):
    j = max(0, len(P) - 1 - k)
    return _unit(P[j] - P[-1])


def _seg_tangents(seg):
    """(forward unit tangent at start, backward unit tangent at end)."""
    if seg[0] == "L":
        d = _unit(seg[2] - seg[1])
        return d, -d
    _, p0, p1, c, R, a0, sw = seg
    sg = 1.0 if sw > 0 else -1.0
    a1 = a0 + sw
    f0 = sg * np.array([-math.sin(a0), math.cos(a0)])
    f1 = sg * np.array([-math.sin(a1), math.cos(a1)])
    return f0, -f1


def _tangent_ok(seg, t0, t1, max_deg=4.0):
    """When a piece must join its neighbours smoothly (G1), a line or arc is only acceptable
    if its own end tangents agree with the required ones."""
    if t0 is None and t1 is None:
        return True
    lim = math.cos(math.radians(max_deg))
    a, b = _seg_tangents(seg)
    if t0 is not None and float(a @ t0) < lim:
        return False
    if t1 is not None and float(b @ t1) < lim:
        return False
    return True


def fit_piece(P, tol, t0=None, t1=None, depth=0):
    n = len(P)
    if n <= 2 or math.hypot(*(P[-1] - P[0])) < 1e-9 and n <= 3:
        return [("L", P[0].copy(), P[-1].copy())]
    seg = fit_line(P, tol)
    if seg and _tangent_ok(seg, t0, t1):
        return [seg]
    seg = fit_arc(P, tol)
    if seg and _tangent_ok(seg, t0, t1):
        return [seg]
    if n <= 4 or depth > 24:
        return [("L", P[i].copy(), P[i + 1].copy()) for i in range(n - 1)]
    return _fit_greedy(P, tol, t0, t1)


def _cubic_ok(P, t0, t1, tol):
    ctrl, err, _ = fit_cubic(P, t0, t1, tol)
    return ctrl, err <= tol


def _tan_back_at(P, j, k=4):
    """Backward unit tangent at index j (points against the direction of travel)."""
    a = max(0, j - k)
    b = min(len(P) - 1, j + k)
    return _unit(P[a] - P[b])


def _fit_greedy(P, tol, t0=None, t1=None):
    """Fewest cubic segments: from each start, extend the segment as far as one cubic stays within
    tolerance (exponential then binary search), then continue G1 from its end tangent."""
    n = len(P)
    segs = []
    i = 0
    tin = t0 if t0 is not None else _tangent_start(P)
    while i < n - 1:
        rest = P[i:]
        tend = t1 if t1 is not None else _tangent_end(rest)
        ctrl, ok = _cubic_ok(rest, tin, tend, tol)
        if ok:
            segs.append(("C", ctrl[0].copy(), ctrl[1], ctrl[2], ctrl[3].copy()))
            break
        lo, hi = i + 1, n - 1          # lo: known fit (2 points always fit), hi: known failure
        best = None
        step = 4
        probe = min(i + step, hi - 1)
        while probe > lo:
            ctrl, ok = _cubic_ok(P[i:probe + 1], tin, _tan_back_at(P, probe), tol)
            if ok:
                lo, best = probe, ctrl
                step *= 2
                probe = min(i + step, hi - 1)
                if probe <= lo:
                    break
            else:
                hi = probe
                break
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ctrl, ok = _cubic_ok(P[i:mid + 1], tin, _tan_back_at(P, mid), tol)
            if ok:
                lo, best = mid, ctrl
            else:
                hi = mid
        j = lo
        if best is None:
            segs.append(("L", P[i].copy(), P[j].copy()))
            tin = _unit(P[j] - P[i])
        else:
            segs.append(("C", best[0].copy(), best[1], best[2], best[3].copy()))
            tin = -_tan_back_at(P, j)
        i = j
    return segs


# ---------------------------------------------------------------- anchor reduction
def _sample(sg, k=24):
    t = sg[0]
    if t == "L":
        return np.linspace(sg[1], sg[2], k)
    if t == "C":
        u = np.linspace(0, 1, k)
        return _bez([sg[1], sg[2], sg[3], sg[4]], u)
    if t == "A":
        _, p0, p1, c, R, a0, sw = sg
        a = a0 + np.linspace(0, 1, k) * sw
        return np.stack([c[0] + R * np.cos(a), c[1] + R * np.sin(a)], 1)
    return None


def _fwd_start(sg):
    if sg[0] == "C":
        v = sg[2] - sg[1]
        if math.hypot(*v) < 1e-9:
            v = sg[3] - sg[1]
        return _unit(v)
    if sg[0] == "L":
        return _unit(sg[2] - sg[1])
    f0, _ = _seg_tangents(sg)
    return f0


def _fwd_end(sg):
    if sg[0] == "C":
        v = sg[4] - sg[3]
        if math.hypot(*v) < 1e-9:
            v = sg[4] - sg[2]
        return _unit(v)
    if sg[0] == "L":
        return _unit(sg[2] - sg[1])
    _, b = _seg_tangents(sg)
    return -b


def reduce_anchors(segs, tol, closed, max_kink_deg=25.0):
    """Merge neighbouring segments into one cubic wherever their joint is smooth (not a corner) and
    one cubic still follows both within tolerance. Collinear lines merge into one line. Repeats until
    nothing merges. For closed chains the joint across the start point is merged as well."""
    if not segs or segs[0][0] in ("CIRCLE", "ELLIPSE"):
        return segs
    lim = math.cos(math.radians(max_kink_deg))
    segs = list(segs)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        i = 0
        while i < len(segs) - (0 if closed and len(segs) > 2 else 1):
            j = (i + 1) % len(segs)
            a, b = segs[i], segs[j]
            if float(_fwd_end(a) @ _fwd_start(b)) < lim:
                i += 1
                continue
            if a[0] == "L" and b[0] == "L" and fit_line(np.vstack([_sample(a, 8), _sample(b, 8)[1:]]), tol):
                merged = ("L", a[1].copy(), b[2].copy())
            else:
                pts = np.vstack([_sample(a), _sample(b)[1:]])
                ctrl, err, _ = fit_cubic(pts, _fwd_start(a), -_fwd_end(b), tol)
                if err > tol:
                    i += 1
                    continue
                merged = ("C", a[1].copy() if a[0] != "C" else a[1].copy(), ctrl[1], ctrl[2],
                          (b[2] if b[0] in ("L", "A") else b[4]).copy())
            if j == 0:  # wrapped joint of a closed chain: merged segment becomes the first
                segs = [merged] + segs[1:i]
            else:
                segs[i:i + 2] = [merged]
            changed = True
    return segs


# ---------------------------------------------------------------- Illustrator-style anchors
def _seg_len(sg):
    S = _sample(sg, 16)
    return float(np.hypot(*np.diff(S, axis=0).T).sum())


def _extrema(S, min_gap, min_amp=0.6):
    """Indices where the curve turns in x or in y (horizontal/vertical tangent), with hysteresis so
    pixel noise does not create extrema."""
    out = []
    for c in (0, 1):
        v = S[:, c]
        last_i, last_v, direction = 0, v[0], 0
        for i in range(1, len(v)):
            d = v[i] - last_v
            if direction >= 0 and d > 0:
                last_i, last_v, direction = i, v[i], 1
            elif direction <= 0 and d < 0:
                last_i, last_v, direction = i, v[i], -1
            elif abs(d) >= min_amp:
                # reversed by more than min_amp: the previous max/min was a real extremum
                if 0 < last_i < len(v) - 1:
                    out.append(last_i)
                last_i, last_v, direction = i, v[i], -direction
    out = sorted(set(out))
    kept = []
    for i in out:
        if i < min_gap or i > len(S) - 1 - min_gap:
            continue
        if kept and i - kept[-1] < min_gap:
            continue
        kept.append(i)
    return kept


def _axis_tangent(S, i, k=3):
    d = S[min(len(S) - 1, i + k)] - S[max(0, i - k)]
    if abs(d[0]) >= abs(d[1]):
        return np.array([math.copysign(1.0, d[0]), 0.0])
    return np.array([0.0, math.copysign(1.0, d[1])])


def _shift_start(sg, p):
    d = p - sg[1]
    if sg[0] == "L":
        return ("L", p.copy(), sg[2])
    if sg[0] == "C":
        return ("C", p.copy(), sg[2] + d, sg[3], sg[4])
    return None


def drop_tiny(segs, min_len):
    """Remove near-duplicate anchors: a segment shorter than min_len is absorbed by its successor
    (whose start moves onto the tiny segment's start, so the outline stays closed and pinned ends
    stay put)."""
    if len(segs) < 2 or segs[0][0] in ("CIRCLE", "ELLIPSE"):
        return segs
    out = list(segs)
    i = 0
    while i < len(out) - 1 and len(out) > 1:
        sg = out[i]
        if sg[0] in ("L", "C") and _seg_len(sg) < min_len:
            nxt = _shift_start(out[i + 1], sg[1]) if out[i + 1][0] in ("L", "C") else None
            if nxt is not None:
                out[i:i + 2] = [nxt]
                continue
        i += 1
    return out


def restyle(segs, tol, s, closed, extrema=True, corner_deg=25.0):
    """Re-anchor smooth stretches the way type designers and Illustrator users expect: an anchor at
    every horizontal/vertical extreme of the curve with an axis-aligned handle, as few anchors as
    possible in between. Corners and straight lines are kept as they are."""
    if not segs or segs[0][0] in ("CIRCLE", "ELLIPSE"):
        return segs
    lim = math.cos(math.radians(corner_deg))
    # split into smooth stretches at corners
    stretches, cur = [], [segs[0]]
    for a, b in zip(segs[:-1], segs[1:]):
        if float(_fwd_end(a) @ _fwd_start(b)) < lim:
            stretches.append(cur)
            cur = [b]
        else:
            cur.append(b)
    stretches.append(cur)
    out = []
    for st in stretches:
        if all(sg[0] == "L" for sg in st):
            out += st
            continue
        pts = [_sample(sg, max(8, int(_seg_len(sg) / 0.75))) for sg in st]
        S = np.vstack([pts[0]] + [p[1:] for p in pts[1:]])
        t_start, t_end = _fwd_start(st[0]), -_fwd_end(st[-1])
        ex = _extrema(S, max(6, int(round(10 * s)))) if extrema else []
        cuts = [0] + ex + [len(S) - 1]
        for a, b in zip(cuts[:-1], cuts[1:]):
            Q = S[a:b + 1].copy()
            ta = t_start if a == 0 else _axis_tangent(S, a)
            tb = t_end if b == len(S) - 1 else -_axis_tangent(S, b)
            if b > 0 and b < len(S) - 1:
                Q[-1] = S[b]
            out += _fit_greedy(Q, tol, ta, tb)
    return out


# ---------------------------------------------------------------- whole-shape tests
def fit_circle_loop(P, tol):
    if len(P) < 12:
        return None
    x, y = P[:, 0], P[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    rhs = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r2 = cx * cx + cy * cy - sol[2]
    if r2 <= 0:
        return None
    R = math.sqrt(r2)
    err = np.abs(np.hypot(x - cx, y - cy) - R).max()
    if err > tol:
        return None
    direction = 1 if _signed_area(P) > 0 else -1
    return ("CIRCLE", np.array([cx, cy]), R, direction)


def fit_ellipse_loop(P, tol):
    if len(P) < 16:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(P.astype(np.float32))
    a, b = w / 2.0, h / 2.0
    if a < 1 or b < 1:
        return None
    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)
    X = P[:, 0] - cx
    Y = P[:, 1] - cy
    xl = ct * X + st * Y
    yl = -st * X + ct * Y
    r = np.sqrt((xl / a) ** 2 + (yl / b) ** 2)
    dist = np.abs(r - 1) * np.hypot(xl, yl) / np.maximum(r, 1e-9)
    if dist.max() > tol * 1.15:
        return None
    direction = 1 if _signed_area(P) > 0 else -1
    return ("ELLIPSE", np.array([cx, cy]), a, b, th, direction)


def _signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def _tls(P):
    c = P.mean(0)
    X = P - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    d = vt[0]
    if d @ (P[-1] - P[0]) < 0:
        d = -d
    return c, d, float(np.abs(X @ np.array([-d[1], d[0]])).max())


def _is_line(P, tol):
    return len(P) >= 2 and _tls(P)[2] <= tol


def straight_runs(P, tol, min_len):
    """Greedy maximal straight runs (index pairs) of at least min_len points."""
    n = len(P)
    runs = []
    i = 0
    last_end = 0
    while i < n - 1:
        j = min(n - 1, i + min_len)
        if j - i < min_len or not _is_line(P[i:j + 1], tol):
            i += max(1, min_len // 4)
            continue
        best, step = j, 4
        while best < n - 1:
            cand = min(n - 1, best + step)
            if _is_line(P[i:cand + 1], tol):
                best = cand
                step *= 2
            elif step == 1:
                break
            else:
                step = max(1, step // 2)
        a = i
        while a - 1 >= last_end and _is_line(P[a - 1:best + 1], tol):
            a -= 1
        runs.append((a, best))
        last_end = best
        i = best
    return [r for r in _filter_runs(P, runs, min_len)]


def _bow(P):
    """Systematic bend of a near-straight run: amplitude of the quadratic in its residual."""
    c, d, _ = _tls(P)
    nrm = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ nrm
    if len(P) < 5 or np.ptp(t) < 1e-9:
        return 0.0
    tt = t / (np.ptp(t) / 2)
    A = np.c_[tt * tt, tt, np.ones_like(tt)]
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    return abs(float(coef[0]))


def _filter_runs(P, runs, min_len, bow_max=0.22, chain_deg=40.0):
    """Reject chords of gentle curves: runs that bend, and runs that continue a neighbour run
    at a shallow angle (a sequence of near-collinear chords is a curve, not a polygon)."""
    def central(a, b):
        m = (b - a) // 7
        return P[a + m:b - m + 1]
    runs = [r for r in runs if _bow(central(*r)) <= bow_max]
    if len(runs) < 2:
        return runs
    dirs = [_tls(P[a:b + 1])[1] for a, b in runs]
    bad = set()
    lim = math.cos(math.radians(chain_deg))
    for i in range(len(runs) - 1):
        gap = runs[i + 1][0] - runs[i][1]
        if gap <= min_len and float(dirs[i] @ dirs[i + 1]) > lim:
            # nearly collinear neighbours: same edge (keep, merged later) only if jointly straight
            if not _is_line(P[runs[i][0]:runs[i + 1][1] + 1], 1e9) or \
               _tls(P[runs[i][0]:runs[i + 1][1] + 1])[2] > 2.5 * _tls(P[runs[i][0]:runs[i][1] + 1])[2] + 0.6:
                bad.update((i, i + 1))
    return [r for j, r in enumerate(runs) if j not in bad]


def run_tol(tol):
    # a straight run must be straighter than the general tolerance, or gentle arcs get chopped into chords
    return max(0.45, 0.5 * tol)


def run_len(s):
    return max(8, int(round(8 * s)))


def _fillet(la, lb, Q, tol):
    """Circle tangent to both lines that best fits the gap points Q. Returns (TA, TB, C, r) or None."""
    cA, dA = la
    cB, dB = lb
    X = _intersect((cA, dA), (cB, dB), min_sin=0.05)
    if X is None:
        return None
    a = -dA
    b = dB
    cosphi = max(-1.0, min(1.0, float(a @ b)))
    phi = math.acos(cosphi)  # interior angle at X
    if phi < 0.05 or phi > math.pi - 0.05:
        return None
    u = _unit(a + b)
    sh, ch = math.sin(phi / 2), math.cos(phi / 2)

    def resid(t):
        C = X + u * t
        r = t * sh
        return np.abs(np.hypot(*(Q - C).T) - r)

    dmax = float(np.hypot(*(Q - X).T).max()) * 4 + 1
    lo, hi = 1e-6, dmax
    for _ in range(80):  # golden-section on mean residual
        m1 = lo + (hi - lo) * 0.382
        m2 = lo + (hi - lo) * 0.618
        if resid(m1).mean() < resid(m2).mean():
            hi = m2
        else:
            lo = m1
    t = (lo + hi) / 2
    if resid(t).max() > tol:
        return None
    C = X + u * t
    r = t * sh
    TA = X + a * (t * ch)
    TB = X + b * (t * ch)
    return TA, TB, C, r


def _arc_tuple(p0, p1, C, r):
    v0, v1 = p0 - C, p1 - C
    a0 = math.atan2(v0[1], v0[0])
    sw = math.atan2(v0[0] * v1[1] - v0[1] * v1[0], v0 @ v1)
    return ("A", p0.copy(), p1.copy(), C.copy(), float(r), a0, sw)


def fit_smooth_piece(P, tol, s, t0=None, t1=None):
    """Straight runs become lines; the stretches between them become tangent fillets when a
    circle fits, otherwise curves with tangents inherited from the neighbouring lines (G1)."""
    n = len(P)
    runs = straight_runs(P, run_tol(tol), run_len(s))
    if not runs:
        return fit_piece(P, tol, t0, t1)
    # merge adjacent collinear runs (noise can break one edge into two)
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= run_len(s) and _is_line(P[merged[-1][0]:b + 1], run_tol(tol) * 1.5):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    lines = []
    for a, b in merged:
        c, d, _ = _tls(P[a:b + 1])
        pa = P[0].copy() if a == 0 else c + ((P[a] - c) @ d) * d
        pb = P[-1].copy() if b == n - 1 else c + ((P[b] - c) @ d) * d
        lines.append({"a": a, "b": b, "c": c, "d": d, "pa": pa, "pb": pb, "pin_a": a == 0, "pin_b": b == n - 1})
    gaps = {}
    for i in range(len(lines) - 1):
        A, B = lines[i], lines[i + 1]
        if B["a"] <= A["b"]:
            X = None if (A["pin_b"] or B["pin_a"]) else _intersect((A["c"], A["d"]), (B["c"], B["d"]))
            if X is None or math.hypot(*(X - A["pb"])) > 4 * tol:
                X = (A["pb"] + B["pa"]) / 2
            A["pb"], B["pa"] = X, X.copy()
            gaps[i] = ("join",)
            continue
        Q = P[A["b"]:B["a"] + 1]
        fil = None if (A["pin_b"] or B["pin_a"]) else _fillet((A["c"], A["d"]), (B["c"], B["d"]), Q, tol)
        if fil is not None:
            TA, TB, C, r = fil
            # tangent points must stay on their lines' own extent
            if (TA - A["pa"]) @ A["d"] > 0 and (B["pb"] - TB) @ B["d"] > 0:
                A["pb"], B["pa"] = TA, TB
                gaps[i] = ("arc", C, r)
                continue
        gaps[i] = ("curve",)
    segs = []
    if lines[0]["a"] > 0:
        Q = P[:lines[0]["a"] + 1].copy()
        Q[-1] = lines[0]["pa"]
        segs += fit_piece(Q, tol, t0, -lines[0]["d"])
    for i, L in enumerate(lines):
        segs.append(("L", L["pa"].copy(), L["pb"].copy()))
        if i in gaps:
            g = gaps[i]
            B = lines[i + 1]
            if g[0] == "arc":
                segs.append(_arc_tuple(L["pb"], B["pa"], g[1], g[2]))
            elif g[0] == "join":
                pass  # shared vertex already set
            else:
                Q = P[L["b"]:B["a"] + 1].copy()
                Q[0], Q[-1] = L["pb"], B["pa"]
                segs += fit_piece(Q, tol, L["d"], -B["d"])
    last = lines[-1]
    if last["b"] < n - 1:
        Q = P[last["b"]:].copy()
        Q[0] = last["pb"]
        segs += fit_piece(Q, tol, last["d"], t1)
    return segs


# ---------------------------------------------------------------- chain driver
def ls_line(P, tol, margin):
    """Total-least-squares line through the piece, ignoring rounded ends. Returns (centroid, dir) or None."""
    n = len(P)
    m = min(margin, max(0, (n - 4) // 4))
    Q = P[m:n - m] if n - 2 * m >= 3 else P
    if len(Q) < 2:
        return None
    c = Q.mean(0)
    X = Q - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    if np.abs(X @ nrm).max() > tol:
        return None
    # rounded ends may stray a little further, but not much
    if np.abs((P - c) @ nrm).max() > 3.0 * tol:
        return None
    if d @ (P[-1] - P[0]) < 0:
        d = -d
    return c, d


def _intersect(l1, l2, min_sin=0.17):
    c1, d1 = l1
    c2, d2 = l2
    cr = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cr) < min_sin:
        return None
    w = c2 - c1
    t = (w[0] * d2[1] - w[1] * d2[0]) / cr
    return c1 + t * d1


def _project(p, l):
    c, d = l
    return c + ((p - c) @ d) * d


def _fit_cornered(pieces, closed, s, tol, sigma):  # noqa: C901
    """pieces: lattice point arrays; consecutive pieces share their corner point."""
    sm = [smooth_open(p, sigma) for p in pieces]
    margin = max(2, int(round(1.5 * s)))
    lines = [ls_line(p, tol, margin) if len(p) >= 4 else None for p in sm]
    npc = len(pieces)
    cps = [p[0].copy() for p in pieces] + [pieces[-1][-1].copy()]
    reach = 3.0 * s + 2
    # internal corners (open: 1..npc-1; closed: all, cps[npc] is cps[0])
    idxs = range(npc) if closed else range(1, npc)
    for j in idxs:
        la = lines[j - 1] if j > 0 else (lines[-1] if closed else None)
        lb = lines[j]
        cp = cps[j]
        if la is not None and lb is not None:
            x = _intersect(la, lb)
            if x is not None and math.hypot(*(x - cp)) <= reach:
                cp = x
            else:
                cp = (_project(cp, la) + _project(cp, lb)) / 2
        elif la is not None:
            cp = _project(cp, la)
        elif lb is not None:
            cp = _project(cp, lb)
        cps[j] = cp
    if closed:
        cps[npc] = cps[0]
    segs = []
    for i in range(npc):
        a, b = cps[i], cps[i + 1]
        if lines[i] is not None:
            segs.append(("L", a.copy(), b.copy()))
        else:
            Q = sm[i].copy()
            Q[0], Q[-1] = a, b
            segs += fit_smooth_piece(Q, tol, s)
    return segs


def fit_chain(chain, s, tol, sigma, allow_primitives=True, corner_angle=40.0, anchor_style="minimal"):
    _fit_chain(chain, s, tol, sigma, allow_primitives, corner_angle)
    # second pass over each smooth stretch as a whole (between corners): refitting the already
    # smoothed geometry in one go removes anchors the piecewise first pass could not see
    if anchor_style == "extrema":
        chain.segs = restyle(chain.segs, tol * 0.6, s, chain.closed, extrema=True)
    else:
        chain.segs = reduce_anchors(restyle(chain.segs, tol * 0.6, s, chain.closed, extrema=False),
                                    tol * 0.4, chain.closed)
    chain.segs = drop_tiny(chain.segs, max(1.0, 0.8 * s))


def _fit_chain(chain, s, tol, sigma, allow_primitives=True, corner_angle=40.0):
    P = chain.pts
    if chain.closed:
        corners = prune_corners(P, detect_corners(P, True, s, corner_angle), True, s)
        if not corners:
            S = smooth_closed(P, sigma)
            if allow_primitives:
                prim = fit_circle_loop(S, tol) or fit_ellipse_loop(S, tol)
                if prim:
                    chain.segs = [prim]
                    return
            runs = straight_runs(S, run_tol(tol), run_len(s))
            if runs:
                # start the loop in the middle of the longest straight run, on its fitted line,
                # so the pinned start point never sits where a fillet or corner is needed
                a, b = max(runs, key=lambda r: r[1] - r[0])
                c, d, _ = _tls(S[a:b + 1])
                r0 = (a + b) // 2
                S = S.copy()
                S[r0] = c + ((S[r0] - c) @ d) * d
                Q = np.vstack([S[r0:], S[:r0 + 1]])
                chain.segs = reduce_anchors(fit_smooth_piece(Q, tol, s), tol * 0.6, True)
                chain.pts = np.vstack([P[r0:], P[:r0]])
                return
            n = len(S)
            i1 = n // 2
            Q = np.vstack([S, S[:1]])
            first = Q[: i1 + 1]
            second = Q[i1:]
            tA = _unit(S[1] - S[-1])
            tB = _unit(S[i1 + 1] - S[i1 - 1])
            chain.segs = reduce_anchors(fit_piece(first, tol, tA, -tB) + fit_piece(second, tol, tB, -tA),
                                        tol * 0.6, True)
            return
        c0 = corners[0]
        n = len(P)
        Pr = np.vstack([P[c0:], P[:c0]])
        cs = [(c - c0) % n for c in corners] + [n]
        Pc = np.vstack([Pr, Pr[:1]])
        pieces = [Pc[a:b + 1] for a, b in zip(cs[:-1], cs[1:])]
        chain.segs = reduce_anchors(_fit_cornered(pieces, True, s, tol, sigma), tol * 0.6, True)
        chain.pts = Pr
        return
    corners = prune_corners(P, detect_corners(P, False, s, corner_angle), False, s)
    cuts = [0] + [c for c in corners if 0 < c < len(P) - 1] + [len(P) - 1]
    pieces = [P[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:])]
    chain.segs = reduce_anchors(_fit_cornered(pieces, False, s, tol, sigma), tol * 0.6, False)


def reverse_segs(segs):
    out = []
    for sg in reversed(segs):
        t = sg[0]
        if t == "L":
            out.append(("L", sg[2], sg[1]))
        elif t == "C":
            out.append(("C", sg[4], sg[3], sg[2], sg[1]))
        elif t == "A":
            _, p0, p1, c, R, a0, sw = sg
            out.append(("A", p1, p0, c, R, a0 + sw, -sw))
        elif t == "CIRCLE":
            out.append(("CIRCLE", sg[1], sg[2], -sg[3]))
        elif t == "ELLIPSE":
            out.append(("ELLIPSE", sg[1], sg[2], sg[3], sg[4], -sg[5]))
    return out


def seg_start(sg):
    t = sg[0]
    if t in ("L", "C", "A"):
        return sg[1]
    if t == "CIRCLE":
        return sg[1] + np.array([sg[2], 0.0])
    _, c, a, b, th, _d = sg
    return c + a * np.array([math.cos(th), math.sin(th)])


def seg_end(sg):
    t = sg[0]
    if t == "L":
        return sg[2]
    if t == "C":
        return sg[4]
    if t == "A":
        return sg[2]
    return seg_start(sg)


def arc_to_beziers(c, R, a0, sweep, rx=None, ry=None, theta=0.0):
    """Approximate a circular (or elliptical with rx,ry,theta) arc by cubic Beziers (<=90deg each)."""
    rx = R if rx is None else rx
    ry = R if ry is None else ry
    n = max(1, int(math.ceil(abs(sweep) / (math.pi / 2) - 1e-9)))
    d = sweep / n
    k = 4.0 / 3.0 * math.tan(d / 4.0)
    ct, st = math.cos(theta), math.sin(theta)

    def pt(a):
        x, y = rx * math.cos(a), ry * math.sin(a)
        return np.array([c[0] + ct * x - st * y, c[1] + st * x + ct * y])

    def der(a):
        x, y = -rx * math.sin(a), ry * math.cos(a)
        return np.array([ct * x - st * y, st * x + ct * y])

    out = []
    a = a0
    for _ in range(n):
        b = a + d
        p0, p3 = pt(a), pt(b)
        out.append((p0, p0 + k * der(a), p3 - k * der(b), p3))
        a = b
    return out
