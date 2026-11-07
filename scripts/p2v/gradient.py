"""Gradient detection and fitting (general: linear at any angle, radial at any centre, any number of stops).

How it works
  1. Quantisation slices a smooth gradient into colour bands. A band boundary differs from a real
     edge: across a real edge the colour jumps by (about) the full palette difference within a few
     pixels; across a band boundary the colour changes only by the local slope. For every pair of
     touching regions we sample the source on both sides of the boundary at a distance beyond the
     edge blur and compare that change with the palette difference. Low ratio = soft boundary.
  2. Regions joined by soft boundaries are merged (union-find) into one shape.
  3. For each merged shape a gradient model is fitted on its source pixels:
       linear  — angle search (coarse + refine); colour profile along the axis, binned
       radial  — centre search (coarse grid + refine); colour profile along the radius
     The model with the lowest residual wins if it explains the colour variation clearly better than
     a flat fill. Colour stops are extracted from the binned profile with a recursive simplification
     (a stop is added only where linear interpolation would miss the profile by more than tolerance).
  4. Separate shapes whose gradients look alike (same family, same direction) are tested for one
     shared gradient spanning all of them; it is kept when it fits the union as well as the parts.
Colours are fitted in encoded sRGB, the space SVG/PDF interpolate in by default.
"""
import math
import numpy as np
import cv2


class Gradient:
    __slots__ = ("kind", "p0", "p1", "center", "radius", "stops", "mean", "res")

    def __init__(self, kind, stops, mean, res, p0=None, p1=None, center=None, radius=None):
        self.kind = kind          # "linear" | "radial"
        self.p0, self.p1 = p0, p1  # linear axis endpoints (working px)
        self.center, self.radius = center, radius
        self.stops = stops        # list of (offset 0..1, (r,g,b) 0..1)
        self.mean = mean
        self.res = res

    def eval(self, xy):
        if self.kind == "linear":
            v = self.p1 - self.p0
            t = ((xy - self.p0) @ v) / max(float(v @ v), 1e-9)
        else:
            t = np.hypot(*(xy - self.center).T) / max(self.radius, 1e-9)
        t = np.clip(t, 0, 1)
        offs = np.array([o for o, _ in self.stops])
        cols = np.array([c for _, c in self.stops])
        return np.stack([np.interp(t, offs, cols[:, c]) for c in range(3)], 1)

    def describe(self, s=1):
        d = {"type": self.kind, "stops": len(self.stops)}
        if self.kind == "linear":
            v = self.p1 - self.p0
            d["angle_deg"] = round(math.degrees(math.atan2(v[1], v[0])) % 360, 1)
        else:
            d["center"] = [round(float(self.center[0] / s), 1), round(float(self.center[1] / s), 1)]
            d["radius"] = round(float(self.radius / s), 1)
        return d


# ------------------------------------------------------------------ 1-2: soft boundaries & merging
def soft_groups(comp, clab, pal_lab, lab, r, ratio_thr=0.7, min_pairs=6, pde_max=32.0):
    """Return group id per component (union of components joined by soft boundaries)."""
    H, W = comp.shape
    n = int(comp.max()) + 1
    keys, ratios = [], []
    for axis in (1, 0):
        if axis == 1:
            a, b = comp[:, :-1], comp[:, 1:]
        else:
            a, b = comp[:-1, :], comp[1:, :]
        ys, xs = np.nonzero(a != b)
        if len(ys) == 0:
            continue
        ca, cb = a[ys, xs], b[ys, xs]
        ok = (clab[ca] >= 0) & (clab[cb] >= 0)
        ys, xs, ca, cb = ys[ok], xs[ok], ca[ok], cb[ok]
        def probe(d):
            """Samples d px away on each side, but only where they still lie in their own region."""
            if axis == 1:
                p1 = (ys, np.clip(xs - d, 0, W - 1))
                p2 = (ys, np.clip(xs + 1 + d, 0, W - 1))
            else:
                p1 = (np.clip(ys - d, 0, H - 1), xs)
                p2 = (np.clip(ys + 1 + d, 0, H - 1), xs)
            ok = (comp[p1] == ca) & (comp[p2] == cb)
            return np.linalg.norm(lab[p1] - lab[p2], axis=1), ok

        across = np.zeros(len(ys))
        valid = np.zeros(len(ys), bool)
        for d in sorted({r, max(1, r // 2), 1}, reverse=True):
            v, ok = probe(d)
            take = ok & ~valid
            across[take] = v[take]
            valid |= ok
        # second probe twice as far: a real edge saturates, a gradient keeps growing
        far, okf = probe(2 * r)
        pde = np.linalg.norm(pal_lab[clab[ca]] - pal_lab[clab[cb]], axis=1)
        ratio = across / np.maximum(pde, 1.0)
        grow = np.where(okf, far / np.maximum(across, 1.0), 0.0)
        # neighbouring bands of one gradient are close in colour; a soft shadow edge between
        # unrelated colours (gold on black) must not fuse them into one fill
        soft = valid & (pde < pde_max) & ((ratio < ratio_thr) | ((grow > 1.5) & (ratio < 1.2)))
        ratios.append(soft.astype(np.float32))
        lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
        keys.append(lo.astype(np.int64) * n + hi)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if keys:
        k = np.concatenate(keys)
        rt = np.concatenate(ratios)
        order = np.argsort(k, kind="stable")
        k, rt = k[order], rt[order]
        cut = np.flatnonzero(np.diff(k)) + 1
        for seg_k, seg_r in zip(np.split(k, cut), np.split(rt, cut)):
            if len(seg_r) < min_pairs:
                continue
            if float(np.mean(seg_r)) > 0.6:  # most of the shared boundary is soft
                i, j = int(seg_k[0] // n), int(seg_k[0] % n)
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def absorb_fringes(comp, clab, carea, rgb, rad, blend_tol=0.06):
    """Thin regions whose pixels are blends of their neighbours' colours (anti-aliasing between a
    gradient fill and its surroundings) are absorbed into the neighbour they share most boundary with.
    Thin regions of a genuinely distinct colour (hairlines) are kept."""
    H, W = comp.shape
    n = int(comp.max()) + 1
    flat = comp.ravel()
    cnt = np.bincount(flat, minlength=n).astype(float)
    mean = np.stack([np.bincount(flat, weights=rgb[..., c].ravel(), minlength=n) for c in range(3)], 1)
    mean /= np.maximum(cnt, 1)[:, None]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    order = np.argsort(comp, axis=None, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(n + 1))
    ys_all, xs_all = np.unravel_index(order, comp.shape)
    comp = comp.copy()
    for c in np.argsort(cnt):
        if clab[c] < 0 or cnt[c] == 0:
            continue
        ys, xs = ys_all[bounds[c]:bounds[c + 1]], xs_all[bounds[c]:bounds[c + 1]]
        if len(ys) == 0:
            continue
        y0, y1 = max(0, ys.min() - rad - 1), min(H, ys.max() + rad + 2)
        x0, x1 = max(0, xs.min() - rad - 1), min(W, xs.max() + rad + 2)
        sub = comp[y0:y1, x0:x1]
        m = (sub == c)
        if not m.any():
            continue
        inner = int(cv2.erode(m.astype(np.uint8), ker).sum())
        if inner > 0.15 * m.sum():
            continue
        ring = cv2.dilate(m.astype(np.uint8), cross).astype(bool) & ~m
        nb = sub[ring]
        nb = nb[nb >= 0]
        if nb.size == 0:
            continue
        vals, k = np.unique(nb, return_counts=True)
        vals = [v for v in vals if clab[v] >= 0]
        if not vals:
            continue
        px = rgb[y0:y1, x0:x1][m]
        if len(px) > 3000:
            px = px[np.random.default_rng(0).choice(len(px), 3000, replace=False)]
        # blend test against pairs of neighbour colours (and single neighbours)
        best = np.full(len(px), 9.0)
        cols = [mean[v] for v in vals]
        for i in range(len(cols)):
            best = np.minimum(best, np.linalg.norm(px - cols[i], axis=1))
            for j in range(i + 1, len(cols)):
                A, B = cols[i], cols[j]
                ab = B - A
                t = np.clip(((px - A) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
                best = np.minimum(best, np.linalg.norm(px - (A + t[:, None] * ab), axis=1))
        if float((best < blend_tol).mean()) < 0.7:
            continue
        # mark as fringe; pixels are re-assigned individually below (watershed on the source),
        # so the boundary follows the real edge instead of jumping sliver by sliver
        sub[m] = -3
        cnt[c] = 0
    fr = comp == -3
    if fr.any():
        markers = np.where(fr, 0, comp + 1).astype(np.int32)
        img = np.ascontiguousarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
        cv2.watershed(img, markers)
        # watershed lines (-1) and any leftovers take a neighbouring basin
        k3 = np.ones((3, 3), np.uint8)
        for _ in range(4):
            hole = markers <= 0
            if not hole.any():
                break
            dil = cv2.dilate(np.where(hole, 0, markers).astype(np.float32), k3).astype(np.int32)
            markers = np.where(hole, dil, markers)
        comp = np.where(fr | (markers <= 0) & fr, markers - 1, comp)
        comp[comp < 0] = 0
    # compact ids
    used = np.unique(comp)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    comp2 = remap[comp].astype(np.int32)
    clab2 = clab[used]
    carea2 = np.bincount(comp2.ravel(), minlength=len(used)).astype(np.int64)
    return comp2, clab2, carea2


# ------------------------------------------------------------------ 3: model fitting
def _profile(t, cols, nb):
    lo, hi = np.percentile(t, 0.5), np.percentile(t, 99.5)
    if hi - lo < 1e-6:
        return None
    b = np.clip(((t - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
    cnt = np.bincount(b, minlength=nb).astype(float)
    sums = np.stack([np.bincount(b, weights=cols[:, c], minlength=nb) for c in range(3)], 1)
    means = sums / np.maximum(cnt, 1)[:, None]
    res = float(((cols - means[b]) ** 2).sum(1).mean())
    return res, lo, hi, means, cnt


def _linear_search(xy, cols, nb):
    best = (1e9, 0.0)
    for deg in range(0, 180, 4):
        th = math.radians(deg)
        u = np.array([math.cos(th), math.sin(th)])
        p = _profile(xy @ u, cols, nb)
        if p and p[0] < best[0]:
            best = (p[0], deg)
    for step in (1.0, 0.25):
        base = best[1]
        for k in range(-4, 5):
            deg = base + k * step
            th = math.radians(deg)
            u = np.array([math.cos(th), math.sin(th)])
            p = _profile(xy @ u, cols, nb)
            if p and p[0] < best[0]:
                best = (p[0], deg)
    th = math.radians(best[1])
    return best[0], np.array([math.cos(th), math.sin(th)])


def _radial_search(xy, cols, nb):
    lo, hi = xy.min(0), xy.max(0)
    ext = hi - lo
    lo2, hi2 = lo - 0.25 * ext, hi + 0.25 * ext
    best = (1e9, None)
    span = hi2 - lo2
    cx_grid = np.linspace(lo2[0], hi2[0], 9)
    cy_grid = np.linspace(lo2[1], hi2[1], 9)
    for _ in range(4):
        for cx in cx_grid:
            for cy in cy_grid:
                c = np.array([cx, cy])
                p = _profile(np.hypot(*(xy - c).T), cols, nb)
                if p and p[0] < best[0]:
                    best = (p[0], c)
        span = span / 3.0
        c = best[1]
        cx_grid = np.linspace(c[0] - span[0], c[0] + span[0], 7)
        cy_grid = np.linspace(c[1] - span[1], c[1] + span[1], 7)
    return best


def _stops(means, cnt, tol):
    """Recursive simplification of the binned profile into colour stops."""
    idx = np.nonzero(cnt > 0)[0]
    if len(idx) == 0:
        return []
    nb = len(cnt)
    t = (idx + 0.5) / nb
    C = means[idx]
    keep = {0, len(idx) - 1}

    def rec(i, j):
        if j - i < 2:
            return
        w = (t[i + 1:j] - t[i]) / max(t[j] - t[i], 1e-9)
        interp = C[i] + w[:, None] * (C[j] - C[i])
        err = np.abs(C[i + 1:j] - interp).max(1)
        k = int(err.argmax())
        if err[k] > tol:
            m = i + 1 + k
            keep.add(m)
            rec(i, m)
            rec(m, j)

    rec(0, len(idx) - 1)
    ks = sorted(keep)
    stops = [(float(t[k]), tuple(float(v) for v in np.clip(C[k], 0, 1))) for k in ks]
    # pin ends to 0 and 1 so the gradient covers the full axis
    stops[0] = (0.0, stops[0][1])
    stops[-1] = (1.0, stops[-1][1])
    return stops


def fit_gradient(xy, cols, s, stop_tol=0.018, min_std=0.025, gain=0.45, kinds=("linear", "radial")):
    """xy: (N,2) working-px coordinates, cols: (N,3) sRGB 0..1. Returns Gradient or None (flat)."""
    if len(xy) < 50:
        return None
    if len(xy) > 25000:
        sel = np.random.default_rng(0).choice(len(xy), 25000, replace=False)
        xy, cols = xy[sel], cols[sel]
    mean = cols.mean(0)
    flat = float(((cols - mean) ** 2).sum(1).mean())
    if math.sqrt(flat / 3) < min_std:
        return None
    ext = float(np.hypot(*(xy.max(0) - xy.min(0))))
    nb = int(np.clip(ext / (2.0 * s), 16, 256))
    lin_res, u = _linear_search(xy, cols, nb) if "linear" in kinds else (1e9, None)
    rad_res, c = _radial_search(xy, cols, nb) if "radial" in kinds else (1e9, None)
    if rad_res < 0.8 * lin_res:
        kind, res = "radial", rad_res
        t = np.hypot(*(xy - c).T)
    else:
        kind, res = "linear", lin_res
        t = xy @ u
    if res > gain * flat:
        return None
    _, lo, hi, means, cnt = _profile(t, cols, nb)
    stops = _stops(means, cnt, stop_tol)
    if len(stops) < 2:
        return None
    if kind == "linear":
        cen = xy.mean(0)
        base = cen - u * (cen @ u)
        return Gradient("linear", stops, mean, res, p0=base + u * lo, p1=base + u * hi)
    # radial: offsets measured from the centre (t from 0), remap lo..hi to lo/hi..1
    R = hi
    stops = [(lo / R + o * (hi - lo) / R, col) for o, col in stops]
    if stops[0][0] > 1e-6:
        stops = [(0.0, stops[0][1])] + stops
    return Gradient("radial", stops, mean, res, center=c, radius=R)


# ------------------------------------------------------------------ driver
def detect(comp, clab, carea, palette, pal_lab, lab, rgb, s, band):
    """Merge soft-bounded regions and fit gradients.

    Returns (comp2, clab2, carea2, palette2, gradients{label: Gradient}).
    Merged groups whose colour turns out flat keep a flat fill (their mean colour).
    """
    r = int(max(2 * s, round(band)))
    comp, clab, carea = absorb_fringes(comp, clab, carea, rgb, max(1, int(round(max(0.8 * s, 0.6 * band)))))
    gid = soft_groups(comp, clab, pal_lab, lab, r)
    uniq, inv = np.unique(gid, return_inverse=True)
    comp2 = inv[comp].astype(np.int32)
    n2 = len(uniq)
    members = [[] for _ in range(n2)]
    for c, g in enumerate(inv):
        members[g].append(c)
    clab2 = np.empty(n2, np.int32)
    carea2 = np.zeros(n2, np.int64)
    pal = [tuple(p) for p in palette]
    grads = {}
    order = np.argsort(comp2, axis=None, kind="stable")
    bounds = np.searchsorted(comp2.ravel()[order], np.arange(n2 + 1))
    ys_all, xs_all = np.unravel_index(order, comp2.shape)
    min_single = int(400 * s * s)
    ero = int(max(2, round(band))) + 1  # fit on interiors: edge pixels are blends with the outside  # a lone region is only tested for an internal gradient if sizeable
    for g in range(n2):
        ms = members[g]
        carea2[g] = int(sum(int(carea[m]) for m in ms))
        if clab[ms[0]] < 0 or (len(ms) == 1 and carea2[g] < min_single):
            clab2[g] = clab[ms[0]]
            continue
        a, b = bounds[g], bounds[g + 1]
        ys, xs = ys_all[a:b], xs_all[a:b]
        ys, xs = _interior(comp2, g, ys, xs, ero)
        xy = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float64)
        cols = rgb[ys, xs].astype(np.float64)
        gr = fit_gradient(xy, cols, s)
        if gr is None and len(ms) == 1:
            clab2[g] = clab[ms[0]]
            continue
        lab_new = len(pal)
        pal.append(tuple(cols.mean(0)) if gr is None else tuple(gr.mean))
        clab2[g] = lab_new
        if gr is not None:
            grads[lab_new] = (gr, xy, cols)
    grads = _share(grads, s)
    comp2, clab2, carea2 = _absorb_orphans(comp2, clab2, carea2, grads, rgb, ys_all, xs_all, bounds)
    comp2, clab2, carea2 = _fuse_same_gradient(comp2, clab2, grads)
    if grads:
        comp2, clab2, carea2 = refine_edges(comp2, clab2, grads, rgb, ero)
    return comp2, clab2, carea2, np.array(pal, np.float32), grads


def refine_edges(comp, clab, grads, rgb, d):
    """Re-place the outline of every gradient shape at the 50 % coverage line.

    Band-wise quantisation puts the edge of a gradient shape at a different threshold in every band
    (a bright band meets black at another point than a dark band does), which shows as steps along
    the outline. Near each edge between a gradient shape G and a flat shape F, every pixel is
    re-classified with a two-colour coverage model: the gradient's own predicted colour at that
    position versus F's colour. The pixel goes to G when it is at least half G. The outline then
    follows the real edge continuously. Thin shapes keep their core."""
    H, W = comp.shape
    n = len(clab)
    is_grad = np.array([clab[i] in grads for i in range(n)])
    # pairs (gradient comp, flat comp) that touch, and the edge pixels per pair
    pair_px = {}
    for a, b, sa, sb in ((comp[:, :-1], comp[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                         (comp[:-1, :], comp[1:, :], (slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        m = (a != b) & (is_grad[a] ^ is_grad[b]) & (clab[a] >= 0) & (clab[b] >= 0)
        ys, xs = np.nonzero(m)
        ga, gb = a[ys, xs], b[ys, xs]
        G = np.where(is_grad[ga], ga, gb)
        F = np.where(is_grad[ga], gb, ga)
        for gg, ff, y, x in zip(G.tolist(), F.tolist(), ys.tolist(), xs.tolist()):
            pair_px.setdefault((gg, ff), []).append((y, x))
    if not pair_px:
        return comp, clab, np.bincount(comp.ravel(), minlength=n).astype(np.int64)
    bnd = np.zeros((H, W), bool)
    bnd[:, :-1] |= comp[:, :-1] != comp[:, 1:]
    bnd[:, 1:] |= comp[:, :-1] != comp[:, 1:]
    bnd[:-1, :] |= comp[:-1, :] != comp[1:, :]
    bnd[1:, :] |= comp[:-1, :] != comp[1:, :]
    dist = cv2.distanceTransform((~bnd).astype(np.uint8), cv2.DIST_L2, 3)
    thick = np.zeros(n, np.float32)
    np.maximum.at(thick, comp.ravel(), dist.ravel())
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    flat_col = {}
    out = comp.copy()
    for (gg, ff), pts in pair_px.items():
        pts = np.array(pts)
        y0, x0 = max(0, pts[:, 0].min() - d - 2), max(0, pts[:, 1].min() - d - 2)
        y1, x1 = min(H, pts[:, 0].max() + d + 3), min(W, pts[:, 1].max() + d + 3)
        e = np.zeros((y1 - y0, x1 - x0), np.uint8)
        e[pts[:, 0] - y0, pts[:, 1] - x0] = 1
        near = cv2.dilate(e, ker) > 0
        sc = comp[y0:y1, x0:x1]
        sel = near & ((sc == gg) | (sc == ff)) & (dist[y0:y1, x0:x1] < np.minimum(d, 0.5 * thick[sc]))
        ys, xs = np.nonzero(sel)
        if len(ys) == 0:
            continue
        ys, xs = ys + y0, xs + x0
        if ff not in flat_col:
            fm = (comp == ff) & (dist >= d)
            flat_col[ff] = rgb[fm].mean(0) if fm.any() else rgb[comp == ff].mean(0)
        Fc = flat_col[ff][None, :]
        xy = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float64)
        Gc = grads[clab[gg]].eval(xy)
        c = rgb[ys, xs].astype(np.float64)
        ab = Gc - Fc
        t = ((c - Fc) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9)
        out[ys, xs] = np.where(t >= 0.5, gg, ff)
    return out, clab, np.bincount(out.ravel(), minlength=n).astype(np.int64)


def _fuse_same_gradient(comp, clab, grads):
    """Touching shapes painted with the very same gradient become one shape (no seam between them)."""
    n = len(clab)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in ((comp[:, :-1], comp[:, 1:]), (comp[:-1, :], comp[1:, :])):
        m = a != b
        for x, y in set(zip(a[m].tolist(), b[m].tolist())):
            gx, gy = grads.get(clab[x]), grads.get(clab[y])
            if gx is not None and gx is gy:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx
    root = np.array([find(i) for i in range(n)])
    used, inv = np.unique(root, return_inverse=True)
    comp2 = inv[comp].astype(np.int32)
    return comp2, clab[used], np.bincount(comp2.ravel(), minlength=len(used)).astype(np.int64)


def _absorb_rims(comp, clab, carea, grads, s, max_width_src=1.3):
    """Hair-thin slivers (mean width under ~1 source pixel) lying along a gradient shape are rim
    noise (JPEG ringing, bevel glints); each would cost a whole extra outline of anchors. They join
    the neighbour they share most boundary with."""
    if not grads:
        return comp, clab, carea
    n = len(clab)
    perim = np.zeros(n)
    pair_cnt = {}
    for a, b in ((comp[:, :-1], comp[:, 1:]), (comp[:-1, :], comp[1:, :])):
        m = a != b
        aa, bb = a[m], b[m]
        perim += np.bincount(aa, minlength=n) + np.bincount(bb, minlength=n)
        keys = np.minimum(aa, bb).astype(np.int64) * n + np.maximum(aa, bb)
        u, c = np.unique(keys, return_counts=True)
        for k, cnt in zip(u.tolist(), c.tolist()):
            pair_cnt[k] = pair_cnt.get(k, 0) + cnt
    nb = [dict() for _ in range(n)]
    for k, cnt in pair_cnt.items():
        x, y = divmod(k, n)
        nb[x][y] = cnt
        nb[y][x] = cnt
    target = np.arange(n)
    for o in range(n):
        if clab[o] < 0 or clab[o] in grads:
            continue
        width = 2.0 * carea[o] / max(perim[o], 1) / s
        if width >= max_width_src:
            continue
        if not any(clab[q] in grads for q in nb[o]):
            continue
        cands = [(cnt, q) for q, cnt in nb[o].items() if clab[q] >= 0]
        if cands:
            target[o] = max(cands)[1]
    for i in range(n):
        j, hops = i, 0
        while target[j] != j and hops < n:
            j = target[j]
            hops += 1
        target[i] = j
    used, inv = np.unique(target, return_inverse=True)
    comp2 = inv[comp].astype(np.int32)
    return comp2, clab[used], np.bincount(comp2.ravel(), minlength=len(used)).astype(np.int64)


def _absorb_orphans(comp, clab, carea, grads, rgb, ys_all, xs_all, bounds, tol=0.05, max_share=2.0,
                    thin_w=6.0):
    """Small leftover regions inside or along a gradient shape (bits the band merge missed) join the
    gradient shape when the gradient itself predicts their colour."""
    if not grads:
        return comp, clab, carea
    n = len(clab)
    pairs = set()
    perim = np.zeros(n)
    for a, b in ((comp[:, :-1], comp[:, 1:]), (comp[:-1, :], comp[1:, :])):
        m = a != b
        perim += np.bincount(a[m], minlength=n) + np.bincount(b[m], minlength=n)
        for x, y in zip(a[m].tolist(), b[m].tolist()):
            pairs.add((x, y))
            pairs.add((y, x))
    flat_mean = {}
    nbrs = [[] for _ in range(n)]
    for x, y in pairs:
        nbrs[x].append(y)
    target = np.arange(n)
    for o in np.argsort(carea):
        if clab[o] < 0 or clab[o] in grads:
            continue
        cand = [g for g in nbrs[o] if clab[g] in grads and carea[o] <= max_share * carea[g]]
        if not cand:
            continue
        ys, xs = ys_all[bounds[o]:bounds[o + 1]], xs_all[bounds[o]:bounds[o + 1]]
        xy = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float64)
        col = rgb[ys, xs].astype(np.float64)
        # thin slivers (mean width = 2*area/perimeter) may be anti-aliased blends of the
        # gradient and the shape next to it
        thin = 2.0 * carea[o] / max(perim[o], 1) < thin_w
        others = []
        if thin:
            for q in nbrs[o]:
                if clab[q] >= 0 and clab[q] not in grads:
                    if q not in flat_mean:
                        qy, qx = ys_all[bounds[q]:bounds[q + 1]], xs_all[bounds[q]:bounds[q + 1]]
                        flat_mean[q] = rgb[qy, qx].reshape(-1, 3).mean(0)
                    others.append(flat_mean[q])
        best = None
        for g in cand:
            pred = grads[clab[g]].eval(xy)
            d = np.abs(pred - col).mean(1)
            for oc in others:
                ab = oc[None, :] - pred
                t = np.clip(((col - pred) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
                d = np.minimum(d, np.abs(pred + t[:, None] * ab - col).mean(1))
            err = float(d.mean())
            lim = tol * 1.6 if thin else tol
            if err < lim and (best is None or err < best[0]):
                best = (err, g)
        if best:
            target[o] = best[1]
            carea[best[1]] += carea[o]
    # resolve chains of reassignment and compact
    for i in range(n):
        j = i
        while target[j] != j:
            j = target[j]
        target[i] = j
    used, inv = np.unique(target, return_inverse=True)
    comp2 = inv[comp].astype(np.int32)
    return comp2, clab[used], np.bincount(comp2.ravel(), minlength=len(used)).astype(np.int64)


def _interior(comp, g, ys, xs, e):
    H, W = comp.shape
    y0, y1 = max(0, ys.min() - 1), min(H, ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(W, xs.max() + 2)
    m = (comp[y0:y1, x0:x1] == g).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
    er = cv2.erode(m, k)
    if er.sum() < 0.3 * m.sum():
        return ys, xs
    iy, ix = np.nonzero(er)
    return iy + y0, ix + x0


def _share(grads, s):
    """Grow shared gradients greedily: a shape joins a group when one gradient still fits the union
    about as well as the separate gradients did. Largest shapes seed first."""
    items = sorted(grads.items(), key=lambda kv: -len(kv[1][1]))
    out = {k: v[0] for k, v in items}
    used = set()
    for ki, (gi, xyi, ci) in items:
        if ki in used:
            continue
        group, joint = [ki], None
        for kj, (gj, xyj, cj) in items:
            if kj in used or kj in group or gi.kind != gj.kind:
                continue
            if gi.kind == "radial" and np.hypot(*(gi.center - gj.center)) > 0.15 * max(gi.radius, gj.radius):
                continue
            if gi.kind == "linear":
                ui = _unit(gi.p1 - gi.p0)
                uj = _unit(gj.p1 - gj.p0)
                if abs(float(ui @ uj)) < math.cos(math.radians(8)):
                    continue
            trial = group + [kj]
            xy = np.vstack([grads[k][1] for k in trial])
            cols = np.vstack([grads[k][2] for k in trial])
            cand = fit_gradient(xy, cols, s, kinds=(gi.kind,))
            parts = sum(grads[k][0].res * len(grads[k][1]) for k in trial) / len(xy)
            if cand is not None and cand.res <= 1.6 * parts + 2e-4:
                group, joint = trial, cand
        if joint is not None:
            for k in group:
                out[k] = joint
                used.add(k)
    return out


def _unit(v):
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def merge_fragments(comp, clab, carea, grads, rgb, lab, s, amin_src=80.0, de_max=20.0,
                    rim_width_src=1.3, de_rim=45.0):
    """Minimum feature size for gradient artwork: a fragment smaller than amin_src source px whose
    colour is close (dE < de_max) to a neighbour joins that neighbour. Small details that contrast
    with everything around them (a dot, a registered mark) stay."""
    from .imgio import rgb_to_lab
    n = len(clab)
    amin = amin_src * s * s
    pairs = {}
    for a, b in ((comp[:, :-1], comp[:, 1:]), (comp[:-1, :], comp[1:, :])):
        m = a != b
        keys = np.minimum(a[m], b[m]).astype(np.int64) * n + np.maximum(a[m], b[m])
        u, c = np.unique(keys, return_counts=True)
        for k, cnt in zip(u.tolist(), c.tolist()):
            pairs[k] = pairs.get(k, 0) + cnt
    nb = [dict() for _ in range(n)]
    for k, cnt in pairs.items():
        x, y = divmod(k, n)
        nb[x][y] = cnt
        nb[y][x] = cnt
    order = np.argsort(comp, axis=None, kind="stable")
    bounds = np.searchsorted(comp.ravel()[order], np.arange(n + 1))
    ys_all, xs_all = np.unravel_index(order, comp.shape)
    mean_lab = np.zeros((n, 3))
    for i in range(n):
        ys, xs = ys_all[bounds[i]:bounds[i + 1]], xs_all[bounds[i]:bounds[i + 1]]
        if len(ys):
            mean_lab[i] = lab[ys, xs].mean(0)
    perim = np.zeros(n)
    for k, cnt in pairs.items():
        x, y = divmod(k, n)
        perim[x] += cnt
        perim[y] += cnt
    target = np.arange(n)
    for o in np.argsort(carea):
        if clab[o] < 0 or clab[o] in grads:
            continue
        # hair-thin rims along a gradient shape (JPEG ringing, bevel glints) count as fragments
        # whatever their length: mean width = 2 * area / perimeter
        width_src = 2.0 * carea[o] / max(perim[o], 1) / s
        rim = width_src < rim_width_src and any(clab[q] in grads for q in nb[o])
        if carea[o] >= amin and not rim:
            continue
        lim = de_rim if rim else de_max
        ys, xs = ys_all[bounds[o]:bounds[o + 1]], xs_all[bounds[o]:bounds[o + 1]]
        if len(ys) == 0:
            continue
        best = None
        for q, cnt in nb[o].items():
            q = int(target[q])
            if q == o or clab[q] < 0:
                continue
            if clab[q] in grads:
                xy = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float64)
                ref = rgb_to_lab(grads[clab[q]].eval(xy).astype(np.float32).reshape(-1, 1, 3)).reshape(-1, 3).mean(0)
            else:
                ref = mean_lab[q]
            de = float(np.linalg.norm(mean_lab[o] - ref))
            if de < lim and (best is None or (de, -cnt) < best[:2]):
                best = (de, -cnt, q)
        if best is not None:
            target[o] = best[2]
    for i in range(n):
        j, hops = i, 0
        while target[j] != j and hops < n:
            j = target[j]
            hops += 1
        target[i] = j
    used, inv = np.unique(target, return_inverse=True)
    comp2 = inv[comp].astype(np.int32)
    return comp2, clab[used], np.bincount(comp2.ravel(), minlength=len(used)).astype(np.int64)
