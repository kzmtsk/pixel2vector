"""Stage 1: working-resolution image -> palette label map -> clean 4-connected regions."""
import numpy as np
import cv2
from .imgio import composite, rgb_to_lab, lab_to_rgb

TRANSPARENT = -2  # palette index reserved for transparent pixels


def working_scale(h, w, category, max_scale=4):
    budget = {"logo": 8.0e6, "lineart": 8.0e6, "illustration": 6.0e6, "photo": 2.0e6}[category]
    s = int(np.floor(np.sqrt(budget / float(h * w))))
    return int(max(1, min(max_scale, s)))


def upscale(rgba, s):
    if s == 1:
        return rgba.copy()
    h, w = rgba.shape[:2]
    # premultiply so transparent fringes do not bleed dark colour
    pm = rgba.copy()
    pm[..., :3] *= pm[..., 3:4]
    up = cv2.resize(pm, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
    up = np.clip(up, 0, 1)
    a = up[..., 3:4]
    rgb = np.where(a > 1e-4, up[..., :3] / np.maximum(a, 1e-4), 0)
    return np.concatenate([np.clip(rgb, 0, 1), a], axis=2).astype(np.float32)


def _kmeans(pts, k, seed=0):
    k = max(1, min(k, len(pts)))
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05)
    _, _, cen = cv2.kmeans(pts.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return cen


def _merge_close(cen, pts, de=5.0):
    """Merge k-means centres closer than `de` (Lab): k was an upper bound, not a promise."""
    cen = [c.astype(np.float64) for c in cen]
    lbl = _assign(pts, np.array(cen, np.float32))
    w = list(np.bincount(lbl, minlength=len(cen)).astype(float) + 1e-6)
    while len(cen) > 1:
        best = (1e9, -1, -1)
        for i in range(len(cen)):
            for j in range(i + 1, len(cen)):
                dd = float(np.linalg.norm(cen[i] - cen[j]))
                if dd < best[0]:
                    best = (dd, i, j)
        if best[0] >= de:
            break
        _, i, j = best
        cen[i] = (cen[i] * w[i] + cen[j] * w[j]) / (w[i] + w[j])
        w[i] += w[j]
        del cen[j], w[j]
    return np.array(cen, np.float32)


def _assign(lab_flat, cen, chunk=1 << 20):
    out = np.empty(len(lab_flat), np.int32)
    for i in range(0, len(lab_flat), chunk):
        x = lab_flat[i:i + chunk]
        d = ((x[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def quantize(rgba_w, category, n_colors, s=1, opaque_mask=None):
    """Return (labels int32 HxW with TRANSPARENT for clear pixels, palette RGB float (K,3))."""
    H, W = rgba_w.shape[:2]
    alpha = rgba_w[..., 3]
    opaque = alpha >= 0.5 if opaque_mask is None else opaque_mask
    # straight (un-premultiplied) colour: semi-transparent edge pixels keep their true hue
    # instead of turning into a lighter fringe colour
    rgb = np.where(opaque[..., None], rgba_w[..., :3], composite(rgba_w)).astype(np.float32)
    if category == "photo":
        u8 = (rgb * 255).astype(np.uint8)
        u8 = cv2.pyrMeanShiftFiltering(u8, sp=max(4, int(round(min(H, W) / 160))), sr=18)
        rgb = u8.astype(np.float32) / 255.0
    elif category == "illustration":
        u8 = (rgb * 255).astype(np.uint8)
        u8 = cv2.bilateralFilter(u8, 5, 20, 5)
        rgb = u8.astype(np.float32) / 255.0
    lab = rgb_to_lab(rgb)

    if category == "lineart":
        L8 = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(L8[opaque] if opaque.any() else L8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        labels = (L8 > thr).astype(np.int32)  # 0 ink, 1 paper
        palette_lab = np.array([
            lab[labels == 0].mean(0) if (labels == 0).any() else [0, 0, 0],
            lab[labels == 1].mean(0) if (labels == 1).any() else [100, 0, 0],
        ], np.float32)
    else:
        pts = lab[opaque]
        if len(pts) == 0:
            pts = lab.reshape(-1, 3)
        if len(pts) > 250000:
            pts = pts[np.random.default_rng(1).choice(len(pts), 250000, replace=False)]
        cen = _merge_close(_kmeans(pts, n_colors), pts)
        labels = _assign(lab.reshape(-1, 3), cen).reshape(H, W)
        palette_lab = cen

    labels = labels.astype(np.int32)
    if category in ("logo", "illustration"):
        labels, palette_lab = suppress_fringes(labels, lab, palette_lab, opaque, s)
    labels[~opaque] = TRANSPARENT
    labels, palette_lab = _compact(labels, palette_lab)
    # refine palette from region interiors (excludes anti-aliased fringes)
    palette = refine_palette(labels, lab, palette_lab)
    return labels, palette


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab))), float(t)


def edge_band_width(lab):
    """Typical width (px) of colour transitions: soft/blurred sources have wide bands."""
    L = lab[..., 0].astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    hi = float(np.percentile(g, 99.5))
    if hi < 1e-3:
        return 1.0
    band = float((g > 0.12 * hi).sum())
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    edges = float((cv2.Canny(L8, 20, 60) > 0).sum())
    if edges < 1:
        return 1.0
    return max(1.0, min(40.0, band / edges))


def suppress_fringes(labels, lab, pal, opaque, s=None, mix_rgb=0.07):
    """Remove anti-aliasing fringe clusters.

    A fringe cluster (a) is thin — almost nothing survives an erosion of ~1 source pixel — and
    (b) its colour lies on the line between two other palette colours (a blend). Its pixels are
    re-assigned to the nearest remaining colour, which places the boundary at the blend midpoint.
    """
    K = len(pal)
    if K <= 2:
        return labels, pal
    r = max(1 if s is None else int(round(0.8 * s)), int(round(0.6 * edge_band_width(lab))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    keep = np.ones(K, bool)
    prgb = lab_to_rgb(pal.reshape(-1, 1, 3).astype(np.float32)).reshape(-1, 3)  # blends are linear in encoded RGB
    sizes = np.bincount(labels[opaque].ravel(), minlength=K)
    order = np.argsort(sizes)  # test small clusters first
    for i in order:
        m = (labels == i) & opaque
        tot = int(m.sum())
        if tot == 0:
            keep[i] = False
            continue
        inner = int(cv2.erode(m.astype(np.uint8), ker).sum())
        if inner > 0.12 * tot:
            continue
        others = [j for j in range(K) if j != i and keep[j]]
        if len(others) < 2:
            continue
        # per-pixel blend test: is each pixel explainable as a mix of two remaining colours?
        px = lab[m]
        if len(px) > 4000:
            px = px[np.random.default_rng(0).choice(len(px), 4000, replace=False)]
        px = lab_to_rgb(px.reshape(-1, 1, 3).astype(np.float32)).reshape(-1, 3)
        best = np.full(len(px), 9.0)
        for a in range(len(others)):
            for b in range(a + 1, len(others)):
                A, B = prgb[others[a]], prgb[others[b]]
                ab = B - A
                t = np.clip(((px - A) @ ab) / max(ab @ ab, 1e-9), 0, 1)
                d = np.linalg.norm(px - (A + t[:, None] * ab), axis=1)
                best = np.minimum(best, d)
        is_mix = float((best < mix_rgb).mean()) > 0.7
        if is_mix:
            keep[i] = False
    if keep.all():
        return labels, pal
    kept = np.nonzero(keep)[0]
    sub = pal[kept]
    m = ~keep[np.clip(labels, 0, K - 1)] & (labels >= 0)
    if m.any():
        px = lab[m]
        d = ((px[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        labels = labels.copy()
        labels[m] = kept[d.argmin(1)]
    return labels, pal


def _compact(labels, pal):
    used = np.unique(labels[labels >= 0])
    remap = np.full(len(pal), -1, np.int32)
    remap[used] = np.arange(len(used))
    out = labels.copy()
    m = labels >= 0
    out[m] = remap[labels[m]]
    return out, pal[used]


def refine_palette(labels, lab, palette_lab):
    out = []
    ker = np.ones((3, 3), np.uint8)
    for i in range(len(palette_lab)):
        m = (labels == i).astype(np.uint8)
        e = cv2.erode(m, ker, iterations=2)
        sel = e.astype(bool) if e.sum() > 20 else m.astype(bool)
        out.append(np.median(lab[sel], axis=0) if sel.any() else palette_lab[i])
    return lab_to_rgb(np.array(out, np.float32).reshape(-1, 1, 3)).reshape(-1, 3)


def components(labels):
    """4-connected component map. Returns (comp int32 HxW, comp_label array, comp_area array)."""
    H, W = labels.shape
    comp = np.full((H, W), -1, np.int32)
    comp_label, comp_area = [], []
    nxt = 0
    for lab in np.unique(labels):
        m = (labels == lab).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        sel = cc > 0
        comp[sel] = cc[sel] - 1 + nxt
        comp_label.extend([int(lab)] * (n - 1))
        comp_area.extend(st[1:, cv2.CC_STAT_AREA].tolist())
        nxt += n - 1
    return comp, np.array(comp_label, np.int32), np.array(comp_area, np.int64)


def despeckle(labels, min_area, passes=3, protect_transparent=True):
    """Merge connected components smaller than min_area into the neighbour sharing most boundary."""
    H, W = labels.shape
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    for _ in range(passes):
        comp, clab, carea = components(labels)
        small = np.nonzero(carea < min_area)[0]
        if len(small) == 0:
            break
        # bounding boxes via one pass
        order = small[np.argsort(carea[small])]
        ys, xs = np.nonzero(np.isin(comp, small))
        cid = comp[ys, xs]
        bb = {}
        for c in order:
            bb[c] = [H, W, -1, -1]
        # vectorised bbox
        srt = np.argsort(cid)
        cid_s, ys_s, xs_s = cid[srt], ys[srt], xs[srt]
        bounds = np.flatnonzero(np.diff(cid_s)) + 1
        starts = np.r_[0, bounds]
        ends = np.r_[bounds, len(cid_s)]
        for a, b in zip(starts, ends):
            c = cid_s[a]
            bb[c] = [ys_s[a:b].min(), xs_s[a:b].min(), ys_s[a:b].max(), xs_s[a:b].max()]
        changed = 0
        for c in order:
            if protect_transparent and clab[c] == TRANSPARENT:
                continue
            y0, x0, y1, x1 = bb[c]
            y0, x0 = max(0, y0 - 1), max(0, x0 - 1)
            y1, x1 = min(H - 1, y1 + 1), min(W - 1, x1 + 1)
            sub_c = comp[y0:y1 + 1, x0:x1 + 1]
            sub_l = labels[y0:y1 + 1, x0:x1 + 1]
            m = (sub_c == c)
            ring = cv2.dilate(m.astype(np.uint8), cross).astype(bool) & ~m
            nb = sub_l[ring]
            nb = nb[nb != clab[c]]
            if protect_transparent:
                nb = nb[nb != TRANSPARENT]
            if nb.size == 0:
                continue
            vals, cnt = np.unique(nb, return_counts=True)
            new = vals[cnt.argmax()]
            sub_l[m] = new
            changed += 1
        if changed == 0:
            break
    return labels
