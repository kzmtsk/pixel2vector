"""Image analysis: classify into one of four categories and suggest parameters.

Categories
  logo          flat-colour logo / icon, few colours, crisp edges
  illustration  many flat colour regions (flat-shaded artwork, NBP-style flat renders)
  lineart       monochrome drawing / stamp / signature / handwriting
  photo         continuous tone; vectorised as a posterized stylisation
"""
import numpy as np
import cv2
from .imgio import composite, rgb_to_lab


def _sample(rgb, max_side=512):
    h, w = rgb.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        rgb = cv2.resize(rgb, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return rgb


def _kmeans_lab(lab_pts, k, attempts=2):
    k = max(1, min(k, len(lab_pts)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, lbl, cen = cv2.kmeans(lab_pts.astype(np.float32), k, None, crit, attempts, cv2.KMEANS_PP_CENTERS)
    return lbl.ravel(), cen


def suggest_palette_size(lab_img, max_k=32, merge_de=9.0, min_share=0.002):
    pts = lab_img.reshape(-1, 3)
    if len(pts) > 60000:
        idx = np.random.default_rng(0).choice(len(pts), 60000, replace=False)
        pts = pts[idx]
    lbl, cen = _kmeans_lab(pts, max_k)
    w = np.bincount(lbl, minlength=len(cen)).astype(float)
    # agglomerative merge of near-identical centres
    cen = [c.astype(float) for c in cen]
    w = list(w)
    merged = True
    while merged and len(cen) > 1:
        merged = False
        best = (1e9, -1, -1)
        for i in range(len(cen)):
            for j in range(i + 1, len(cen)):
                d = np.linalg.norm(cen[i] - cen[j])
                if d < best[0]:
                    best = (d, i, j)
        if best[0] < merge_de:
            _, i, j = best
            tw = w[i] + w[j]
            cen[i] = (cen[i] * w[i] + cen[j] * w[j]) / tw
            w[i] = tw
            del cen[j], w[j]
            merged = True
    w = np.array(w) / max(1.0, sum(w))
    return int(max(2, (w >= min_share).sum()))


def analyze(rgba, has_alpha):
    rgb = composite(rgba)
    H, W = rgb.shape[:2]
    small = _sample(rgb)
    if has_alpha:
        # judge colours on opaque pixels only (transparent areas are not a colour)
        sm_a = _sample(rgba)
        op = sm_a[..., 3] >= 0.5
        if op.any():
            small = np.where(op[..., None], sm_a[..., :3], small)
    lab = rgb_to_lab(small)
    L = lab[..., 0]

    # chroma
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    gray = float(np.percentile(chroma, 95)) < 8.0

    # colour concentration (5-bit histogram)
    q = (np.clip(small, 0, 1) * 31).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    counts = np.sort(np.bincount(key.ravel()))[::-1]
    cum = np.cumsum(counts) / counts.sum()
    n95 = int(np.searchsorted(cum, 0.95) + 1)

    # flatness: share of pixels whose 3x3 neighbourhood is (almost) uniform in Lab
    k = np.ones((3, 3), np.float32) / 9.0
    var = np.zeros(L.shape, np.float32)
    for c in range(3):
        ch = lab[..., c]
        m = cv2.filter2D(ch, -1, k)
        m2 = cv2.filter2D(ch * ch, -1, k)
        var += np.maximum(m2 - m * m, 0)
    std = np.sqrt(var)
    flat = float((std < 2.5).mean())

    # bimodality for lineart
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    thr, bw = cv2.threshold(L8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = L8[L8 <= thr]
    light = L8[L8 > thr]
    bimodal = 0.0
    if dark.size and light.size:
        md, ml = np.median(dark), np.median(light)
        near = (np.abs(L8.astype(int) - md) < 45) | (np.abs(L8.astype(int) - ml) < 45)
        bimodal = float(near.mean()) if (ml - md) > 80 else 0.0
    ink_share = float((L8 <= thr).mean())

    if gray and bimodal > 0.93 and n95 <= 40:
        cat = "lineart"
    elif n95 <= 48 and flat >= 0.80:
        cat = "logo"
    elif flat >= 0.55:
        cat = "illustration"
    else:
        cat = "photo"

    if cat == "lineart":
        k_suggest = 2
    elif cat == "photo":
        k_suggest = 16
    else:
        lab_pts = lab[op] if has_alpha and op.any() else lab
        k_suggest = suggest_palette_size(lab_pts, max_k=40 if cat == "illustration" else 24)

    # reliability note: how confident is the call
    margin = {
        "lineart": min(bimodal - 0.93, 0.07) / 0.07,
        "logo": min(flat - 0.80, 0.2) / 0.2,
        "illustration": min(flat - 0.55, 0.80 - flat) / 0.125,
        "photo": min(0.55 - flat, 0.3) / 0.3,
    }[cat]
    confidence = "high" if margin > 0.5 else ("medium" if margin > 0.15 else "low")

    return {
        "width": W,
        "height": H,
        "megapixels": round(W * H / 1e6, 3),
        "has_alpha": has_alpha,
        "category": cat,
        "confidence": confidence,
        "suggested_colors": int(k_suggest),
        "stats": {
            "grayscale": gray,
            "colors_95pct": n95,
            "flat_share": round(flat, 3),
            "bimodal_share": round(bimodal, 3),
            "ink_share": round(ink_share, 3),
        },
    }
