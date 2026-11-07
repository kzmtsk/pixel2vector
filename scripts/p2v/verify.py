"""Stage 6: verification — rasterise the SVG and measure it against the source."""
import subprocess
import tempfile
import os
import numpy as np
import cv2
from PIL import Image
from .imgio import composite, rgb_to_lab


def render_svg(svg_path, w, h, bg="white"):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "r.png")
        cmd = ["rsvg-convert", "-w", str(w), "-h", str(h), "-b", bg, "-o", out, svg_path]
        subprocess.run(cmd, check=True, capture_output=True)
        im = Image.open(out).convert("RGB")
        return np.asarray(im).astype(np.float32) / 255.0


def compare(rgba_src, svg_path, bg_rgb=(1, 1, 1)):
    H, W = rgba_src.shape[:2]
    bg_hex = "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in bg_rgb)
    ref = composite(rgba_src, bg_rgb)
    ren = render_svg(svg_path, W, H, bg_hex)
    de = np.linalg.norm(rgb_to_lab(ref) - rgb_to_lab(ren), axis=2)
    # edge-tolerant view: ignore 1px bands around source edges (anti-aliasing is resampled, not an error)
    g = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(g, 30, 90), np.ones((3, 3), np.uint8)) > 0
    interior = ~edges
    res = {
        "mean_dE": round(float(de.mean()), 2),
        "p95_dE": round(float(np.percentile(de, 95)), 2),
        "share_dE_gt10": round(float((de > 10).mean()), 4),
        "interior_mean_dE": round(float(de[interior].mean()) if interior.any() else 0.0, 2),
        "interior_share_dE_gt10": round(float((de[interior] > 10).mean()) if interior.any() else 0.0, 4),
    }
    return res, ref, ren, de


def preview(ref, svg_path, de, out_png, target_h=520, bg_hex="#FFFFFF"):
    H, W = ref.shape[:2]
    f = target_h / H
    tw = max(1, int(round(W * f)))
    src = cv2.resize((ref * 255).astype(np.uint8), (tw, target_h), interpolation=cv2.INTER_NEAREST)
    vec = (render_svg(svg_path, tw, target_h, bg_hex) * 255).astype(np.uint8)
    heat = np.clip(de / 25.0, 0, 1)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    heat = cv2.resize(heat, (tw, target_h), interpolation=cv2.INTER_NEAREST)
    gap = np.full((target_h, 8, 3), 255, np.uint8)
    Image.fromarray(np.hstack([src, gap, vec, gap, heat])).save(out_png)
