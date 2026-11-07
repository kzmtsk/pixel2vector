"""End-to-end pipeline."""
import os
import json
import time
import numpy as np
from .imgio import load_rgba, to_hex
from .analyze import analyze
from .quantize import working_scale, upscale, quantize, despeckle, components, edge_band_width, TRANSPARENT
from .imgio import composite, rgb_to_lab
from .graph import build_graph
from .fit import fit_chain
from .model import assemble
from .gradient import detect as detect_gradients
from .export import Scene, write_svg, write_pdf, write_eps, write_dxf
from . import verify

PRESETS = {
    #               tol(src px) sigma(src px) min_area(src px^2) corner_deg  primitives
    "logo":         (0.30,       0.55,        1.5,               40.0,       True),
    "lineart":      (0.35,       0.60,        2.0,               45.0,       True),
    "illustration": (0.45,       0.70,        4.0,               45.0,       True),
    "photo":        (0.90,       1.10,        0.0,               60.0,       False),
}
PRECISION = {"high": 0.6, "normal": 1.0, "low": 1.8}


def _grad_report(grads, regions, s):
    out = {}
    for lab, g in grads.items():
        e = out.setdefault(id(g), dict(g.describe(s), shapes=0,
                                         stop_colors=[to_hex(c) for _, c in g.stops]))
        e["shapes"] += sum(1 for r in regions if r.label == lab)
    return list(out.values())


def detect_background(labels):
    border = np.r_[labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    border = border[border >= 0]
    if border.size == 0:
        return None
    vals, cnt = np.unique(border, return_counts=True)
    i = int(cnt.argmax())
    return int(vals[i]) if cnt[i] / border.size >= 0.5 else None


def run(path, out_dir, formats=("svg",), mode="stacked", category=None, colors=None,
        background="auto", precision="normal", cmyk=None, width_mm=None, basename=None, gradients="auto",
        do_verify=True, log=print):
    t0 = time.time()
    rgba, has_alpha = load_rgba(path)
    H0, W0 = rgba.shape[:2]
    info = analyze(rgba, has_alpha)
    cat = category or info["category"]
    ncol = int(colors or info["suggested_colors"])
    if cat == "lineart":
        ncol = 2
    s = working_scale(H0, W0, cat)
    tol0, sig0, amin0, cdeg, prims = PRESETS[cat]
    pf = PRECISION[precision]
    tol = max(0.45, tol0 * s * pf)
    sigma = max(0.8, sig0 * s)
    log("[p2v] %s  %dx%d  category=%s  colors=%d  scale=x%d" % (os.path.basename(path), W0, H0, cat, ncol, s))

    up = upscale(rgba, s)
    labels, palette = quantize(up, cat, ncol, s=s)
    N = labels.size
    min_area = max(4, int(round(amin0 * s * s)))
    if cat == "photo":
        min_area = max(min_area, int(N * 2.5e-4))
    elif cat == "illustration":
        min_area = max(min_area, int(N * 1e-5))
    labels = despeckle(labels, min_area)
    comp, clab, carea = components(labels)
    work_rgb = np.where((up[..., 3] >= 0.5)[..., None], up[..., :3], composite(up)).astype(np.float32)
    work_lab = rgb_to_lab(work_rgb)
    band = edge_band_width(rgb_to_lab(composite(up)))
    grads = {}
    if gradients == "on" or (gradients == "auto" and cat in ("logo", "illustration")):
        pal_lab = rgb_to_lab(np.asarray(palette, np.float32).reshape(-1, 1, 3)).reshape(-1, 3)
        comp, clab, carea, palette, grads = detect_gradients(comp, clab, carea, palette, pal_lab,
                                                             work_lab, work_rgb, s, band)
    t1 = time.time()
    chains = build_graph(comp)
    t2 = time.time()
    # soft sources (blurred, upscaled, heavy JPEG) carry no geometry finer than their edge band:
    # widen smoothing, tolerance and feature windows accordingly
    s_eff = max(float(s), band / 2.5)
    tol = max(tol, 0.22 * band * pf)
    sigma = max(sigma, 0.45 * band)
    for ch in chains:
        fit_chain(ch, s_eff, tol, sigma, allow_primitives=prims, corner_angle=cdeg)
    t3 = time.time()
    regions = assemble(chains, clab, carea)

    dropped = set()
    bg_label = detect_background(np.where(comp >= 0, clab[comp], -1))
    if background == "auto":
        background = "remove" if (cat == "lineart" and not has_alpha) else "keep"
    if background == "remove" and bg_label is not None:
        dropped.add(bg_label)

    pt_per_px = 1.0
    if width_mm:
        pt_per_px = float(width_mm) / 25.4 * 72.0 / W0

    base = basename or os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    outputs = {}
    scene = Scene(W0, H0, s, palette, regions, chains, dropped, mode=mode, cmyk=cmyk,
                  pt_per_px=pt_per_px, title=base, gradients=grads)
    # the SVG is always produced (it drives verification)
    svg_path = os.path.join(out_dir, base + ".svg")
    write_svg(scene, svg_path)
    if "svg" in formats:
        outputs["svg"] = svg_path
    if "pdf" in formats:
        p = os.path.join(out_dir, base + ".pdf")
        write_pdf(scene, p)
        outputs["pdf"] = p
    if "eps" in formats:
        p = os.path.join(out_dir, base + ".eps")
        write_eps(scene, p)
        outputs["eps"] = p
    if "dxf" in formats:
        dscene = Scene(W0, H0, s, palette, regions, chains, dropped, mode="outline",
                       pt_per_px=pt_per_px, title=base)
        p = os.path.join(out_dir, base + ".dxf")
        write_dxf(dscene, p)
        outputs["dxf"] = p
    t4 = time.time()

    report = {
        "input": os.path.basename(path),
        "analysis": info,
        "used": {"category": cat, "colors": int(len(palette)), "scale": s, "mode": mode,
                 "background": background, "precision": precision,
                 "tolerance_src_px": round(tol / s, 3), "edge_band_src_px": round(band / s, 2)},
        "palette": [to_hex(c) for c in palette],
        "dropped_colors": [to_hex(palette[l]) for l in dropped],
        "graph": {"regions": len([r for r in regions if r.label >= 0]), "chains": len(chains),
                  "nodes_closed_loops": sum(1 for c in chains if c.closed)},
        "segments": scene.stats(),
        "gradients": _grad_report(grads, regions, s),
        "outputs": outputs,
        "timing_s": {"quantize": round(t1 - t0, 2), "graph": round(t2 - t1, 2),
                     "fit": round(t3 - t2, 2), "export": round(t4 - t3, 2)},
    }
    if do_verify:
        bg_rgb = tuple(palette[bg_label]) if (bg_label is not None and bg_label in dropped) else (1, 1, 1)
        # verification compares the filled rendition, even when the delivered mode is outline
        vpath = svg_path
        if mode == "outline":
            vpath = os.path.join(out_dir, base + "._verify.svg")
            write_svg(Scene(W0, H0, s, palette, regions, chains, dropped, mode="stacked", title=base,
                            gradients=grads), vpath)
        res, ref, ren, de = verify.compare(rgba, vpath, bg_rgb)
        bg_hex = "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in bg_rgb)
        prev = os.path.join(out_dir, base + "_preview.png")
        verify.preview(ref, vpath, de, prev, bg_hex=bg_hex)
        if vpath != svg_path:
            os.remove(vpath)
        report["verify"] = res
        report["preview"] = prev
    if "svg" not in formats and os.path.exists(svg_path):
        os.remove(svg_path)
    with open(os.path.join(out_dir, base + "_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
