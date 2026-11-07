#!/usr/bin/env python3
"""pixel2vector — raster to vector via a shared-boundary Vector Graph.

usage:
  python3 vectorize.py IMAGE --out DIR [--formats svg,pdf,eps,dxf] [--mode stacked|adjoining|outline]
                       [--category logo|illustration|lineart|photo] [--colors N]
                       [--background auto|keep|remove] [--precision high|normal|low]
                       [--width-mm MM] [--cmyk-map map.json] [--name BASENAME] [--no-verify]
                       [--gradients auto|on|off] [--anchors minimal|extrema]

cmyk-map JSON: {"#1A2B3C": [100, 60, 0, 20], ...}  (palette hex -> C,M,Y,K percent; read report palette first)
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2v.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="pixel2vector")
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="svg")
    ap.add_argument("--mode", default="stacked", choices=["stacked", "adjoining", "outline"])
    ap.add_argument("--category", choices=["logo", "illustration", "lineart", "photo"])
    ap.add_argument("--colors", type=int)
    ap.add_argument("--background", default="auto", choices=["auto", "keep", "remove"])
    ap.add_argument("--precision", default="normal", choices=["high", "normal", "low"])
    ap.add_argument("--width-mm", type=float)
    ap.add_argument("--cmyk-map")
    ap.add_argument("--gradients", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--anchors", default="minimal", choices=["minimal", "extrema"])
    ap.add_argument("--name")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    fmts = tuple(f.strip().lower() for f in a.formats.split(",") if f.strip())
    bad = set(fmts) - {"svg", "pdf", "eps", "dxf"}
    if bad:
        ap.error("unknown format(s): %s" % ", ".join(sorted(bad)))
    cmyk = None
    if a.cmyk_map:
        with open(a.cmyk_map) as f:
            cmyk = json.load(f)
    rep = run(a.image, a.out, formats=fmts, mode=a.mode, category=a.category, colors=a.colors,
              background=a.background, precision=a.precision, cmyk=cmyk, width_mm=a.width_mm,
              basename=a.name, gradients=a.gradients, anchors=a.anchors, do_verify=not a.no_verify, log=lambda m: print(m, file=sys.stderr))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
