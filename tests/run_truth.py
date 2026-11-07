#!/usr/bin/env python3
"""Ground-truth gradient suite: rasterise known vectors, vectorise them back, print what came out.

usage: python3 tests/run_truth.py [OUTDIR]
Expected (truth): g1 linear 30.03 deg | g2 radial centre (170,160) | g3 one linear 0 deg shared by
3 shapes | g4 linear 90 deg background + flat red square + flat black circle.
"""
import os
import sys
import json
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "scripts"))
from p2v.pipeline import run  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p2v_truth"
os.makedirs(out, exist_ok=True)
for f in sorted(os.listdir(os.path.join(here, "truth"))):
    if not f.endswith(".svg"):
        continue
    png = os.path.join(out, f[:-4] + ".png")
    subprocess.run(["rsvg-convert", os.path.join(here, "truth", f), "-o", png], check=True)
    rep = run(png, out, formats=("svg",), log=lambda m: None)
    g = [(x["type"], x.get("angle_deg", x.get("center")), x["stops"], x["shapes"]) for x in rep["gradients"]]
    print(f, "regions=%d" % rep["graph"]["regions"], "gradients=%s" % json.dumps(g),
          "interior_dE=%.2f" % rep["verify"]["interior_mean_dE"])
