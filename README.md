# pixel2vector

Raster → vector engine built around a shared-boundary Vector Graph (Claude skill + standalone CLI).

```
pip install -r requirements.txt          # + rsvg-convert for verification
python3 scripts/analyze.py image.png
python3 scripts/vectorize.py image.png --out out --formats svg,pdf,eps,dxf --mode stacked
```

Pipeline: analyse (logo / illustration / lineart / photo) → upscale for sub-pixel edges → Lab k-means
with blend-fringe suppression → 4-connected regions → shared-boundary graph (nodes + chains) →
corner detection, least-squares lines, tangent fillets, arcs, Béziers, circle/ellipse primitives →
stacked / adjoining / outline export → rasterised ΔE verification with preview.

See SKILL.md for the agent workflow.
