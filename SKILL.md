---
name: pixel2vector
description: Raster görseli (PNG/JPG/WebP) vektöre çevirir — logo, illüstrasyon, çizim/mühür, fotoğraf. Önce görseli sınıflandırır, çıktı hedefini sorar, SVG/PDF/EPS/DXF üretir ve sonucu ölçerek doğrular.
---

# pixel2vector

From-scratch vectoriser built around a **shared-boundary Vector Graph**: every boundary between two
colour regions is stored once and referenced by both neighbours, so shapes meet exactly (no hairline
gaps, no overlaps) and a cutter traces each edge once. Geometry is fitted with a primitive ladder —
straight line → tangent fillet / circular arc → cubic Bézier — plus whole-shape circle and ellipse
detection, corner detection with least-squares corner placement, and sub-pixel boundaries recovered
from anti-aliasing (the source is upscaled before quantisation). Smooth colour transitions are
rebuilt as real gradients — linear at any angle or radial at any centre, any number of stops, one
gradient per shape or one shared across several shapes — instead of being posterised into bands.

Talk to the user in their language (Turkish for Kazım). Keep narration minimal.

## 0. Engine setup (once per session)

The engine is the `scripts/` folder of the repo `github.com/kzmtsk/pixel2vector` (if this skill's own
directory already contains `scripts/vectorize.py`, use that instead and skip the clone):

```bash
[ -f /tmp/pixel2vector/scripts/vectorize.py ] || git clone --depth 1 https://github.com/kzmtsk/pixel2vector /tmp/pixel2vector
ENGINE=/tmp/pixel2vector/scripts
pip install --break-system-packages -q numpy opencv-python-headless pillow ezdxf
which rsvg-convert || echo "rsvg-convert missing: verification needs it (apt package librsvg2-bin)"
```

Work on a copy of the user's upload, never the original. Each run writes into its own output folder.

## 1. Analyse

```bash
python3 $ENGINE/analyze.py INPUT
```

Returns `category` (logo | illustration | lineart | photo), `confidence`, `suggested_colors`, stats.
Tell the user in one line what it is and how many colours you see. If `confidence` is `low`, name the
runner-up category in the same line (e.g. logo vs illustration) — the question in step 2 settles it.

Category meaning:
- **logo** — few flat colours, crisp edges (icons, wordmarks, low-res client logos)
- **illustration** — many flat colour regions (flat-shaded art, NBP flat renders)
- **lineart** — monochrome drawing, stamp, signature, handwriting, scan of ink
- **photo** — continuous tone; output is a posterised stylisation, not a faithful copy. Say so.

Gradients are detected automatically for logo and illustration (`--gradients auto`). Force them with
`--gradients on` (e.g. a photo-classified image that is really a gradient graphic) or switch them off
with `--gradients off` when the user wants flat colours only.

## 2. Ask the output target — always, before processing

Use AskUserQuestion (multiSelect). Never skip this, even if the target seems obvious.

Question: "Vektör çıktısı nereye gidecek?" Options:
1. **Baskı (PDF/EPS)** — offset/dijital baskı
2. **Kesim / lazer / plotter (DXF)**
3. **Web / ekran (SVG)**
4. **Illustrator / InDesign'da düzenleme**

In the same AskUserQuestion call, add only the questions that are genuinely open:
- Background: keep or remove — ask when the image has a solid background and the target is not
  obvious (cutting and lineart default to remove; transparent PNGs keep their transparency anyway).
- Category, when confidence was low (offer the two candidates).
- Colour count, when the suggestion looks doubtful against what you see in the image.

If print is chosen, also ask (plain text is fine) for the final physical width in mm, if known
(`--width-mm`), and whether they want CMYK values assigned to the palette colours. CMYK is never
auto-converted (a naive RGB→CMYK is wrong for print); the user supplies values per palette colour
after seeing the palette from the first run, via `--cmyk-map`.

## 3. Map targets to runs

| Target | Command flags | Files delivered |
|---|---|---|
| Web | `--formats svg --mode stacked` | `.svg` |
| Print | `--formats pdf,eps --mode stacked [--width-mm W] [--cmyk-map M]` | `.pdf`, `.eps` |
| Cutting | `--formats dxf,svg --mode outline --background remove [--width-mm W]` | `.dxf` (+ outline `.svg` preview) |
| Illustrator/InDesign | `--formats svg,pdf --mode adjoining` | `.svg` (groups per colour, ids `color-NN-HEX`, gradient groups `gradient-NN`), `.pdf` |

Gradients in each format: SVG `linearGradient` / `radialGradient` (userSpaceOnUse, editable stops),
PDF axial / radial shading, EPS PostScript Level 3 `shfill`. DXF carries outlines only. A CMYK map
applies to flat colours; gradient stops stay RGB — say so when print + CMYK + gradients meet.

Modes:
- **stacked** — largest regions painted first; a hole is left out when the shape filling it is painted
  on top, so lower shapes run underneath. No seams on screen, register-safe in print.
- **adjoining** — every shape keeps exact holes and touches its neighbours edge-to-edge on the shared
  boundary. Best for editing (select same fill, recolour). Can show hairline seams in some renderers.
- **outline** — each boundary once, as a hairline stroke. DXF uses true ARC / CIRCLE / ELLIPSE / SPLINE /
  LINE entities, units mm, one layer per colour.

When several targets are chosen, run once per distinct mode, same output folder, different `--name`
(e.g. `logo_print`, `logo_edit`). Illustrator opens the PDF directly; tell the user they can save it
as `.ai` from there.

Run:

```bash
python3 $ENGINE/vectorize.py INPUT --out OUTDIR --name BASENAME --formats ... --mode ... \
  [--category C] [--colors N] [--background keep|remove] [--precision high|normal|low]
```

stdout is the JSON report (also saved as `BASENAME_report.json`); progress goes to stderr.

## 4. Verify before delivering

Every run rasterises its own SVG and compares it to the source (ΔE in Lab), and writes
`BASENAME_preview.png` (source | vector | error heatmap). Always Read the preview image, and check:

| category | pass if |
|---|---|
| logo, lineart | `verify.interior_mean_dE` ≤ 1.5 and `verify.share_dE_gt10` ≤ 0.03 |
| illustration | `interior_mean_dE` ≤ 3 and `share_dE_gt10` ≤ 0.06 |
| photo | stylisation — report numbers, judge by eye |

Also look at the preview for: lost small details (raise `--colors` or use `--precision high`), a
colour that should not be there (lower `--colors`), wrong category behaviour (`--category`), or a
background that should have been dropped. Retry at most twice, changing one thing at a time, and say
what you changed. Edge errors in the heatmap are normal (anti-aliasing is resampled, not an error).

## 5. Deliver

Send the output files with SendUserFile (display: attach) plus the preview PNG (render). Then a short
summary in the user's language: category, palette (hex list), segment mix from `segments`
(L lines, A arcs, C curves, CIRCLE, ELLIPSE), gradients from `gradients` (type, angle or centre,
stop count, how many shapes share it), the two verify numbers, and anything that fell short.
No step-by-step recap.

## Known limits (state them only when relevant)

- Strokes are traced as filled outlines (both edges), not as centrelines.
- Photos become posterised art; for a faithful look raise `--colors` (24–40).
- Heavily blurred or upscaled sources lose corner sharpness; the engine widens its tolerance to the
  measured edge blur, so geometry stays clean but tight corners may soften.
- No symmetry enforcement yet; no gradient meshes (free-form metal reflections are approximated by
  the best linear/radial fit, possibly split across a few shapes).
- EPS gradients (`shfill`) have not been render-checked in a PostScript interpreter yet.
- Working resolution is capped (logo/lineart ≈ 8 MP, illustration ≈ 6 MP, photo ≈ 2 MP after the
  internal upscale); very large inputs are processed at 1×.
