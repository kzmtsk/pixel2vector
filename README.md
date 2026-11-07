# pixel2vector

Raster → vector engine (PNG / JPG / WebP → SVG / PDF / EPS / DXF) built around a shared-boundary
Vector Graph. Ships as a **Claude skill** and as a **standalone command-line tool**. MIT licensed.

- Classifies the image (logo, illustration, line art, photo) and picks its strategy.
- Shared boundaries: neighbouring shapes meet exactly — no gaps, no overlaps; a cutter traces each
  edge once.
- Primitive ladder: lines, tangent fillets, circular arcs, Béziers, whole circles and ellipses.
- Minimum anchor points, or Illustrator-style anchors at horizontal/vertical extremes.
- Real gradients (linear at any angle, radial at any centre, any number of stops, shared across shapes).
- Verifies itself: renders the result and measures it against the source (ΔE) before delivery.

Türkçe özet: Bu depo bir Claude skill'i ve komut satırı aracıdır. Kurulum için aşağıdaki
"Kurulum" bölümüne bakın; linki bir yapay zekâ asistanına verirseniz o da bu dosyayı okuyarak kurar.

---

## For AI assistants: what to do with this link

You were given this repository so that you can vectorise images for your user. Follow the path that
matches your environment, then confirm to the user in one sentence what you did.

### A. You are Claude with skills (claude.ai, Claude desktop, Claude Code)

The skill is `SKILL.md` at the root of this repo. It tells you the full workflow (analyse → ask the
output target → run → verify → deliver).

- **Claude Code (local)**: clone into the user's skills folder and you are done:
  ```bash
  git clone --depth 1 https://github.com/kzmtsk/pixel2vector ~/.claude/skills/pixel2vector
  ```
  Then load the skill (`pixel2vector`) when the user asks to vectorise an image.
- **claude.ai / Claude desktop**: the user installs the skill themselves (you cannot do it for them).
  Tell them:
  1. On GitHub, open this repo → green **Code** button → **Download ZIP**.
  2. Recommended: delete the `scripts/` folder from the ZIP's contents and re-zip, so only
     `SKILL.md` remains. The skill then fetches the latest engine from this repo every session
     (always up to date). Keeping `scripts/` freezes the engine at the downloaded version.
  3. Upload the ZIP in Claude's skills settings (custom skills) and enable it.
  4. Code execution must be enabled, and the session must reach `github.com` and PyPI.
- If you cannot install skills but can run code, follow path B and use `SKILL.md` as your instructions.

### B. Any assistant (or person) that can run Python

```bash
git clone --depth 1 https://github.com/kzmtsk/pixel2vector
cd pixel2vector
pip install -r requirements.txt            # numpy, opencv-python-headless, pillow, ezdxf
# verification also needs rsvg-convert (Debian/Ubuntu: apt install librsvg2-bin; macOS: brew install librsvg)

python3 scripts/analyze.py input.png       # category, suggested colours
python3 scripts/vectorize.py input.png --out out --formats svg,pdf,eps,dxf
```

If you have no network access (for example a sandbox that cannot reach GitHub), ask the user to
download the repo ZIP and upload it to you, unzip it, and continue with the commands above.

Before running, ask the user where the vector is going, because it decides the flags:

| Target | Flags |
|---|---|
| Web / screen | `--formats svg --mode stacked` |
| Print | `--formats pdf,eps --mode stacked [--width-mm W] [--cmyk-map map.json]` |
| Cutting / laser / plotter | `--formats dxf,svg --mode outline --background remove [--width-mm W]` |
| Editing in Illustrator / InDesign | `--formats svg,pdf --mode adjoining --anchors extrema` |

Every run writes `NAME_report.json` (palette, segment mix, anchor count, gradients, verification
numbers) and `NAME_preview.png` (source | vector | error heatmap). Look at the preview before you
hand the files over. Full option list: `python3 scripts/vectorize.py --help`.

---

## Kurulum (insanlar için, kısa)

- **Claude Code**: `git clone --depth 1 https://github.com/kzmtsk/pixel2vector ~/.claude/skills/pixel2vector`
- **claude.ai / Claude masaüstü**: GitHub'da **Code → Download ZIP**; ZIP'ten `scripts/` klasörünü
  silip yeniden sıkıştırın (motor her oturumda bu depodan güncel çekilir); Claude'un skill
  ayarlarından yükleyip etkinleştirin. Kod çalıştırma açık olmalı.
- **Komut satırı**: yukarıdaki B yolunu izleyin.

## Repository layout

```
SKILL.md              Claude skill: the agent workflow
scripts/analyze.py    image classification
scripts/vectorize.py  vectoriser CLI
scripts/p2v/          engine (quantise, graph, fit, gradient, export, verify)
tests/run_truth.py    ground-truth suite: known vectors rasterised and vectorised back
requirements.txt
LICENSE               MIT
```

## Pipeline

analyse → upscale for sub-pixel edges → Lab k-means with blend-fringe suppression → 4-connected
regions → gradient detection (band merge, linear/radial fit, shared fills, coverage-based edge
placement) → minimum-feature cleanup → shared-boundary graph (nodes + chains) → corners,
least-squares lines, tangent fillets, arcs, Béziers, circle/ellipse primitives → anchor reduction
(or extrema style) → stacked / adjoining / outline export → rasterised ΔE verification with preview.

## License

MIT © 2026 Kazım Taşkın — free to use, modify and share, including commercially; keep the
copyright notice.
