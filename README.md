# pixel2vector

**[English](#english) · [Türkçe](#türkçe)**

---

<a id="english"></a>
## English

Raster → vector engine (PNG / JPG / WebP → SVG / PDF / EPS / DXF) built around a shared-boundary
Vector Graph. Ships as a **Claude skill** and as a **standalone command-line tool**. MIT licensed.

- Classifies the image (logo, illustration, line art, photo) and picks its strategy.
- Shared boundaries: neighbouring shapes meet exactly — no gaps, no overlaps; a cutter traces each
  edge once.
- Primitive ladder: lines, tangent fillets, circular arcs, Béziers, whole circles and ellipses.
- Minimum anchor points, or Illustrator-style anchors at horizontal/vertical extremes.
- Real gradients (linear at any angle, radial at any centre, any number of stops, shared across shapes).
- Verifies itself: renders the result and measures it against the source (ΔE) before delivery.

### For AI assistants: what to do with this link

You were given this repository so that you can vectorise images for your user. Follow the path that
matches your environment, then tell the user in one sentence what you did. Talk to the user in their
own language.

#### A. You are Claude with skills (claude.ai, Claude desktop, Claude Code)

The skill is `SKILL.md` at the root of this repo. It holds the full workflow (analyse → ask the
output target → run → verify → deliver).

- **Claude Code (local)**: clone into the user's skills folder and you are done:
  ```bash
  git clone --depth 1 https://github.com/kzmtsk/pixel2vector ~/.claude/skills/pixel2vector
  ```
  Then use the skill `pixel2vector` whenever the user asks to vectorise an image.
- **claude.ai / Claude desktop**: the user installs the skill themselves (you cannot do it for them).
  Tell them:
  1. On GitHub, open this repo → green **Code** button → **Download ZIP**.
  2. Recommended: remove the `scripts/` folder from the ZIP's contents and re-zip, so only `SKILL.md`
     remains. The skill then fetches the latest engine from this repo every session (always up to
     date). Keeping `scripts/` freezes the engine at the downloaded version.
  3. Upload the ZIP in Claude's skills settings (custom skills) and enable it.
  4. Code execution must be enabled, and the session must reach `github.com` and PyPI.
- If you cannot install skills but can run code, follow path B and use `SKILL.md` as your instructions.

#### B. Any assistant (or person) that can run Python

```bash
git clone --depth 1 https://github.com/kzmtsk/pixel2vector
cd pixel2vector
pip install -r requirements.txt            # numpy, opencv-python-headless, pillow, ezdxf
# verification also needs rsvg-convert (Debian/Ubuntu: apt install librsvg2-bin; macOS: brew install librsvg)

python3 scripts/analyze.py input.png       # category, suggested colours
python3 scripts/vectorize.py input.png --out out --formats svg,pdf,eps,dxf
```

No network access (a sandbox that cannot reach GitHub)? Ask the user to download the repo ZIP and
upload it to you, unzip it, and continue with the commands above.

Before running, ask the user where the vector is going — it decides the flags:

| Target | Flags |
|---|---|
| Web / screen | `--formats svg --mode stacked` |
| Print | `--formats pdf,eps --mode stacked [--width-mm W] [--cmyk-map map.json]` |
| Cutting / laser / plotter | `--formats dxf,svg --mode outline --background remove [--width-mm W]` |
| Editing in Illustrator / InDesign | `--formats svg,pdf --mode adjoining --anchors extrema` |

Every run writes `NAME_report.json` (palette, segment mix, anchor count, gradients, verification
numbers) and `NAME_preview.png` (source | vector | error heatmap). Look at the preview before you
hand the files over. All options: `python3 scripts/vectorize.py --help`.

### Repository layout

```
SKILL.md              Claude skill: the agent workflow
scripts/analyze.py    image classification
scripts/vectorize.py  vectoriser CLI
scripts/p2v/          engine (quantise, graph, fit, gradient, export, verify)
tests/run_truth.py    ground-truth suite: known vectors rasterised and vectorised back
requirements.txt
LICENSE               MIT
```

### Pipeline

analyse → upscale for sub-pixel edges → Lab k-means with blend-fringe suppression → 4-connected
regions → gradient detection (band merge, linear/radial fit, shared fills, coverage-based edge
placement) → minimum-feature cleanup → shared-boundary graph (nodes + chains) → corners,
least-squares lines, tangent fillets, arcs, Béziers, circle/ellipse primitives → anchor reduction
(or extrema style) → stacked / adjoining / outline export → rasterised ΔE verification with preview.

### License

MIT © 2026 Kazım Taşkın — free to use, modify and share, including commercially; keep the
copyright notice.

---

<a id="türkçe"></a>
## Türkçe

Raster → vektör motoru (PNG / JPG / WebP → SVG / PDF / EPS / DXF), ortak sınırlı bir Vector Graph
üzerine kurulu. Hem **Claude skill'i** hem **bağımsız komut satırı aracı** olarak çalışır. MIT lisanslı.

- Görseli sınıflandırır (logo, illüstrasyon, çizim, fotoğraf) ve stratejisini ona göre seçer.
- Ortak sınırlar: komşu şekiller tam oturur — boşluk ya da bindirme yok; kesim makinesi her kenardan
  bir kez geçer.
- Geometri basamakları: doğru, teğet köşe yayı, daire yayı, Bézier, tam daire ve elips.
- En az anchor noktası ya da Illustrator usulü yatay/dikey uç noktalarda anchor.
- Gerçek degradeler (her açıda doğrusal, her merkezde dairesel, sınırsız durak, şekiller arası ortak).
- Kendini doğrular: sonucu görüntüye çevirip kaynakla ölçer (ΔE), sonra teslim eder.

### Yapay zekâ asistanları için: bu linkle ne yapmalı

Bu depo, kullanıcın için görselleri vektöre çevirebilesin diye sana verildi. Ortamına uyan yolu izle,
ardından ne yaptığını kullanıcıya tek cümleyle söyle. Kullanıcıyla onun dilinde konuş.

#### A. Skill destekli Claude'san (claude.ai, Claude masaüstü, Claude Code)

Skill, deponun kökündeki `SKILL.md` dosyasıdır. Tüm akış orada yazar (analiz → çıktı hedefini sor →
çalıştır → doğrula → teslim et).

- **Claude Code (yerel)**: kullanıcının skill klasörüne klonla, bu kadar:
  ```bash
  git clone --depth 1 https://github.com/kzmtsk/pixel2vector ~/.claude/skills/pixel2vector
  ```
  Kullanıcı bir görseli vektöre çevirmek istediğinde `pixel2vector` skill'ini kullan.
- **claude.ai / Claude masaüstü**: skill'i kullanıcı kendisi kurar (sen onun yerine kuramazsın).
  Ona şunu söyle:
  1. GitHub'da bu depoyu aç → yeşil **Code** düğmesi → **Download ZIP**.
  2. Önerilen: ZIP'in içinden `scripts/` klasörünü çıkarıp yeniden sıkıştır, sadece `SKILL.md`
     kalsın. Böylece skill motorun en güncel halini her oturumda bu depodan çeker. `scripts/`
     klasörü kalırsa motor indirildiği günkü sürümde donar.
  3. ZIP'i Claude'un skill ayarlarından (özel skill'ler) yükle ve etkinleştir.
  4. Kod çalıştırma açık olmalı; oturum `github.com` ve PyPI'ye erişebilmeli.
- Skill kuramıyor ama kod çalıştırabiliyorsan B yolunu izle, `SKILL.md`'yi talimatın olarak kullan.

#### B. Python çalıştırabilen her asistan (ya da kişi)

```bash
git clone --depth 1 https://github.com/kzmtsk/pixel2vector
cd pixel2vector
pip install -r requirements.txt            # numpy, opencv-python-headless, pillow, ezdxf
# doğrulama için rsvg-convert de gerekir (Debian/Ubuntu: apt install librsvg2-bin; macOS: brew install librsvg)

python3 scripts/analyze.py girdi.png       # kategori, önerilen renk sayısı
python3 scripts/vectorize.py girdi.png --out cikti --formats svg,pdf,eps,dxf
```

İnternet erişimi yoksa (GitHub'a ulaşamayan bir ortam), kullanıcıdan depo ZIP'ini indirip sana
yüklemesini iste, aç ve yukarıdaki komutlarla devam et.

Çalıştırmadan önce kullanıcıya vektörün nereye gideceğini sor — ayarları bu belirler:

| Hedef | Ayarlar |
|---|---|
| Web / ekran | `--formats svg --mode stacked` |
| Baskı | `--formats pdf,eps --mode stacked [--width-mm G] [--cmyk-map harita.json]` |
| Kesim / lazer / plotter | `--formats dxf,svg --mode outline --background remove [--width-mm G]` |
| Illustrator / InDesign'da düzenleme | `--formats svg,pdf --mode adjoining --anchors extrema` |

Her çalıştırma `AD_report.json` (palet, parça dağılımı, anchor sayısı, degradeler, doğrulama
sonuçları) ve `AD_preview.png` (kaynak | vektör | hata haritası) yazar. Dosyaları teslim etmeden önce
ön izlemeye bak. Tüm seçenekler: `python3 scripts/vectorize.py --help`.

### Depo yapısı

```
SKILL.md              Claude skill'i: agent akışı
scripts/analyze.py    görsel sınıflandırma
scripts/vectorize.py  vektörleştirme komut satırı aracı
scripts/p2v/          motor (renk ayrımı, graf, eğri oturtma, degrade, dışa aktarım, doğrulama)
tests/run_truth.py    doğru cevabı bilinen testler: bilinen vektörler piksele çevrilip geri vektörlenir
requirements.txt
LICENSE               MIT
```

### Lisans

MIT © 2026 Kazım Taşkın — ticari dahil serbestçe kullanılabilir, değiştirilebilir ve paylaşılabilir;
telif notu korunmalıdır.
