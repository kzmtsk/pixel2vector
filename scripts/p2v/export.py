"""Stage 5: serialisers — SVG, PDF, EPS, DXF.

Paint modes
  stacked    regions painted largest-first; a hole is omitted when the region that fills it is
             painted later, so the lower shape runs underneath (no seams on screen, register-safe
             in print). Holes that must show what is below (or a removed background) stay cut.
  adjoining  every region keeps its exact holes; shapes meet edge-to-edge on shared boundaries.
             Best for editing (select-by-colour, recolour) — may show hairline seams when rendered.
  outline    each shared boundary chain once, as a hairline stroke (cutting / engraving / plotting).
"""
import math
import numpy as np
from .fit import arc_to_beziers, seg_start
from .graph import OUTSIDE
from .imgio import to_hex


class Scene:
    def __init__(self, W, H, s, palette, regions, chains, dropped_labels, mode="stacked",
                 cmyk=None, pt_per_px=1.0, title="pixel2vector"):
        self.W, self.H, self.s = W, H, s
        self.palette = palette
        self.regions = regions
        self.chains = chains
        self.dropped = set(dropped_labels)
        self.mode = mode
        self.cmyk = {k.upper(): v for k, v in (cmyk or {}).items()}
        self.pt_per_px = pt_per_px
        self.title = title
        self.by_id = {r.id: r for r in regions}
        self.visible = [r for r in regions if r.label >= 0 and r.label not in self.dropped]
        order = sorted(self.visible, key=lambda r: -r.area)
        for i, r in enumerate(order):
            r.order = i
        self.paint_order = order

    # ------------------------------------------------------------ helpers
    def hexcol(self, label):
        return to_hex(self.palette[label])

    def is_painted(self, rid):
        if rid == OUTSIDE:
            return False
        r = self.by_id.get(rid)
        return r is not None and r.label >= 0 and r.label not in self.dropped

    def loops_for(self, r):
        loops = [r.outer]
        if self.mode == "stacked":
            for h in r.holes:
                keep = False
                for nb in h.neighbours:
                    if not self.is_painted(nb) or self.by_id[nb].order < r.order:
                        keep = True
                        break
                if keep:
                    loops.append(h)
        else:
            loops += r.holes
        return loops

    def outline_chains(self):
        out = []
        for ch in self.chains:
            a, b = self.is_painted(ch.right), self.is_painted(ch.left)
            if not (a or b):
                continue
            # chains on the image frame against a dropped/absent background are still outlines
            owner = None
            cands = [rid for rid in (ch.right, ch.left) if self.is_painted(rid)]
            owner = min(cands, key=lambda rid: self.by_id[rid].area)
            out.append((ch, self.by_id[owner].label))
        return out

    def stats(self):
        cnt = {"L": 0, "A": 0, "C": 0, "CIRCLE": 0, "ELLIPSE": 0}
        for ch in self.chains:
            for sg in ch.segs:
                cnt[sg[0]] += 1
        return cnt


def _f(v):
    s = ("%.3f" % v).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ================================================================ SVG
def _svg_loop(segs, k):
    out = []
    first = True
    for sg in segs:
        t = sg[0]
        if t in ("CIRCLE", "ELLIPSE"):
            if t == "CIRCLE":
                _, c, R, d = sg
                a = b = R / k
                th = 0.0
            else:
                _, c, a, b, th, d = sg
                a, b = a / k, b / k
            cx, cy = c[0] / k, c[1] / k
            ct, st = math.cos(th), math.sin(th)
            p0 = (cx + a * ct, cy + a * st)
            p1 = (cx - a * ct, cy - a * st)
            fl = 1 if d > 0 else 0
            rot = _f(math.degrees(th))
            out.append("M%s %sA%s %s %s 0 %d %s %sA%s %s %s 0 %d %s %sZ" % (
                _f(p0[0]), _f(p0[1]), _f(a), _f(b), rot, fl, _f(p1[0]), _f(p1[1]),
                _f(a), _f(b), rot, fl, _f(p0[0]), _f(p0[1])))
            first = True
            continue
        if first:
            p = sg[1]
            out.append("M%s %s" % (_f(p[0] / k), _f(p[1] / k)))
            first = False
        if t == "L":
            p = sg[2]
            out.append("L%s %s" % (_f(p[0] / k), _f(p[1] / k)))
        elif t == "C":
            _, p0, c1, c2, p1 = sg
            out.append("C%s %s %s %s %s %s" % tuple(_f(v / k) for v in (c1[0], c1[1], c2[0], c2[1], p1[0], p1[1])))
        elif t == "A":
            _, p0, p1, c, R, a0, sw = sg
            out.append("A%s %s 0 %d %d %s %s" % (_f(R / k), _f(R / k), 1 if abs(sw) > math.pi else 0,
                                                 1 if sw > 0 else 0, _f(p1[0] / k), _f(p1[1] / k)))
    if not first:
        out.append("Z")
    return "".join(out)


def write_svg(scene, path):
    k = scene.s
    W, H = scene.W, scene.H
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%d" height="%d" viewBox="0 0 %d %d">' % (W, H, W, H),
             '<title>%s</title>' % scene.title]
    if scene.mode == "outline":
        groups = {}
        for ch, lab in scene.outline_chains():
            groups.setdefault(lab, []).append(_svg_loop_open(ch.segs, k))
        for i, lab in enumerate(sorted(groups)):
            hx = scene.hexcol(lab)
            lines.append('<g id="color-%02d-%s" fill="none" stroke="%s" stroke-width="0.5">' % (i + 1, hx[1:], hx))
            lines.append('<path d="%s"/>' % " ".join(groups[lab]))
            lines.append("</g>")
    elif scene.mode == "adjoining":
        groups = {}
        for r in scene.visible:
            groups.setdefault(r.label, []).append(r)
        for i, lab in enumerate(sorted(groups, key=lambda l: -sum(r.area for r in groups[l]))):
            hx = scene.hexcol(lab)
            lines.append('<g id="color-%02d-%s" fill="%s" fill-rule="evenodd">' % (i + 1, hx[1:], hx))
            for r in groups[lab]:
                d = "".join(_svg_loop(l.segs, k) for l in scene.loops_for(r))
                lines.append('<path id="r%d" d="%s"/>' % (r.id, d))
            lines.append("</g>")
    else:  # stacked
        lines.append('<g id="stacked" fill-rule="evenodd">')
        for r in scene.paint_order:
            d = "".join(_svg_loop(l.segs, k) for l in scene.loops_for(r))
            lines.append('<path id="r%d" class="c%s" fill="%s" d="%s"/>' % (r.id, scene.hexcol(r.label)[1:], scene.hexcol(r.label), d))
        lines.append("</g>")
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _svg_loop_open(segs, k):
    d = _svg_loop(segs, k)
    # outlines are open strokes; closing Z on an open chain would add a stray line
    if segs and segs[0][0] not in ("CIRCLE", "ELLIPSE"):
        a, b = seg_start(segs[0]), _seg_last(segs)
        if math.hypot(*(a - b)) > 1e-6 and d.endswith("Z"):
            d = d[:-1]
    return d


def _seg_last(segs):
    sg = segs[-1]
    t = sg[0]
    return sg[2] if t in ("L", "A") else (sg[4] if t == "C" else seg_start(sg))


# ================================================================ cubic flattening (PDF)
def loop_to_cubics(segs):
    """Yield subpaths: list of ('M',p) / ('L',p) / ('C',c1,c2,p) commands, arcs converted."""
    out = []
    first = True
    for sg in segs:
        t = sg[0]
        if t in ("CIRCLE", "ELLIPSE"):
            if t == "CIRCLE":
                _, c, R, d = sg
                bez = arc_to_beziers(c, R, 0.0, 2 * math.pi * d)
            else:
                _, c, a, b, th, d = sg
                bez = arc_to_beziers(c, None, 0.0, 2 * math.pi * d, rx=a, ry=b, theta=th)
            out.append(("M", bez[0][0]))
            for b in bez:
                out.append(("C", b[1], b[2], b[3]))
            out.append(("Z",))
            first = True
            continue
        if first:
            out.append(("M", sg[1]))
            first = False
        if t == "L":
            out.append(("L", sg[2]))
        elif t == "C":
            out.append(("C", sg[2], sg[3], sg[4]))
        elif t == "A":
            _, p0, p1, c, R, a0, sw = sg
            for b in arc_to_beziers(c, R, a0, sw):
                out.append(("C", b[1], b[2], b[3]))
    if not first:
        out.append(("Z",))
    return out


def _fill_op_color(scene, label, stroke=False):
    hx = scene.hexcol(label)
    if hx.upper() in scene.cmyk:
        c, m, y, kk = [v / 100.0 for v in scene.cmyk[hx.upper()]]
        return "%s %s %s %s %s" % (_f(c), _f(m), _f(y), _f(kk), "K" if stroke else "k")
    r, g, b = scene.palette[label]
    return "%s %s %s %s" % (_f(r), _f(g), _f(b), "RG" if stroke else "rg")


def _pdf_path(cmds, k):
    o = []
    for c in cmds:
        if c[0] == "M":
            o.append("%s %s m" % (_f(c[1][0] / k), _f(c[1][1] / k)))
        elif c[0] == "L":
            o.append("%s %s l" % (_f(c[1][0] / k), _f(c[1][1] / k)))
        elif c[0] == "C":
            o.append("%s %s %s %s %s %s c" % tuple(_f(v / k) for v in (c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1])))
        else:
            o.append("h")
    return "\n".join(o)


def write_pdf(scene, path):
    k = scene.s
    W, H = scene.W, scene.H
    sc = scene.pt_per_px
    body = ["q", "%s 0 0 %s 0 %s cm" % (_f(sc), _f(-sc), _f(H * sc))]
    if scene.mode == "outline":
        body.append("%s w" % _f(0.25 / sc))
        for ch, lab in scene.outline_chains():
            cmds = [c for c in loop_to_cubics(ch.segs)]
            if ch.segs[0][0] not in ("CIRCLE", "ELLIPSE") and not ch.closed and cmds and cmds[-1][0] == "Z":
                cmds = cmds[:-1]
            body.append(_fill_op_color(scene, lab, stroke=True))
            body.append(_pdf_path(cmds, k))
            body.append("S")
    else:
        regs = scene.paint_order if scene.mode == "stacked" else sorted(scene.visible, key=lambda r: r.label)
        for r in regs:
            body.append(_fill_op_color(scene, r.label))
            for l in scene.loops_for(r):
                body.append(_pdf_path(loop_to_cubics(l.segs), k))
            body.append("f*")
    body.append("Q")
    content = ("\n".join(body)).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %s %s] /Contents 4 0 R /Resources << >> >>" % (_f(W * sc), _f(H * sc))).encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        ("<< /Title (%s) /Producer (pixel2vector) /Creator (pixel2vector) >>" % scene.title).encode("latin-1", "replace"),
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 5 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    with open(path, "wb") as f:
        f.write(out)


# ================================================================ EPS (native arcs)
def _eps_loop(segs, k):
    o = []
    first = True
    for sg in segs:
        t = sg[0]
        if t == "CIRCLE":
            _, c, R, d = sg
            o.append("%s %s moveto %s %s %s 0 %s %s closepath" % (
                _f((c[0] + R) / k), _f(c[1] / k), _f(c[0] / k), _f(c[1] / k), _f(R / k),
                "360" if d > 0 else "-360", "arc" if d > 0 else "arcn"))
            first = True
            continue
        if t == "ELLIPSE":
            _, c, a, b, th, d = sg
            ct, st = math.cos(th), math.sin(th)
            o.append("%s %s moveto matrix currentmatrix %s %s translate %s rotate %s %s scale 0 0 1 0 %s %s setmatrix closepath" % (
                _f((c[0] + a * ct) / k), _f((c[1] + a * st) / k), _f(c[0] / k), _f(c[1] / k), _f(math.degrees(th)),
                _f(a / k), _f(b / k), "360" if d > 0 else "-360", "arc" if d > 0 else "arcn"))
            first = True
            continue
        if first:
            o.append("%s %s moveto" % (_f(sg[1][0] / k), _f(sg[1][1] / k)))
            first = False
        if t == "L":
            o.append("%s %s lineto" % (_f(sg[2][0] / k), _f(sg[2][1] / k)))
        elif t == "C":
            o.append("%s %s %s %s %s %s curveto" % tuple(_f(v / k) for v in (sg[2][0], sg[2][1], sg[3][0], sg[3][1], sg[4][0], sg[4][1])))
        elif t == "A":
            _, p0, p1, c, R, a0, sw = sg
            o.append("%s %s %s %s %s %s" % (_f(c[0] / k), _f(c[1] / k), _f(R / k), _f(math.degrees(a0)),
                                             _f(math.degrees(a0 + sw)), "arc" if sw > 0 else "arcn"))
    if not first:
        o.append("closepath")
    return "\n".join(o)


def write_eps(scene, path):
    k = scene.s
    W, H = scene.W, scene.H
    sc = scene.pt_per_px
    bw, bh = int(math.ceil(W * sc)), int(math.ceil(H * sc))
    L = ["%!PS-Adobe-3.0 EPSF-3.0",
         "%%%%BoundingBox: 0 0 %d %d" % (bw, bh),
         "%%%%HiResBoundingBox: 0 0 %s %s" % (_f(W * sc), _f(H * sc)),
         "%%%%Title: %s" % scene.title,
         "%%Creator: pixel2vector",
         "%%EndComments",
         "gsave",
         "0 %s translate %s %s scale" % (_f(H * sc), _f(sc), _f(-sc))]

    def col(label, stroke=False):
        hx = scene.hexcol(label)
        if hx.upper() in scene.cmyk:
            return "%s %s %s %s setcmykcolor" % tuple(_f(v / 100.0) for v in scene.cmyk[hx.upper()])
        r, g, b = scene.palette[label]
        return "%s %s %s setrgbcolor" % (_f(r), _f(g), _f(b))

    if scene.mode == "outline":
        L.append("%s setlinewidth" % _f(0.25 / sc))
        for ch, lab in scene.outline_chains():
            seg = _eps_loop(ch.segs, k)
            if not ch.closed and seg.endswith("closepath") and ch.segs[0][0] not in ("CIRCLE", "ELLIPSE"):
                seg = seg[: -len("closepath")]
            L += [col(lab), "newpath", seg, "stroke"]
    else:
        regs = scene.paint_order if scene.mode == "stacked" else sorted(scene.visible, key=lambda r: r.label)
        for r in regs:
            L.append(col(r.label))
            L.append("newpath")
            for l in scene.loops_for(r):
                L.append(_eps_loop(l.segs, k))
            L.append("eofill")
    L += ["grestore", "showpage", "%%EOF"]
    with open(path, "w", encoding="latin-1") as f:
        f.write("\n".join(L) + "\n")


# ================================================================ DXF (true primitives)
def write_dxf(scene, path):
    import ezdxf
    from ezdxf import colors as dxfcolors
    k = scene.s
    H = scene.H
    mm = scene.pt_per_px * 25.4 / 72.0  # mm per source pixel

    def P(p):
        return (p[0] / k * mm, (H - p[1] / k) * mm)

    doc = ezdxf.new("R2010", setup=False)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    layers = {}

    def layer(label):
        if label not in layers:
            hx = scene.hexcol(label)
            name = "C%02d_%s" % (len(layers) + 1, hx[1:])
            ly = doc.layers.add(name)
            r, g, b = [int(round(v * 255)) for v in scene.palette[label]]
            ly.rgb = (r, g, b)
            layers[label] = name
        return layers[label]

    if scene.mode == "outline":
        items = scene.outline_chains()
    else:
        # filled modes in CAD terms: every visible region's own closed contours (shared edges duplicated)
        items = []
        for r in scene.visible:
            for l in scene.loops_for(r):
                items.append((type("X", (), {"segs": l.segs})(), r.label))
    for ch, lab in items:
        ly = {"layer": layer(lab)}
        for sg in ch.segs:
            t = sg[0]
            if t == "L":
                msp.add_line(P(sg[1]), P(sg[2]), dxfattribs=ly)
            elif t == "C":
                msp.add_open_spline([P(sg[1]), P(sg[2]), P(sg[3]), P(sg[4])], degree=3, dxfattribs=ly)
            elif t == "A":
                _, p0, p1, c, R, a0, sw = sg
                a1 = a0 + sw
                if sw > 0:
                    st, en = -math.degrees(a1), -math.degrees(a0)
                else:
                    st, en = -math.degrees(a0), -math.degrees(a1)
                msp.add_arc(P(c), R / k * mm, st % 360, en % 360, dxfattribs=ly)
            elif t == "CIRCLE":
                msp.add_circle(P(sg[1]), sg[2] / k * mm, dxfattribs=ly)
            elif t == "ELLIPSE":
                _, c, a, b, th, d = sg
                if b > a:
                    a, b = b, a
                    th += math.pi / 2
                major = (a / k * mm * math.cos(th), -a / k * mm * math.sin(th))
                msp.add_ellipse(P(c), major_axis=major, ratio=max(1e-6, min(1.0, b / a)), dxfattribs=ly)
    doc.saveas(path)
