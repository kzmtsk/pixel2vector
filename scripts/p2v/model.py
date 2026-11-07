"""Stage 4: assemble regions (outer boundary + holes) from shared chains."""
import numpy as np
from .fit import reverse_segs
from .graph import OUTSIDE


class Loop:
    __slots__ = ("segs", "area", "neighbours", "chain_ids")

    def __init__(self, segs, area, neighbours, chain_ids):
        self.segs = segs
        self.area = area
        self.neighbours = neighbours
        self.chain_ids = chain_ids


class Region:
    __slots__ = ("id", "label", "area", "outer", "holes", "order")

    def __init__(self, rid, label, area):
        self.id = rid
        self.label = label
        self.area = area
        self.outer = None
        self.holes = []
        self.order = 0


def _poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def assemble(chains, comp_label, comp_area):
    per = {}
    for ch in chains:
        for side, other, fwd in ((ch.right, ch.left, True), (ch.left, ch.right, False)):
            if side == OUTSIDE:
                continue
            per.setdefault(side, []).append((ch, fwd, other))

    regions = []
    for rid, items in per.items():
        reg = Region(rid, int(comp_label[rid]), int(comp_area[rid]))
        loops = []
        open_items = []
        for ch, fwd, other in items:
            if ch.closed:
                pts = ch.pts if fwd else ch.pts[::-1]
                segs = ch.segs if fwd else reverse_segs(ch.segs)
                loops.append(Loop(segs, _poly_area(pts), {other}, [ch.idx]))
            else:
                open_items.append((ch, fwd, other))
        # link open chains at nodes
        starts = {}
        for it in open_items:
            ch, fwd, _ = it
            p = ch.pts[0] if fwd else ch.pts[-1]
            starts.setdefault((int(p[0]), int(p[1])), []).append(it)
        used = set()

        def out_dir(it):
            ch, fwd, _ = it
            d = (ch.pts[1] - ch.pts[0]) if fwd else (ch.pts[-2] - ch.pts[-1])
            return d

        def in_dir(it):
            ch, fwd, _ = it
            return (ch.pts[-1] - ch.pts[-2]) if fwd else (ch.pts[0] - ch.pts[1])

        for it0 in open_items:
            if id(it0) in used:
                continue
            seq = [it0]
            used.add(id(it0))
            cur = it0
            guard = 0
            while True:
                guard += 1
                if guard > 1_000_000:
                    break
                ch, fwd, _ = cur
                e = ch.pts[-1] if fwd else ch.pts[0]
                key = (int(e[0]), int(e[1]))
                s0 = it0[0].pts[0] if it0[1] else it0[0].pts[-1]
                if key == (int(s0[0]), int(s0[1])):
                    break
                cands = [c for c in starts.get(key, []) if id(c) not in used]
                if not cands:
                    break
                if len(cands) > 1:
                    din = in_dir(cur)

                    def rank(c):
                        dout = out_dir(c)
                        cr = din[0] * dout[1] - din[1] * dout[0]
                        dt = din[0] * dout[0] + din[1] * dout[1]
                        # right turn (cr>0 in y-down) first, then straight, then left
                        return (0 if cr > 0 else (1 if dt > 0 else 2))
                    cands.sort(key=rank)
                nxt = cands[0]
                used.add(id(nxt))
                seq.append(nxt)
                cur = nxt
            # build loop (sequence may close only when the end reaches start node)
            segs, pts, nbs, cids = [], [], set(), []
            for ch, fwd, other in seq:
                segs += ch.segs if fwd else reverse_segs(ch.segs)
                pts.append(ch.pts if fwd else ch.pts[::-1])
                nbs.add(other)
                cids.append(ch.idx)
            P = np.vstack(pts)
            loops.append(Loop(segs, _poly_area(P), nbs, cids))
        if not loops:
            continue
        # split sequences that closed early are already separate; outer = largest |area|
        loops.sort(key=lambda l: -abs(l.area))
        reg.outer = loops[0]
        reg.holes = loops[1:]
        regions.append(reg)
    return regions
