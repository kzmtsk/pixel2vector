"""Stage 2: the Vector Graph topology.

Boundaries live on the pixel-corner lattice. Every boundary between two regions is stored
exactly once as a *chain* (a maximal run of lattice edges separating the same pair of regions),
running between *nodes* (lattice vertices where 3+ regions meet, or checkerboard crossings).
Both neighbouring regions reference the same chain, so their outlines coincide exactly —
no hairline gaps, no overlaps, and a cutter traces each boundary once.
"""
import numpy as np

OUTSIDE = -1  # region id for the area outside the image frame


class Chain:
    __slots__ = ("pts", "right", "left", "closed", "segs", "idx")

    def __init__(self, pts, right, left, closed):
        self.pts = pts          # (n,2) float array of lattice points (x,y), y down
        self.right = right      # region id on the right-hand side when walking pts forward
        self.left = left
        self.closed = closed
        self.segs = None        # fitted geometry (list of segments), filled by fit stage
        self.idx = -1


def build_graph(comp):
    """comp: HxW int32 region ids (>=0). Returns list[Chain]."""
    H, W = comp.shape
    P = np.full((H + 2, W + 2), OUTSIDE, np.int32)
    P[1:-1, 1:-1] = comp
    # horizontal lattice edge (x,y)->(x+1,y): pixel above = P[y, x+1], below = P[y+1, x+1]
    up = P[0:H + 1, 1:W + 1]
    dn = P[1:H + 2, 1:W + 1]
    has_h = up != dn                    # shape (H+1, W)
    # vertical lattice edge (x,y)->(x,y+1): pixel left = P[y+1, x], right = P[y+1, x+1]
    lf = P[1:H + 1, 0:W + 1]
    rt = P[1:H + 1, 1:W + 2]
    has_v = lf != rt                    # shape (H, W+1)

    deg = np.zeros((H + 1, W + 1), np.int8)
    deg[:, :-1] += has_h
    deg[:, 1:] += has_h
    deg[:-1, :] += has_v
    deg[1:, :] += has_v

    vis_h = np.zeros_like(has_h)
    vis_v = np.zeros_like(has_v)

    # python-side lookups (fast enough; arrays converted to lists of rows is slower)
    def edges_at(x, y):
        out = []
        if x < W and has_h[y, x] and not vis_h[y, x]:
            out.append((1, 0))
        if x > 0 and has_h[y, x - 1] and not vis_h[y, x - 1]:
            out.append((-1, 0))
        if y < H and has_v[y, x] and not vis_v[y, x]:
            out.append((0, 1))
        if y > 0 and has_v[y - 1, x] and not vis_v[y - 1, x]:
            out.append((0, -1))
        return out

    def mark(x, y, d):
        dx, dy = d
        if dx == 1:
            vis_h[y, x] = True
        elif dx == -1:
            vis_h[y, x - 1] = True
        elif dy == 1:
            vis_v[y, x] = True
        else:
            vis_v[y - 1, x] = True

    def sides(x, y, d):
        """(right, left) region ids for edge leaving (x,y) in direction d (image coords, y down)."""
        dx, dy = d
        if dx == 1:      # right side is below
            return int(dn[y, x]), int(up[y, x])
        if dx == -1:     # right side is above
            return int(up[y, x - 1]), int(dn[y, x - 1])
        if dy == 1:      # right side is left pixel
            return int(lf[y, x]), int(rt[y, x])
        return int(rt[y - 1, x]), int(lf[y - 1, x])

    def walk(x0, y0, d0, stop_at_nodes=True):
        right, left = sides(x0, y0, d0)
        pts = [(x0, y0)]
        x, y, d = x0, y0, d0
        while True:
            mark(x, y, d)
            x, y = x + d[0], y + d[1]
            pts.append((x, y))
            if (x, y) == (x0, y0):
                return pts, right, left, True
            if stop_at_nodes and deg[y, x] != 2:
                return pts, right, left, False
            nxt = edges_at(x, y)
            if not nxt:
                return pts, right, left, False
            # at a degree-2 vertex there is exactly one unvisited continuation
            d = nxt[0]

    chains = []
    ny, nx = np.nonzero(deg > 2)
    for y, x in zip(ny.tolist(), nx.tolist()):
        for d in edges_at(x, y):
            # edges may have been consumed by a previous walk in this loop
            dx, dy = d
            if dx == 1 and vis_h[y, x]:
                continue
            if dx == -1 and vis_h[y, x - 1]:
                continue
            if dy == 1 and vis_v[y, x]:
                continue
            if dy == -1 and vis_v[y - 1, x]:
                continue
            pts, r, l, closed = walk(x, y, d)
            chains.append(Chain(np.array(pts, np.float64), r, l, False))

    # remaining edges form closed loops through degree-2 vertices only
    while True:
        hy, hx = np.nonzero(has_h & ~vis_h)
        if len(hy) == 0:
            break
        for y, x in zip(hy.tolist(), hx.tolist()):
            if vis_h[y, x]:
                continue
            pts, r, l, closed = walk(x, y, (1, 0), stop_at_nodes=False)
            arr = np.array(pts[:-1], np.float64)  # drop duplicated start point
            chains.append(Chain(arr, r, l, True))
    for i, c in enumerate(chains):
        c.idx = i
    return chains
