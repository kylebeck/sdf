import itertools
import numpy as np
from . import backend as bk

_min = bk.minimum
_max = bk.maximum

def union(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b_elem in bs:
            d2 = b_elem(p)
            K = k or getattr(b_elem, '_k', None)
            if K is None:
                d1 = _min(d1, d2)
            else:
                h = bk.clip(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m - K * h * (1 - h)
        return d1
    return f

def difference(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b_elem in bs:
            d2 = b_elem(p)
            K = k or getattr(b_elem, '_k', None)
            if K is None:
                d1 = _max(d1, -d2)
            else:
                h = bk.clip(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
                m = d1 + (-d2 - d1) * h
                d1 = m + K * h * (1 - h)
        return d1
    return f

def intersection(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b_elem in bs:
            d2 = b_elem(p)
            K = k or getattr(b_elem, '_k', None)
            if K is None:
                d1 = _max(d1, d2)
            else:
                h = bk.clip(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m + K * h * (1 - h)
        return d1
    return f

def blend(a, *bs, k=0.5):
    def f(p):
        d1 = a(p)
        for b_elem in bs:
            d2 = b_elem(p)
            K = k or getattr(b_elem, '_k', None)
            d1 = K * d2 + (1 - K) * d1
        return d1
    return f

def negate(other):
    def f(p):
        return -other(p)
    return f

def dilate(other, r):
    def f(p):
        return other(p) - r
    return f

def erode(other, r):
    def f(p):
        return other(p) + r
    return f

def shell(other, thickness):
    def f(p):
        return bk.abs(other(p)) - thickness / 2
    return f

def repeat(other, spacing, count=None, padding=0, k=None):
    count = np.array(count) if count is not None else None
    spacing = np.array(spacing)
    K = k if k is not None else getattr(other, '_k', None)

    def neighbors(dim, padding, spacing):
        try:
            padding = [padding[i] for i in range(dim)]
        except Exception:
            padding = [padding] * dim
        try:
            spacing = [spacing[i] for i in range(dim)]
        except Exception:
            spacing = [spacing] * dim
        for i, s in enumerate(spacing):
            if s == 0:
                padding[i] = 0
        axes = [list(range(-p, p + 1)) for p in padding]
        return list(itertools.product(*axes))

    def f(p):
        sp = bk.as_tensor(spacing, p)
        q = bk.where(sp != 0, p / sp, bk.as_tensor(0.0, p))
        if count is None:
            index = bk.round(q)
        else:
            cnt = bk.as_tensor(count, p)
            index = bk.clip(bk.round(q), -cnt, cnt)

        indexes = [index + bk.as_tensor(n, p) for n in neighbors(p.shape[-1], padding, spacing)]
        A = [other(p - sp * i) for i in indexes]
        a = A[0]
        for b_item in A[1:]:
            if K is not None and K > 0:
                h = bk.clip(0.5 + 0.5 * (b_item - a) / K, 0, 1)
                m = b_item + (a - b_item) * h
                a = m - K * h * (1 - h)
            else:
                a = _min(a, b_item)
        return a
    return f

class SpatialHashGrid:
    def __init__(self, cell_size):
        self.cell_size = float(cell_size)
        self.grid = {}
        self.points = []
        self.radii = []

    def _cell_coords(self, point):
        return tuple(np.floor(np.asarray(point) / self.cell_size).astype(int))

    def insert(self, point, radius, idx):
        cell = self._cell_coords(point)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(idx)
        self.points.append(point)
        self.radii.append(radius)

    def is_valid(self, candidate, candidate_radius, max_search_radius):
        cand = np.asarray(candidate)
        cell = self._cell_coords(cand)
        search_radius = candidate_radius + max_search_radius
        search_range = max(1, int(np.ceil(search_radius / self.cell_size)))
        dim = len(cand)
        ranges = [range(cell[i] - search_range, cell[i] + search_range + 1) for i in range(dim)]
        for neighbor_cell in itertools.product(*ranges):
            if neighbor_cell in self.grid:
                for idx in self.grid[neighbor_cell]:
                    pt = self.points[idx]
                    r = self.radii[idx]
                    dist_sq = np.sum((cand - pt) ** 2)
                    req_dist = candidate_radius + r
                    if dist_sq < req_dist * req_dist:
                        return False
        return True

    def query_range(self, min_pt, max_pt):
        dim = len(min_pt)
        min_cell = np.floor(np.asarray(min_pt) / self.cell_size).astype(int)
        max_cell = np.floor(np.asarray(max_pt) / self.cell_size).astype(int)
        ranges = [range(min_cell[i], max_cell[i] + 1) for i in range(dim)]
        indices = []
        for cell in itertools.product(*ranges):
            if cell in self.grid:
                indices.extend(self.grid[cell])
        return indices


