import functools
import math
import numpy as np

from . import core, dn, d3, ease, backend as bk

# Constants

ORIGIN = np.array((0, 0))

X = np.array((1, 0))
Y = np.array((0, 1))

UP = Y

# SDF Class

_ops = {}

class SDF2:
    def __init__(self, f):
        self.f = f
    def __call__(self, p):
        res = self.f(p)
        if bk.is_tensor(res):
            return res.reshape(-1, 1)
        return res.reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        self._k = k
        return self

def sdf2(f):
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    return wrapper

def op2(f):
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op23(f):
    def wrapper(*args, **kwargs):
        return d3.SDF3(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

# Helpers

def _length(a):
    return bk.norm(a, axis=-1)

def _normalize(a):
    return bk.normalize(a)

def _dot(a, b):
    return bk.dot(a, b)

def _vec(*arrs):
    return bk.vec(*arrs)

_min = bk.minimum
_max = bk.maximum

# Primitives

@sdf2
def circle(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - bk.as_tensor(center, p)) - radius
    return f

@sdf2
def line(normal=UP, point=ORIGIN):
    normal = _normalize(normal)
    def f(p):
        return _dot(bk.as_tensor(point, p) - p, normal)
    return f

@sdf2
def slab(x0=None, y0=None, x1=None, y1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(line(X, (x0, 0)))
    if x1 is not None:
        fs.append(line(-X, (x1, 0)))
    if y0 is not None:
        fs.append(line(Y, (0, y0)))
    if y1 is not None:
        fs.append(line(-Y, (0, y1)))
    return intersection(*fs, k=k)

@sdf2
def rectangle(size=1, center=ORIGIN, a=None, b_pt=None):
    if a is not None and b_pt is not None:
        a = np.array(a)
        b_pt = np.array(b_pt)
        size = b_pt - a
        center = a + size / 2
        return rectangle(size, center)
    size = np.array(size)
    def f(p):
        q = bk.abs(p - bk.as_tensor(center, p)) - bk.as_tensor(size / 2, p)
        return _length(_max(q, 0)) + _min(bk.amax(q, axis=-1), 0)
    return f

@sdf2
def rounded_rectangle(size, radius, center=ORIGIN):
    size = np.array(size)
    try:
        r0, r1, r2, r3 = radius
    except TypeError:
        r0 = r1 = r2 = r3 = radius
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        if bk.is_tensor(p):
            import torch
            r = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype).unsqueeze(-1)
            c0 = (x > 0) & (y > 0)
            c1 = (x > 0) & (y <= 0)
            c2 = (x <= 0) & (y <= 0)
            c3 = (x <= 0) & (y > 0)
            r[c0] = r0
            r[c1] = r1
            r[c2] = r2
            r[c3] = r3
        else:
            r = np.zeros(len(p)).reshape((-1, 1))
            r[np.logical_and(x > 0, y > 0)] = r0
            r[np.logical_and(x > 0, y <= 0)] = r1
            r[np.logical_and(x <= 0, y <= 0)] = r2
            r[np.logical_and(x <= 0, y > 0)] = r3
        q = bk.abs(p) - bk.as_tensor(size / 2, p) + r
        if bk.is_tensor(p):
            return (
                _min(_max(q[..., 0], q[..., 1]), 0).unsqueeze(-1) +
                _length(_max(q, 0)).unsqueeze(-1) - r)
        return (
            _min(_max(q[:,0], q[:,1]), 0).reshape((-1, 1)) +
            _length(_max(q, 0)).reshape((-1, 1)) - r)
    return f

@sdf2
def equilateral_triangle(r=1):
    k = math.sqrt(3)
    def f(p):
        x = bk.abs(p[..., 0]) - r
        y = p[..., 1] + r / k
        cond = (x + k * y) > 0
        qx = (x - k * y) / 2
        qy = (-k * x - y) / 2
        px = bk.where(cond, qx, x)
        py = bk.where(cond, qy, y)
        px = px - bk.clip(px, -2 * r, 0)
        p_vec = _vec(px, py)
        return -_length(p_vec) * bk.sign(py)
    return f

@sdf2
def hexagon(r):
    r_val = r * math.sqrt(3) / 2
    k0 = -math.sqrt(3) / 2
    k1 = 0.5
    k2 = math.tan(math.pi / 6)
    def f(p):
        p_abs = bk.abs(p)
        k_vec = bk.as_tensor([k0, k1], p)
        dot_val = _dot(p_abs, k_vec)
        min_dot = _min(dot_val, 0)
        if bk.is_tensor(min_dot):
            min_dot_col = min_dot.unsqueeze(-1)
        else:
            min_dot_col = np.reshape(min_dot, (-1, 1))
        p2 = p_abs - 2 * k_vec * min_dot_col
        px = p2[..., 0] - bk.clip(p2[..., 0], -k2 * r_val, k2 * r_val)
        py = p2[..., 1] - r_val
        p3 = _vec(px, py)
        return _length(p3) * bk.sign(py)
    return f

@sdf2
def regular_polygon(n, r):
    if n < 3:
        raise ValueError("n must be at least 3")
    an = math.pi / n
    he = r * math.cos(an)
    tan_an = math.tan(an)
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        ang = bk.arctan2(y, x)
        ang_fold = ang - 2 * an * bk.floor((ang + an) / (2 * an))
        d = bk.hypot(x, y)
        px = d * bk.cos(ang_fold)
        py = bk.abs(d * bk.sin(ang_fold))
        py_clipped = bk.clip(py, 0, he * tan_an)
        dx = px - he
        dy = py - py_clipped
        dist = bk.hypot(dx, dy)
        sign_val = bk.where(px < he, -1.0, 1.0)
        return sign_val * dist
    return f

@sdf2
def rounded_x(w, r):
    def f(p):
        p_abs = bk.abs(p)
        q_val = _min(p_abs[..., 0] + p_abs[..., 1], w) * 0.5
        q_vec = _vec(q_val, q_val)
        return _length(p_abs - q_vec) - r
    return f

@sdf2
def vesica(r, d):
    b = math.sqrt(r * r - d * d)
    def f(p):
        p_abs = bk.abs(p)
        x = p_abs[..., 0]
        y = p_abs[..., 1]
        cond = (y - b) * d > x * b
        p_b = _vec(x, y - b)
        p_d = _vec(x + d, y)
        d1 = _length(p_b)
        d2 = _length(p_d) - r
        return bk.where(cond, d1, d2)
    return f

@sdf2
def polygon(points):
    pts = [np.array(pt, dtype=float) for pt in points]
    n = len(pts)
    def f(p):
        v0 = bk.as_tensor(pts[0], p)
        d = _dot(p - v0, p - v0)
        s = 0 * p[..., 0] + 1.0
        px = p[..., 0]
        py = p[..., 1]
        for i in range(n):
            j = (i + n - 1) % n
            vi = pts[i]
            vj = pts[j]
            e = vj - vi
            e_sq = np.dot(e, e)
            vi_t = bk.as_tensor(vi, p)
            e_t = bk.as_tensor(e, p)
            w_t = p - vi_t
            t = bk.clip(_dot(w_t, e_t) / e_sq, 0, 1)
            if bk.is_tensor(t):
                t_col = t.unsqueeze(-1)
            else:
                t_col = np.reshape(t, (-1, 1))
            b_t = w_t - e_t * t_col
            dist_sq = _dot(b_t, b_t)
            d = _min(d, dist_sq)
            
            c1 = py >= vi[1]
            c2 = py < vj[1]
            c3 = (e[0] * (py - vi[1])) > (e[1] * (px - vi[0]))
            c = c1 & c2 & c3
            c_inv = (~c1) & (~c2) & (~c3)
            flip = c | c_inv
            s = bk.where(flip, -s, s)
        return s * bk.sqrt(d)
    return f

@sdf2
def line_segment(a, b, r=0):
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    ba_arr = b_arr - a_arr
    ba_sq = np.dot(ba_arr, ba_arr)
    def f(p):
        a_t = bk.as_tensor(a_arr, p)
        ba_t = bk.as_tensor(ba_arr, p)
        pa = p - a_t
        h = bk.clip(_dot(pa, ba_t) / ba_sq, 0, 1)
        if bk.is_tensor(h):
            h_col = h.unsqueeze(-1)
        else:
            h_col = np.reshape(h, (-1, 1))
        d_vec = pa - ba_t * h_col
        return _length(d_vec) - r
    return f

@sdf2
def bezier(a, b, c, r=0, steps=65):
    a_pt = np.array(a, dtype=float)
    b_pt = np.array(b, dtype=float)
    c_pt = np.array(c, dtype=float)
    t_vals = np.linspace(0, 1, steps)
    curve_pts = np.stack([(1 - t)**2 * a_pt + 2 * (1 - t) * t * b_pt + t**2 * c_pt for t in t_vals], axis=0)
    def f(p):
        curve_t = bk.as_tensor(curve_pts, p)
        if bk.is_tensor(p):
            diff = p.unsqueeze(1) - curve_t.unsqueeze(0)
            dists = bk.norm(diff, axis=-1)
            min_dist = dists.min(dim=1).values
        else:
            diff = p[:, np.newaxis, :] - curve_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            min_dist = np.min(dists, axis=1)
        return min_dist - r
    return f

# Positioning

@op2
def translate(other, offset):
    def f(p):
        return other(p - bk.as_tensor(offset, p))
    return f

@op2
def scale(other, factor):
    try:
        x, y = factor
    except TypeError:
        x = y = factor
    s = (x, y)
    m = min(x, y)
    def f(p):
        return other(p / bk.as_tensor(s, p)) * m
    return f

@op2
def rotate(other, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    matrix = np.array([
        [c, -s],
        [s, c],
    ]).T
    def f(p):
        return other(bk.matmul(p, matrix))
    return f

@op2
def circular_array(other, count, k=None):
    angles = [i / count * 2 * math.pi for i in range(count)]
    K = k if k is not None else getattr(other, '_k', None)
    return union(*[other.rotate(a) for a in angles], k=K)

# Alterations

@op2
def elongate(other, size):
    def f(p):
        q = bk.abs(p) - bk.as_tensor(size, p)
        x = q[..., 0]
        y = q[..., 1]
        if bk.is_tensor(x):
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
        else:
            x = x.reshape((-1, 1))
            y = y.reshape((-1, 1))
        w = _min(_max(x, y), 0)
        return other(_max(q, 0)) + w
    return f

# 2D => 3D Operations

@op23
def extrude(other, h):
    def f(p):
        d_val = other(p[..., [0, 1]])
        w = _vec(d_val.reshape(-1), bk.abs(p[..., 2]) - h / 2)
        return _min(_max(w[..., 0], w[..., 1]), 0) + _length(_max(w, 0))
    return f

@op23
def extrude_to(a, b, h, e=ease.linear):
    def f(p):
        xy = p[..., [0, 1]]
        z = p[..., 2]
        d1 = a(xy)
        d2 = b(xy)
        t = e(bk.clip(z / h, -0.5, 0.5) + 0.5)
        if bk.is_symbolic(t):
            t_col = t
        elif bk.is_tensor(t):
            t_col = t.unsqueeze(-1)
        else:
            t_col = np.reshape(t, (-1, 1))
        d = d1 + (d2 - d1) * t_col
        w = _vec(d.reshape(-1), bk.abs(z) - h / 2)
        return _min(_max(w[..., 0], w[..., 1]), 0) + _length(_max(w, 0))
    return f

@op23
def revolve(other, offset=0):
    def f(p):
        xy = p[..., [0, 1]]
        q = _vec(_length(xy) - offset, p[..., 2])
        return other(q)
    return f

# Common

union = op2(dn.union)
difference = op2(dn.difference)
intersection = op2(dn.intersection)
blend = op2(dn.blend)
negate = op2(dn.negate)
dilate = op2(dn.dilate)
erode = op2(dn.erode)
shell = op2(dn.shell)
repeat = op2(dn.repeat)

def _poisson_disc_2d(domain, r_min, r_max=None, k_samples=30, variable_radius=None,
                     containment_mode='strict', seed=None, bounds=None):
    if domain is None:
        raise ValueError("domain (an SDF2 or SDF3 object) must be provided to specify the Poisson field boundary.")
    if containment_mode not in ('strict', 'center'):
        raise ValueError(f"Invalid containment_mode '{containment_mode}'. Expected 'strict' or 'center'.")

    rng = np.random.default_rng(seed)

    if bounds is None:
        bounds = core._estimate_bounds(domain, dim=2)
    (x0, y0), (x1, y1) = bounds

    if r_max is None:
        r_max = r_min

    def get_radius(p):
        if variable_radius is not None:
            try:
                val = float(variable_radius(p, domain))
            except TypeError:
                val = float(variable_radius(p))
            if r_max is not None and r_max > r_min:
                return float(np.clip(val, r_min, r_max))
            return max(r_min, val)
        return r_min

    def is_contained(p, r):
        val = float(domain(np.asarray(p, dtype=float).reshape(1, 2))[0, 0])
        if containment_mode == 'strict':
            return val + r <= 1e-6
        return val <= 1e-6

    cell_size = r_min / math.sqrt(2)
    grid = dn.SpatialHashGrid(cell_size)

    samples = []
    radii = []

    initial_pt = None
    initial_r = None
    for _ in range(10000):
        rx = rng.uniform(x0, x1)
        ry = rng.uniform(y0, y1)
        pt = np.array([rx, ry])
        r = get_radius(pt)
        if is_contained(pt, r):
            initial_pt = pt
            initial_r = r
            break

    if initial_pt is None:
        return np.empty((0, 2)), np.empty((0,))

    grid.insert(initial_pt, initial_r, 0)
    samples.append(initial_pt)
    radii.append(initial_r)

    active = [0]

    passes = [r_max]
    if r_max > r_min:
        cur_r = r_max
        while cur_r > r_min + 1e-6:
            next_r = max(r_min, cur_r * 0.75)
            if abs(next_r - cur_r) < 1e-6:
                break
            cur_r = next_r
            passes.append(cur_r)

    for pass_r in passes:
        reseed_tries = 0
        while reseed_tries < 500:
            if not active and pass_r < r_max:
                rx = rng.uniform(x0, x1)
                ry = rng.uniform(y0, y1)
                pt = np.array([rx, ry])
                r = min(pass_r, get_radius(pt))
                if is_contained(pt, r) and grid.is_valid(pt, r, r_max):
                    idx = len(samples)
                    grid.insert(pt, r, idx)
                    samples.append(pt)
                    radii.append(r)
                    active.append(idx)
                    reseed_tries = 0
                else:
                    reseed_tries += 1

            while active:
                idx_in_active = rng.integers(0, len(active))
                parent_idx = active[idx_in_active]
                parent_pt = samples[parent_idx]
                parent_r = radii[parent_idx]

                found = False
                for _ in range(k_samples):
                    angle = rng.uniform(0, 2 * math.pi)
                    r_cand_est = min(pass_r, get_radius(parent_pt)) if r_max > r_min else get_radius(parent_pt)
                    min_dist = parent_r + r_cand_est
                    dist = rng.uniform(min_dist, 2 * min_dist)
                    cand = parent_pt + np.array([math.cos(angle), math.sin(angle)]) * dist

                    if cand[0] < x0 or cand[0] > x1 or cand[1] < y0 or cand[1] > y1:
                        continue

                    cand_r = min(pass_r, get_radius(cand)) if r_max > r_min else get_radius(cand)

                    if is_contained(cand, cand_r) and grid.is_valid(cand, cand_r, r_max):
                        new_idx = len(samples)
                        grid.insert(cand, cand_r, new_idx)
                        samples.append(cand)
                        radii.append(cand_r)
                        active.append(new_idx)
                        found = True
                        break

                if not found:
                    active.pop(idx_in_active)

            if pass_r == r_max:
                break

    if not samples:
        return np.empty((0, 2)), np.empty((0,))

    return np.array(samples), np.array(radii)

def _smooth_reduce_torch(d_matrix, K):
    import torch
    if K is None or K <= 0 or d_matrix.shape[1] <= 1:
        return d_matrix.min(dim=1, keepdim=True).values
    d_sorted, _ = torch.sort(d_matrix, dim=1)
    d_acc = d_sorted[:, 0:1]
    M = d_matrix.shape[1]
    for col in range(1, M):
        d_next = d_sorted[:, col:col+1]
        diff = d_next - d_acc
        if (diff >= K).all():
            break
        h = torch.clamp(0.5 + 0.5 * diff / K, 0.0, 1.0)
        m = d_next + (d_acc - d_next) * h
        d_acc = m - K * h * (1.0 - h)
    return d_acc

def _smooth_reduce_numpy(d_matrix, K):
    if K is None or K <= 0 or d_matrix.shape[1] <= 1:
        return d_matrix.min(axis=1, keepdims=True)
    d_sorted = np.sort(d_matrix, axis=1)
    d_acc = d_sorted[:, 0:1]
    M = d_matrix.shape[1]
    for col in range(1, M):
        d_next = d_sorted[:, col:col+1]
        diff = d_next - d_acc
        if np.all(diff >= K):
            break
        h = np.clip(0.5 + 0.5 * diff / K, 0.0, 1.0)
        m = d_next + (d_acc - d_next) * h
        d_acc = m - K * h * (1.0 - h)
    return d_acc

@op2
def poisson_array(
    other,
    r_min,
    r_max=None,
    domain=None,
    bounds=None,
    k_samples=30,
    variable_radius=None,
    scale_to_radius=False,
    base_radius=None,
    containment_mode='strict',
    seed=None,
    random_rotation=False,
    k=None
):
    if domain is None:
        raise ValueError("domain (an SDF2 or SDF3 object) must be provided to specify the Poisson field boundary.")
    if containment_mode not in ('strict', 'center'):
        raise ValueError(f"Invalid containment_mode '{containment_mode}'. Expected 'strict' or 'center'.")

    is_domain_3d = isinstance(domain, d3.SDF3)
    if not is_domain_3d and not isinstance(domain, SDF2):
        try:
            domain(np.zeros((1, 3)))
            is_domain_3d = True
        except Exception:
            is_domain_3d = False

    if is_domain_3d:
        centers_3d, radii = d3._poisson_disc_3d(
            domain=domain,
            r_min=r_min,
            r_max=r_max,
            k_samples=k_samples,
            variable_radius=variable_radius,
            containment_mode=containment_mode,
            seed=seed,
            bounds=bounds
        )
        if len(centers_3d) > 0:
            centers = centers_3d[:, :2]
        else:
            centers = np.empty((0, 2))
    else:
        centers, radii = _poisson_disc_2d(
            domain=domain,
            r_min=r_min,
            r_max=r_max,
            k_samples=k_samples,
            variable_radius=variable_radius,
            containment_mode=containment_mode,
            seed=seed,
            bounds=bounds
        )

    base_r = base_radius if base_radius is not None else 1.0
    if len(radii) > 0 and scale_to_radius:
        scales = radii / base_r
    else:
        scales = np.ones(len(radii), dtype=np.float32)

    rng = np.random.default_rng(seed)
    if random_rotation and len(centers) > 0:
        angles = rng.uniform(0, 2 * math.pi, len(centers))
    else:
        angles = np.zeros(len(centers), dtype=np.float32)

    matrices = []
    for ang in angles:
        cos_a = math.cos(-ang)
        sin_a = math.sin(-ang)
        matrices.append([[cos_a, -sin_a], [sin_a, cos_a]])
    matrices = np.array(matrices, dtype=np.float32) if len(centers) > 0 else np.empty((0, 2, 2), dtype=np.float32)
    centers = np.array(centers, dtype=np.float32) if len(centers) > 0 else np.empty((0, 2), dtype=np.float32)
    scales = np.array(scales, dtype=np.float32) if len(centers) > 0 else np.empty((0,), dtype=np.float32)

    r_bound = scales * base_r
    K = k if k is not None else getattr(other, '_k', None)

    def f(p):
        if len(centers) == 0:
            if bk.is_tensor(p):
                import torch
                return torch.full(p.shape[:-1] + (1,), 1e9, device=p.device, dtype=p.dtype)
            return np.full(p.shape[:-1] + (1,), 1e9)

        p_2d = p.reshape(-1, 2)
        N = p_2d.shape[0]
        is_t = bk.is_tensor(p)

        if is_t:
            import torch
            p_min = p_2d.min(dim=0).values
            p_max = p_2d.max(dim=0).values
            p_center = (p_min + p_max) * 0.5
            p_half_diag = torch.linalg.norm((p_max - p_min) * 0.5)

            centers_t = bk.as_tensor(centers, p)
            scales_t = bk.as_tensor(scales, p)
            matrices_t = bk.as_tensor(matrices, p)
            r_bound_t = bk.as_tensor(r_bound, p)

            c_dists = torch.linalg.norm(centers_t - p_center, dim=1)
            active = c_dists <= (p_half_diag + r_bound_t + 0.1)
            active_indices = torch.where(active)[0]

            if len(active_indices) == 0:
                active_indices = torch.argmin(c_dists).unsqueeze(0)

            C_act = centers_t[active_indices]
            S_act = scales_t[active_indices]
            M_act = matrices_t[active_indices]
            M = len(active_indices)

            max_pass = 5000000
            if N * M > max_pass:
                chunk_sz = max(1000, max_pass // M)
                chunks = []
                for start_idx in range(0, N, chunk_sz):
                    p_sub = p_2d[start_idx : start_idx + chunk_sz]
                    n_sub = p_sub.shape[0]
                    P_shifted = (p_sub.unsqueeze(1) - C_act.unsqueeze(0)) / S_act.unsqueeze(0).unsqueeze(2)
                    P_rotated = torch.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                    P_flat = P_rotated.reshape(-1, 2)
                    d_flat = other(P_flat)
                    d_matrix = d_flat.reshape(n_sub, M) * S_act.unsqueeze(0)
                    chunks.append(_smooth_reduce_torch(d_matrix, K))
                d_min = torch.cat(chunks, dim=0)
            else:
                P_shifted = (p_2d.unsqueeze(1) - C_act.unsqueeze(0)) / S_act.unsqueeze(0).unsqueeze(2)
                P_rotated = torch.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                P_flat = P_rotated.reshape(-1, 2)
                d_flat = other(P_flat)
                d_matrix = d_flat.reshape(N, M) * S_act.unsqueeze(0)
                d_min = _smooth_reduce_torch(d_matrix, K)

            return d_min.reshape(p.shape[:-1] + (1,))
        else:
            p_min = p_2d.min(axis=0)
            p_max = p_2d.max(axis=0)
            p_center = (p_min + p_max) * 0.5
            p_half_diag = np.linalg.norm((p_max - p_min) * 0.5)

            c_dists = np.linalg.norm(centers - p_center, axis=1)
            active = c_dists <= (p_half_diag + r_bound + 0.1)
            active_indices = np.where(active)[0]

            if len(active_indices) == 0:
                active_indices = np.array([np.argmin(c_dists)])

            C_act = centers[active_indices]
            S_act = scales[active_indices]
            M_act = matrices[active_indices]
            M = len(active_indices)

            max_pass = 5000000
            if N * M > max_pass:
                chunk_sz = max(1000, max_pass // M)
                chunks = []
                for start_idx in range(0, N, chunk_sz):
                    p_sub = p_2d[start_idx : start_idx + chunk_sz]
                    n_sub = p_sub.shape[0]
                    P_shifted = (p_sub[:, None, :] - C_act[None, :, :]) / S_act[None, :, None]
                    P_rotated = np.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                    P_flat = P_rotated.reshape(-1, 2)
                    d_flat = other(P_flat)
                    d_matrix = d_flat.reshape(n_sub, M) * S_act[None, :]
                    chunks.append(_smooth_reduce_numpy(d_matrix, K))
                d_min = np.concatenate(chunks, axis=0)
            else:
                P_shifted = (p_2d[:, None, :] - C_act[None, :, :]) / S_act[None, :, None]
                P_rotated = np.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                P_flat = P_rotated.reshape(-1, 2)
                d_flat = other(P_flat)
                d_matrix = d_flat.reshape(N, M) * S_act[None, :]
                d_min = _smooth_reduce_numpy(d_matrix, K)

            return d_min.reshape(p.shape[:-1] + (1,))

    return f

