import functools
import math
import numpy as np

from . import core, dn, d2, ease, backend as bk

# Constants

ORIGIN = np.array((0, 0, 0))

X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))

UP = Z

# SDF Class

_ops = {}

class SDF3:
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
        return getattr(self.f, name)
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
    def generate(self, *args, **kwargs):
        return core.generate(self, *args, **kwargs)
    def save(self, path, *args, **kwargs):
        return core.save(path, self, *args, **kwargs)
    def show_slice(self, *args, **kwargs):
        return core.show_slice(self, *args, **kwargs)

def sdf3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    return wrapper

def op3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op32(f):
    def wrapper(*args, **kwargs):
        return d2.SDF2(f(*args, **kwargs))
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

@sdf3
def sphere(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - bk.as_tensor(center, p)) - radius
    return f

@sdf3
def plane(normal=UP, point=ORIGIN):
    normal = _normalize(normal)
    def f(p):
        return _dot(bk.as_tensor(point, p) - p, normal)
    return f

@sdf3
def slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None:
        fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None:
        fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None:
        fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None:
        fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None:
        fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)

@sdf3
def box(size=1, center=ORIGIN, a=None, b_pt=None):
    if a is not None and b_pt is not None:
        a = np.array(a)
        b_pt = np.array(b_pt)
        size = b_pt - a
        center = a + size / 2
        return box(size, center)
    size = np.array(size)
    def f(p):
        q = bk.abs(p - bk.as_tensor(center, p)) - bk.as_tensor(size / 2, p)
        return _length(_max(q, 0)) + _min(bk.amax(q, axis=-1), 0)
    return f

@sdf3
def rounded_box(size, radius):
    size = np.array(size)
    def f(p):
        q = bk.abs(p) - bk.as_tensor(size / 2, p) + radius
        return _length(_max(q, 0)) + _min(bk.amax(q, axis=-1), 0) - radius
    return f

@sdf3
def wireframe_box(size, thickness):
    size = np.array(size)
    def g(a_val, b_val, c_val):
        return _length(_max(_vec(a_val, b_val, c_val), 0)) + _min(_max(a_val, _max(b_val, c_val)), 0)
    def f(p):
        p_sub = bk.abs(p) - bk.as_tensor(size / 2 + thickness / 2, p)
        q = bk.abs(p_sub + thickness / 2) - thickness / 2
        px, py, pz = p_sub[..., 0], p_sub[..., 1], p_sub[..., 2]
        qx, qy, qz = q[..., 0], q[..., 1], q[..., 2]
        return _min(_min(g(px, qy, qz), g(qx, py, qz)), g(qx, qy, pz))
    return f

@sdf3
def torus(r1, r2):
    def f(p):
        xy = p[..., [0, 1]]
        z = p[..., 2]
        a = _length(xy) - r1
        b_val = _length(_vec(a, z)) - r2
        return b_val
    return f

@sdf3
def capsule(a_pt, b_pt, radius):
    a_pt = np.array(a_pt)
    b_pt = np.array(b_pt)
    def f(p):
        a_t = bk.as_tensor(a_pt, p)
        b_t = bk.as_tensor(b_pt, p)
        pa = p - a_t
        ba = b_t - a_t
        h = bk.clip(_dot(pa, ba) / _dot(ba, ba), 0, 1)
        if bk.is_tensor(h):
            h = h.unsqueeze(-1)
        else:
            h = h.reshape((-1, 1))
        return _length(pa - ba * h) - radius
    return f

@sdf3
def cylinder(radius):
    def f(p):
        return _length(p[..., [0, 1]]) - radius
    return f

@sdf3
def capped_cylinder(a_pt, b_pt, radius):
    a_pt = np.array(a_pt)
    b_pt = np.array(b_pt)
    def f(p):
        a_t = bk.as_tensor(a_pt, p)
        b_t = bk.as_tensor(b_pt, p)
        ba = b_t - a_t
        pa = p - a_t
        baba = _dot(ba, ba)
        paba = _dot(pa, ba)
        if bk.is_tensor(paba):
            paba = paba.unsqueeze(-1)
            baba = baba.unsqueeze(-1) if bk.is_tensor(baba) else baba
        else:
            paba = paba.reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - radius * baba
        y = bk.abs(paba - baba * 0.5) - baba * 0.5
        if not bk.is_tensor(x):
            x = x.reshape((-1, 1))
            y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = bk.where(
            _max(x, y) < 0,
            -_min(x2, y2),
            bk.where(x > 0, x2, 0) + bk.where(y > 0, y2, 0))
        return bk.sign(d) * bk.sqrt(bk.abs(d)) / baba
    return f

@sdf3
def rounded_cylinder(ra, rb, h):
    def f(p):
        d_val = _vec(
            _length(p[..., [0, 1]]) - ra + rb,
            bk.abs(p[..., 2]) - h / 2 + rb)
        return (
            _min(_max(d_val[..., 0], d_val[..., 1]), 0) +
            _length(_max(d_val, 0)) - rb)
    return f

@sdf3
def capped_cone(a_pt, b_pt, ra, rb):
    a_pt = np.array(a_pt)
    b_pt = np.array(b_pt)
    def f(p):
        a_t = bk.as_tensor(a_pt, p)
        b_t = bk.as_tensor(b_pt, p)
        rba = rb - ra
        baba = _dot(b_t - a_t, b_t - a_t)
        papa = _dot(p - a_t, p - a_t)
        paba = _dot(p - a_t, b_t - a_t) / baba
        x = bk.sqrt(papa - paba * paba * baba)
        cax = _max(0, x - bk.where(paba < 0.5, ra, rb))
        cay = bk.abs(paba - 0.5) - 0.5
        k_val = rba * rba + baba
        f_val = bk.clip((rba * (x - ra) + paba * baba) / k_val, 0, 1)
        cbx = x - ra - f_val * rba
        cby = paba - f_val
        s = bk.where((cbx < 0) & (cay < 0), -1, 1)
        return s * bk.sqrt(_min(
            cax * cax + cay * cay * baba,
            cbx * cbx + cby * cby * baba))
    return f

@sdf3
def rounded_cone(r1, r2, h):
    def f(p):
        q = _vec(_length(p[..., [0, 1]]), p[..., 2])
        b_val = (r1 - r2) / h
        a_val = math.sqrt(1 - b_val * b_val)
        k_val = _dot(q, _vec(-b_val, a_val))
        c1 = _length(q) - r1
        c2 = _length(q - bk.as_tensor([0, h], q)) - r2
        c3 = _dot(q, _vec(a_val, b_val)) - r1
        return bk.where(k_val < 0, c1, bk.where(k_val > a_val * h, c2, c3))
    return f

@sdf3
def ellipsoid(size):
    size = np.array(size)
    def f(p):
        sz = bk.as_tensor(size, p)
        k0 = _length(p / sz)
        k1 = _length(p / (sz * sz))
        return k0 * (k0 - 1) / k1
    return f

@sdf3
def pyramid(h):
    def f(p):
        a_val = bk.abs(p[..., [0, 1]]) - 0.5
        w = a_val[..., 1] > a_val[..., 0]
        if bk.is_tensor(a_val):
            a_val = bk.where(w.unsqueeze(-1), a_val[..., [1, 0]], a_val)
        else:
            a_val[w] = a_val[:, [1, 0]][w]
        px = a_val[..., 0]
        py = p[..., 2]
        pz = a_val[..., 1]
        m2 = h * h + 0.25
        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py
        s = _max(-qx, 0)
        t = bk.clip((qy - 0.5 * pz) / (m2 + 0.25), 0, 1)
        a_sq = m2 * (qx + s) ** 2 + qy * qy
        b_sq = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = bk.where(
            _min(qy, -qx * m2 - qy * 0.5) > 0,
            0, _min(a_sq, b_sq))
        return bk.sqrt((d2 + qz * qz) / m2) * bk.sign(_max(qz, -py))
    return f

# Platonic Solids

@sdf3
def tetrahedron(r):
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return (_max(bk.abs(x + y) - z, bk.abs(x - y) + z) - r) / math.sqrt(3)
    return f

@sdf3
def octahedron(r):
    def f(p):
        if bk.is_tensor(p):
            import torch
            return (torch.sum(bk.abs(p), dim=-1) - r) * math.tan(math.radians(30))
        return (np.sum(np.abs(p), axis=-1) - r) * np.tan(np.radians(30))
    return f

@sdf3
def dodecahedron(r):
    x, y, z = _normalize(((1 + math.sqrt(5)) / 2, 1, 0))
    def f(p):
        p_scaled = bk.abs(p / r)
        vec_xyz = bk.as_tensor((x, y, z), p)
        vec_zxy = bk.as_tensor((z, x, y), p)
        vec_yzx = bk.as_tensor((y, z, x), p)
        a_val = _dot(p_scaled, vec_xyz)
        b_val = _dot(p_scaled, vec_zxy)
        c_val = _dot(p_scaled, vec_yzx)
        q = (_max(_max(a_val, b_val), c_val) - x) * r
        return q
    return f

@sdf3
def icosahedron(r):
    r *= 0.8506507174597755
    x, y, z = _normalize(((math.sqrt(5) + 3) / 2, 1, 0))
    w = math.sqrt(3) / 3
    def f(p):
        p_scaled = bk.abs(p / r)
        vec_xyz = bk.as_tensor((x, y, z), p)
        vec_zxy = bk.as_tensor((z, x, y), p)
        vec_yzx = bk.as_tensor((y, z, x), p)
        vec_www = bk.as_tensor((w, w, w), p)
        a_val = _dot(p_scaled, vec_xyz)
        b_val = _dot(p_scaled, vec_zxy)
        c_val = _dot(p_scaled, vec_yzx)
        d_val = _dot(p_scaled, vec_www) - x
        return _max(_max(_max(a_val, b_val), c_val) - x, d_val) * r
    return f

# Positioning

@op3
def translate(other, offset):
    def f(p):
        return other(p - bk.as_tensor(offset, p))
    return f

@op3
def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))
    def f(p):
        return other(p / bk.as_tensor(s, p)) * m
    return f

@op3
def multmatrix(other, matrix):
    M = np.array(matrix)
    if M.shape == (3, 4):
        M = np.vstack((M, [0, 0, 0, 1]))
    elif M.shape == (4, 4):
        pass
    else:
        raise ValueError("Matrix must be 3x4 or 4x4")
    
    M_inv = np.linalg.inv(M)
    A = M[:3, :3]
    _, S, _ = np.linalg.svd(A)
    m = np.min(S)
    
    R_inv_T = M_inv[:3, :3].T
    t_inv = M_inv[:3, 3]
    
    def f(p):
        p_orig = bk.matmul(p, R_inv_T)
        p_orig = p_orig + bk.as_tensor(t_inv, p)
        return other(p_orig) * m
    return f

@op3
def rotate(other, angle, vector=Z):
    v = bk.to_numpy(vector)
    x, y, z = _normalize(v)
    s = math.sin(angle)
    c = math.cos(angle)
    m = 1 - c
    matrix = np.array([
        [m*x*x + c, m*x*y + z*s, m*z*x - y*s],
        [m*x*y - z*s, m*y*y + c, m*y*z + x*s],
        [m*z*x + y*s, m*y*z - x*s, m*z*z + c],
    ]).T
    def f(p):
        return other(bk.matmul(p, matrix))
    return f

@op3
def rotate_to(other, a, b_pt):
    a = _normalize(bk.to_numpy(a))
    b_pt = _normalize(bk.to_numpy(b_pt))
    dot_val = np.dot(b_pt, a)
    if dot_val == 1:
        return other
    if dot_val == -1:
        return rotate(other, math.pi, _perpendicular(a))
    angle = math.acos(dot_val)
    v = _normalize(np.cross(b_pt, a))
    return rotate(other, angle, v)

def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

@op3
def orient(other, axis):
    return rotate_to(other, UP, axis)

@op3
def circular_array(other, count, offset=0, k=None):
    K = k if k is not None else getattr(other, '_k', None)
    other = other.translate(X * offset)
    da = 2 * math.pi / count
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        d = bk.hypot(x, y)
        a_val = bk.arctan2(y, x) % da
        d1 = other(_vec(bk.cos(a_val - da) * d, bk.sin(a_val - da) * d, z))
        d2 = other(_vec(bk.cos(a_val) * d, bk.sin(a_val) * d, z))
        if K is not None and K > 0:
            h = bk.clip(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
            m = d2 + (d1 - d2) * h
            return m - K * h * (1 - h)
        return _min(d1, d2)
    return f

# Alterations

@op3
def elongate(other, size):
    def f(p):
        q = bk.abs(p) - bk.as_tensor(size, p)
        x = q[..., 0]
        y = q[..., 1]
        z = q[..., 2]
        if bk.is_tensor(x):
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            z = z.unsqueeze(-1)
        else:
            x = x.reshape((-1, 1))
            y = y.reshape((-1, 1))
            z = z.reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return other(_max(q, 0)) + w
    return f

@op3
def twist(other, k):
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        c = bk.cos(k * z)
        s = bk.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    return f

@op3
def bend(other, k):
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        c = bk.cos(k * x)
        s = bk.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    return f

@op3
def bend_linear(other, p0, p1, v, e=ease.linear):
    p0_arr = np.array(p0)
    p1_arr = np.array(p1)
    v_arr = -np.array(v)
    ab_arr = p1_arr - p0_arr
    ab_sq = np.dot(ab_arr, ab_arr)
    def f(p):
        p0_t = bk.as_tensor(p0_arr, p)
        ab_t = bk.as_tensor(ab_arr, p)
        v_t = bk.as_tensor(v_arr, p)
        t = bk.clip(_dot(p - p0_t, ab_t) / ab_sq, 0, 1)
        t_e = e(t)
        if bk.is_symbolic(t_e):
            pass
        elif bk.is_tensor(t_e):
            t_e = t_e.reshape((-1, 1))
        else:
            t_e = np.reshape(t_e, (-1, 1))
        return other(p + t_e * v_t)
    return f

@op3
def bend_radial(other, r0, r1, dz, e=ease.linear):
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r = bk.hypot(x, y)
        t = bk.clip((r - r0) / (r1 - r0), 0, 1)
        z2 = z - dz * e(t)
        return other(_vec(x, y, z2))
    return f

@op3
def transition_linear(f0, f1, p0=-Z, p1=Z, e=ease.linear):
    p0_arr = np.array(p0)
    p1_arr = np.array(p1)
    ab_arr = p1_arr - p0_arr
    ab_sq = np.dot(ab_arr, ab_arr)
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        p0_t = bk.as_tensor(p0_arr, p)
        ab_t = bk.as_tensor(ab_arr, p)
        t = bk.clip(_dot(p - p0_t, ab_t) / ab_sq, 0, 1)
        t_e = e(t)
        if bk.is_symbolic(t_e):
            pass
        elif bk.is_tensor(t_e):
            t_e = t_e.reshape((-1, 1))
        else:
            t_e = np.reshape(t_e, (-1, 1))
        return t_e * d2 + (1 - t_e) * d1
    return f

@op3
def transition_radial(f0, f1, r0=0, r1=1, e=ease.linear):
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        x = p[..., 0]
        y = p[..., 1]
        r = bk.hypot(x, y)
        t = bk.clip((r - r0) / (r1 - r0), 0, 1)
        t_e = e(t)
        if bk.is_symbolic(t_e):
            pass
        elif bk.is_tensor(t_e):
            t_e = t_e.reshape((-1, 1))
        else:
            t_e = np.reshape(t_e, (-1, 1))
        return t_e * d2 + (1 - t_e) * d1
    return f

@op3
def wrap_around(other, x0, x1, r=None, e=ease.linear):
    p0 = X * x0
    p1 = X * x1
    v = -Y
    if r is None:
        r = np.linalg.norm(p1 - p0) / (2 * math.pi)
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        d = bk.hypot(x, y) - r
        a = bk.arctan2(y, x)
        t = (a + math.pi) / (2 * math.pi)
        t_e = e(t)
        if bk.is_symbolic(t_e):
            pass
        elif bk.is_tensor(t_e):
            t_e = t_e.reshape((-1, 1))
            d = d.reshape((-1, 1))
        else:
            t_e = np.reshape(t_e, (-1, 1))
            d = np.reshape(d, (-1, 1))
        p0_t = bk.as_tensor(p0, p)
        p1_t = bk.as_tensor(p1, p)
        v_t = bk.as_tensor(v, p)
        q = p0_t + (p1_t - p0_t) * t_e + v_t * d
        qx = q[..., 0]
        qy = q[..., 1]
        return other(_vec(qx, qy, z))
    return f

# 3D => 2D Operations

@op32
def slice(other):
    # TODO: support specifying a slice plane
    # TODO: probably a better way to do this
    s = slab(z0=-1e-9, z1=1e-9)
    a = other & s
    b = other.negate() & s
    def f(p):
        x = p[..., 0]
        y = p[..., 1]
        p3 = _vec(x, y, 0 * x)
        if bk.is_tensor(p3):
            A = a(p3).reshape(-1)
            B = -b(p3).reshape(-1)
        else:
            A = np.reshape(a(p3), -1)
            B = np.reshape(-b(p3), -1)
        return bk.where(A <= 0, B, A)
    return f

# Common

union = op3(dn.union)
difference = op3(dn.difference)
intersection = op3(dn.intersection)
blend = op3(dn.blend)
negate = op3(dn.negate)
dilate = op3(dn.dilate)
erode = op3(dn.erode)
shell = op3(dn.shell)
repeat = op3(dn.repeat)

def _random_rotation_matrix_3d(rng):
    u1, u2, u3 = rng.uniform(0, 1, 3)
    q0 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q1 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q2 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q3 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

def _poisson_disc_3d(domain, r_min, r_max=None, k_samples=30, variable_radius=None,
                     containment_mode='strict', seed=None, bounds=None):
    if domain is None:
        raise ValueError("domain (an SDF2 or SDF3 object) must be provided to specify the Poisson field boundary.")
    if containment_mode not in ('strict', 'center'):
        raise ValueError(f"Invalid containment_mode '{containment_mode}'. Expected 'strict' or 'center'.")

    rng = np.random.default_rng(seed)

    is_domain_2d = isinstance(domain, d2.SDF2)
    if not is_domain_2d and not isinstance(domain, SDF3):
        try:
            domain(np.zeros((1, 2)))
            is_domain_2d = True
        except Exception:
            is_domain_2d = False

    if bounds is None:
        if is_domain_2d:
            b2d = core._estimate_bounds(domain, dim=2)
            bounds = ((b2d[0][0], b2d[0][1], -0.1), (b2d[1][0], b2d[1][1], 0.1))
        else:
            bounds = core._estimate_bounds(domain, dim=3)
    (x0, y0, z0), (x1, y1, z1) = bounds

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
        p_arr = np.asarray(p, dtype=float)
        if is_domain_2d:
            val = float(domain(p_arr[:2].reshape(1, 2))[0, 0])
        else:
            val = float(domain(p_arr.reshape(1, 3))[0, 0])
        if containment_mode == 'strict':
            return val + r <= 1e-6
        return val <= 1e-6

    cell_size = r_min / math.sqrt(3)
    grid = dn.SpatialHashGrid(cell_size)

    samples = []
    radii = []

    initial_pt = None
    initial_r = None
    for _ in range(10000):
        rx = rng.uniform(x0, x1)
        ry = rng.uniform(y0, y1)
        rz = rng.uniform(z0, z1)
        pt = np.array([rx, ry, rz])
        r = get_radius(pt)
        if is_contained(pt, r):
            initial_pt = pt
            initial_r = r
            break

    if initial_pt is None:
        return np.empty((0, 3)), np.empty((0,))

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
                rz = rng.uniform(z0, z1)
                pt = np.array([rx, ry, rz])
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
                    u = rng.uniform(-1, 1)
                    phi = rng.uniform(0, 2 * math.pi)
                    sin_theta = math.sqrt(max(0.0, 1.0 - u * u))
                    dir_3d = np.array([sin_theta * math.cos(phi), sin_theta * math.sin(phi), u])

                    r_cand_est = min(pass_r, get_radius(parent_pt)) if r_max > r_min else get_radius(parent_pt)
                    min_dist = parent_r + r_cand_est
                    dist = rng.uniform(min_dist, 2 * min_dist)
                    cand = parent_pt + dir_3d * dist

                    if cand[0] < x0 or cand[0] > x1 or cand[1] < y0 or cand[1] > y1 or cand[2] < z0 or cand[2] > z1:
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
        return np.empty((0, 3)), np.empty((0,))

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

@op3
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

    is_domain_2d = isinstance(domain, d2.SDF2)
    if not is_domain_2d and not isinstance(domain, SDF3):
        try:
            domain(np.zeros((1, 2)))
            is_domain_2d = True
        except Exception:
            is_domain_2d = False

    if is_domain_2d:
        centers_2d, radii = d2._poisson_disc_2d(
            domain=domain,
            r_min=r_min,
            r_max=r_max,
            k_samples=k_samples,
            variable_radius=variable_radius,
            containment_mode=containment_mode,
            seed=seed,
            bounds=bounds
        )
        if len(centers_2d) > 0:
            centers = np.column_stack([centers_2d, np.zeros(len(centers_2d))])
        else:
            centers = np.empty((0, 3))
    else:
        centers, radii = _poisson_disc_3d(
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
        matrices = [_random_rotation_matrix_3d(rng) for _ in range(len(centers))]
    else:
        matrices = [np.eye(3) for _ in range(len(centers))]

    matrices = np.array(matrices, dtype=np.float32) if len(centers) > 0 else np.empty((0, 3, 3), dtype=np.float32)
    centers = np.array(centers, dtype=np.float32) if len(centers) > 0 else np.empty((0, 3), dtype=np.float32)
    scales = np.array(scales, dtype=np.float32) if len(centers) > 0 else np.empty((0,), dtype=np.float32)

    r_bound = scales * base_r
    K = k if k is not None else getattr(other, '_k', None)

    def f(p):
        if len(centers) == 0:
            if bk.is_tensor(p):
                import torch
                return torch.full(p.shape[:-1] + (1,), 1e9, device=p.device, dtype=p.dtype)
            return np.full(p.shape[:-1] + (1,), 1e9)

        p_3d = p.reshape(-1, 3)
        N = p_3d.shape[0]
        is_t = bk.is_tensor(p)

        if is_t:
            import torch
            p_min = p_3d.min(dim=0).values
            p_max = p_3d.max(dim=0).values
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
                    p_sub = p_3d[start_idx : start_idx + chunk_sz]
                    n_sub = p_sub.shape[0]
                    P_shifted = (p_sub.unsqueeze(1) - C_act.unsqueeze(0)) / S_act.unsqueeze(0).unsqueeze(2)
                    P_rotated = torch.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                    P_flat = P_rotated.reshape(-1, 3)
                    d_flat = other(P_flat)
                    d_matrix = d_flat.reshape(n_sub, M) * S_act.unsqueeze(0)
                    chunks.append(_smooth_reduce_torch(d_matrix, K))
                d_min = torch.cat(chunks, dim=0)
            else:
                P_shifted = (p_3d.unsqueeze(1) - C_act.unsqueeze(0)) / S_act.unsqueeze(0).unsqueeze(2)
                P_rotated = torch.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                P_flat = P_rotated.reshape(-1, 3)
                d_flat = other(P_flat)
                d_matrix = d_flat.reshape(N, M) * S_act.unsqueeze(0)
                d_min = _smooth_reduce_torch(d_matrix, K)

            return d_min.reshape(p.shape[:-1] + (1,))
        else:
            p_min = p_3d.min(axis=0)
            p_max = p_3d.max(axis=0)
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
                    p_sub = p_3d[start_idx : start_idx + chunk_sz]
                    n_sub = p_sub.shape[0]
                    P_shifted = (p_sub[:, None, :] - C_act[None, :, :]) / S_act[None, :, None]
                    P_rotated = np.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                    P_flat = P_rotated.reshape(-1, 3)
                    d_flat = other(P_flat)
                    d_matrix = d_flat.reshape(n_sub, M) * S_act[None, :]
                    chunks.append(_smooth_reduce_numpy(d_matrix, K))
                d_min = np.concatenate(chunks, axis=0)
            else:
                P_shifted = (p_3d[:, None, :] - C_act[None, :, :]) / S_act[None, :, None]
                P_rotated = np.einsum('nki,kij->nkj', P_shifted, M_act) if random_rotation else P_shifted
                P_flat = P_rotated.reshape(-1, 3)
                d_flat = other(P_flat)
                d_matrix = d_flat.reshape(N, M) * S_act[None, :]
                d_min = _smooth_reduce_numpy(d_matrix, K)

            return d_min.reshape(p.shape[:-1] + (1,))

    return f

