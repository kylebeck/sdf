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
def circular_array(other, count, offset=0):
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
