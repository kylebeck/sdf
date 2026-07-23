import functools
import math
import numpy as np

from . import dn, d3, ease, backend as bk

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
def circular_array(other, count):
    angles = [i / count * 2 * math.pi for i in range(count)]
    return union(*[other.rotate(a) for a in angles])

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
