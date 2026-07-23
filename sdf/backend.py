import math
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

def is_tensor(x):
    return HAS_TORCH and isinstance(x, torch.Tensor)

def as_tensor(x, ref=None):
    if not HAS_TORCH:
        return np.asarray(x)
    if is_tensor(x):
        if ref is not None and is_tensor(ref):
            return x.to(device=ref.device, dtype=ref.dtype)
        return x
    if ref is not None:
        if is_tensor(ref):
            return torch.tensor(x, device=ref.device, dtype=ref.dtype)
        return np.asarray(x)
    return np.asarray(x)

def to_numpy(x):
    if is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def norm(a, axis=-1):
    if is_tensor(a):
        return torch.linalg.norm(a, dim=axis)
    return np.linalg.norm(a, axis=axis)

def normalize(a):
    if is_tensor(a):
        n = torch.linalg.norm(a, dim=-1, keepdim=True)
        n = torch.where(n == 0, torch.ones_like(n), n)
        return a / n
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return a / n

def dot(a, b):
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        a_tensor = as_tensor(a, ref)
        b_tensor = as_tensor(b, ref)
        return torch.sum(a_tensor * b_tensor, dim=-1)
    return np.sum(a * b, axis=-1)

def vec(*arrs):
    if any(is_tensor(a) for a in arrs):
        ref = next(a for a in arrs if is_tensor(a))
        tensors = [as_tensor(a, ref) for a in arrs]
        return torch.stack(tensors, dim=-1)
    return np.stack(arrs, axis=-1)

def minimum(a, b):
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        return torch.minimum(as_tensor(a, ref), as_tensor(b, ref))
    return np.minimum(a, b)

def maximum(a, b):
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        return torch.maximum(as_tensor(a, ref), as_tensor(b, ref))
    return np.maximum(a, b)

def clip(x, a_min, a_max):
    if is_tensor(x):
        ref = x
        min_t = as_tensor(a_min, ref) if a_min is not None else None
        max_t = as_tensor(a_max, ref) if a_max is not None else None
        return torch.clamp(x, min=min_t, max=max_t)
    return np.clip(x, a_min, a_max)

def abs(x):
    if is_tensor(x):
        return torch.abs(x)
    return np.abs(x)

def sign(x):
    if is_tensor(x):
        return torch.sign(x)
    return np.sign(x)

def sqrt(x):
    if is_tensor(x):
        return torch.sqrt(x)
    return np.sqrt(x)

def sin(x):
    if is_tensor(x):
        return torch.sin(x)
    return np.sin(x)

def cos(x):
    if is_tensor(x):
        return torch.cos(x)
    return np.cos(x)

def tan(x):
    if is_tensor(x):
        return torch.tan(x)
    return np.tan(x)

def radians(x):
    if is_tensor(x):
        return torch.deg2rad(x)
    return np.radians(x)

def hypot(x, y):
    x_t = is_tensor(x)
    y_t = is_tensor(y)
    if x_t or y_t:
        ref = x if x_t else y
        return torch.hypot(as_tensor(x, ref), as_tensor(y, ref))
    return np.hypot(x, y)

def arctan2(y, x):
    x_t = is_tensor(x)
    y_t = is_tensor(y)
    if x_t or y_t:
        ref = y if y_t else x
        return torch.atan2(as_tensor(y, ref), as_tensor(x, ref))
    return np.arctan2(y, x)

def where(condition, x, y):
    cond_t = is_tensor(condition)
    x_t = is_tensor(x)
    y_t = is_tensor(y)
    if cond_t or x_t or y_t:
        ref = x if x_t else (y if y_t else condition)
        cond_tensor = as_tensor(condition, ref).bool()
        x_tensor = as_tensor(x, ref)
        y_tensor = as_tensor(y, ref)
        return torch.where(cond_tensor, x_tensor, y_tensor)
    return np.where(condition, x, y)

def matmul(a, b):
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        return torch.matmul(as_tensor(a, ref), as_tensor(b, ref))
    return np.dot(a, b)

def amax(a, axis=None):
    if is_tensor(a):
        if axis is None:
            return torch.max(a)
        return torch.max(a, dim=axis).values
    return np.amax(a, axis=axis)

def round(x):
    if is_tensor(x):
        return torch.round(x)
    return np.round(x)

def floor(x):
    if is_tensor(x):
        return torch.floor(x)
    return np.floor(x)

def ceil(x):
    if is_tensor(x):
        return torch.ceil(x)
    return np.ceil(x)
