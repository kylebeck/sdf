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

def is_symbolic(x):
    from .taichi_backend import SymbolicNode
    return isinstance(x, SymbolicNode)

def _to_sym_code(val):
    from .taichi_backend import _to_code
    return _to_code(val)

def as_tensor(x, ref=None):
    if is_symbolic(ref) or is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(_to_sym_code(x))
    if not HAS_TORCH:
        return np.asarray(x)
    if is_tensor(x):
        if ref is not None and is_tensor(ref):
            dtype = ref.dtype if (ref.dtype != torch.bool or x.dtype == torch.bool) else x.dtype
            return x.to(device=ref.device, dtype=dtype)
        return x
    if ref is not None:
        if is_tensor(ref):
            dtype = ref.dtype if (ref.dtype != torch.bool or isinstance(x, bool)) else torch.float32
            return torch.tensor(x, device=ref.device, dtype=dtype)
        return np.asarray(x)
    return np.asarray(x)

def to_numpy(x):
    if is_symbolic(x):
        return x
    if is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def norm(a, axis=-1):
    if is_symbolic(a):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"({_to_sym_code(a)}).norm()")
    if is_tensor(a):
        return torch.linalg.norm(a, dim=axis)
    return np.linalg.norm(a, axis=axis)

def normalize(a):
    if is_symbolic(a):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"({_to_sym_code(a)}).normalized()")
    if is_tensor(a):
        n = torch.linalg.norm(a, dim=-1, keepdim=True)
        n = torch.where(n == 0, torch.ones_like(n), n)
        return a / n
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return a / n

def dot(a, b):
    if is_symbolic(a) or is_symbolic(b):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"({_to_sym_code(a)}).dot({_to_sym_code(b)})")
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        a_tensor = as_tensor(a, ref)
        b_tensor = as_tensor(b, ref)
        return torch.sum(a_tensor * b_tensor, dim=-1)
    return np.sum(a * b, axis=-1)

def vec(*arrs):
    if any(is_symbolic(x) for x in arrs):
        from .taichi_backend import SymbolicNode
        elems = ", ".join(_to_sym_code(x) for x in arrs)
        return SymbolicNode(f"ti.Vector([{elems}])")
    if any(is_tensor(a) for a in arrs):
        ref = next(a for a in arrs if is_tensor(a))
        tensors = [as_tensor(a, ref) for a in arrs]
        return torch.stack(tensors, dim=-1)
    return np.stack(arrs, axis=-1)

def minimum(a, b):
    if is_symbolic(a) or is_symbolic(b):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.min({_to_sym_code(a)}, {_to_sym_code(b)})")
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        return torch.minimum(as_tensor(a, ref), as_tensor(b, ref))
    return np.minimum(a, b)

def maximum(a, b):
    if is_symbolic(a) or is_symbolic(b):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.max({_to_sym_code(a)}, {_to_sym_code(b)})")
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        return torch.maximum(as_tensor(a, ref), as_tensor(b, ref))
    return np.maximum(a, b)

def clip(x, a_min, a_max):
    if is_symbolic(x) or is_symbolic(a_min) or is_symbolic(a_max):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.min(ti.max({_to_sym_code(x)}, {_to_sym_code(a_min)}), {_to_sym_code(a_max)})")
    if is_tensor(x):
        ref = x
        min_t = as_tensor(a_min, ref) if a_min is not None else None
        max_t = as_tensor(a_max, ref) if a_max is not None else None
        return torch.clamp(x, min=min_t, max=max_t)
    return np.clip(x, a_min, a_max)

def abs(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.abs({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.abs(x)
    return np.abs(x)

def sign(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        c = _to_sym_code(x)
        return SymbolicNode(f"ti.select({c} > 0.0, 1.0, ti.select({c} < 0.0, -1.0, 0.0))")
    if is_tensor(x):
        return torch.sign(x)
    return np.sign(x)

def sqrt(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.sqrt({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.sqrt(x)
    return np.sqrt(x)

def sin(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.sin({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.sin(x)
    return np.sin(x)

def cos(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.cos({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.cos(x)
    return np.cos(x)

def tan(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.tan({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.tan(x)
    return np.tan(x)

def radians(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"({_to_sym_code(x)} * {math.pi / 180.0:.8f})")
    if is_tensor(x):
        return torch.deg2rad(x)
    return np.radians(x)

def hypot(x, y):
    if is_symbolic(x) or is_symbolic(y):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.sqrt(({_to_sym_code(x)})**2 + ({_to_sym_code(y)})**2)")
    x_t = is_tensor(x)
    y_t = is_tensor(y)
    if x_t or y_t:
        ref = x if x_t else y
        return torch.hypot(as_tensor(x, ref), as_tensor(y, ref))
    return np.hypot(x, y)

def arctan2(y, x):
    if is_symbolic(x) or is_symbolic(y):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.atan2({_to_sym_code(y)}, {_to_sym_code(x)})")
    x_t = is_tensor(x)
    y_t = is_tensor(y)
    if x_t or y_t:
        ref = y if y_t else x
        return torch.atan2(as_tensor(y, ref), as_tensor(x, ref))
    return np.arctan2(y, x)

def where(condition, x, y):
    if is_symbolic(condition) or is_symbolic(x) or is_symbolic(y):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.select({_to_sym_code(condition)}, {_to_sym_code(x)}, {_to_sym_code(y)})")
    cond_t = is_tensor(condition)
    x_t = is_tensor(x)
    y_t = is_tensor(y)
    if cond_t or x_t or y_t:
        ref = x if x_t else (y if y_t else None)
        if ref is None:
            cond_tensor = as_tensor(condition).bool()
            x_tensor = torch.tensor(x, device=cond_tensor.device, dtype=torch.float32)
            y_tensor = torch.tensor(y, device=cond_tensor.device, dtype=torch.float32)
            return torch.where(cond_tensor, x_tensor, y_tensor)
        cond_tensor = as_tensor(condition, ref).bool()
        x_tensor = as_tensor(x, ref)
        y_tensor = as_tensor(y, ref)
        return torch.where(cond_tensor, x_tensor, y_tensor)
    return np.where(condition, x, y)

def matmul(a, b):
    if is_symbolic(a) or is_symbolic(b):
        from .taichi_backend import SymbolicNode
        b_arr = np.asarray(b, dtype=float)
        if b_arr.ndim == 2:
            r0 = ", ".join(f"{x:.8f}" for x in b_arr[0])
            r1 = ", ".join(f"{x:.8f}" for x in b_arr[1])
            r2 = ", ".join(f"{x:.8f}" for x in b_arr[2])
            mat_str = f"ti.Matrix([[{r0}], [{r1}], [{r2}]])"
            return SymbolicNode(f"({_to_sym_code(a)} @ {mat_str})")
        return SymbolicNode(f"({_to_sym_code(a)} @ {_to_sym_code(b)})")
    a_t = is_tensor(a)
    b_t = is_tensor(b)
    if a_t or b_t:
        ref = a if a_t else b
        return torch.matmul(as_tensor(a, ref), as_tensor(b, ref))
    return np.dot(a, b)

def amax(a, axis=None):
    if is_symbolic(a):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"({a.code}).max()")
    if is_tensor(a):
        if axis is None:
            return torch.max(a)
        return torch.max(a, dim=axis).values
    return np.amax(a, axis=axis)

def round(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.round({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.round(x)
    return np.round(x)

def floor(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.floor({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.floor(x)
    return np.floor(x)

def ceil(x):
    if is_symbolic(x):
        from .taichi_backend import SymbolicNode
        return SymbolicNode(f"ti.ceil({_to_sym_code(x)})")
    if is_tensor(x):
        return torch.ceil(x)
    return np.ceil(x)
