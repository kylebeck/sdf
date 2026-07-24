import math
import linecache
import numpy as np

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    ti = None
    HAS_TAICHI = False

class SymbolicNode:
    """
    Symbolic expression node used to trace Python SDF trees into Taichi C-style GPU shader code.
    """
    def __init__(self, code, shape=None):
        self.code = str(code)
        self.shape = shape

    def __str__(self):
        return self.code

    def __repr__(self):
        return f"SymbolicNode({self.code})"

    @property
    def dtype(self):
        return float

    def reshape(self, *args):
        return self

    def unsqueeze(self, dim):
        return self

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 2 and idx[0] == Ellipsis:
                idx = idx[1]

        if isinstance(idx, int):
            comps = ['x', 'y', 'z', 'w']
            if idx < 4:
                return SymbolicNode(f"{self.code}.{comps[idx]}")
            return SymbolicNode(f"{self.code}[{idx}]")

        if isinstance(idx, list):
            comps = ['x', 'y', 'z', 'w']
            sub_vec = ", ".join(f"{self.code}.{comps[i]}" for i in idx)
            return SymbolicNode(f"ti.Vector([{sub_vec}])", shape=(len(idx),))

        if idx == slice(None):
            return self

        return SymbolicNode(f"{self.code}[{idx}]")

    def __add__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} + {other_code})")

    def __radd__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({other_code} + {self.code})")

    def __sub__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} - {other_code})")

    def __rsub__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({other_code} - {self.code})")

    def __mul__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} * {other_code})")

    def __rmul__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({other_code} * {self.code})")

    def __truediv__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} / {other_code})")

    def __rtruediv__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({other_code} / {self.code})")

    def __mod__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} % {other_code})")

    def __neg__(self):
        return SymbolicNode(f"(-{self.code})")

    def __and__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} and {other_code})")

    def __or__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} or {other_code})")

    def __gt__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} > {other_code})")

    def __lt__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} < {other_code})")

    def __ge__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} >= {other_code})")

    def __le__(self, other):
        other_code = _to_code(other)
        return SymbolicNode(f"({self.code} <= {other_code})")


def _to_code(val):
    if isinstance(val, SymbolicNode):
        return val.code
    if isinstance(val, (int, float)):
        return f"{float(val):.8f}"
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=float).flatten()
        elems = ", ".join(f"{x:.8f}" for x in arr)
        return f"ti.Vector([{elems}])"
    return str(val)

def trace_sdf_to_code(sdf_obj):
    p = SymbolicNode("p", shape=(3,))
    res = sdf_obj(p)
    return res.code

_TAICHI_KERNEL_CACHE = {}

def init_taichi(arch=None):
    if not HAS_TAICHI:
        raise RuntimeError("Taichi is not installed. Install via `pip install taichi`.")
    runtime = ti.lang.impl.get_runtime()
    if not getattr(runtime, "prog", None):
        if arch is None:
            arch = ti.metal
        try:
            ti.init(arch=arch, log_level=ti.WARN)
        except Exception:
            ti.init(arch=ti.cpu, log_level=ti.WARN)

def get_taichi_evaluator(sdf_obj, arch=None):
    init_taichi(arch=arch)
    expr_code = trace_sdf_to_code(sdf_obj)

    if expr_code in _TAICHI_KERNEL_CACHE:
        return _TAICHI_KERNEL_CACHE[expr_code]

    idx = len(_TAICHI_KERNEL_CACHE)
    func_name = f"sdf_func_{idx}"
    kernel_name = f"eval_kernel_{idx}"

    code_str = f"""@ti.func
def {func_name}(p: ti.template()):
    return {expr_code}

@ti.kernel
def {kernel_name}(
    vol: ti.template(),
    x0: ti.f32, y0: ti.f32, z0: ti.f32,
    dx: ti.f32, dy: ti.f32, dz: ti.f32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32
):
    for i, j, k in ti.ndrange(nx, ny, nz):
        p = ti.Vector([x0 + float(i) * dx, y0 + float(j) * dy, z0 + float(k) * dz])
        vol[i, j, k] = float({func_name}(p))
"""
    filename = f"<taichi_sdf_{idx}.py>"
    linecache.cache[filename] = (
        len(code_str),
        None,
        [line + "\n" for line in code_str.splitlines()],
        filename
    )

    compiled_code = compile(code_str, filename, "exec")
    exec_scope = {"ti": ti, "math": math}
    exec(compiled_code, exec_scope)
    kernel_fn = exec_scope[kernel_name]

    _TAICHI_KERNEL_CACHE[expr_code] = kernel_fn
    return kernel_fn

def evaluate_grid_taichi(sdf_obj, X, Y, Z, arch=None):
    init_taichi(arch=arch)

    nx, ny, nz = len(X), len(Y), len(Z)
    x0, y0, z0 = float(X[0]), float(Y[0]), float(Z[0])
    dx = float(X[1] - X[0]) if nx > 1 else 1.0
    dy = float(Y[1] - Y[0]) if ny > 1 else 1.0
    dz = float(Z[1] - Z[0]) if nz > 1 else 1.0

    vol_field = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
    kernel_fn = get_taichi_evaluator(sdf_obj, arch=arch)

    kernel_fn(vol_field, x0, y0, z0, dx, dy, dz, nx, ny, nz)
    return vol_field.to_numpy().astype(np.float64)
