from functools import partial
from multiprocessing.pool import ThreadPool
from skimage import measure

import multiprocessing
import itertools
import numpy as np
import time

from . import progress, stl, backend as b

WORKERS = multiprocessing.cpu_count()
SAMPLES = 2 ** 22
BATCH_SIZE = 32

def _marching_cubes(volume, level=0):
    verts, faces, _, _ = measure.marching_cubes(volume, level)
    return verts, faces

def _surface_nets(volume, level=0):
    try:
        from . import _meshing
        return _meshing.surface_nets_extract(volume)
    except ImportError:
        raise NotImplementedError("Cython extension not compiled. Run 'python setup.py build_ext --inplace' or 'pip install -e .'")

def _gradient(sdf, P, epsilon=1e-5):
    # TODO: Implement numerical gradient for Dual Contouring
    # Compute central difference for each axis
    raise NotImplementedError("Gradient not yet implemented")

def _dual_contouring(sdf, X, Y, Z, volume, qef_threshold):
    try:
        from . import _meshing
        return _meshing.dual_contouring_extract(volume, qef_threshold)
    except ImportError:
        raise NotImplementedError("Cython extension not compiled. Run 'python setup.py build_ext --inplace' or 'pip install -e .'")

def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def _skip(sdf, job):
    X, Y, Z = job
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    z = (z0 + z1) / 2
    r = abs(sdf(np.array([(x, y, z)])).reshape(-1)[0])
    d = np.linalg.norm(np.array((x-x0, y-y0, z-z0)))
    if r <= d:
        return False, 0
    corners = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))))
    values = sdf(corners).reshape(-1)
    same = np.all(values > 0) if values[0] > 0 else np.all(values < 0)
    sign = 1 if values[0] > 0 else -1
    return same, sign

def _worker(sdf, job_info, step, sparse):
    indices, job = job_info
    X, Y, Z = job
    if sparse:
        same, sign = _skip(sdf, job)
        if same:
            return indices, None, sign
    P = _cartesian_product(X, Y, Z)
    shape = (len(X), len(Y), len(Z))
    volume = sdf(P).reshape(shape)
    return indices, volume, 0

def _estimate_bounds(sdf):
    # TODO: raise exception if bound estimation fails
    s = 16
    x0 = y0 = z0 = -1e9
    x1 = y1 = z1 = 1e9
    prev = None
    for i in range(32):
        X = np.linspace(x0, x1, s)
        Y = np.linspace(y0, y1, s)
        Z = np.linspace(z0, z1, s)
        d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
        threshold = np.linalg.norm(d) / 2
        if threshold == prev:
            break
        prev = threshold
        P = _cartesian_product(X, Y, Z)
        volume = sdf(P).reshape((len(X), len(Y), len(Z)))
        where = np.argwhere(np.abs(volume) <= threshold)
        x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
        x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2
    return ((x0, y0, z0), (x1, y1, z1))

def generate(
        sdf,
        step=None, bounds=None, samples=SAMPLES,
        workers=WORKERS, batch_size=BATCH_SIZE,
        verbose=True, sparse=True, method='marching_cubes',
        qef_threshold=1e-3, device='auto', gpu_batch_size=2**20):

    start = time.time()

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if step is None and samples is not None:
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (volume / samples) ** (1 / 3)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    if verbose:
        print('min %g, %g, %g' % (x0, y0, z0))
        print('max %g, %g, %g' % (x1, y1, z1))
        print('step %g, %g, %g' % (dx, dy, dz))

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    num_samples = len(X) * len(Y) * len(Z)

    target_device = 'numpy'
    if device == 'auto':
        if b.HAS_TORCH:
            import torch
            if torch.backends.mps.is_available():
                target_device = torch.device('mps')
            elif torch.cuda.is_available():
                target_device = torch.device('cuda')
            else:
                target_device = torch.device('cpu')
    elif device in ('numpy', None):
        target_device = 'numpy'
    else:
        if not b.HAS_TORCH:
            if verbose:
                print(f"Warning: PyTorch not installed. Falling back to NumPy CPU for device={device}")
            target_device = 'numpy'
        else:
            import torch
            target_device = torch.device(device)

    if target_device != 'numpy':
        import torch
        if verbose:
            print(f"Evaluating {num_samples} volume samples on PyTorch device: {target_device}")

        X_t = torch.as_tensor(X, dtype=torch.float32)
        Y_t = torch.as_tensor(Y, dtype=torch.float32)
        Z_t = torch.as_tensor(Z, dtype=torch.float32)

        grid_x, grid_y, grid_z = torch.meshgrid(X_t, Y_t, Z_t, indexing='ij')
        P_all = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        vol_flat = torch.empty(num_samples, dtype=torch.float32, device='cpu')

        num_batches = (num_samples + gpu_batch_size - 1) // gpu_batch_size
        bar = progress.Bar(num_batches, enabled=verbose)
        for i in range(0, num_samples, gpu_batch_size):
            batch_P = P_all[i : i + gpu_batch_size].to(target_device)
            with torch.no_grad():
                res = sdf(batch_P)
            vol_flat[i : i + batch_P.shape[0]] = res.reshape(-1).cpu()
            bar.increment(1)
        bar.done()

        global_vol = vol_flat.reshape(len(X), len(Y), len(Z)).numpy().astype(np.float64)

    else:
        s = batch_size
        x_indices = list(range(0, len(X), s))
        y_indices = list(range(0, len(Y), s))
        z_indices = list(range(0, len(Z), s))
        
        batches = []
        for ix in x_indices:
            for iy in y_indices:
                for iz in z_indices:
                    job = (X[ix:ix+s], Y[iy:iy+s], Z[iz:iz+s])
                    batches.append(((ix, iy, iz), job))
                    
        num_batches = len(batches)

        if verbose:
            print('%d samples in %d batches with %d workers' %
                (num_samples, num_batches, workers))

        global_vol = np.zeros((len(X), len(Y), len(Z)), dtype=np.float64)
        skipped = empty = nonempty = 0
        bar = progress.Bar(num_batches, enabled=verbose)
        pool = ThreadPool(workers)
        f = partial(_worker, sdf, step=(dx, dy, dz), sparse=sparse)
        for (ix, iy, iz), vol_block, sign in pool.imap(f, batches):
            bar.increment(1)
            if vol_block is None:
                skipped += 1
                sx, sy, sz = min(s, len(X)-ix), min(s, len(Y)-iy), min(s, len(Z)-iz)
                global_vol[ix:ix+sx, iy:iy+sy, iz:iz+sz] = sign * 1e3
            else:
                nonempty += 1
                sx, sy, sz = vol_block.shape
                global_vol[ix:ix+sx, iy:iy+sy, iz:iz+sz] = vol_block
        bar.done()

        if verbose:
            print('%d skipped, %d nonempty blocks evaluated' % (skipped, nonempty))
        
    try:
        if method == 'marching_cubes':
            verts, faces = _marching_cubes(global_vol)
        elif method == 'surface_nets':
            verts, faces = _surface_nets(global_vol)
        elif method == 'dual_contouring':
            verts, faces = _dual_contouring(sdf, X, Y, Z, global_vol, qef_threshold)
        else:
            raise ValueError(f"Unknown meshing method: {method}")
    except Exception as e:
        print(f"Meshing exception in global evaluation ({method}): {e}")
        verts, faces = np.empty((0, 3)), np.empty((0, 3), dtype=int)

    scale = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
    offset = np.array([X[0], Y[0], Z[0]])
    if len(verts) > 0:
        verts = verts * scale + offset

    if verbose:
        triangles = len(faces)
        seconds = time.time() - start
        print('%d triangles in %g seconds' % (triangles, seconds))

    return verts, faces

def save(path, *args, **kwargs):
    verts, faces = generate(*args, **kwargs)
    if path.lower().endswith('.stl'):
        points = verts[faces].reshape((-1, 3))
        stl.write_binary_stl(path, points)
    else:
        mesh = _mesh(verts, faces)
        mesh.write(path)

def _mesh(verts, faces):
    import meshio
    cells = [('triangle', faces)]
    return meshio.Mesh(verts, cells)

def _debug_triangles(X, Y, Z):
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]

    p = 0.25
    x0, x1 = x0 + (x1 - x0) * p, x1 - (x1 - x0) * p
    y0, y1 = y0 + (y1 - y0) * p, y1 - (y1 - y0) * p
    z0, z1 = z0 + (z1 - z0) * p, z1 - (z1 - z0) * p

    v = [
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z0),
        (x0, y1, z1),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z0),
        (x1, y1, z1),
    ]

    return [
        v[3], v[5], v[7],
        v[5], v[3], v[1],
        v[0], v[6], v[4],
        v[6], v[0], v[2],
        v[0], v[5], v[1],
        v[5], v[0], v[4],
        v[5], v[6], v[7],
        v[6], v[5], v[4],
        v[6], v[3], v[7],
        v[3], v[6], v[2],
        v[0], v[3], v[2],
        v[3], v[0], v[1],
    ]

def sample_slice(
        sdf, w=1024, h=1024,
        x=None, y=None, z=None, bounds=None):

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if x is not None:
        X = np.array([x])
        Y = np.linspace(y0, y1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], Y[0], Y[-1])
        axes = 'ZY'
    elif y is not None:
        Y = np.array([y])
        X = np.linspace(x0, x1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], X[0], X[-1])
        axes = 'ZX'
    elif z is not None:
        Z = np.array([z])
        X = np.linspace(x0, x1, w)
        Y = np.linspace(y0, y1, h)
        extent = (Y[0], Y[-1], X[0], X[-1])
        axes = 'YX'
    else:
        raise Exception('x, y, or z position must be specified')

    P = _cartesian_product(X, Y, Z)
    return sdf(P).reshape((w, h)), extent, axes

def show_slice(*args, **kwargs):
    import matplotlib.pyplot as plt
    show_abs = kwargs.pop('abs', False)
    a, extent, axes = sample_slice(*args, **kwargs)
    if show_abs:
        a = np.abs(a)
    im = plt.imshow(a, extent=extent, origin='lower')
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.colorbar(im)
    plt.show()
