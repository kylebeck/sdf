import math
import numpy as np
from . import backend as bk

def sample_volume_octree(
    sdf,
    X, Y, Z,
    device='auto',
    block_size=16,
    safety_factor=1.25,
    gpu_batch_size=2**20,
    verbose=True
):
    """
    Hierarchical Adaptive Octree Sampler for Signed Distance Fields.
    Evaluates coarse block bounding spheres first, prunes non-surface space,
    and evaluates fine voxel coordinates on active surface blocks only.
    """
    nx, ny, nz = len(X), len(Y), len(Z)
    dx = float(X[1] - X[0]) if nx > 1 else 1.0
    dy = float(Y[1] - Y[0]) if ny > 1 else 1.0
    dz = float(Z[1] - Z[0]) if nz > 1 else 1.0

    global_vol = np.zeros((nx, ny, nz), dtype=np.float64)

    # 1. Build coarse block partition
    s = block_size
    x_indices = list(range(0, nx, s))
    y_indices = list(range(0, ny, s))
    z_indices = list(range(0, nz, s))

    blocks = []
    block_centers = []
    block_radii = []

    for ix in x_indices:
        sx = min(s, nx - ix)
        for iy in y_indices:
            sy = min(s, ny - iy)
            for iz in z_indices:
                sz = min(s, nz - iz)
                cx = X[ix] + (sx - 1) * dx * 0.5
                cy = Y[iy] + (sy - 1) * dy * 0.5
                cz = Z[iz] + (sz - 1) * dz * 0.5
                r_cell = 0.5 * math.sqrt((sx * dx)**2 + (sy * dy)**2 + (sz * dz)**2)
                blocks.append((ix, iy, iz, sx, sy, sz))
                block_centers.append((cx, cy, cz))
                block_radii.append(r_cell)

    num_blocks = len(blocks)
    if verbose:
        print(f"Adaptive Octree: Partitioned grid ({nx}x{ny}x{nz}) into {num_blocks} blocks of size ~{s}^3")

    # 2. Evaluate coarse block centers to detect active surface blocks
    centers_arr = np.array(block_centers, dtype=np.float32)

    target_device = 'numpy'
    if device == 'auto':
        if bk.HAS_TORCH:
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
        if bk.HAS_TORCH:
            import torch
            target_device = torch.device(device)

    if target_device != 'numpy':
        import torch
        centers_tensor = torch.as_tensor(centers_arr, device=target_device)
        with torch.no_grad():
            center_dists = sdf(centers_tensor).reshape(-1).cpu().numpy()
    else:
        center_dists = sdf(centers_arr).reshape(-1)

    active_blocks = []
    pruned_count = 0
    total_grid_voxels = nx * ny * nz
    active_voxels = 0

    for i, (ix, iy, iz, sx, sy, sz) in enumerate(blocks):
        d_center = float(center_dists[i])
        r_bound = block_radii[i] * safety_factor

        if abs(d_center) > r_bound:
            # Block is completely outside or completely inside surface!
            sign = 1.0 if d_center > 0 else -1.0
            global_vol[ix:ix+sx, iy:iy+sy, iz:iz+sz] = sign * 1e3
            pruned_count += 1
        else:
            active_blocks.append((ix, iy, iz, sx, sy, sz))
            active_voxels += sx * sy * sz

    prune_pct = (pruned_count / num_blocks) * 100.0 if num_blocks > 0 else 0.0
    eval_pct = (active_voxels / total_grid_voxels) * 100.0 if total_grid_voxels > 0 else 0.0

    if verbose:
        print(f"Adaptive Octree: {pruned_count}/{num_blocks} blocks pruned ({prune_pct:.1f}% space skipped).")
        print(f"Adaptive Octree: Evaluating fine grid for {active_voxels}/{total_grid_voxels} voxels ({eval_pct:.1f}% active surface).")

    if not active_blocks:
        return global_vol

    # 3. Build coordinate points for active surface blocks only
    active_points_list = []
    slice_maps = []

    for (ix, iy, iz, sx, sy, sz) in active_blocks:
        sub_X = X[ix:ix+sx]
        sub_Y = Y[iy:iy+sy]
        sub_Z = Z[iz:iz+sz]
        grid_x, grid_y, grid_z = np.meshgrid(sub_X, sub_Y, sub_Z, indexing='ij')
        pts = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
        active_points_list.append(pts)
        slice_maps.append((ix, iy, iz, sx, sy, sz))

    all_active_pts = np.vstack(active_points_list).astype(np.float32)
    num_active_pts = len(all_active_pts)

    # 4. Evaluate active points on GPU/CPU
    if target_device != 'numpy':
        import torch
        P_all = torch.as_tensor(all_active_pts, dtype=torch.float32)
        vol_flat = torch.empty(num_active_pts, dtype=torch.float32, device='cpu')

        for i in range(0, num_active_pts, gpu_batch_size):
            batch_P = P_all[i : i + gpu_batch_size].to(target_device)
            with torch.no_grad():
                res = sdf(batch_P)
            vol_flat[i : i + batch_P.shape[0]] = res.reshape(-1).cpu()

        evaluated_dists = vol_flat.numpy().astype(np.float64)
    else:
        evaluated_dists = sdf(all_active_pts).reshape(-1).astype(np.float64)

    # 5. Insert evaluated active voxel distances back into global_vol
    curr_offset = 0
    for (ix, iy, iz, sx, sy, sz) in slice_maps:
        n_pts = sx * sy * sz
        block_vals = evaluated_dists[curr_offset : curr_offset + n_pts].reshape(sx, sy, sz)
        global_vol[ix:ix+sx, iy:iy+sy, iz:iz+sz] = block_vals
        curr_offset += n_pts

    return global_vol
