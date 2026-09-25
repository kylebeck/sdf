import numpy as np

def weld_quad_mesh(verts, quads, decimals=5):
    """
    Deduplicates coincident vertices and eliminates degenerate sliver quads.
    """
    if len(verts) == 0 or len(quads) == 0:
        return verts, quads

    u_verts, inv = np.unique(np.round(verts, decimals), axis=0, return_inverse=True)
    new_quads = inv[quads]
    
    # Keep only quads where all 4 vertices are strictly distinct
    q0, q1, q2, q3 = new_quads[:, 0], new_quads[:, 1], new_quads[:, 2], new_quads[:, 3]
    valid_mask = (q0 != q1) & (q0 != q2) & (q0 != q3) & (q1 != q2) & (q1 != q3) & (q2 != q3)
    valid_quads = new_quads[valid_mask]
    
    # Compact vertex array by removing unreferenced vertices
    used_v = np.unique(valid_quads)
    remap = np.full(len(u_verts), -1, dtype=np.int32)
    remap[used_v] = np.arange(len(used_v))
    final_verts = u_verts[used_v]
    final_quads = remap[valid_quads]
    
    return final_verts, final_quads

def get_point_evaluator(sdf_obj, device='auto', verbose=0):
    if device in ('taichi', 'auto'):
        try:
            from . import taichi_backend
            if bool(taichi_backend.HAS_TAICHI):
                _ = taichi_backend.trace_sdf_to_code(sdf_obj)
                arch = None
                return lambda pts: taichi_backend.evaluate_points_and_gradients_taichi(
                    sdf_obj, pts, arch=arch, verbose=verbose
                )
        except Exception:
            if device == 'taichi':
                raise

    def _cpu_eval(pts, eps=1e-5):
        dists = sdf_obj(pts).reshape(-1).astype(np.float64)
        dx = np.array([eps, 0.0, 0.0])
        dy = np.array([0.0, eps, 0.0])
        dz = np.array([0.0, 0.0, eps])

        fx1 = sdf_obj(pts + dx).reshape(-1)
        fx0 = sdf_obj(pts - dx).reshape(-1)
        fy1 = sdf_obj(pts + dy).reshape(-1)
        fy0 = sdf_obj(pts - dy).reshape(-1)
        fz1 = sdf_obj(pts + dz).reshape(-1)
        fz0 = sdf_obj(pts - dz).reshape(-1)

        gx = (fx1 - fx0) / (2.0 * eps)
        gy = (fy1 - fy0) / (2.0 * eps)
        gz = (fz1 - fz0) / (2.0 * eps)

        grads = np.stack([gx, gy, gz], axis=-1).astype(np.float64)
        norms = np.linalg.norm(grads, axis=-1, keepdims=True)
        grads /= np.where(norms > 1e-12, norms, 1.0)
        return dists, grads

    return _cpu_eval

def relax_quad_mesh(
        sdf, verts, quads,
        iters=8, alpha=0.4, crease_angle_deg=35.0,
        snap_to_surface=True, device='auto', verbose=False):
    """
    Projected Tangential Relaxation using Taichi JIT and Cython C execution.
    """
    V = len(verts)
    if V == 0 or len(quads) == 0:
        return verts, quads

    verts = np.ascontiguousarray(verts, dtype=np.float64)

    e0 = quads[:, [0, 1]]
    e1 = quads[:, [1, 2]]
    e2 = quads[:, [2, 3]]
    e3 = quads[:, [3, 0]]
    all_edges = np.stack([e0, e1, e2, e3], axis=1).reshape(-1, 2)
    f_indices = np.repeat(np.arange(len(quads)), 4)
    sorted_edges = np.sort(all_edges, axis=1)

    edge_map = {}
    for idx, edge in enumerate(sorted_edges):
        key = (edge[0], edge[1])
        edge_map.setdefault(key, []).append(f_indices[idx])

    neighbors = [set() for _ in range(V)]
    for u, v in sorted_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    d1 = verts[quads[:, 2]] - verts[quads[:, 0]]
    d2 = verts[quads[:, 3]] - verts[quads[:, 1]]
    fnormals = np.cross(d1, d2)
    fn_len = np.linalg.norm(fnormals, axis=-1, keepdims=True)
    fnormals = fnormals / np.where(fn_len > 1e-12, fn_len, 1.0)

    cos_thresh = np.cos(np.radians(crease_angle_deg))
    vertex_crease_edges = [[] for _ in range(V)]

    for (u, v), f_list in edge_map.items():
        if len(f_list) == 2:
            dot = np.dot(fnormals[f_list[0]], fnormals[f_list[1]])
            if dot < cos_thresh:
                vertex_crease_edges[u].append(v)
                vertex_crease_edges[v].append(u)
        elif len(f_list) == 1:
            vertex_crease_edges[u].append(v)
            vertex_crease_edges[v].append(u)

    v_type = np.zeros(V, dtype=np.int8)
    crease_tangents = np.zeros((V, 3), dtype=np.float64)
    crease_nbrs = np.full((V, 2), -1, dtype=np.int32)

    for i in range(V):
        c_nbrs = vertex_crease_edges[i]
        if len(c_nbrs) == 0:
            v_type[i] = 0
        elif len(c_nbrs) == 2:
            v_type[i] = 1
            crease_nbrs[i, 0] = c_nbrs[0]
            crease_nbrs[i, 1] = c_nbrs[1]
            t = verts[c_nbrs[1]] - verts[c_nbrs[0]]
            norm_t = np.linalg.norm(t)
            if norm_t > 1e-12:
                crease_tangents[i] = t / norm_t
            else:
                v_type[i] = 2
        else:
            v_type[i] = 2

    counts = [len(nbrs) for nbrs in neighbors]
    adj_offsets = np.zeros(V + 1, dtype=np.int32)
    adj_offsets[1:] = np.cumsum(counts)
    if V > 0 and sum(counts) > 0:
        adj_indices = np.concatenate([list(nbrs) for nbrs in neighbors]).astype(np.int32)
    else:
        adj_indices = np.empty(0, dtype=np.int32)

    eval_fn = get_point_evaluator(sdf, device=device, verbose=int(verbose))

    try:
        from . import _meshing
        use_cython_relax = True
    except ImportError:
        use_cython_relax = False

    smooth_idx = np.where(v_type == 0)[0]
    crease_idx = np.where(v_type == 1)[0]
    neighbor_lists = [np.array(list(nbrs), dtype=np.int32) for nbrs in neighbors]

    for it in range(iters):
        dists, grads = eval_fn(verts)

        if use_cython_relax:
            _meshing.relax_iteration_c(
                verts, adj_offsets, adj_indices, crease_nbrs,
                crease_tangents, v_type, grads, dists,
                float(alpha), bool(snap_to_surface)
            )
        else:
            delta = np.zeros_like(verts)
            for i in smooth_idx:
                nbrs = neighbor_lists[i]
                if len(nbrs) > 0:
                    delta[i] = np.mean(verts[nbrs], axis=0) - verts[i]
            for i in crease_idx:
                c0, c1 = crease_nbrs[i]
                delta[i] = 0.5 * (verts[c0] + verts[c1]) - verts[i]

            if len(smooth_idx) > 0:
                n_smooth = grads[smooth_idx]
                dot_s = np.sum(delta[smooth_idx] * n_smooth, axis=-1, keepdims=True)
                delta[smooth_idx] -= dot_s * n_smooth

            if len(crease_idx) > 0:
                t_crease = crease_tangents[crease_idx]
                dot_c = np.sum(delta[crease_idx] * t_crease, axis=-1, keepdims=True)
                delta[crease_idx] = dot_c * t_crease

            verts += alpha * delta

            if snap_to_surface:
                verts -= dists.reshape(-1, 1) * grads

    if snap_to_surface:
        dists, grads = eval_fn(verts)
        verts -= dists.reshape(-1, 1) * grads

    return verts, quads
