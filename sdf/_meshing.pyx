# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, fmin, fmax

def surface_nets_extract(volume, callback=None, bint return_quads=False):
    """
    Cython implementation of Surface Nets extraction.
    """
    cdef double[:, :, :] vol = np.ascontiguousarray(volume, dtype=np.float64)
    cdef int nx = vol.shape[0]
    cdef int ny = vol.shape[1]
    cdef int nz = vol.shape[2]
    
    cdef np.ndarray[np.int32_t, ndim=3] vindex = np.full((nx-1, ny-1, nz-1), -1, dtype=np.int32)
    cdef list verts = []
    cdef int vcount = 0
    
    cdef int x, y, z, i, j, k
    cdef bint has_pos, has_neg
    
    # 1. Identify active voxels and place vertices
    for x in range(nx - 1):
        if callback is not None:
            callback(x + 1, nx - 1)
        for y in range(ny - 1):
            for z in range(nz - 1):
                has_pos = False
                has_neg = False
                
                # Check 8 corners
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            if vol[x+i, y+j, z+k] > 0:
                                has_pos = True
                            else:
                                has_neg = True
                                
                if has_pos and has_neg:
                    # Place vertex at centroid
                    verts.append((x + 0.5, y + 0.5, z + 0.5))
                    vindex[x, y, z] = vcount
                    vcount += 1
                    
    cdef list faces = []
    cdef int v00, v01, v10, v11
    
    # 2. X-edges
    for x in range(nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                if (vol[x, y, z] > 0) != (vol[x+1, y, z] > 0):
                    v00 = vindex[x, y-1, z-1]
                    v01 = vindex[x, y, z-1]
                    v10 = vindex[x, y-1, z]
                    v11 = vindex[x, y, z]
                    
                    if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                        if return_quads:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11, v01))
                            else:
                                faces.append((v00, v01, v11, v10))
                        else:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11))
                                faces.append((v00, v11, v01))
                            else:
                                faces.append((v00, v01, v11))
                                faces.append((v00, v11, v10))

    # 3. Y-edges
    for x in range(1, nx - 1):
        for y in range(ny - 1):
            for z in range(1, nz - 1):
                if (vol[x, y, z] > 0) != (vol[x, y+1, z] > 0):
                    v00 = vindex[x-1, y, z-1]
                    v01 = vindex[x, y, z-1]
                    v10 = vindex[x-1, y, z]
                    v11 = vindex[x, y, z]
                    
                    if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                        if return_quads:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v01, v11, v10))
                            else:
                                faces.append((v00, v10, v11, v01))
                        else:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v01, v11))
                                faces.append((v00, v11, v10))
                            else:
                                faces.append((v00, v10, v11))
                                faces.append((v00, v11, v01))

    # 4. Z-edges
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(nz - 1):
                if (vol[x, y, z] > 0) != (vol[x, y, z+1] > 0):
                    v00 = vindex[x-1, y-1, z]
                    v01 = vindex[x, y-1, z]
                    v10 = vindex[x-1, y, z]
                    v11 = vindex[x, y, z]
                    
                    if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                        if return_quads:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11, v01))
                            else:
                                faces.append((v00, v01, v11, v10))
                        else:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11))
                                faces.append((v00, v11, v01))
                            else:
                                faces.append((v00, v01, v11))
                                faces.append((v00, v11, v10))
                            
    cdef int f_dim = 4 if return_quads else 3
    return np.array(verts, dtype=np.float64).reshape(-1, 3), np.array(faces, dtype=np.int32).reshape(-1, f_dim)

cdef inline void eigen_symmetric3x3(double A[3][3], double eigvals[3], double eigvecs[3][3]) noexcept nogil:
    cdef double D[3][3]
    cdef int i, j, it, p, q, r
    cdef double app, aqq, apq, eta, t, c, s, tmp_ip, tmp_iq
    cdef double max_off
    cdef double tmp_val, tmp_col
    
    for i in range(3):
        for j in range(3):
            D[i][j] = A[i][j]
            eigvecs[i][j] = 1.0 if i == j else 0.0
            
    for it in range(25):
        # Find maximum off-diagonal element
        p = 0
        q = 1
        max_off = fabs(D[0][1])
        if fabs(D[0][2]) > max_off:
            max_off = fabs(D[0][2])
            p = 0
            q = 2
        if fabs(D[1][2]) > max_off:
            max_off = fabs(D[1][2])
            p = 1
            q = 2
            
        if max_off < 1e-15:
            break
            
        app = D[p][p]
        aqq = D[q][q]
        apq = D[p][q]
        
        eta = (app - aqq) / (2.0 * apq)
        if eta >= 0.0:
            t = 1.0 / (eta + sqrt(1.0 + eta * eta))
        else:
            t = -1.0 / (-eta + sqrt(1.0 + eta * eta))
            
        c = 1.0 / sqrt(1.0 + t * t)
        s = t * c
        
        # Apply Jacobi rotation to D
        D[p][p] = app + t * apq
        D[q][q] = aqq - t * apq
        D[p][q] = 0.0
        D[q][p] = 0.0
        
        for i in range(3):
            if i != p and i != q:
                tmp_ip = D[i][p]
                tmp_iq = D[i][q]
                D[i][p] = c * tmp_ip + s * tmp_iq
                D[p][i] = D[i][p]
                D[i][q] = -s * tmp_ip + c * tmp_iq
                D[q][i] = D[i][q]
                
        # Accumulate rotation into eigvecs (columns of eigvecs are eigenvectors)
        for i in range(3):
            tmp_ip = eigvecs[i][p]
            tmp_iq = eigvecs[i][q]
            eigvecs[i][p] = c * tmp_ip + s * tmp_iq
            eigvecs[i][q] = -s * tmp_ip + c * tmp_iq

    # Extract eigenvalues
    eigvals[0] = D[0][0]
    eigvals[1] = D[1][1]
    eigvals[2] = D[2][2]
    
    # Sort eigenvalues descending and permute eigenvector columns
    for i in range(2):
        for j in range(i + 1, 3):
            if eigvals[j] > eigvals[i]:
                tmp_val = eigvals[i]
                eigvals[i] = eigvals[j]
                eigvals[j] = tmp_val
                for r in range(3):
                    tmp_col = eigvecs[r][i]
                    eigvecs[r][i] = eigvecs[r][j]
                    eigvecs[r][j] = tmp_col

cdef inline void solve_qef(double ATA[3][3], double ATb[3], double mass_point[3], double qef_threshold, double out_v[3]) noexcept nogil:
    cdef double eigvals[3]
    cdef double eigvecs[3][3]
    cdef double r[3]
    cdef double x[3]
    cdef double max_eig, threshold, proj
    cdef int i, k
    
    # Compute residual r = ATb - ATA * mass_point
    for i in range(3):
        r[i] = ATb[i] - (ATA[i][0] * mass_point[0] + ATA[i][1] * mass_point[1] + ATA[i][2] * mass_point[2])
        x[i] = 0.0
        
    eigen_symmetric3x3(ATA, eigvals, eigvecs)
    
    max_eig = eigvals[0]
    if max_eig < 1e-12:
        out_v[0] = mass_point[0]
        out_v[1] = mass_point[1]
        out_v[2] = mass_point[2]
        return
        
    threshold = qef_threshold * max_eig
    # SVD pseudo-inverse: accumulate projections onto eigenvectors where eigval >= threshold
    for k in range(3):
        if eigvals[k] >= threshold:
            proj = (eigvecs[0][k] * r[0] + eigvecs[1][k] * r[1] + eigvecs[2][k] * r[2]) / eigvals[k]
            x[0] += proj * eigvecs[0][k]
            x[1] += proj * eigvecs[1][k]
            x[2] += proj * eigvecs[2][k]
            
    out_v[0] = mass_point[0] + x[0]
    out_v[1] = mass_point[1] + x[1]
    out_v[2] = mass_point[2] + x[2]


cdef inline void get_normal(double[:, :, :] vol, int x, int y, int z, int nx, int ny, int nz, double nOut[3]) noexcept nogil:
    cdef double dx = vol[min(x+1, nx-1), y, z] - vol[max(x-1, 0), y, z]
    cdef double dy = vol[x, min(y+1, ny-1), z] - vol[x, max(y-1, 0), z]
    cdef double dz = vol[x, y, min(z+1, nz-1)] - vol[x, y, max(z-1, 0)]
    cdef double norm = (dx*dx + dy*dy + dz*dz)**0.5
    if norm > 0:
        nOut[0] = dx / norm
        nOut[1] = dy / norm
        nOut[2] = dz / norm
    else:
        nOut[0] = 1.0
        nOut[1] = 0.0
        nOut[2] = 0.0

cdef inline void add_edge(double v1, double v2, double p1[3], double p2[3], double n1[3], double n2[3], double ATA[3][3], double ATb[3], double mass_point[3], int* num_edges) noexcept nogil:
    cdef double t, norm, dot
    cdef double p[3]
    cdef double n[3]
    
    if (v1 > 0) != (v2 > 0):
        t = v1 / (v1 - v2)
        p[0] = p1[0] + t * (p2[0] - p1[0])
        p[1] = p1[1] + t * (p2[1] - p1[1])
        p[2] = p1[2] + t * (p2[2] - p1[2])
        n[0] = n1[0] + t * (n2[0] - n1[0])
        n[1] = n1[1] + t * (n2[1] - n1[1])
        n[2] = n1[2] + t * (n2[2] - n1[2])
        
        norm = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2])**0.5
        if norm > 0:
            n[0] /= norm
            n[1] /= norm
            n[2] /= norm
            
        ATA[0][0] += n[0]*n[0]
        ATA[0][1] += n[0]*n[1]
        ATA[0][2] += n[0]*n[2]
        ATA[1][0] += n[1]*n[0]
        ATA[1][1] += n[1]*n[1]
        ATA[1][2] += n[1]*n[2]
        ATA[2][0] += n[2]*n[0]
        ATA[2][1] += n[2]*n[1]
        ATA[2][2] += n[2]*n[2]
        
        dot = n[0]*p[0] + n[1]*p[1] + n[2]*p[2]
        ATb[0] += n[0]*dot
        ATb[1] += n[1]*dot
        ATb[2] += n[2]*dot
        
        mass_point[0] += p[0]
        mass_point[1] += p[1]
        mass_point[2] += p[2]
        num_edges[0] += 1

def dual_contouring_extract(volume, float qef_threshold=0.01, callback=None, bint return_quads=False, normal_volume=None):
    """
    Cython implementation of Dual Contouring extraction with SVD QEF solving.
    """
    cdef double[:, :, :] vol = np.ascontiguousarray(volume, dtype=np.float64)
    cdef int nx = vol.shape[0]
    cdef int ny = vol.shape[1]
    cdef int nz = vol.shape[2]
    
    cdef bint has_nvol = normal_volume is not None
    cdef double[:, :, :, :] nvol
    if has_nvol:
        nvol = np.ascontiguousarray(normal_volume, dtype=np.float64)
        
    cdef np.ndarray[np.int32_t, ndim=3] vindex = np.full((nx-1, ny-1, nz-1), -1, dtype=np.int32)
    cdef list verts = []
    cdef int vcount = 0
    
    cdef int x, y, z, i, j, k, idx
    cdef double vx, vy, vz
    cdef double c_pts[8][3]
    cdef double c_v[8]
    cdef double c_n[8][3]
    
    cdef double ATA[3][3]
    cdef double ATb[3]
    cdef double out_v[3]
    cdef double mass_point[3]
    cdef int num_edges
    
    # Edges defined by corner indices
    cdef int edges[12][2]
    edges[0][0] = 0; edges[0][1] = 4
    edges[1][0] = 2; edges[1][1] = 6
    edges[2][0] = 1; edges[2][1] = 5
    edges[3][0] = 3; edges[3][1] = 7
    edges[4][0] = 0; edges[4][1] = 2
    edges[5][0] = 4; edges[5][1] = 6
    edges[6][0] = 1; edges[6][1] = 3
    edges[7][0] = 5; edges[7][1] = 7
    edges[8][0] = 0; edges[8][1] = 1
    edges[9][0] = 4; edges[9][1] = 5
    edges[10][0] = 2; edges[10][1] = 3
    edges[11][0] = 6; edges[11][1] = 7
    
    # 1. Identify active voxels and solve QEF
    for x in range(nx - 1):
        if callback is not None:
            callback(x + 1, nx - 1)
        for y in range(ny - 1):
            for z in range(nz - 1):
                # Evaluate 8 corners
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            idx = i*4 + j*2 + k
                            c_pts[idx][0] = x + i
                            c_pts[idx][1] = y + j
                            c_pts[idx][2] = z + k
                            c_v[idx] = vol[x+i, y+j, z+k]
                            if has_nvol:
                                c_n[idx][0] = nvol[x+i, y+j, z+k, 0]
                                c_n[idx][1] = nvol[x+i, y+j, z+k, 1]
                                c_n[idx][2] = nvol[x+i, y+j, z+k, 2]
                            else:
                                get_normal(vol, x+i, y+j, z+k, nx, ny, nz, c_n[idx])
                
                # Reset QEF
                ATA[0][0] = 0; ATA[0][1] = 0; ATA[0][2] = 0
                ATA[1][0] = 0; ATA[1][1] = 0; ATA[1][2] = 0
                ATA[2][0] = 0; ATA[2][1] = 0; ATA[2][2] = 0
                ATb[0] = 0; ATb[1] = 0; ATb[2] = 0
                mass_point[0] = 0; mass_point[1] = 0; mass_point[2] = 0
                num_edges = 0
                
                # Accumulate QEF for each active edge
                for i in range(12):
                    add_edge(c_v[edges[i][0]], c_v[edges[i][1]], 
                             c_pts[edges[i][0]], c_pts[edges[i][1]], 
                             c_n[edges[i][0]], c_n[edges[i][1]], 
                             ATA, ATb, mass_point, &num_edges)
                
                if num_edges > 0:
                    mass_point[0] /= num_edges
                    mass_point[1] /= num_edges
                    mass_point[2] /= num_edges
                    
                    solve_qef(ATA, ATb, mass_point, qef_threshold, out_v)
                    
                    if fabs(out_v[0] - mass_point[0]) > 1.0 or \
                       fabs(out_v[1] - mass_point[1]) > 1.0 or \
                       fabs(out_v[2] - mass_point[2]) > 1.0:
                        verts.append((mass_point[0], mass_point[1], mass_point[2]))
                    else:
                        vx = fmin(fmax(out_v[0], <double>x), <double>(x + 1))
                        vy = fmin(fmax(out_v[1], <double>y), <double>(y + 1))
                        vz = fmin(fmax(out_v[2], <double>z), <double>(z + 1))
                        verts.append((vx, vy, vz))
                        
                    vindex[x, y, z] = vcount
                    vcount += 1
                    
    cdef list faces = []
    cdef int v00, v01, v10, v11
    
    # 2. X-edges
    for x in range(nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                if (vol[x, y, z] > 0) != (vol[x+1, y, z] > 0):
                    v00 = vindex[x, y-1, z-1]
                    v01 = vindex[x, y, z-1]
                    v10 = vindex[x, y-1, z]
                    v11 = vindex[x, y, z]
                    
                    if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                        if return_quads:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11, v01))
                            else:
                                faces.append((v00, v01, v11, v10))
                        else:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11))
                                faces.append((v00, v11, v01))
                            else:
                                faces.append((v00, v01, v11))
                                faces.append((v00, v11, v10))

    # 3. Y-edges
    for x in range(1, nx - 1):
        for y in range(ny - 1):
            for z in range(1, nz - 1):
                if (vol[x, y, z] > 0) != (vol[x, y+1, z] > 0):
                    v00 = vindex[x-1, y, z-1]
                    v01 = vindex[x, y, z-1]
                    v10 = vindex[x-1, y, z]
                    v11 = vindex[x, y, z]
                    
                    if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                        if return_quads:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v01, v11, v10))
                            else:
                                faces.append((v00, v10, v11, v01))
                        else:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v01, v11))
                                faces.append((v00, v11, v10))
                            else:
                                faces.append((v00, v10, v11))
                                faces.append((v00, v11, v01))

    # 4. Z-edges
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(nz - 1):
                if (vol[x, y, z] > 0) != (vol[x, y, z+1] > 0):
                    v00 = vindex[x-1, y-1, z]
                    v01 = vindex[x, y-1, z]
                    v10 = vindex[x-1, y, z]
                    v11 = vindex[x, y, z]
                    
                    if v00 >= 0 and v01 >= 0 and v10 >= 0 and v11 >= 0:
                        if return_quads:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11, v01))
                            else:
                                faces.append((v00, v01, v11, v10))
                        else:
                            if vol[x, y, z] > 0:
                                faces.append((v00, v10, v11))
                                faces.append((v00, v11, v01))
                            else:
                                faces.append((v00, v01, v11))
                                faces.append((v00, v11, v10))
                            
    cdef int f_dim = 4 if return_quads else 3
    return np.array(verts, dtype=np.float64).reshape(-1, 3), np.array(faces, dtype=np.int32).reshape(-1, f_dim)

def relax_iteration_c(
    np.ndarray[np.float64_t, ndim=2] verts,
    np.ndarray[np.int32_t, ndim=1] adj_offsets,
    np.ndarray[np.int32_t, ndim=1] adj_indices,
    np.ndarray[np.int32_t, ndim=2] crease_nbrs,
    np.ndarray[np.float64_t, ndim=2] crease_tangents,
    np.ndarray[np.int8_t, ndim=1] v_type,
    np.ndarray[np.float64_t, ndim=2] grads,
    np.ndarray[np.float64_t, ndim=1] dists,
    double alpha,
    bint snap_to_surface=True
):
    cdef int V = verts.shape[0]
    cdef double[:, :] v_view = verts
    cdef int[:] off_view = adj_offsets
    cdef int[:] ind_view = adj_indices
    cdef int[:, :] cn_view = crease_nbrs
    cdef double[:, :] ct_view = crease_tangents
    cdef signed char[:] vt_view = v_type
    cdef double[:, :] g_view = grads
    cdef double[:] d_view = dists
    
    cdef int i, j, start, end, count, nbr, c0, c1
    cdef double mx, my, mz, dx, dy, dz, dot, nx, ny, nz, tx, ty, tz, d
    
    with nogil:
        for i in range(V):
            if vt_view[i] == 2:
                if snap_to_surface:
                    d = d_view[i]
                    nx = g_view[i, 0]; ny = g_view[i, 1]; nz = g_view[i, 2]
                    v_view[i, 0] -= d * nx
                    v_view[i, 1] -= d * ny
                    v_view[i, 2] -= d * nz
                continue
                
            if vt_view[i] == 0:
                start = off_view[i]
                end = off_view[i+1]
                count = end - start
                if count > 0:
                    mx = 0.0; my = 0.0; mz = 0.0
                    for j in range(start, end):
                        nbr = ind_view[j]
                        mx += v_view[nbr, 0]
                        my += v_view[nbr, 1]
                        mz += v_view[nbr, 2]
                    dx = (mx / count) - v_view[i, 0]
                    dy = (my / count) - v_view[i, 1]
                    dz = (mz / count) - v_view[i, 2]
                    
                    nx = g_view[i, 0]; ny = g_view[i, 1]; nz = g_view[i, 2]
                    dot = dx * nx + dy * ny + dz * nz
                    dx -= dot * nx
                    dy -= dot * ny
                    dz -= dot * nz
                    
                    v_view[i, 0] += alpha * dx
                    v_view[i, 1] += alpha * dy
                    v_view[i, 2] += alpha * dz
                    
                    if snap_to_surface:
                        d = d_view[i]
                        v_view[i, 0] -= d * nx
                        v_view[i, 1] -= d * ny
                        v_view[i, 2] -= d * nz
                        
            elif vt_view[i] == 1:
                c0 = cn_view[i, 0]
                c1 = cn_view[i, 1]
                dx = 0.5 * (v_view[c0, 0] + v_view[c1, 0]) - v_view[i, 0]
                dy = 0.5 * (v_view[c0, 1] + v_view[c1, 1]) - v_view[i, 1]
                dz = 0.5 * (v_view[c0, 2] + v_view[c1, 2]) - v_view[i, 2]
                
                tx = ct_view[i, 0]
                ty = ct_view[i, 1]
                tz = ct_view[i, 2]
                dot = dx * tx + dy * ty + dz * tz
                
                v_view[i, 0] += alpha * (dot * tx)
                v_view[i, 1] += alpha * (dot * ty)
                v_view[i, 2] += alpha * (dot * tz)
                
                if snap_to_surface:
                    d = d_view[i]
                    nx = g_view[i, 0]; ny = g_view[i, 1]; nz = g_view[i, 2]
                    v_view[i, 0] -= d * nx
                    v_view[i, 1] -= d * ny
                    v_view[i, 2] -= d * nz

