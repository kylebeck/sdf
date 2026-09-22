# cython: language_level=3
import numpy as np
cimport numpy as np

def surface_nets_extract(volume, callback=None):
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
                        if vol[x, y, z] > 0:
                            faces.append((v00, v10, v11))
                            faces.append((v00, v11, v01))
                        else:
                            faces.append((v00, v01, v11))
                            faces.append((v00, v11, v10))
                            
    return np.array(verts, dtype=np.float64).reshape(-1, 3), np.array(faces, dtype=np.int32).reshape(-1, 3)

cdef inline double det3x3(double M[3][3]):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
            M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
            M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))

cdef inline bint invert3x3(double M[3][3], double invOut[3][3], double threshold):
    cdef double det = det3x3(M)
    if abs(det) < threshold:
        return False
        
    cdef double invdet = 1.0 / det
    invOut[0][0] = (M[1][1] * M[2][2] - M[2][1] * M[1][2]) * invdet
    invOut[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invdet
    invOut[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invdet
    invOut[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invdet
    invOut[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invdet
    invOut[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * invdet
    invOut[2][0] = (M[1][0] * M[2][1] - M[2][0] * M[1][1]) * invdet
    invOut[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * invdet
    invOut[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * invdet
    return True

cdef inline void get_normal(double[:, :, :] vol, int x, int y, int z, int nx, int ny, int nz, double nOut[3]):
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

cdef inline void add_edge(double v1, double v2, double p1[3], double p2[3], double n1[3], double n2[3], double ATA[3][3], double ATb[3], double mass_point[3], int* num_edges):
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

def dual_contouring_extract(volume, float qef_threshold, callback=None):
    """
    Cython implementation of Dual Contouring extraction.
    """
    cdef double[:, :, :] vol = np.ascontiguousarray(volume, dtype=np.float64)
    cdef int nx = vol.shape[0]
    cdef int ny = vol.shape[1]
    cdef int nz = vol.shape[2]
    
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
    cdef double invATA[3][3]
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
                    
                    if invert3x3(ATA, invATA, qef_threshold):
                        vx = invATA[0][0]*ATb[0] + invATA[0][1]*ATb[1] + invATA[0][2]*ATb[2]
                        vy = invATA[1][0]*ATb[0] + invATA[1][1]*ATb[1] + invATA[1][2]*ATb[2]
                        vz = invATA[2][0]*ATb[0] + invATA[2][1]*ATb[1] + invATA[2][2]*ATb[2]
                        
                        # Clip to avoid huge spikes if QEF is ill-conditioned but passed threshold
                        if vx < x - 1 or vx > x + 2 or vy < y - 1 or vy > y + 2 or vz < z - 1 or vz > z + 2:
                            verts.append((mass_point[0], mass_point[1], mass_point[2]))
                        else:
                            verts.append((vx, vy, vz))
                    else:
                        verts.append((mass_point[0], mass_point[1], mass_point[2]))
                        
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
                        if vol[x, y, z] > 0:
                            faces.append((v00, v10, v11))
                            faces.append((v00, v11, v01))
                        else:
                            faces.append((v00, v01, v11))
                            faces.append((v00, v11, v10))
                            
    return np.array(verts, dtype=np.float64).reshape(-1, 3), np.array(faces, dtype=np.int32).reshape(-1, 3)
