from sdf import sphere, box
import sdf.core
import time

s = sphere(1)

print("Marching Cubes...")
start = time.time()
verts_mc, faces_mc = sdf.core.generate(s, step=0.05, method='marching_cubes')
print(f"MC: {len(verts_mc)} verts, {len(faces_mc)} faces in {time.time()-start:.2f}s")

print("Surface Nets...")
start = time.time()
verts_sn, faces_sn = sdf.core.generate(s, step=0.05, method='surface_nets')
print(f"SN: {len(verts_sn)} verts, {len(faces_sn)} faces in {time.time()-start:.2f}s")

print("Dual Contouring...")
start = time.time()
verts_dc, faces_dc = sdf.core.generate(s, step=0.05, method='dual_contouring', qef_threshold=1e-3)
print(f"DC: {len(verts_dc)} verts, {len(faces_dc)} faces in {time.time()-start:.2f}s")
