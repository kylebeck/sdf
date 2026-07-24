import time
from sdf import sphere, box, torus, generate

# Create compound object
s = (sphere(1) | box(1.5).translate((0.5, 0.5, 0)) - torus(1.2, 0.2)).repeat((2.5, 2.5, 2.5), count=(2, 2, 2))

print("\n--- Benchmarking Uniform GPU Sampling (adaptive=False) ---")
t0 = time.time()
v_dense, f_dense = generate(s, step=0.02, device='auto', adaptive=False, verbose=True)
t_dense = time.time() - t0

print("\n--- Benchmarking Adaptive Octree GPU Sampling (adaptive=True) ---")
t0 = time.time()
v_opt, f_opt = generate(s, step=0.02, device='auto', adaptive=True, verbose=True)
t_opt = time.time() - t0

print("\n==========================================")
print(f"Uniform Grid GPU Time:  {t_dense:.3f}s ({len(v_dense)} verts, {len(f_dense)} faces)")
print(f"Adaptive Octree Time:   {t_opt:.3f}s ({len(v_opt)} verts, {len(f_opt)} faces)")
if t_opt > 0:
    print(f"Octree Speedup:         {t_dense / t_opt:.2f}x")
print("==========================================")
