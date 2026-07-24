import time
from sdf import sphere, box, torus, generate

# Create compound object
s = sphere(1) | box(1.5).translate((0.5, 0.5, 0)) - torus(1.2, 0.2)

print("\n--- Benchmarking CPU (NumPy) ---")
t0 = time.time()
v_cpu, f_cpu = generate(s, step=0.03, device='numpy', verbose=True)
t_cpu = time.time() - t0

print("\n--- Benchmarking PyTorch GPU (MPS) ---")
t0 = time.time()
v_torch, f_torch = generate(s, step=0.03, device='mps', verbose=True)
t_torch = time.time() - t0

print("\n--- Benchmarking Taichi JIT Shader GPU (Metal) ---")
t0 = time.time()
v_taichi, f_taichi = generate(s, step=0.03, device='taichi', verbose=True)
t_taichi = time.time() - t0

print("\n==========================================")
print(f"NumPy CPU Time:     {t_cpu:.3f}s ({len(v_cpu)} verts, {len(f_cpu)} faces)")
print(f"PyTorch MPS Time:   {t_torch:.3f}s ({len(v_torch)} verts, {len(f_torch)} faces)")
print(f"Taichi Metal Time:  {t_taichi:.3f}s ({len(v_taichi)} verts, {len(f_taichi)} faces)")
print("==========================================")
