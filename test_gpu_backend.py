import time
import torch
import numpy as np
from sdf import sphere, box, torus, union, difference, generate

print(f"PyTorch Version: {torch.__version__}")
print(f"Apple Silicon MPS Available: {torch.backends.mps.is_available()}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Create a complex compound object
s = sphere(1) | box(1.5).translate((0.5, 0.5, 0)) - torus(1.2, 0.2)

print("\n--- Benchmarking CPU (NumPy) ---")
t0 = time.time()
verts_cpu, faces_cpu = generate(s, step=0.04, device='numpy', verbose=True)
t_cpu = time.time() - t0

print("\n--- Benchmarking GPU (PyTorch Auto) ---")
t0 = time.time()
verts_gpu, faces_gpu = generate(s, step=0.04, device='auto', verbose=True)
t_gpu = time.time() - t0

print("\n==========================================")
print(f"NumPy CPU Time:  {t_cpu:.3f}s ({len(verts_cpu)} verts, {len(faces_cpu)} faces)")
print(f"PyTorch GPU Time: {t_gpu:.3f}s ({len(verts_gpu)} verts, {len(faces_gpu)} faces)")
if t_gpu > 0:
    print(f"Speedup: {t_cpu / t_gpu:.2f}x")
print("==========================================")
