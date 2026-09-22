# RFC: Neural SDF Baking and Quad Meshing Strategy

## Context
The `sdf` library provides a powerful, programmatic API for generating 3D models via constructive solid geometry. Currently, meshes are extracted using isosurface extraction techniques (e.g., Marching Cubes, Dual Contouring) which output dense, triangulated meshes.

To support production-grade pipelines—which often require clean, feature-aligned quadrilateral topologies for deformation, UV mapping, and subdivision—we need an advanced meshing strategy. Recent state-of-the-art research demonstrates that Neural Signed Distance Fields (Neural SDFs) coupled with Neural Frame Fields can reliably generate these high-quality quad meshes.

## Proposal
We propose implementing **Neural Baking and Quad Extraction** as a new, optional mesh generation strategy directly within the library. 

Since the library operates fundamentally as a programmatic generator without a live viewport UI, this feature does not need to be split into a "real-time vs. offline" paradigm. Instead, it will simply be invoked as an alternative generation strategy, for example:
`f.save('out.obj', strategy='neural_quad')`.

### The Strategy Pipeline
When the neural generation strategy is invoked, the pipeline will execute the following steps:

1. **Dataset Generation (Sampling):**
   The existing procedural SDF `f(p)` will be sampled heavily within its bounding box to generate a ground-truth dataset of point-distance pairs $(x, y, z) \rightarrow d$, alongside analytical gradients (normals).
2. **Neural Field Baking (Optimization):**
   A lightweight neural network (MLP) will be instantiated and trained on the fly (via PyTorch or a similar accelerated backend) to overfit the procedural geometry.
   - **Eikonal Regularization:** The network will use PyTorch's `autograd` to calculate spatial gradients natively. We will apply an Eikonal loss (`||∇d|| = 1`) to ensure the neural field is a mathematically valid SDF.
   - **Tangent-Projected Cross Fields:** Instead of naively predicting vectors, the network will predict a single 3D guidance vector. This vector is dynamically projected onto the local tangent plane (orthogonal to the autograd normal) to construct an inherently orthogonal cross-field `(u, v, n)`. This guarantees mathematical correctness without penalty losses.
   - **Implicit Hessian Alignment:** Pre-computing explicit principal curvatures fails on umbilic geometries (e.g., spheres or planes) due to mathematical ambiguity, generating noisy training targets. Instead, we must use PyTorch `autograd` to calculate the Neural Hessian (directional derivatives) and apply an *implicit alignment loss* that forces the cross-field to align with the curvature natively, allowing it to smoothly bridge singular regions.
3. **Quad Extraction (The Solver):**
   Generating the final quads from the baked neural field requires a robust parameterization solver.
   - **Surface Projection:** The base surface is extracted (or virtually sampled) using the Neural SDF. Any queried point can be perfectly snapped to the surface using Newton-Raphson steps (`p = p - SDF(p) * ∇SDF(p)`).
   - **Global Parameterization (C++ Requirement):** Neural networks intrinsically output continuous fields, not discrete manifold topologies. Python-based workarounds (like streamline tracing or shrinkwrapping) will fail on arbitrary geometries (e.g., non-zero genus shapes). As proven by SOTA architectures (e.g., NeurCross), the Python/ML backend must *exclusively* be responsible for baking the fields. It will then hand off the dense triangulated zero-isosurface and the predicted cross-field data directly to a rigorous C++ Mixed-Integer Quadrangulation (MIQ) solver (like `libQEx` or `libigl`) to weave the seamless global $(u, v)$ parameterization.
   - **Isoline Tracing:** The C++ solver extracts the integer iso-lines of the parameterization to form the final, watertight manifold quad mesh.

## Technical Feasibility & Lift
- **Compatibility:** Our existing API inherently models the exact mapping `(N, 3) -> (N, 1)` needed to train the network.
- **Backend:** A deep learning backend (e.g., PyTorch) will need to be introduced as an optional dependency for this specific generation strategy.
- **Next Steps:** Develop a small, standalone neural baking prototype to validate the optimization speed and fidelity of an MLP overfitting our procedural SDFs.

## Performance & Integration Guidelines
To ensure the library remains performant and avoids memory leaks, the integration of deep learning and C++ solvers must adhere to the following constraints:

1. **Zero-Copy Marshaling for Dataset Generation:**
   Dataset generation (computing distances, analytical normals, and principal curvatures) must not bottleneck on the CPU. We will leverage the existing Taichi backend to execute these operations on the GPU and utilize `taichi.to_torch()` or `DLPack` for zero-copy memory transfers into the PyTorch training loop.
2. **Strict Garbage Collection:**
   The PyTorch dependency must be imported lazily to keep the library's base footprint small. After the neural baking phase concludes, explicit garbage collection (`torch.cuda.empty_cache()`, `del` tensors) must be invoked to prevent GPU memory leaks across multiple generation calls.
3. **Asynchronous GPU Querying for C-Extensions:**
   When the C++ Dual Contouring algorithm requests distance samples, it must not trigger individual, GIL-blocking Python callbacks per voxel. The C++ backend must batch queries and request evaluations asynchronously, allowing the neural field to process them en masse on the GPU.
4. **GIL Safety and Multithreading in the Solver:**
   The parameterization solver (e.g., `libigl` bindings) will perform massive sparse matrix solves. All Cython/pybind11 wrappers for these routines must explicitly release the Python GIL (`with nogil:`), allowing the host application to remain responsive or execute parallel Python threads while the C++ solver runs utilizing OpenMP/MKL.
