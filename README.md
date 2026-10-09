# sdf: High-Performance Procedural CAD & Signed Distance Functions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Hardware: CUDA | MPS | Taichi](https://img.shields.io/badge/Acceleration-CUDA%20%7C%20Apple%20Silicon%20%7C%20Taichi-success.svg)]()
[![Meshing: Dual Contouring | Surface Nets](https://img.shields.io/badge/Meshing-Dual%20Contouring%20%7C%20Quads-orange.svg)]()
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Project-72a4f2?style=flat&logo=kofi&logoColor=white)](https://ko-fi.com/A0P828G4OQ)

<p align="left">
  <a href="https://ko-fi.com/A0P828G4OQ" target="_blank">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" height="36" alt="Support this project on Ko-fi" />
  </a>
</p>

Generate 3D meshes based on **SDFs (Signed Distance Functions)** with a clean, pythonic API.

This repository is an actively maintained, modernized fork of [Michael Fogleman's `sdf`](https://github.com/fogleman/sdf), engineered for 3D printing, computational geometry, procedural design, and technical CAD modeling.

Special thanks to [Inigo Quilez](https://iquilezles.org/) for his pioneering documentation and formulas for signed distance functions:
- [3D Signed Distance Functions](https://iquilezles.org/www/articles/distfunctions/distfunctions.htm)
- [2D Signed Distance Functions](https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm)

---

## Why This Fork? (What's New)

The original `sdf` library introduced an exceptionally elegant API for procedural solid modeling. This fork builds upon that foundation to provide industrial-grade performance, modern meshing algorithms, and advanced CAD operations required for complex mechanical assemblies and organic consumer products:

| Feature Area | Original `fogleman/sdf` | Modernized Fork (`kylebeck/sdf`) |
| :--- | :--- | :--- |
| **Compute Backends** | CPU NumPy / ThreadPool | **PyTorch GPU** (NVIDIA CUDA & Apple Silicon MPS) + **Taichi JIT** shaders + vectorized CPU fallback |
| **Meshing Engines** | Standard Marching Cubes (`skimage`) | **Dual Contouring** (with SVD QEF solving), **Surface Nets**, and Cython-accelerated Marching Cubes |
| **Sharp CAD Features** | Beveled/faceted edge artifacts | **Preserves sharp edges & corners** via Quadric Error Function (QEF) minimization in Dual Contouring |
| **Topology & Quads** | Triangles only | **Native Quad Mesh export (`.obj`)** with feature welding and iterative manifold relaxation |
| **Volume Sampling** | Uniform dense grid | **Adaptive Octree sampling**: hierarchically prunes empty voxel space to slash generation time |
| **Procedural Patterns** | Cartesian `repeat` & `circular_array` | **Poisson disc blue-noise arrays (`poisson_array`)** in 2D & 3D with variable radius weighting |
| **Array Smoothing** | Hard boolean cuts only | **Smooth blend factor (`k`)** support across circular and Poisson arrays |
| **Transforms & Primitives** | Translation, Euler rotation, scale | **4x4 Affine matrix transforms (`multmatrix`)**, Bézier curves, line segments, regular polygons, vesica |
| **CLI Experience** | Basic text progress | **Responsive Charm-inspired progress engine** with live throughput, slice counters, and summary metrics |

---

## Quickstart

<img width=350 align="right" src="docs/images/example.png">

Here is a complete example that generates the canonical [Constructive Solid Geometry (CSG)](https://en.wikipedia.org/wiki/Constructive_solid_geometry) model shown on the right. Note the intuitive operator syntax for union (`|`), difference (`-`), and intersection (`&`):

```python
from sdf import *

# Constructive Solid Geometry
f = sphere(1) & box(1.5)

c = cylinder(0.5)
f -= c.orient(X) | c.orient(Y) | c.orient(Z)

# Export watertight binary STL (or .obj, .ply, .3mf)
f.save('out.stl')
```

### Sharp Edge Preservation & Quad Mesh Export

To preserve crisp mechanical 90° edges without excessive subdivision, or to export clean quad topologies for subdivision modeling in Blender or CAD:

```python
# Extract sharp mechanical features with Dual Contouring
f.save('out.obj', method='dual_contouring')  # Produces native quad topology
```

---

## Examples Gallery

| [gearlike.py](examples/gearlike.py) | [knurling.py](examples/knurling.py) | [blobby.py](examples/blobby.py) | [weave.py](examples/weave.py) |
| :---: | :---: | :---: | :---: |
| ![gearlike](docs/images/gearlike.png) | ![knurling](docs/images/knurling.png) | ![blobby](docs/images/blobby.png) | ![weave](docs/images/weave.png) |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- C/C++ build tools (GCC, Clang, or MSVC) for building the high-speed Cython meshing extension

### Setup in a Virtual Environment

```bash
# Clone the repository
git clone https://github.com/kylebeck/sdf.git
cd sdf

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install core package in editable mode (builds Cython extensions automatically)
pip install -e .
```

Verify that the installation was successful:

```bash
python examples/example.py  # Generates out.stl
```

### Optional GPU & Shader Acceleration

Install the optional acceleration backends to unlock GPU compute:

```bash
# Install PyTorch for CUDA or Apple Silicon (MPS) acceleration:
pip install -e ".[gpu]"       # or: pip install torch

# Install Taichi for JIT GPU shader compilation:
pip install -e ".[taichi]"    # or: pip install taichi

# Install all performance extras:
pip install -e ".[all]"
```

If you modify `sdf/_meshing.pyx` during development, rebuild the extension in-place:

```bash
python setup.py build_ext --inplace
```

---

## Meshing Strategies & Export

The `save()` and `generate()` functions provide flexible control over meshing algorithms, compute hardware, and surface relaxation.

```python
f.save(
    'part.obj',
    step=0.02,                   # Grid resolution step
    method='dual_contouring',    # 'dual_contouring', 'surface_nets', or 'marching_cubes'
    device='auto',               # 'auto', 'cuda', 'mps', 'taichi', or 'numpy'
    adaptive=True,               # Adaptive octree spatial acceleration
    qef_threshold=1e-3,          # Feature sharpness tolerance for Dual Contouring
    iters=8,                     # Relaxation iterations for quad meshes
    alpha=0.4,                   # Relaxation step factor
    crease_angle=35.0            # Crease-preserving threshold (degrees)
)
```

### Meshing Algorithms

1. **Dual Contouring (`method='dual_contouring'`)**:
   - **Best for:** Mechanical parts, sharp corners, brackets, enclosures, and quad mesh export.
   - Preserves sharp edges by solving the Quadric Error Function (QEF) via Singular Value Decomposition (SVD) inside surface voxels.
   - When saving to `.obj`, outputs **native quadrilateral faces**.
2. **Surface Nets (`method='surface_nets'`)**:
   - **Best for:** Smooth organic surfaces, ergonomic grips, and bio-inspired forms.
   - Positions vertices smoothly on the zero-crossing iso-surface with balanced triangle edge ratios.
3. **Marching Cubes (`method='marching_cubes'`)**:
   - **Best for:** Standard triangulated iso-surfaces.

### File Formats
- **`.stl`**: Written using high-speed binary STL output.
- **`.obj`**: Supports native quad face definitions (`f v1 v2 v3 v4`) when using Dual Contouring, as well as triangulated meshes.
- **`.ply`, `.vtk`, `.3mf`, `.off`**: Over 20 formats supported via [meshio](https://github.com/nschloe/meshio).

---

## Compute Acceleration & Adaptive Octree

### 1. Hardware Backends (`device`)
The library provides two distinct computational models: **Vectorized Tensor Evaluation** (PyTorch / NumPy) and **Fused JIT Shader Compilation** (Taichi).

- **`'auto'`** *(default)*: Automatically selects the fastest available tensor backend: NVIDIA CUDA $\rightarrow$ Apple Silicon MPS $\rightarrow$ PyTorch CPU $\rightarrow$ NumPy CPU.
- **`'cuda'`**: Evaluates vectorized point batches using PyTorch on NVIDIA GPUs.
- **`'mps'`**: Evaluates vectorized point batches using Apple Silicon Metal Performance Shaders (M1/M2/M3/M4).
- **`'taichi'`**: Symbolically compiles the entire CSG tree into a **single fused GPU compute shader** via [Taichi Lang](https://taichi-lang.org/) (compiling to native Metal or CUDA).
- **`'numpy'`**: Evaluates batches across multiple CPU worker threads.

> [!NOTE]
> **Why is Taichi its own device option?**
> While PyTorch evaluates your SDF graph operation-by-operation using tensor allocations in VRAM, Taichi traces the SDF symbolically and JIT-compiles the entire CSG tree into a **single fused GPU compute kernel** (using Metal on macOS or CUDA on NVIDIA). This eliminates intermediate memory allocations and kernel dispatch overhead, making it blazingly fast for analytical geometry. `'auto'` defaults to PyTorch because PyTorch offers 100% drop-in compatibility for any arbitrary Python code, while `'taichi'` is an opt-in shader compiler for maximum raw throughput.

### 2. Adaptive Octree Sampling (`adaptive=True`)
Traditional volume samplers evaluate every voxel in a bounding box, wasting compute on empty air or dense interiors. The built-in adaptive octree sampler hierarchically subdivides space, calculating distances only in blocks near the zero-crossing boundary. This dramatically accelerates high-resolution generation and reduces memory footprint.

---

## Key New Features

### 1. Poisson Disc Organic Arrays (`poisson_array`)
Creates non-periodic blue-noise point distributions across 2D boundaries or 3D volumes. Ideal for organic air vents, speaker grilles, drainage arrays, and textured grips:

```python
from sdf import *

# Define domain boundary and recurring feature
domain = rectangle((80, 80))
hole = circle(1.5)

# Generate blue-noise perforation pattern
perforations = hole.poisson_array(
    r_min=2.5,                  # Minimum separation radius
    domain=domain,              # Boundary constraint (SDF2 or SDF3)
    variable_radius=None,       # Optional scalar field function r(p)
    containment_mode='strict',  # 'strict' (fully inside) or 'center'
    seed=42,                    # Deterministic seed
    k=0.5                       # Smooth blend factor between adjacent instances
)

plate = box((90, 90, 3)) - perforations.extrude(5)
plate.save('grille.stl')
```

### 2. Smooth Array Blending (`circular_array` & `poisson_array`)
Array operations support polynomial smooth minimum ($smin$) transitions between adjacent pattern elements by passing the `k` parameter or chaining `.k()`:

```python
# Smoothly blended radial petals/ribs
rib = capsule(-X * 2, X * 2, 0.4)
fluted_body = rib.circular_array(count=8, offset=3.0, k=0.5)
```

### 3. Affine Matrix Transformations (`multmatrix`)
Transform SDF geometry by any arbitrary $3 \times 4$ or $4 \times 4$ affine transformation matrix for precision kinematic alignment and compound CAD assemblies:

```python
matrix = [
    [0.866, -0.5,  0.0, 10.0],
    [0.5,    0.866, 0.0,  5.0],
    [0.0,    0.0,   1.0, -2.5],
    [0.0,    0.0,   0.0,  1.0]
]

transformed_shape = shape.multmatrix(matrix)
```

### 4. Expanded 2D Primitives
The 2D toolkit includes primitives for complex sketch profiles and sweeps:
- `line_segment(a, b, r=0)`: Rounded or exact line segments between 2D points.
- `bezier(a, b, c, r=0, steps=65)`: Quadratic Bézier curves.
- `regular_polygon(n, r)`: Regular $n$-sided polygons (pentagons, octagons, etc.).
- `vesica(r, d)`: Symmetrical lens/vesica piscis curves.

---

## API Reference

### 3D Primitives

#### sphere
<img width=128 align="right" src="docs/images/sphere.png">

`sphere(radius=1, center=ORIGIN)`

```python
f = sphere()            # Unit sphere
f = sphere(2)           # Custom radius
f = sphere(1, (1, 2, 3))# Translated sphere
```

#### box
<img width=128 align="right" src="docs/images/box2.png">

`box(size=1, center=ORIGIN, a=None, b=None)`

```python
f = box(1)                           # 1x1x1 cube
f = box((1, 2, 3))                   # Rectangular prism
f = box(a=(-1, -1, -1), b=(3, 4, 5)) # Bounding-box definition
```

#### rounded_box
<img width=128 align="right" src="docs/images/rounded_box.png">

`rounded_box(size, radius)`

```python
f = rounded_box((1, 2, 3), 0.25)
```

#### wireframe_box
<img width=128 align="right" src="docs/images/wireframe_box.png">

`wireframe_box(size, thickness)`

```python
f = wireframe_box((1, 2, 3), 0.05)
```

#### torus
<img width=128 align="right" src="docs/images/torus.png">

`torus(r1, r2)`

```python
f = torus(1, 0.25) # Major radius 1, minor radius 0.25
```

#### capsule
<img width=128 align="right" src="docs/images/capsule.png">

`capsule(a, b, radius)`

```python
f = capsule(-Z, Z, 0.5)
```

#### cylinder / capped_cylinder / rounded_cylinder
<img width=128 align="right" src="docs/images/capped_cylinder.png">

```python
f = cylinder(0.5)                    # Infinite cylinder along Z
f = capped_cylinder(-Z, Z, 0.5)      # Flat-ended cylinder between points
f = rounded_cylinder(0.5, 0.1, 2.0)  # Cylinder with rounded edges (ra, rb, h)
```

#### capped_cone / rounded_cone
<img width=128 align="right" src="docs/images/capped_cone.png">

```python
f = capped_cone(-Z, Z, 1.0, 0.5)     # Truncated cone between points (a, b, ra, rb)
f = rounded_cone(0.75, 0.25, 2.0)    # Cone with rounded spherical ends (r1, r2, h)
```

#### ellipsoid
<img width=128 align="right" src="docs/images/ellipsoid.png">

`ellipsoid(size)`

```python
f = ellipsoid((1, 2, 3))
```

#### pyramid
<img width=128 align="right" src="docs/images/pyramid.png">

`pyramid(h)`

```python
f = pyramid(1)
```

#### Platonic Solids
- `tetrahedron(r)`
- `octahedron(r)`
- `dodecahedron(r)`
- `icosahedron(r)`

#### Infinite Primitives
- `plane(normal=UP, point=ORIGIN)`: Infinite cutting plane.
- `slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None)`: Useful for slicing shapes between axis-aligned bounds.

---

### Text & Images

#### text
![Text](docs/images/text-large.png)

`text(font_name, text, width=None, height=None, pixels=PIXELS, points=512)`

```python
FONT = 'Arial'
TEXT = 'Hello, world!'
w, h = measure_text(FONT, TEXT)

f = rounded_box((w + 1, h + 1, 0.2), 0.1)
f -= text(FONT, TEXT).extrude(1)
```

#### image
![Image Mask](docs/images/butterfly.png)

`image(path_or_array, width=None, height=None, pixels=PIXELS)`

```python
IMAGE = 'examples/butterfly.png'
w, h = measure_image(IMAGE)

f = rounded_box((w * 1.1, h * 1.1, 0.1), 0.05)
f |= image(IMAGE).extrude(1) & slab(z0=0, z1=0.075)
```

---

### Positioning & Transforms

- `translate(offset)`: Translate by `(dx, dy, dz)` or vector `X * d`.
- `scale(factor)`: Uniform or non-uniform scaling.
- `rotate(angle, vector=Z)`: Rotate by `angle` radians around `vector`.
- `rotate_to(a, b)`: Rotates vector `a` onto vector `b`.
- `orient(axis)`: Rotates shape such that `+Z` points along `axis`.
- `multmatrix(matrix)`: Applies a $3 \times 4$ or $4 \times 4$ affine transformation matrix.

```python
f = box(1).translate((0, 0, 2)).rotate(pi / 4, Z)
```

---

### Boolean Operations & Smoothing

Boolean operations can be constructed using mathematical operators or explicit functional calls:

```python
a = box((3, 3, 0.5))
b = sphere()

# Union
f = a | b
f = union(a, b)

# Difference
f = a - b
f = difference(a, b)

# Intersection
f = a & b
f = intersection(a, b)
```

#### Smooth Booleans
Polynomial smooth minimum blends geometry at intersections:

```python
# Operator syntax (attach .k() to right operand)
f = a | b.k(0.25)
f = a - b.k(0.25)
f = a & b.k(0.25)

# Functional syntax
f = union(a, b, k=0.25)
f = difference(a, b, k=0.25)
f = intersection(a, b, k=0.25)
```

#### Smoothing Guidance (`k`)

Smoothing in `sdf` is powered by the polynomial smooth minimum ($smin$). The smoothing parameter $k$ controls the blend transition radius between surfaces.

##### Key Considerations & Rules of Thumb:
- **Blend Distance (Active Zone):** Smoothing occurs only in regions where component surfaces are within distance $k$ of each other ($|d_1 - d_2| < k$). Outside this zone, primitives retain their exact geometry.
- **Seam Fillet Expansion ($\frac{k}{4}$):** At the intersection joint ($d_1 = d_2$), `smooth_union` expands outward by up to $\frac{k}{4}$. Set $k \approx 4 \times r_{\text{fillet}}$ for a desired fillet radius.
- **Scale $k$ Relative to Feature Size:** Keep $k$ small relative to component dimensions (typically $10\%$ to $50\%$ of feature size). Setting $k$ larger than component dimensions will swallow fine features and turn the shape into a blob.
- **Attach `.k()` to the Right-Hand Operand:** When using binary operators (`|`, `-`, `&`), `_k` is evaluated on the right operand:
  - `a | b.k(0.25)` $\rightarrow$ **Correct** (applies smooth union with $k=0.25$).
  - `a.k(0.25) | b` $\rightarrow$ **Incorrect** (`_k` attached to `a` is ignored).
- **Avoid Post-Chaining `.k()`:** `(a | b).k(0.25)` sets `_k` on the *result* of the union after evaluation has already occurred.
- **Use Functional Syntax for Clarity:** `union(a, b, k=0.25)` or `difference(a, b, k=0.25)` avoids operand-ordering ambiguity.
- **Smooth Operations vs. `blend()`:** `union(a, b, k=...)` creates local fillets at joints. In contrast, `blend(a, b, k=...)` interpolates distance fields globally across all of space.

---

### Repetition & Arrays

- `repeat(spacing, count=None, padding=0)`: Repeats underlying geometry linearly in 1D, 2D, or 3D.
- `circular_array(count, offset=0, k=None)`: Radial repetition around Z with optional smooth blend factor $k$.
- `poisson_array(r_min, r_max=None, domain=..., k=None)`: Blue-noise non-periodic point distribution with optional smooth blending.

---

### Spatial Deformations

- `twist(k)`: Twists shape along Z by angle $k \cdot z$.
- `bend(k)`: Cylindrically bends geometry along an axis.
- `bend_linear(p0, p1, v, e=ease.linear)`: Targeted directional bend between points `p0` and `p1` with easing curves.
- `bend_radial(r0, r1, dz, e=ease.linear)`: Radial dish/dome deformation for diaphragms, ergonomic buttons, and thumb rests.
- `transition_linear(f0, f1, p0, p1, e=ease.linear)`: Continuous geometric morphing between shapes along an axis.
- `transition_radial(f0, f1, r0, r1, e=ease.linear)`: Radial morphing from `f0` to `f1`.
- `wrap_around(x0, x1, r=None, e=ease.linear)`: Wraps planar geometry around the Z axis.
- `elongate(size)`: Stretches shapes along dimensions while preserving end radii.
- `dilate(r)`: Expands geometry outward by distance `r`.
- `erode(r)`: Contracts geometry inward by distance `r`.
- `shell(thickness)`: Creates a hollow, constant-wall shell.
- `blend(a, b, k=0.5)`: Global distance field cross-fade.

---

### 2D to 3D Operations

> [!NOTE]
> When importing `from sdf import *`, 3D boolean functions (`union`, `difference`, `intersection`) shadow 2D functions in the global scope.
> To combine 2D shapes before extruding, use operator syntax (`|`, `&`, `-`) or qualify with `d2`:
> ```python
> profile = circle(2) | rectangle((1, 4))
> # Or: profile = d2.union(circle(2), rectangle((1, 4)))
> f = profile.extrude(10)
> ```

- `extrude(h)`: Linear extrusion along the Z axis.
- `extrude_to(other, h, e=ease.linear)`: Lofts between two distinct 2D cross-sections across height `h` using easing transitions.
- `revolve(offset=0)`: Lathes a 2D profile 360° around the Z axis.

---

### 3D to 2D Operations

- `slice()`: Slices a 3D SDF at the $Z=0$ plane to produce a 2D cross-section SDF.

---

### 2D Primitives

- `circle(radius=1, center=ORIGIN)`
- `line(normal=UP, point=ORIGIN)`
- `slab(x0=None, y0=None, x1=None, y1=None, k=None)`
- `rectangle(size=1, center=ORIGIN, a=None, b=None)`
- `rounded_rectangle(size, radius, center=ORIGIN)`
- `equilateral_triangle(r=1)`
- `hexagon(r)`
- `regular_polygon(n, r)`: Regular polygon with $n$ vertices and circumradius $r$.
- `rounded_x(w, r)`
- `vesica(r, d)`: Symmetrical lens formed by intersecting arcs.
- `line_segment(a, b, r=0)`: Line segment between points with optional radius/thickness.
- `bezier(a, b, c, r=0, steps=65)`: Quadratic Bézier curve.
- `polygon(points)`: Arbitrary 2D polygon defined by counter-clockwise vertices.

---

## Support & Sponsoring

If this library helps power your 3D printing projects, computational design research, or mechanical CAD workflows, consider supporting continued maintenance and development:

<p align="left">
  <a href="https://ko-fi.com/A0P828G4OQ" target="_blank">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" height="42" alt="Support this project on Ko-fi" />
  </a>
</p>

Your support directly funds work on accelerated meshing pipelines, enhanced CAD primitives, automated manifold verification, and expanded export integrations.

---

## License & Attribution

- Released under the [MIT License](LICENSE.md).
- Original concept and implementation by [Michael Fogleman](https://github.com/fogleman/sdf).
- Modernized enhancements, GPU backends, Dual Contouring, adaptive octree, and CAD operations developed by [Kyle Beck](https://github.com/kylebeck).
