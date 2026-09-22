import numpy as np
import sdf

def test_circular_array_2d_smoothing():
    print("Testing 2D circular_array smoothing...")
    item = sdf.circle(0.5).translate((1.0, 0))
    hard_arr = item.circular_array(6)
    smooth_arr1 = item.circular_array(6, k=0.3)
    smooth_arr2 = item.k(0.3).circular_array(6)

    # Point midway between circle 0 (1, 0) and circle 1 (0.5, 0.866)
    p = np.array([[0.75, 0.433]])
    d_hard = float(hard_arr(p)[0, 0])
    d_smooth1 = float(smooth_arr1(p)[0, 0])
    d_smooth2 = float(smooth_arr2(p)[0, 0])

    print(f"  Point (0.75, 0.433): hard={d_hard:.4f}, smooth1={d_smooth1:.4f}, smooth2={d_smooth2:.4f}")
    assert d_smooth1 < d_hard, "Smooth union should produce smaller (more inside/filleted) distance"
    assert abs(d_smooth1 - d_smooth2) < 1e-5, ".k() chaining and k parameter should produce identical results"
    print("  2D circular_array smoothing PASSED")

def test_circular_array_3d_smoothing():
    print("Testing 3D circular_array smoothing...")
    item = sdf.sphere(0.5)
    hard_arr = item.circular_array(6, offset=1.0)
    smooth_arr1 = item.circular_array(6, offset=1.0, k=0.3)
    smooth_arr2 = item.k(0.3).circular_array(6, offset=1.0)

    p = np.array([[0.75, 0.433, 0.0]])
    d_hard = float(hard_arr(p)[0, 0])
    d_smooth1 = float(smooth_arr1(p)[0, 0])
    d_smooth2 = float(smooth_arr2(p)[0, 0])

    print(f"  Point (0.75, 0.433, 0): hard={d_hard:.4f}, smooth1={d_smooth1:.4f}, smooth2={d_smooth2:.4f}")
    assert d_smooth1 < d_hard, "Smooth union should produce smaller distance than hard min"
    assert abs(d_smooth1 - d_smooth2) < 1e-5, ".k() chaining and k parameter should match"
    print("  3D circular_array smoothing PASSED")

def test_poisson_array_2d_smoothing():
    print("Testing 2D poisson_array smoothing...")
    domain = sdf.rectangle((4, 4))
    item = sdf.circle(0.3)
    hard_field = item.poisson_array(r_min=0.4, domain=domain, seed=42)
    smooth_field1 = item.poisson_array(r_min=0.4, domain=domain, seed=42, k=0.2)
    smooth_field2 = item.k(0.2).poisson_array(r_min=0.4, domain=domain, seed=42)

    grid_pts = np.random.default_rng(123).uniform(-1.5, 1.5, (100, 2))
    d_hard = hard_field(grid_pts)
    d_smooth1 = smooth_field1(grid_pts)
    d_smooth2 = smooth_field2(grid_pts)

    assert np.all(d_smooth1 <= d_hard + 1e-6), "Smooth reduction should always be <= hard min"
    assert np.allclose(d_smooth1, d_smooth2, atol=1e-5), ".k() chaining and k parameter should match"
    assert np.any(d_smooth1 < d_hard - 1e-4), "Smooth union should smooth out touching/overlapping geometry"
    print("  2D poisson_array smoothing PASSED")

def test_poisson_array_3d_smoothing():
    print("Testing 3D poisson_array smoothing...")
    domain = sdf.box((3, 3, 3))
    item = sdf.sphere(0.3)
    hard_field = item.poisson_array(r_min=0.45, domain=domain, seed=42)
    smooth_field1 = item.poisson_array(r_min=0.45, domain=domain, seed=42, k=0.25)
    smooth_field2 = item.k(0.25).poisson_array(r_min=0.45, domain=domain, seed=42)

    grid_pts = np.random.default_rng(456).uniform(-1, 1, (100, 3))
    d_hard = hard_field(grid_pts)
    d_smooth1 = smooth_field1(grid_pts)
    d_smooth2 = smooth_field2(grid_pts)

    assert np.all(d_smooth1 <= d_hard + 1e-6), "Smooth reduction should always be <= hard min"
    assert np.allclose(d_smooth1, d_smooth2, atol=1e-5), ".k() chaining and k parameter should match"
    assert np.any(d_smooth1 < d_hard - 1e-4), "Smooth union should smooth touching geometry in 3D"
    print("  3D poisson_array smoothing PASSED")

def test_repeat_smoothing():
    print("Testing repeat smoothing...")
    item = sdf.circle(0.4)
    hard_rep = item.repeat(spacing=(1.0, 1.0), count=(2, 2), padding=1)
    smooth_rep1 = item.repeat(spacing=(1.0, 1.0), count=(2, 2), padding=1, k=0.2)
    smooth_rep2 = item.k(0.2).repeat(spacing=(1.0, 1.0), count=(2, 2), padding=1)

    pts = np.array([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]])
    d_hard = hard_rep(pts)
    d_smooth1 = smooth_rep1(pts)
    d_smooth2 = smooth_rep2(pts)

    assert np.all(d_smooth1 <= d_hard + 1e-6)
    assert np.allclose(d_smooth1, d_smooth2, atol=1e-5)
    assert np.any(d_smooth1 < d_hard - 1e-4)
    print("  repeat smoothing PASSED")

if __name__ == "__main__":
    test_circular_array_2d_smoothing()
    test_circular_array_3d_smoothing()
    test_poisson_array_2d_smoothing()
    test_poisson_array_3d_smoothing()
    test_repeat_smoothing()
    print("\nALL ARRAY SMOOTHING TESTS PASSED SUCCESSFULLY!")
