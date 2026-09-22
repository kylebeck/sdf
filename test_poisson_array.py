import math
import numpy as np
import sdf
from sdf import bk

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

def test_missing_domain_raises():
    print("Testing missing domain validation...")
    try:
        sdf.circle(0.1).poisson_array(r_min=0.3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "domain" in str(e)

    try:
        sdf.sphere(0.1).poisson_array(r_min=0.3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "domain" in str(e)
    print("  missing domain PASSED")

def test_invalid_containment_mode_raises():
    print("Testing invalid containment_mode validation...")
    try:
        sdf.circle(0.1).poisson_array(r_min=0.3, domain=sdf.circle(1.0), containment_mode='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "containment_mode" in str(e)
    print("  invalid containment_mode PASSED")

def test_poisson_array_2d_basic():
    print("Testing 2D poisson_array basic...")
    domain = sdf.rectangle((4, 4))
    item = sdf.circle(0.1)
    field = item.poisson_array(r_min=0.4, domain=domain, seed=42)
    
    p_np = np.random.uniform(-3, 3, (100, 2))
    res_np = field(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = field(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  2D poisson_array basic PASSED")

def test_poisson_array_2d_variable_radius_and_scaling():
    print("Testing 2D poisson_array variable radius & scaling...")
    domain = sdf.rectangle((6, 6))
    item = sdf.circle(0.1)
    field = item.poisson_array(
        r_min=0.2,
        r_max=0.8,
        variable_radius=lambda p: 0.2 + 0.6 * ((p[..., 0] + 3) / 6),
        scale_to_radius=True,
        domain=domain,
        seed=123
    )
    
    p_np = np.random.uniform(-4, 4, (50, 2))
    res_np = field(p_np)
    assert res_np.shape == (50, 1)
    print("  2D poisson_array variable radius & scaling PASSED")

def test_poisson_array_2d_rotation():
    print("Testing 2D poisson_array random rotation...")
    domain = sdf.rectangle((4, 4))
    item = sdf.rectangle((0.2, 0.05))
    field = item.poisson_array(r_min=0.4, domain=domain, random_rotation=True, seed=99)
    
    p_np = np.random.uniform(-2, 2, (50, 2))
    res_np = field(p_np)
    assert res_np.shape == (50, 1)
    print("  2D poisson_array random rotation PASSED")

def test_poisson_array_3d_basic():
    print("Testing 3D poisson_array basic...")
    domain = sdf.box((4, 4, 4))
    item = sdf.sphere(0.1)
    field = item.poisson_array(r_min=0.5, domain=domain, seed=42)
    
    p_np = np.random.uniform(-3, 3, (100, 3))
    res_np = field(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = field(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  3D poisson_array basic PASSED")

def test_poisson_array_3d_with_2d_domain():
    print("Testing 3D poisson_array with 2D domain (scattering.py scenario)...")
    domain = sdf.rectangle((20, 20))
    item = sdf.sphere(0.5)
    field = item.poisson_array(
        r_min=0.1,
        r_max=0.6,
        variable_radius=lambda p: 0.1 + 0.5 * ((p[..., 0] + 3) / 6),
        scale_to_radius=True,
        domain=domain,
        seed=42
    )
    
    p_np = np.random.uniform(-10, 10, (100, 3))
    res_np = field(p_np)
    assert res_np.shape == (100, 1)
    print("  3D poisson_array with 2D domain PASSED")

def test_poisson_array_3d_multi_pass_packing():
    print("Testing 3D poisson_array multi-pass packing...")
    domain = sdf.box((6, 6, 6))
    item = sdf.sphere(0.1)
    field = item.poisson_array(
        r_min=0.2,
        r_max=1.0,
        scale_to_radius=True,
        random_rotation=True,
        domain=domain,
        seed=777
    )
    
    p_np = np.random.uniform(-4, 4, (50, 3))
    res_np = field(p_np)
    assert res_np.shape == (50, 1)
    print("  3D poisson_array multi-pass packing PASSED")

if __name__ == '__main__':
    test_missing_domain_raises()
    test_invalid_containment_mode_raises()
    test_poisson_array_2d_basic()
    test_poisson_array_2d_variable_radius_and_scaling()
    test_poisson_array_2d_rotation()
    test_poisson_array_3d_basic()
    test_poisson_array_3d_with_2d_domain()
    test_poisson_array_3d_multi_pass_packing()
    print("\nALL POISSON ARRAY TESTS PASSED SUCCESSFULLY!")
