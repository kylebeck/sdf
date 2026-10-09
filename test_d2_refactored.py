import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

import sdf
from sdf import ease, bk

def test_equilateral_triangle():
    print("Testing equilateral_triangle...")
    obj = sdf.equilateral_triangle(1)
    
    p_np = np.random.uniform(-2, 2, (100, 2))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  equilateral_triangle PASSED")

def test_hexagon():
    print("Testing hexagon...")
    obj = sdf.hexagon(1.5)
    
    p_np = np.random.uniform(-2, 2, (100, 2))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  hexagon PASSED")

def test_regular_polygon():
    print("Testing regular_polygon (pentagon, octagon)...")
    obj5 = sdf.regular_polygon(5, 2.0)
    obj8 = sdf.regular_polygon(8, 1.5)
    
    p_np = np.random.uniform(-3, 3, (100, 2))
    res5_np = obj5(p_np)
    res8_np = obj8(p_np)
    assert res5_np.shape == (100, 1)
    assert res8_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res5_torch = obj5(p_torch)
        res8_torch = obj8(p_torch)
        assert isinstance(res5_torch, torch.Tensor)
        np.testing.assert_allclose(bk.to_numpy(res5_torch), res5_np, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bk.to_numpy(res8_torch), res8_np, rtol=1e-4, atol=1e-4)
    print("  regular_polygon PASSED")

def test_rounded_x():
    print("Testing rounded_x...")
    obj = sdf.rounded_x(1.0, 0.2)
    
    p_np = np.random.uniform(-2, 2, (100, 2))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  rounded_x PASSED")

def test_vesica():
    print("Testing vesica...")
    obj = sdf.vesica(1.5, 0.5)
    
    p_np = np.random.uniform(-2, 2, (100, 2))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  vesica PASSED")

def test_polygon():
    print("Testing polygon...")
    pts = [(-1, -1), (1, -1), (1.5, 0), (0, 2), (-1.5, 0)]
    obj = sdf.polygon(pts)
    
    p_np = np.random.uniform(-3, 3, (100, 2))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  polygon PASSED")

def test_line_segment():
    print("Testing line_segment...")
    obj = sdf.line_segment((-1, -0.5), (2, 1.5), r=0.25)
    
    p_np = np.random.uniform(-3, 3, (100, 2))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  line_segment PASSED")

def test_bezier():
    print("Testing bezier...")
    obj = sdf.bezier((-2, 0), (0, 3), (2, 0), r=0.1)
    
    p_np = np.random.uniform(-3, 3, (100, 2))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  bezier PASSED")

def test_extrude_to():
    print("Testing extrude_to...")
    shape1 = sdf.circle(1.0)
    shape2 = sdf.rectangle((1.5, 1.5))
    obj3d = shape1.extrude_to(shape2, h=4.0, e=ease.in_out_quad)
    
    p_np = np.random.uniform(-2, 2, (100, 3))
    res_np = obj3d(p_np)
    assert res_np.shape == (100, 1)
    
    if HAS_TORCH:
        p_torch = torch.tensor(p_np, dtype=torch.float32)
        res_torch = obj3d(p_torch)
        assert isinstance(res_torch, torch.Tensor)
        assert res_torch.shape == (100, 1)
        np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-4, atol=1e-4)
    print("  extrude_to PASSED")

if __name__ == '__main__':
    test_equilateral_triangle()
    test_hexagon()
    test_regular_polygon()
    test_rounded_x()
    test_vesica()
    test_polygon()
    test_line_segment()
    test_bezier()
    test_extrude_to()
    print("\nALL 2D PRIMITIVES PASSED SUCCESSFULLY!")
