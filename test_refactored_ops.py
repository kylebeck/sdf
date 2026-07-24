import numpy as np
import torch
import sdf
from sdf import ease, bk

def test_bend_linear():
    print("Testing bend_linear...")
    obj = sdf.capsule(-sdf.Z * 2, sdf.Z * 2, 0.25).bend_linear(-sdf.Z, sdf.Z, sdf.X, ease.in_out_quad)
    
    # NumPy
    p_np = np.random.uniform(-2, 2, (100, 3))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    # PyTorch
    p_torch = torch.tensor(p_np, dtype=torch.float32)
    res_torch = obj(p_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape == (100, 1)
    
    # Parity check
    np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-5, atol=1e-5)
    print("  bend_linear PASSED")

def test_bend_radial():
    print("Testing bend_radial...")
    obj = sdf.box((5, 5, 0.25)).bend_radial(1, 2, -1, ease.in_out_quad)
    
    p_np = np.random.uniform(-3, 3, (100, 3))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    p_torch = torch.tensor(p_np, dtype=torch.float32)
    res_torch = obj(p_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape == (100, 1)
    
    np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-5, atol=1e-5)
    print("  bend_radial PASSED")

def test_transition_linear():
    print("Testing transition_linear...")
    obj = sdf.box().transition_linear(sdf.sphere(), e=ease.in_out_quad)
    
    p_np = np.random.uniform(-2, 2, (100, 3))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    p_torch = torch.tensor(p_np, dtype=torch.float32)
    res_torch = obj(p_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape == (100, 1)
    
    np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-5, atol=1e-5)
    print("  transition_linear PASSED")

def test_transition_radial():
    print("Testing transition_radial...")
    obj = sdf.box().transition_radial(sdf.sphere(), e=ease.in_out_quad)
    
    p_np = np.random.uniform(-2, 2, (100, 3))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    p_torch = torch.tensor(p_np, dtype=torch.float32)
    res_torch = obj(p_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape == (100, 1)
    
    np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-5, atol=1e-5)
    print("  transition_radial PASSED")

def test_wrap_around():
    print("Testing wrap_around...")
    obj = sdf.box().wrap_around(0, 1, e=ease.in_out_quad)
    
    p_np = np.random.uniform(-2, 2, (100, 3))
    res_np = obj(p_np)
    assert res_np.shape == (100, 1)
    
    p_torch = torch.tensor(p_np, dtype=torch.float32)
    res_torch = obj(p_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape == (100, 1)
    
    np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-5, atol=1e-5)
    print("  wrap_around PASSED")

def test_slice():
    print("Testing slice...")
    obj3d = sdf.sphere(1)
    obj2d = obj3d.slice()
    
    p_np = np.random.uniform(-2, 2, (100, 2))
    res_np = obj2d(p_np)
    assert res_np.shape == (100, 1)
    
    p_torch = torch.tensor(p_np, dtype=torch.float32)
    res_torch = obj2d(p_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape == (100, 1)
    
    np.testing.assert_allclose(bk.to_numpy(res_torch), res_np, rtol=1e-5, atol=1e-5)
    print("  slice PASSED")

if __name__ == '__main__':
    test_bend_linear()
    test_bend_radial()
    test_transition_linear()
    test_transition_radial()
    test_wrap_around()
    test_slice()
    print("\nALL REFACTORED SDF OPERATIONS PASSED SUCCESSFULLY!")
