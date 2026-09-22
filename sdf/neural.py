import time
import numpy as np

# Try to import torch lazily or handle it at module level with a flag
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_normals_np(f, points, eps=1e-4):
    n = points.shape[0]
    normals = np.zeros((n, 3), dtype=np.float32)
    for i in range(3):
        p_plus = points.copy()
        p_plus[:, i] += eps
        p_minus = points.copy()
        p_minus[:, i] -= eps
        normals[:, i] = (f(p_plus) - f(p_minus)).reshape(-1) / (2 * eps)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms!=0)
    return normals

def get_principal_curvatures_np(f, points, normals, eps=1e-3):
    n = points.shape[0]
    H = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            if j < i:
                H[:, i, j] = H[:, j, i]
                continue
            p_pp = points.copy(); p_pp[:, i] += eps; p_pp[:, j] += eps
            p_pm = points.copy(); p_pm[:, i] += eps; p_pm[:, j] -= eps
            p_mp = points.copy(); p_mp[:, i] -= eps; p_mp[:, j] += eps
            p_mm = points.copy(); p_mm[:, i] -= eps; p_mm[:, j] -= eps
            H[:, i, j] = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)).reshape(-1) / (4 * eps * eps)
            
    dir1 = np.zeros((n, 3), dtype=np.float32)
    dir2 = np.zeros((n, 3), dtype=np.float32)
    
    for i in range(n):
        normal = normals[i]
        if np.linalg.norm(normal) < 1e-5:
            dir1[i], dir2[i] = [1, 0, 0], [0, 1, 0]
            continue
            
        P = np.eye(3) - np.outer(normal, normal)
        H_proj = P @ H[i] @ P
        eigvals, eigvecs = np.linalg.eigh(H_proj)
        
        dot_prods = np.abs(eigvecs.T @ normal)
        idx = np.argsort(dot_prods)[:2]
        
        d1 = eigvecs[:, idx[0]]
        d2 = eigvecs[:, idx[1]]
        dir1[i] = d1 / (np.linalg.norm(d1) + 1e-8)
        dir2[i] = d2 / (np.linalg.norm(d2) + 1e-8)
        
    return dir1, dir2

class RefinedNeuralQuadField:
    # A proxy class if torch is missing, otherwise it's created dynamically
    pass

def _get_model_class():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for neural meshing. Run 'pip install sdf[neural]'")
    
    class Model(nn.Module):
        def __init__(self, hidden_dim=128, num_layers=4):
            super().__init__()
            layers = []
            layers.append(nn.Linear(3, hidden_dim))
            layers.append(nn.Softplus(beta=100))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Softplus(beta=100))
            self.net = nn.Sequential(*layers)
            
            self.head_dist = nn.Linear(hidden_dim, 1)
            self.head_guidance = nn.Linear(hidden_dim, 3)
            
        def forward(self, x):
            feat = self.net(x)
            dist = self.head_dist(feat)
            guidance = self.head_guidance(feat)
            return dist, guidance
    return Model

def bake_field(sdf, bounds, num_samples=100000, epochs=150, batch_size=5000, device='auto', verbose=True):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for neural meshing. Run 'pip install sdf[neural]'")
    
    if device == 'auto':
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            target_device = torch.device('mps')
    else:
        target_device = torch.device(device)
        
    if verbose:
        print(f"Generating {num_samples} training samples for Neural Quad Field...")
        
    t0 = time.time()
    (x0, y0, z0), (x1, y1, z1) = bounds
    
    points_np = np.random.uniform(low=[x0,y0,z0], high=[x1,y1,z1], size=(num_samples, 3)).astype(np.float32)
    distances_np = sdf(points_np).astype(np.float32)
    
    if verbose:
        print(f"Dataset generated in {time.time()-t0:.3f}s. Baking on {target_device}...")
        
    ModelClass = _get_model_class()
    model = ModelClass().to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    t_points = torch.from_numpy(points_np).to(target_device)
    t_distances = torch.from_numpy(distances_np).to(target_device)
    
    model.train()
    t_start = time.time()
    
    for epoch in range(epochs):
        permutation = torch.randperm(t_points.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, t_points.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = t_points[indices].clone().detach().requires_grad_(True)
            dist_gt = t_distances[indices]
            
            optimizer.zero_grad()
            dist_pred, guidance_pred = model(batch_x)
            
            # 1. Autograd for Normals
            d_out = torch.ones_like(dist_pred, requires_grad=False, device=target_device)
            gradients = torch.autograd.grad(
                outputs=dist_pred,
                inputs=batch_x,
                grad_outputs=d_out,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            n_pred = torch.nn.functional.normalize(gradients, dim=1)
            
            # 2. Tangent-Projected Frame (u, v)
            dot_product = torch.sum(guidance_pred * n_pred, dim=1, keepdim=True)
            u_proj = guidance_pred - dot_product * n_pred
            u = torch.nn.functional.normalize(u_proj, dim=1)
            v = torch.cross(n_pred, u, dim=1)
            
            # 3. Autograd for Hessian (to compute Implicit Alignment Loss like NeurCross)
            # We compute the Jacobian of the gradient to get the Hessian matrix
            # Since full batch Hessian is expensive, we can approximate the directional derivative 
            # along u to minimize u^T H v = 0.
            # directional derivative of gradient along u:
            grad_u = torch.autograd.grad(
                outputs=gradients,
                inputs=batch_x,
                grad_outputs=u, # Projecting the Jacobian along u
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            # This grad_u is exactly H * u.
            # We want (H * u) dot v = 0 to align with principal curvatures.
            H_u_v = torch.sum(grad_u * v, dim=1)
            loss_alignment = torch.mean(H_u_v ** 2)
            
            # 4. Losses
            loss_dist = nn.MSELoss()(dist_pred.squeeze(), dist_gt.squeeze())
            grad_norm = torch.norm(gradients, dim=1)
            loss_eikonal = nn.MSELoss()(grad_norm, torch.ones_like(grad_norm))
            
            # Smoothness loss on guidance vector to ensure stability on umbilics
            guidance_laplacian = torch.autograd.grad(
                outputs=guidance_pred,
                inputs=batch_x,
                grad_outputs=torch.ones_like(guidance_pred),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            loss_smooth = torch.mean(guidance_laplacian ** 2)
            
            # Combine losses
            loss = loss_dist + 0.1 * loss_eikonal + 0.1 * loss_alignment + 0.05 * loss_smooth
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        if verbose and (epoch % 30 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:03d} | Total Loss: {epoch_loss/t_points.size()[0]:.6f}")
            
    if verbose:
        print(f"Baking completed in {time.time() - t_start:.2f}s")
        
    del t_points, t_distances
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    return model, target_device

def extract_quad_mesh(model, device, bounds, resolution=32, verbose=True):
    if not HAS_TORCH:
        raise ImportError("PyTorch required.")
        
    if verbose:
        print(f"Extracting Cohesive Quad Mesh (Neural Shrinkwrap, res={resolution})...")
        
    model.eval()
    
    # Generate subdivided cube (genus-0 quad sphere)
    verts = []
    quads = []
    v_dict = {}
    
    (x0, y0, z0), (x1, y1, z1) = bounds
    center = np.array([(x0+x1)/2, (y0+y1)/2, (z0+z1)/2])
    radius = max((x1-x0)/2, (y1-y0)/2, (z1-z0)/2)
    
    def get_v(x, y, z):
        length = np.sqrt(x*x + y*y + z*z) + 1e-8
        # Start the balloon outside the object
        px, py, pz = x/length * radius * 1.5, y/length * radius * 1.5, z/length * radius * 1.5
        px += center[0]
        py += center[1]
        pz += center[2]
        key = (round(px, 5), round(py, 5), round(pz, 5))
        if key not in v_dict:
            v_dict[key] = len(verts)
            verts.append([px, py, pz])
        return v_dict[key]

    for d in range(3):
        for sign in [-1, 1]:
            axes = [0, 1, 2]
            axes.remove(d)
            a1, a2 = axes
            
            for i in range(resolution):
                for j in range(resolution):
                    u0 = -1 + 2 * i / resolution
                    u1 = -1 + 2 * (i + 1) / resolution
                    v0 = -1 + 2 * j / resolution
                    v1 = -1 + 2 * (j + 1) / resolution
                    
                    def make_pt(u, v):
                        p = [0, 0, 0]
                        p[d] = sign
                        p[a1] = u
                        p[a2] = v
                        return get_v(*p)
                    
                    idx00 = make_pt(u0, v0)
                    idx10 = make_pt(u1, v0)
                    idx11 = make_pt(u1, v1)
                    idx01 = make_pt(u0, v1)
                    
                    if sign == 1:
                        quads.append([idx00, idx10, idx11, idx01])
                    else:
                        quads.append([idx00, idx01, idx11, idx10])
                        
    # Neural optimization (Deformable Template / Shrinkwrapping)
    V = torch.tensor(verts, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([V], lr=0.01)
    
    # Pre-compute edges for alignment and Laplacian
    edges = set()
    for q in quads:
        for k in range(4):
            e = tuple(sorted([q[k], q[(k+1)%4]]))
            edges.add(e)
    edges = list(edges)
    E1 = torch.tensor([e[0] for e in edges], device=device)
    E2 = torch.tensor([e[1] for e in edges], device=device)
    
    for step in range(300):
        optimizer.zero_grad()
        
        # 1. SDF Shrinkwrap Loss
        d_pred, guidance_pred = model(V)
        loss_sdf = torch.mean(d_pred**2)
        
        # 2. Cross-Field Alignment Loss
        V_mid = (V[E1] + V[E2]) / 2.0
        d_mid, guidance_mid = model(V_mid)
        
        d_out = torch.ones_like(d_mid, requires_grad=False)
        grad = torch.autograd.grad(d_mid, V_mid, grad_outputs=d_out, create_graph=True)[0]
        n = torch.nn.functional.normalize(grad, dim=1)
        
        dot_product = torch.sum(guidance_mid * n, dim=1, keepdim=True)
        u_proj = guidance_mid - dot_product * n
        u = torch.nn.functional.normalize(u_proj, dim=1)
        v = torch.cross(n, u, dim=1)
        
        edge_vecs = torch.nn.functional.normalize(V[E2] - V[E1], dim=1)
        
        dot_u = torch.sum(edge_vecs * u, dim=1)**2
        dot_v = torch.sum(edge_vecs * v, dim=1)**2
        best_align = torch.max(dot_u, dot_v)
        loss_align = torch.mean(1.0 - best_align)
        
        # 3. Laplacian Uniformity Loss
        edge_lengths = torch.norm(V[E2] - V[E1], dim=1)
        loss_reg = torch.mean((edge_lengths - edge_lengths.mean())**2)
        
        # Total energy to minimize
        loss = loss_sdf + 0.1 * loss_align + 0.05 * loss_reg
        loss.backward()
        optimizer.step()
        
        if verbose and (step % 50 == 0 or step == 299):
            print(f"  Shrinkwrap Step {step:03d} | Loss: {loss.item():.5f} (SDF: {loss_sdf.item():.5f})")

    final_verts = V.detach().cpu().numpy()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    return final_verts, quads
