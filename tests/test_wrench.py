import numpy as np
import torch
from scipy.optimize import minimize

from diffgrasp.wrench import closure_energy, cone_wrenches, hull_distance_sq

torch.set_default_dtype(torch.float64)


def unit(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return x / x.norm(dim=-1, keepdim=True)


def test_robust_grasp_has_zero_closure_energy():
    a = torch.deg2rad(torch.tensor([0.0, 120.0, 240.0]))
    P = torch.stack([torch.cos(a), torch.sin(a), 0 * a], 1)[None]
    assert closure_energy(P, P, torch.zeros(3), 1.0, iters=400).item() < 1e-4


def test_one_sided_grasp_is_penalised():
    P = unit([[0.3, 0, 1.0], [-0.3, 0.2, 1.0], [0, -0.3, 1.0]])[None]
    assert closure_energy(P, P, torch.zeros(3), 1.0).item() > 1.0


def test_hull_distance_matches_an_exact_projection():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(1, 6, 18, generator=g)
    p = 3.0 * torch.randn(1, 6, 1, generator=g)  # a point outside the hull
    fw = hull_distance_sq(W, p, iters=500).item()
    Wn, pn = W[0].numpy(), p[0, :, 0].numpy()
    res = minimize(lambda l: np.sum((Wn @ l - pn) ** 2), np.full(18, 1 / 18), method="SLSQP",
                   bounds=[(0, 1)] * 18, constraints=[{"type": "eq", "fun": lambda l: l.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    np.testing.assert_allclose(fw, res.fun, rtol=1e-4)


def test_closure_gradient_matches_finite_differences():
    P = unit([[0.3, 0, 1.0], [-0.3, 0.2, 1.0], [0, -0.3, 1.0]])[None]
    v = torch.randn(P.shape, generator=torch.Generator().manual_seed(1))
    Pg = P.clone().requires_grad_(True)
    closure_energy(Pg, Pg, torch.zeros(3), 1.0, iters=300).backward()
    h = 1e-6
    num = (closure_energy(P + h * v, P + h * v, torch.zeros(3), 1.0, iters=300)
           - closure_energy(P - h * v, P - h * v, torch.zeros(3), 1.0, iters=300)) / (2 * h)
    np.testing.assert_allclose((Pg.grad * v).sum().item(), num.item(), rtol=0.03)


def test_contact_strength_scales_wrenches():
    P = unit([[1.0, 0, 0], [-1.0, 0, 0]])[None]
    W = cone_wrenches(P, P, torch.zeros(3), 1.0, strength=torch.tensor([[1.0, 0.5]]))
    m = W.shape[-1] // 2
    np.testing.assert_allclose(W[0, :, m:].norm(dim=0).numpy(), 0.5 * W[0, :, :m].norm(dim=0).numpy(), rtol=1e-12)
