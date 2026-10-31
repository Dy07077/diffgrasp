import numpy as np
import pytest
import torch

import diffgrasp as dg
from diffgrasp.objects import Box, Cylinder, Sphere, Torus

T = lambda *x: torch.tensor(x, dtype=torch.float64)  # noqa: E731


def test_primitive_distances():
    assert float(Sphere(0.05)(T(0.1, 0, 0))) == pytest.approx(0.05)
    assert float(Sphere(0.05)(T(0, 0, 0))) == pytest.approx(-0.05)
    assert float(Box((0.1, 0.2, 0.3))(T(0.2, 0, 0))) == pytest.approx(0.1)
    assert float(Box((0.1, 0.2, 0.3))(T(0, 0, 0))) == pytest.approx(-0.1)
    assert float(Box((0.1, 0.1, 0.1))(T(0.2, 0.2, 0))) == pytest.approx(np.sqrt(2) * 0.1)
    assert float(Cylinder(0.1, 0.2)(T(0.3, 0, 0))) == pytest.approx(0.2)
    assert float(Cylinder(0.1, 0.2)(T(0, 0, 0.5))) == pytest.approx(0.3)
    assert float(Torus(0.1, 0.02)(T(0.1, 0, 0))) == pytest.approx(-0.02)
    assert float(Torus(0.1, 0.02)(T(0, 0, 0))) == pytest.approx(0.08)


@pytest.mark.parametrize("name", list(dg.OBJECTS))
def test_sdf_gradient_has_unit_norm(name):
    obj = dg.make_object(name)
    g = torch.Generator().manual_seed(0)
    p = torch.as_tensor(obj.center, dtype=torch.float64) + obj.radius * (2 * torch.rand(400, 3, generator=g, dtype=torch.float64) - 1)
    p = p[obj(p).abs() > 2e-3]
    p.requires_grad_(True)
    (grad,) = torch.autograd.grad(obj(p).sum(), p)
    assert torch.median((grad.norm(dim=-1) - 1).abs()) < 1e-6


@pytest.mark.parametrize("name", list(dg.OBJECTS))
def test_mesh_lies_on_surface_and_faces_outward(name):
    obj = dg.make_object(name)
    V, F = obj.mesh(40)
    voxel = 2.2 * obj.radius / 39
    assert np.abs(obj(torch.as_tensor(V)).numpy()).max() < voxel
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    g = obj.normal(torch.as_tensor(V[F].mean(1))).numpy()
    assert np.mean(np.einsum("ij,ij->i", n, g) > 0) > 0.99


def test_interior_samples_and_surface_rays():
    obj = dg.make_object("donut")
    p = obj.sample_interior(200, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert len(p) == 200 and torch.all(obj(p) < -0.004)
    ball = dg.make_object("ball")
    d = torch.nn.functional.normalize(torch.randn(20, 3, dtype=torch.float64), dim=-1)
    np.testing.assert_allclose(ball.surface_distance_along(d).numpy(), 0.036, atol=1e-6)


@pytest.mark.parametrize("name", list(dg.OBJECTS))
def test_surface_samples_lie_on_the_surface(name):
    obj = dg.make_object(name)
    q = obj.surface_points(5000)
    assert q.shape == (5000, 3)
    assert float(obj(q).abs().max()) < 1e-6
