import numpy as np

from diffgrasp.quality import epsilon_metric, friction_cone_wrenches, is_force_closure


def ring(angles_deg, z=0.0):
    a = np.radians(angles_deg)
    p = np.stack([np.cos(a), np.sin(a), np.full_like(a, z)], 1)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_wrench_shapes():
    P = ring([0, 120, 240])
    assert friction_cone_wrenches(P, P, n_edges=8).shape == (24, 6)


def test_three_equatorial_contacts_are_force_closure():
    P = ring([0, 120, 240])
    assert is_force_closure(P, P, mu=0.5, center=(0, 0, 0))


def test_two_point_contacts_cannot_resist_torsion():
    P = ring([0, 180])
    assert epsilon_metric(P, P, mu=1.0, center=(0, 0, 0)) == 0.0


def test_contacts_on_one_cap_are_not_force_closure():
    P = np.array([[0.3, 0, 1.0], [-0.3, 0.2, 1.0], [0, -0.3, 1.0]])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    assert not is_force_closure(P, P, mu=0.5, center=(0, 0, 0))


def test_quality_increases_with_friction():
    P = ring([0, 120, 240])
    eps = [epsilon_metric(P, P, mu=m, center=(0, 0, 0)) for m in (0.2, 0.5, 1.0)]
    assert eps[0] < eps[1] < eps[2]
