"""Grasp quality: the Ferrari-Canny epsilon metric and penetration checks."""
from __future__ import annotations

import numpy as np

__all__ = ["friction_cone_wrenches", "epsilon_metric", "is_force_closure"]


def friction_cone_wrenches(points, normals, mu=0.5, n_edges=8, center=None, torque_scale=1.0):
    """Primitive contact wrenches of linearised Coulomb friction cones.

    Args:
        points: ``(K, 3)`` contact positions on the object surface.
        normals: ``(K, 3)`` outward object normals (forces push along ``-n``).
        mu: friction coefficient.
        n_edges: number of cone edges.
        center: torque reference point (default: centroid of the points).
        torque_scale: divides torques so forces and torques are comparable
            (use the object's radius).
    Returns:
        ``(K * n_edges, 6)`` unit-normal-force wrenches ``[f, tau / torque_scale]``.
    """
    p = np.asarray(points, float)
    n = np.asarray(normals, float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    c = p.mean(0) if center is None else np.asarray(center, float)
    a = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    phi = 2 * np.pi * np.arange(n_edges) / n_edges
    f = -n[:, None] + mu * (np.cos(phi)[None, :, None] * t1[:, None] + np.sin(phi)[None, :, None] * t2[:, None])
    tau = np.cross((p - c)[:, None], f) / torque_scale
    return np.concatenate([f, tau], -1).reshape(-1, 6)


def epsilon_metric(points, normals, mu=0.5, n_edges=8, center=None, torque_scale=1.0) -> float:
    """Ferrari & Canny (1992) L1 quality: radius of the largest origin-centred
    ball inside the convex hull of the contact wrenches. Zero when the grasp is
    not force closure (the origin is outside the hull).
    """
    from scipy.spatial import ConvexHull, QhullError

    W = friction_cone_wrenches(points, normals, mu, n_edges, center, torque_scale)
    try:
        hull = ConvexHull(W)
    except (QhullError, ValueError):
        return 0.0  # wrenches span less than 6 dimensions
    offsets = hull.equations[:, -1]  # n . w + b <= 0 inside, |n| = 1
    return float(max(0.0, -offsets.max()))


def is_force_closure(points, normals, mu=0.5, **kw) -> bool:
    return epsilon_metric(points, normals, mu, **kw) > 1e-9
