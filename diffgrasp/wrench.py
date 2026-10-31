"""Differentiable force-closure surrogate based on the grasp wrench space.

The Ferrari-Canny metric asks how large a ball around the origin fits inside
the convex hull ``H`` of the primitive contact wrenches (friction-cone edges).
Here we ask a smoother question: are the ``12`` points ``+-delta e_i`` inside
``H``? Each squared distance

    d_i^2(W) = min_{lambda in simplex} |W lambda - p_i|^2

is computed with Frank-Wolfe. By Danskin's theorem its gradient with respect
to the wrenches is ``2 (W lambda* - p_i) lambda*^T`` at the optimal weights,
so ``lambda*`` can be found without autograd and the energy is still exact
to first order. The energy is zero exactly when all twelve points lie in the
hull, which implies a Ferrari-Canny epsilon of at least ``delta / sqrt(6)``.
"""
from __future__ import annotations

import torch

__all__ = ["cone_wrenches", "hull_distance_sq", "closure_energy"]


def cone_wrenches(points, normals, centre, scale, mu=0.5, n_edges=6, strength=None):
    """Primitive wrenches ``(B, 6, K * n_edges)`` of linearised friction cones.

    ``strength`` (``(B, K)``, optional) scales each contact's cone, e.g. to
    fade out contacts that do not touch the object.
    """
    n = torch.nn.functional.normalize(normals, dim=-1)
    a = torch.where(n[..., :1].abs() < 0.9, torch.tensor([1.0, 0, 0], dtype=n.dtype), torch.tensor([0, 1.0, 0], dtype=n.dtype))
    t1 = torch.nn.functional.normalize(torch.cross(n, a, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    phi = 2 * torch.pi * torch.arange(n_edges, dtype=n.dtype) / n_edges
    f = -n[..., None, :] + mu * (torch.cos(phi)[:, None] * t1[..., None, :] + torch.sin(phi)[:, None] * t2[..., None, :])
    r = (points - centre)[..., None, :].expand_as(f)
    tau = torch.cross(r, f, dim=-1) / scale
    W = torch.cat([f, tau], dim=-1)  # (B, K, m, 6)
    if strength is not None:
        W = W * strength[..., None, None]
    return W.flatten(1, 2).transpose(1, 2)


@torch.no_grad()
def _frank_wolfe(W, P, iters):
    """Pairwise Frank-Wolfe for min |W lam - p|^2 over the simplex.

    Each step moves weight from the worst active vertex to the best vertex,
    with an exact line search; unlike plain Frank-Wolfe this converges
    linearly, so a few dozen iterations give accurate optimal weights.
    """
    B, _, M = W.shape
    T = P.shape[-1]
    Wt = W.transpose(1, 2)  # (B, M, 6)
    target = P.transpose(1, 2)  # (B, T, 6)
    # start from the vertex closest to each target
    j0 = torch.cdist(target, Wt).argmin(-1)
    lam = torch.zeros(B, T, M, dtype=W.dtype)
    lam.scatter_(2, j0[..., None], 1.0)
    for _ in range(iters):
        x = torch.einsum("bdm,btm->btd", W, lam)
        r = x - target
        g = torch.einsum("bmd,btd->btm", Wt, r)
        s = g.argmin(-1)
        a = torch.where(lam > 0, g, torch.full_like(g, -torch.inf)).argmax(-1)
        vs = torch.gather(Wt, 1, s[..., None].expand(B, T, 6))
        va = torch.gather(Wt, 1, a[..., None].expand(B, T, 6))
        d = vs - va
        lam_a = torch.gather(lam, 2, a[..., None])[..., 0]
        gamma = (-(r * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-30)).clamp(min=0.0)
        gamma = torch.minimum(gamma, lam_a)
        lam.scatter_add_(2, s[..., None], gamma[..., None])
        lam.scatter_add_(2, a[..., None], -gamma[..., None])
        lam.clamp_(min=0.0)
    return lam


def hull_distance_sq(W, P, iters=80):
    """Squared distances ``(B, T)`` from targets ``P (B, 6, T)`` to conv(W)."""
    lam = _frank_wolfe(W.detach(), P.detach(), iters)
    x = torch.einsum("bdm,btm->btd", W, lam)
    return ((x - P.transpose(1, 2)) ** 2).sum(-1)


def closure_energy(points, normals, centre, scale, mu=0.5, delta=0.05, iters=80, strength=None):
    """Sum of squared distances from ``+-delta e_i`` to the wrench hull."""
    W = cone_wrenches(points, normals, centre, scale, mu, strength=strength)
    I = torch.eye(6, dtype=W.dtype)
    P = delta * torch.cat([I, -I], dim=1).expand(W.shape[0], 6, 12)
    return hull_distance_sq(W, P, iters).sum(-1)
