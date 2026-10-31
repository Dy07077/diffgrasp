"""Signed distance functions (SDFs) for analytic objects, in PyTorch.

Every object maps points ``(..., 3)`` to signed distances ``(...)``:
negative inside, positive outside, with ``|grad| = 1`` almost everywhere.
Objects compose with :class:`Union` (exact ``min``) and :class:`Transformed`.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "SDF", "Sphere", "Box", "Cylinder", "Torus", "Capsule", "Union", "Transformed",
    "OBJECTS", "make_object",
]  # fmt: skip


class SDF:
    """Base class. Subclasses implement :meth:`__call__`; ``radius`` bounds the shape."""

    center = (0.0, 0.0, 0.0)
    radius = 0.1
    name = "object"

    def __call__(self, p: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def normal(self, p: torch.Tensor) -> torch.Tensor:
        """Outward unit normal (normalised SDF gradient), differentiable."""
        with torch.enable_grad():
            q = p if p.requires_grad else p.detach().requires_grad_(True)
            d = self(q)
            (g,) = torch.autograd.grad(d.sum(), q, create_graph=p.requires_grad)
        return torch.nn.functional.normalize(g, dim=-1)

    def surface_distance_along(self, direction: torch.Tensor, iters: int = 40) -> torch.Tensor:
        """Distance from the centre to the surface along unit ``direction`` (bisection)."""
        c = torch.as_tensor(self.center, dtype=direction.dtype)
        lo = torch.zeros(direction.shape[:-1], dtype=direction.dtype)
        hi = torch.full_like(lo, 2.0 * self.radius)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            inside = self(c + mid[..., None] * direction) < 0
            lo = torch.where(inside, mid, lo)
            hi = torch.where(inside, hi, mid)
        return 0.5 * (lo + hi)

    def sample_interior(self, n: int, generator=None, depth: float = 0.004, dtype=torch.float32) -> torch.Tensor:
        """``n`` random points at least ``depth`` inside the object (rejection sampling)."""
        c = torch.as_tensor(self.center, dtype=dtype)
        out = []
        while sum(len(o) for o in out) < n:
            p = c + self.radius * (2 * torch.rand(4 * n, 3, generator=generator, dtype=dtype) - 1)
            with torch.no_grad():
                out.append(p[self(p) < -depth])
        return torch.cat(out)[:n]

    def surface_points(self, n: int = 30000, seed: int = 0, iters: int = 4) -> torch.Tensor:
        """About ``n`` points on the surface, for penetration checks (no meshing needed).

        Random points in a thin shell around the surface are projected along the normal,
        ``p - d(p) n(p)``, which lands on the surface of an exact SDF; the few that do not
        (near a seam of a union) are discarded.
        """
        g = torch.Generator().manual_seed(seed)
        c = torch.as_tensor(self.center, dtype=torch.float64)
        band = 0.05 * self.radius
        out, total = [], 0
        while total < n:
            p = c + 1.1 * self.radius * (2 * torch.rand(16 * n, 3, generator=g, dtype=torch.float64) - 1)
            with torch.no_grad():
                p = p[self(p).abs() < band]
            for _ in range(iters):
                with torch.no_grad():
                    d = self(p)
                p = p - d[:, None] * self.normal(p).detach()
            with torch.no_grad():
                p = p[self(p).abs() < 1e-6]
            out.append(p)
            total += len(p)
        return torch.cat(out)[:n]

    def mesh(self, resolution: int = 64):
        """Triangle mesh of the zero level set (marching cubes)."""
        from skimage.measure import marching_cubes

        c, r = np.asarray(self.center, float), 1.1 * self.radius
        ax = np.linspace(-r, r, resolution)
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1) + c
        with torch.no_grad():
            d = self(torch.as_tensor(g, dtype=torch.float64)).numpy()
        V, F, _, _ = marching_cubes(d, 0.0, spacing=(ax[1] - ax[0],) * 3)
        return V + c - r, F  # marching_cubes already orients faces outward (towards +SDF)


def _t(x, like):
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


class Sphere(SDF):
    def __init__(self, radius: float, center=(0.0, 0.0, 0.0)):
        self.r, self.center, self.radius, self.name = float(radius), tuple(center), float(radius), "sphere"

    def __call__(self, p):
        return torch.linalg.norm(p - _t(self.center, p), dim=-1) - self.r


class Box(SDF):
    """Box with half extents ``half`` and edges rounded by ``rounding``."""

    def __init__(self, half, rounding: float = 0.0, center=(0.0, 0.0, 0.0)):
        self.half = tuple(float(h) for h in half)
        self.rounding = float(rounding)
        self.center = tuple(center)
        self.radius = float(np.linalg.norm(self.half))
        self.name = "box"

    def __call__(self, p):
        q = torch.abs(p - _t(self.center, p)) - (_t(self.half, p) - self.rounding)
        outside = torch.linalg.norm(q.clamp_min(0.0), dim=-1)
        inside = q.max(dim=-1).values.clamp_max(0.0)
        return outside + inside - self.rounding


class Cylinder(SDF):
    """Cylinder along z with radius ``r``, half height ``h`` and rounded rims."""

    def __init__(self, r: float, h: float, rounding: float = 0.0, center=(0.0, 0.0, 0.0)):
        self.r, self.h, self.rounding, self.center = float(r), float(h), float(rounding), tuple(center)
        self.radius = float(np.hypot(r, h))
        self.name = "cylinder"

    def __call__(self, p):
        q = p - _t(self.center, p)
        d = torch.stack([torch.linalg.norm(q[..., :2], dim=-1) - (self.r - self.rounding),
                         torch.abs(q[..., 2]) - (self.h - self.rounding)], -1)
        return torch.linalg.norm(d.clamp_min(0.0), dim=-1) + d.max(-1).values.clamp_max(0.0) - self.rounding


class Torus(SDF):
    """Torus around the z axis with major radius ``R`` and tube radius ``r``."""

    def __init__(self, R: float, r: float, center=(0.0, 0.0, 0.0)):
        self.R, self.r, self.center = float(R), float(r), tuple(center)
        self.radius = float(R + r)
        self.name = "torus"

    def __call__(self, p):
        q = p - _t(self.center, p)
        ring = torch.linalg.norm(q[..., :2], dim=-1) - self.R
        return torch.sqrt(ring**2 + q[..., 2] ** 2) - self.r


class Capsule(SDF):
    def __init__(self, a, b, r: float):
        self.a, self.b, self.r = tuple(a), tuple(b), float(r)
        self.center = tuple(0.5 * (np.asarray(a) + np.asarray(b)))
        self.radius = float(0.5 * np.linalg.norm(np.subtract(b, a)) + r)
        self.name = "capsule"

    def __call__(self, p):
        a, b = _t(self.a, p), _t(self.b, p)
        ab = b - a
        t = (((p - a) * ab).sum(-1) / (ab * ab).sum()).clamp(0.0, 1.0)
        return torch.linalg.norm(p - a - t[..., None] * ab, dim=-1) - self.r


class Union(SDF):
    def __init__(self, *parts, name="union"):
        self.parts = parts
        lo = np.min([np.asarray(s.center) - s.radius for s in parts], 0)
        hi = np.max([np.asarray(s.center) + s.radius for s in parts], 0)
        c = 0.5 * (lo + hi)
        self.center = tuple(c)
        self.radius = float(max(np.linalg.norm(np.asarray(s.center) - c) + s.radius for s in parts))
        self.name = name

    def __call__(self, p):
        return torch.stack([s(p) for s in self.parts], -1).min(-1).values


class Transformed(SDF):
    """``sdf(R^T (p - t))`` for a rotation matrix ``R`` and translation ``t``."""

    def __init__(self, sdf: SDF, R=None, t=(0.0, 0.0, 0.0), name=None):
        self.sdf = sdf
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.t = np.asarray(t, float)
        self.center = tuple(self.R @ np.asarray(sdf.center) + self.t)
        self.radius = sdf.radius
        self.name = name or sdf.name

    def __call__(self, p):
        return self.sdf((p - _t(self.t, p)) @ _t(self.R, p))


def _mug():
    body = Cylinder(0.040, 0.048, rounding=0.004)
    handle = Transformed(Torus(0.026, 0.0065), R=[[1, 0, 0], [0, 0, -1], [0, 1, 0]], t=(0.058, 0.0, 0.0))
    return Union(body, handle, name="mug")


def _bottle():
    body = Cylinder(0.034, 0.075, rounding=0.012, center=(0, 0, -0.02))
    neck = Cylinder(0.014, 0.03, rounding=0.004, center=(0, 0, 0.075))
    shoulder = Sphere(0.032, center=(0, 0, 0.045))
    return Union(body, neck, shoulder, name="bottle")


OBJECTS = {
    "ball": lambda: Sphere(0.036),
    "cube": lambda: Box((0.03, 0.03, 0.03), rounding=0.005),
    "can": lambda: Cylinder(0.033, 0.06, rounding=0.004),
    "mug": _mug,
    "donut": lambda: Torus(0.04, 0.017),
    "bottle": _bottle,
}


def make_object(name: str) -> SDF:
    """Return one of the built-in objects: ``ball, cube, can, mug, donut, bottle``."""
    obj = OBJECTS[name]()
    obj.name = name
    return obj
