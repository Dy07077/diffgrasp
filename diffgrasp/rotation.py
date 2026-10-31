"""Rotation utilities in PyTorch (batched, differentiable)."""
from __future__ import annotations

import torch

__all__ = ["rot_x", "rot_y", "rot_z", "axis_angle_to_matrix", "rotation_6d_to_matrix", "matrix_to_rotation_6d", "frame_from_z"]


def rot_x(theta: torch.Tensor) -> torch.Tensor:
    """Rotation about +x; ``(...,) -> (..., 3, 3)``. Positive angles turn +y towards +z."""
    c, s = torch.cos(theta), torch.sin(theta)
    o, z = torch.ones_like(theta), torch.zeros_like(theta)
    return torch.stack([o, z, z, z, c, -s, z, s, c], dim=-1).reshape(*theta.shape, 3, 3)


def rot_y(theta: torch.Tensor) -> torch.Tensor:
    """Rotation about +y; positive angles turn +z towards +x."""
    c, s = torch.cos(theta), torch.sin(theta)
    o, z = torch.ones_like(theta), torch.zeros_like(theta)
    return torch.stack([c, z, s, z, o, z, -s, z, c], dim=-1).reshape(*theta.shape, 3, 3)


def rot_z(theta: torch.Tensor) -> torch.Tensor:
    """Rotation about +z; positive angles turn +x towards +y."""
    c, s = torch.cos(theta), torch.sin(theta)
    o, z = torch.ones_like(theta), torch.zeros_like(theta)
    return torch.stack([c, -s, z, s, c, z, z, z, o], dim=-1).reshape(*theta.shape, 3, 3)


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula for rotation vectors ``(..., 3)``."""
    theta = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / theta
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = theta[..., None]
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation (Zhou et al., CVPR 2019).

    The two 3-vectors are Gram-Schmidt orthonormalised into the first two
    columns of the rotation matrix; the third column is their cross product.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def frame_from_z(z: torch.Tensor, y_hint: torch.Tensor) -> torch.Tensor:
    """Right-handed frame with third column ``z`` and second column close to ``y_hint``."""
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.nn.functional.normalize(y_hint - (y_hint * z).sum(-1, keepdim=True) * z, dim=-1)
    x = torch.cross(y, z, dim=-1)
    return torch.stack([x, y, z], dim=-1)
