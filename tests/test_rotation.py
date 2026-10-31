import numpy as np
import torch
from scipy.spatial.transform import Rotation

from diffgrasp.rotation import (
    axis_angle_to_matrix,
    frame_from_z,
    matrix_to_rotation_6d,
    rot_x,
    rot_y,
    rot_z,
    rotation_6d_to_matrix,
)


def test_elementary_rotations_match_scipy():
    th = torch.tensor([0.3, -1.2, 2.5], dtype=torch.float64)
    for fn, axis in ((rot_x, "x"), (rot_y, "y"), (rot_z, "z")):
        np.testing.assert_allclose(fn(th).numpy(), Rotation.from_euler(axis, th.numpy()[:, None]).as_matrix(), atol=1e-12)


def test_axis_angle_matches_scipy():
    v = torch.randn(10, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    np.testing.assert_allclose(axis_angle_to_matrix(v).numpy(), Rotation.from_rotvec(v.numpy()).as_matrix(), atol=1e-12)


def test_6d_representation():
    R = torch.as_tensor(Rotation.random(20, random_state=0).as_matrix())
    np.testing.assert_allclose(rotation_6d_to_matrix(matrix_to_rotation_6d(R)).numpy(), R.numpy(), atol=1e-12)
    M = rotation_6d_to_matrix(torch.randn(50, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(1)))
    np.testing.assert_allclose((M.transpose(1, 2) @ M).numpy(), np.broadcast_to(np.eye(3), (50, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(torch.linalg.det(M).numpy(), 1.0, atol=1e-12)


def test_frame_from_z():
    g = torch.Generator().manual_seed(2)
    z, hint = torch.randn(10, 3, generator=g), torch.randn(10, 3, generator=g)
    R = frame_from_z(z, hint)
    np.testing.assert_allclose(R[..., 2].numpy(), torch.nn.functional.normalize(z, dim=-1).numpy(), atol=1e-6)
    np.testing.assert_allclose(torch.linalg.det(R).numpy(), 1.0, atol=1e-5)
