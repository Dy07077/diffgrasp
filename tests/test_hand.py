import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import diffgrasp as dg
from diffgrasp.hand import HandPose
from diffgrasp.rotation import matrix_to_rotation_6d


@pytest.fixture(scope="module")
def hand():
    return dg.HandModel(dtype=torch.float64)


def random_pose(hand, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    R = torch.as_tensor(Rotation.random(n, random_state=seed).as_matrix())
    return HandPose(matrix_to_rotation_6d(R), 0.1 * torch.randn(n, 3, generator=g, dtype=torch.float64),
                    2.0 * torch.randn(n, hand.n_joints, generator=g, dtype=torch.float64))


def test_joint_limits_and_logit_map(hand):
    assert len(dg.JOINT_NAMES) == hand.n_joints == 21
    assert torch.all(hand.lower < hand.upper)
    q = hand.angles(torch.tensor([[-50.0] * 21, [50.0] * 21], dtype=torch.float64))
    assert torch.all(q >= hand.lower - 1e-9) and torch.all(q <= hand.upper + 1e-9)
    q0 = hand.rest_angles[None]
    np.testing.assert_allclose(hand.angles(hand.logits(q0)).numpy(), q0.numpy(), atol=1e-9)


def test_hand_has_human_proportions(hand):
    rest = hand.rest_pose(1)
    flat = torch.zeros(1, 21, dtype=torch.float64).clamp(hand.lower, hand.upper)
    fk = hand.forward(HandPose(rest.rot6d, rest.trans, hand.logits(flat)))
    assert 0.17 < float(fk["tips"][0, 2].norm()) < 0.21  # wrist to middle fingertip


def test_phalanx_lengths_are_preserved(hand):
    fk = hand.forward(random_pose(hand))
    p = fk["p"].reshape(-1, 5, 3, 3)
    seg = (p[:, :, 1:] - p[:, :, :-1]).norm(dim=-1)
    np.testing.assert_allclose(seg.numpy(), np.broadcast_to(hand.lengths[:, :2].numpy(), seg.shape), atol=1e-12)


def test_flexion_curls_fingers_towards_the_palm(hand):
    rest = hand.rest_pose(1)
    q = hand.rest_angles.clone()[None]
    z0 = hand.forward(HandPose(rest.rot6d, rest.trans, hand.logits(q)))["tips"][0, 1, 2]
    q[0, dg.JOINT_NAMES.index("index_mcp")] += 0.5
    z1 = hand.forward(HandPose(rest.rot6d, rest.trans, hand.logits(q)))["tips"][0, 1, 2]
    assert z1 > z0 + 0.02  # the palm faces +z in the wrist frame


def test_global_pose_is_a_rigid_motion(hand):
    pose = random_pose(hand, n=1)
    local = HandPose(hand.rest_pose(1).rot6d, torch.zeros(1, 3, dtype=torch.float64), pose.logits)
    pw = hand.surface_points(hand.forward(pose))[0]
    pl = hand.surface_points(hand.forward(local))[0]
    R = hand.forward(pose)["Rg"][0]
    np.testing.assert_allclose(pw.numpy(), (pl @ R.T + pose.trans[0]).numpy(), atol=1e-12)


def test_gradients_reach_every_parameter(hand):
    pose = random_pose(hand, n=2)
    for t in (pose.rot6d, pose.trans, pose.logits):
        t.requires_grad_(True)
    fk = hand.forward(pose)
    (fk["tips"].pow(2).sum() + hand.surface_points(fk).sum()).backward()
    for t in (pose.rot6d, pose.trans, pose.logits):
        assert torch.isfinite(t.grad).all() and (t.grad.abs().sum(-1) > 0).all()


def test_rest_pose_is_free_of_self_collision(hand):
    obj = dg.make_object("ball")
    far = hand.rest_pose(1)
    far.trans += 1.0
    _, terms = dg.grasp_energy(hand, obj, far, dg.GRASP_TYPES["power"])
    assert float(terms["self"]) == 0.0 and float(terms["penetration"]) == 0.0


def test_render_meshes_are_closed_and_outward(hand):
    for V, F in hand.seg_meshes[:3] + [hand.palm_render]:
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        assert np.all(counts == 2)
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        vol = np.einsum("ij,ij->i", n, V[F[:, 0]]).sum() / 6.0  # divergence theorem
        assert vol > 0
