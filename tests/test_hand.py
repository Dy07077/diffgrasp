import sys

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


def test_signed_distance_is_exact_inside_the_hand(hand):
    fk = hand.forward(hand.rest_pose(1))
    # the axis midpoint of every distal phalanx lies exactly one radius deep
    for f in range(5):
        s = 3 * f + 2
        mid = fk["p"][0, s] + 0.5 * hand.lengths[f, 2] * fk["R"][0, s, :, 1]
        assert float(hand.signed_distance(fk, mid[None, None])) == pytest.approx(-float(hand.radii[f, 2]), abs=1e-9)
    # dense surface samples are on the surface of one part, possibly inside another, never outside
    assert float(hand.signed_distance(fk, hand.dense_surface_points(fk)).max()) < 1e-9


def test_core_model_works_without_scikit_image(monkeypatch):
    # scikit-image is an optional (viz) dependency; only the palm render mesh needs it
    monkeypatch.setitem(sys.modules, "skimage", None)
    monkeypatch.setitem(sys.modules, "skimage.measure", None)
    hand, ball = dg.HandModel(), dg.make_object("ball")
    pose = hand.rest_pose(2)
    dg.grasp_energy(hand, ball, pose, dg.GRASP_TYPES["precision"])
    dg.evaluate(hand, ball, pose, dg.GRASP_TYPES["precision"])
    with pytest.raises(ImportError):
        hand.mesh(hand.forward(pose))


def test_penetration_samples_cover_the_hand(hand):
    # hull vertices alone sit on the rounded rim: every point of the palm must be near a sample
    gap = torch.cdist(hand.palm_dense, hand.palm_samples).min(-1).values
    assert float(gap.max()) < 0.004
    # and so must every point of each capsule (the proximal cap is hidden inside the knuckle)
    for dense, samples in zip(hand.seg_dense, hand.seg_samples):
        dense = dense[dense[:, 1] >= 0]
        assert float(torch.cdist(dense, samples).min(-1).values.max()) < 0.003


def test_penetration_weights_measure_the_skin_each_sample_stands_for(hand):
    from diffgrasp.hand import _voronoi_areas

    # a unit square sampled every 0.1 on the left and every 0.2 on the right: each sample weighs
    # the area of its own cell, so the sparse ones weigh about 4.5 times as much
    rng = np.random.default_rng(0)
    left = np.stack(np.meshgrid(np.arange(0.05, 0.5, 0.1), np.arange(0.05, 1.0, 0.1)), -1).reshape(-1, 2)
    right = np.stack(np.meshgrid(np.arange(0.6, 1.0, 0.2), np.arange(0.1, 1.0, 0.2)), -1).reshape(-1, 2)
    w = _voronoi_areas(np.concatenate([left, right]), rng.random((200000, 2)), 1.0)
    assert w.sum() == pytest.approx(1.0)
    assert 3.5 < w[len(left):].mean() / w[: len(left)].mean() < 5.5
    # on the hand, the weights add up to the sampled skin: capsule sides and tips, and the palm
    L, r = hand.lengths.numpy().ravel(), hand.radii.numpy().ravel()
    V, F = hand.palm_vertices.numpy(), hand.palm_faces
    palm = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    skin = (2 * np.pi * r * (L + r)).sum() + palm
    assert float(hand.sample_weights.sum()) * 3e-5 == pytest.approx(skin, rel=1e-9)
    assert float(hand.sample_weights.min()) > 0
