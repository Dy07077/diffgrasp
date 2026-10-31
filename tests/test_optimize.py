import inspect

import pytest
import torch

import diffgrasp as dg
from diffgrasp.objects import Capsule


def test_optimizer_finds_a_valid_precision_grasp():
    hand, obj = dg.HandModel(), dg.make_object("ball")
    opt = dg.GraspOptimizer(hand, obj, "precision", n_seeds=12, seed=0)
    init = opt.initial_poses()
    e0, _ = dg.grasp_energy(hand, obj, init, opt.contacts)
    res = opt.run(400, init=init)
    assert res.energy.median() < 0.1 * e0.median()
    ev = dg.evaluate(hand, obj, res.pose, opt.contacts, touch=0.002)
    valid = [e for e in ev if e["valid"]]
    assert len(valid) >= 3
    assert max(e["epsilon"] for e in valid) > 0.05
    assert max(e["penetration_mm"] for e in valid) < 1.0


def test_results_are_reproducible():
    hand, obj = dg.HandModel(), dg.make_object("cube")
    a = dg.GraspOptimizer(hand, obj, "precision", n_seeds=4, seed=7).run(30)
    b = dg.GraspOptimizer(hand, obj, "precision", n_seeds=4, seed=7).run(30)
    torch.testing.assert_close(a.pose.logits, b.pose.logits)


def test_penetration_check_catches_a_rod_through_the_palm():
    # a 1 mm rod pierces the broad palmar face far from every hull vertex: vertex samples see
    # nothing, and the object-side half of the dense check sees the full depth
    hand = dg.HandModel()
    pose = hand.rest_pose(1)
    rod = Capsule((0.004, 0.05, -0.06), (0.004, 0.05, 0.06), 0.001)
    fk = hand.forward(pose)
    assert float(torch.relu(-rod(hand.to_world(fk, -1, hand.palm_vertices))).max()) == 0.0
    depth = float(dg.penetration_depth(hand, rod, pose))
    assert depth > 0.01
    e = dg.evaluate(hand, rod, pose, ["palm"])[0]
    assert e["penetration_mm"] == pytest.approx(1e3 * depth) and not e["valid"]


def test_evaluate_uses_the_documented_2mm_contact_threshold():
    assert inspect.signature(dg.evaluate).parameters["touch"].default == 0.002
