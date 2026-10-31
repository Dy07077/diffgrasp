import torch

import diffgrasp as dg


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
