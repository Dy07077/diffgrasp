"""Shared helpers for the example scripts."""
from __future__ import annotations

import pathlib

import numpy as np
import torch

import diffgrasp as dg

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


def synthesize(obj_name, grasp_type, n_seeds=48, iterations=800, seed=0, record_every=None):
    """Optimise a batch of grasps; return the best valid one (by energy) and stats."""
    hand = dg.HandModel()
    obj = dg.make_object(obj_name)
    opt = dg.GraspOptimizer(hand, obj, grasp_type, n_seeds=n_seeds, seed=seed)
    res = opt.run(iterations, record_every=record_every)
    ev = dg.evaluate(hand, obj, res.pose, opt.contacts, touch=0.002)
    order = res.ranking().tolist()
    valid = [i for i in order if ev[i]["valid"]]
    best = valid[0] if valid else order[0]
    stats = {
        "object": obj_name,
        "grasp": grasp_type,
        "seeds": n_seeds,
        "valid_fraction": float(np.mean([e["valid"] for e in ev])),
        "best": ev[best],
        "seconds": res.seconds,
    }
    return hand, obj, opt, res, best, stats


def sdf_numpy(obj):
    return lambda p: obj(torch.as_tensor(np.asarray(p), dtype=torch.float32)).detach().numpy()


def framing(V, OV, scale=0.55):
    """Camera centre and radius that fit both the hand and the object."""
    pts = np.concatenate([V, OV])
    lo, hi = pts.min(0), pts.max(0)
    return 0.5 * (lo + hi), scale * (hi - lo).max()
