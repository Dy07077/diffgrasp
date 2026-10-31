"""Shared helpers for the example scripts."""
from __future__ import annotations

import dataclasses
import pathlib
import platform

import numpy as np
import torch

import diffgrasp as dg

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


# The three validity checks. Each is counted on its own, so one seed can fail several; force closure
# is only evaluated on touching contacts, so a seed with fewer than three of them fails it too.
CHECKS = {
    "fewer_than_3_contacts": lambda e: e["n_contacts"] < 3,
    "not_force_closure": lambda e: e["epsilon"] <= 0,
    "penetration_over_1mm": lambda e: e["penetration_mm"] >= 1.0,
}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a success fraction k / n."""
    p, den = k / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]


def synthesize(obj_name, grasp_type, n_seeds=48, iterations=800, seed=0, record_every=None, weights=None):
    """Optimise a batch of grasps; return the best valid one (by energy) and stats."""
    hand = dg.HandModel()
    obj = dg.make_object(obj_name)
    opt = dg.GraspOptimizer(hand, obj, grasp_type, n_seeds=n_seeds, weights=weights, seed=seed)
    res = opt.run(iterations, record_every=record_every)
    ev = dg.evaluate(hand, obj, res.pose, opt.contacts)
    order = res.ranking().tolist()
    valid = [i for i in order if ev[i]["valid"]]
    best = valid[0] if valid else order[0]
    stats = {
        "object": obj_name,
        "grasp": grasp_type,
        "seeds": n_seeds,
        "valid_fraction": len(valid) / n_seeds,
        "valid_fraction_95ci": wilson(len(valid), n_seeds),
        "failed_checks": {name: sum(map(check, ev)) for name, check in CHECKS.items()},
        "median_valid_epsilon": float(np.median([ev[i]["epsilon"] for i in valid])) if valid else 0.0,
        "best": ev[best],
        "seconds": res.seconds,
        "per_seed": [{**e, "energy": float(res.energy[i])} for i, e in enumerate(ev)],
    }
    return hand, obj, opt, res, best, stats


def run_info(n_seeds=48, iterations=800, seed=0, weights=None):
    """Software, hardware and every setting needed to reproduce a results file."""
    return {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "settings": {
            "seeds_per_object": n_seeds,
            "iterations": iterations,
            "rng_seed": seed,
            "weights": dataclasses.asdict(weights or dg.EnergyWeights()),
            "valid_if": "force closure at mu = 0.5 (Ferrari-Canny eps > 0), >= 3 contacts within 2 mm, "
                        "penetration < 1 mm by diffgrasp.penetration_depth",
        },
    }


def sdf_numpy(obj):
    return lambda p: obj(torch.as_tensor(np.asarray(p), dtype=torch.float32)).detach().numpy()


def framing(V, OV, scale=0.55):
    """Camera centre and radius that fit both the hand and the object."""
    pts = np.concatenate([V, OV])
    lo, hi = pts.min(0), pts.max(0)
    return 0.5 * (lo + hi), scale * (hi - lo).max()
