"""Ablation: the wrench-hull closure term against the unit-normal force balance of Liu et al.

    python examples/ablation_closure.py     # writes assets/ablation.json

Each case runs the same seeds, iterations and weights twice; only the closure term changes.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import diffgrasp as dg
from common import ASSETS, run_info, synthesize

CASES = [("ball", "precision"), ("cube", "precision"), ("can", "precision"), ("mug", "power")]
PAIRS = ("thumb-index", "thumb-middle", "index-middle")


def pairwise_normal_angles(obj, hand, pose):
    """``(B, 3)`` angles in degrees between the object normals at the thumb, index and middle tips.

    Equal normal forces balance only when all three pairs are 120° apart; a thumb opposing two
    side-by-side fingers gives two angles near 180° and one small one.
    """
    fk = hand.forward(pose)
    c, _ = hand.contacts(fk, ["thumb_distal", "index_distal", "middle_distal"])
    n = obj.normal(c).detach()
    angle = lambda a, b: torch.rad2deg(torch.arccos((n[:, a] * n[:, b]).sum(-1).clamp(-1, 1)))  # noqa: E731
    return torch.stack([angle(0, 1), angle(0, 2), angle(1, 2)], -1).numpy()


def main():
    rows = []
    for name, grasp in CASES:
        for model in ("normals", "hull"):
            hand, obj, opt, res, _, stats = synthesize(name, grasp, weights=dg.EnergyWeights(closure_model=model))
            seeds = stats["per_seed"]
            valid = np.array([e["valid"] for e in seeds])
            row = {
                "object": name,
                "grasp": grasp,
                "closure": model,
                "valid_fraction": stats["valid_fraction"],
                "valid_fraction_95ci": stats["valid_fraction_95ci"],
                "failed_checks": stats["failed_checks"],
                # eps is compared among each term's own valid grasps, next to how many there are
                "median_valid_epsilon": stats["median_valid_epsilon"],
                "median_contact_distance_mm": float(np.median([e["contact_distance_mm"] for e in seeds])),
            }
            if grasp == "precision":
                ang = pairwise_normal_angles(obj, hand, res.pose)
                near_120 = np.all(np.abs(ang - 120) < 20, axis=1)
                row["median_pair_angles_deg"] = dict(zip(PAIRS, np.median(ang, 0).round(1).tolist()))
                row["median_pair_angles_valid_deg"] = dict(zip(PAIRS, np.median(ang[valid], 0).round(1).tolist())) if valid.any() else None
                row["share_all_pairs_within_20deg_of_120"] = float(near_120.mean())
                for e, a in zip(seeds, ang):
                    e["pair_angles_deg"] = dict(zip(PAIRS, a.round(1).tolist()))
            row["per_seed"] = seeds
            rows.append(row)
            angles = row.get("median_pair_angles_deg")
            print(f"{name:5s} {grasp:9s} {model:7s} valid {row['valid_fraction']:.0%}  median valid eps "
                  f"{row['median_valid_epsilon']:.3f}  contact {row['median_contact_distance_mm']:.2f} mm  "
                  f"angles {angles}  near 120 {row.get('share_all_pairs_within_20deg_of_120')}  "
                  f"failed checks {row['failed_checks']}", flush=True)
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "ablation.json").write_text(json.dumps({**run_info(), "cases": rows}, indent=1))
    print("wrote", ASSETS / "ablation.json")


if __name__ == "__main__":
    main()
