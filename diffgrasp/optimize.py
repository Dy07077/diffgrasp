"""Batched gradient-based grasp synthesis with annealed noise."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .energy import GRASP_TYPES, EnergyWeights, grasp_energy
from .hand import HandModel, HandPose
from .objects import SDF
from .quality import epsilon_metric
from .rotation import frame_from_z, matrix_to_rotation_6d

__all__ = ["GraspOptimizer", "GraspResult", "evaluate", "penetration_depth", "PREGRASP_DEG"]

# Pre-grasp postures in degrees, joint order as in ``JOINT_NAMES``:
# thumb (cmc_abd, cmc_flex, cmc_rot, mcp, ip), then (abd, mcp, pip, dip) for each finger.
PREGRASP_DEG = {
    "precision": [-10, 50, -80, 0, 5] + [0, 25, 25, 10] * 4,
    "fingertips": [-10, 50, -80, 0, 5] + [0, 25, 25, 10] * 4,
    "power": [-10, 45, -80, 15, 15] + [0, 45, 50, 25] * 4,
}


@dataclass
class GraspResult:
    pose: HandPose  # final poses, one per seed
    energy: torch.Tensor  # (B,)
    terms: dict
    history: list = field(default_factory=list)  # recorded poses during optimisation
    seconds: float = 0.0

    def ranking(self) -> torch.Tensor:
        return torch.argsort(self.energy)


class GraspOptimizer:
    """Optimise many grasps of one object in parallel.

    Args:
        hand: the hand model.
        obj: object SDF.
        grasp_type: ``"precision"``, ``"fingertips"`` or ``"power"`` (or a list of
            contact candidate names).
        n_seeds: number of random initialisations optimised as one batch.
        weights: :class:`EnergyWeights`.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, hand: HandModel, obj: SDF, grasp_type="power", n_seeds=32, weights=None, seed=0):
        self.hand, self.obj = hand, obj
        self.grasp_type = grasp_type if isinstance(grasp_type, str) else "custom"
        self.contacts = GRASP_TYPES[grasp_type] if isinstance(grasp_type, str) else list(grasp_type)
        self.n_seeds = n_seeds
        self.weights = weights or EnergyWeights()
        self.gen = torch.Generator().manual_seed(seed)

    def initial_poses(self, jitter_deg: float = 6.0) -> HandPose:
        """Pre-shaped hands in random orientations with the object in their aperture.

        The hand is pre-shaped for the grasp type (thumb opposing the fingers)
        and translated so that the centroid of its target contact points sits
        at a random point inside the object (so thin parts such as a handle or
        a torus tube get grasped too). Closing onto the object from inside the
        aperture finds opposition grasps far more often than approaching it
        from outside.
        """
        B, h, dt = self.n_seeds, self.hand, self.hand.dtype
        key = self.grasp_type if self.grasp_type in PREGRASP_DEG else "power"
        q0 = torch.deg2rad(torch.tensor(PREGRASP_DEG[key], dtype=dt))
        q = q0 + torch.deg2rad(torch.tensor(jitter_deg, dtype=dt)) * torch.randn(B, h.n_joints, generator=self.gen, dtype=dt)
        q = torch.minimum(torch.maximum(q, h.lower + 1e-3), h.upper - 1e-3)
        rest = h.rest_pose(B)
        local = HandPose(rest.rot6d, rest.trans, h.logits(q))
        c, _ = h.contacts(h.forward(local), self.contacts)
        g = c.mean(1)  # (B, 3) grasp centre in the wrist frame
        z = torch.nn.functional.normalize(torch.randn(B, 3, generator=self.gen, dtype=dt), dim=-1)
        R = frame_from_z(z, torch.randn(B, 3, generator=self.gen, dtype=dt))
        target = self.obj.sample_interior(B, generator=self.gen, dtype=dt)
        trans = target - (R @ g[..., None])[..., 0]
        return HandPose(matrix_to_rotation_6d(R), trans, h.logits(q))

    def run(self, iterations: int = 800, record_every: int | None = None, init: HandPose | None = None,
            lr=(1.5e-2, 2.0e-3, 4.0e-2), noise=0.6, verbose=False) -> GraspResult:
        t0 = time.perf_counter()
        pose = (init or self.initial_poses()).detach()
        params = [p.requires_grad_(True) for p in (pose.rot6d, pose.trans, pose.logits)]
        opt = torch.optim.Adam([{"params": [p], "lr": l} for p, l in zip(params, lr)])
        # cosine decay to 3% of the initial step: explore early, settle precisely at the end
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda it: 0.03 + 0.97 * 0.5 * (1.0 + np.cos(np.pi * min(it / max(iterations - 1, 1), 1.0)))
        )
        base = self.weights
        history = []
        for it in range(iterations):
            frac = it / max(iterations - 1, 1)
            ramp = min(1.0, 0.05 + frac / 0.4)  # penetration and closure fade in
            w = EnergyWeights(**{**base.__dict__, "penetration": base.penetration * ramp, "closure": base.closure * ramp})
            total, terms = grasp_energy(self.hand, self.obj, HandPose(*params), self.contacts, w)
            opt.zero_grad()
            total.sum().backward()
            opt.step()
            sched.step()
            temp = noise * max(0.0, 1.0 - frac / 0.6) ** 2  # annealed exploration noise
            if temp > 0:
                with torch.no_grad():
                    params[2].add_(temp * lr[2] * torch.randn(params[2].shape, generator=self.gen, dtype=params[2].dtype))
                    params[1].add_(temp * lr[1] * torch.randn(params[1].shape, generator=self.gen, dtype=params[1].dtype))
            if record_every and it % record_every == 0:
                history.append(HandPose(*params).detach())
            if verbose and it % 100 == 0:
                print(it, f"{total.min().item():.3f}", {k: f"{v.min().item():.3f}" for k, v in terms.items()})
        final = HandPose(*params).detach()
        total, terms = grasp_energy(self.hand, self.obj, final, self.contacts, base)
        if record_every:
            history.append(final)
        return GraspResult(final, total.detach(), {k: v.detach() for k, v in terms.items()}, history,
                           time.perf_counter() - t0)


def penetration_depth(hand: HandModel, obj: SDF, pose: HandPose, chunk: int = 4096) -> torch.Tensor:
    """Maximum interpenetration ``(B,)`` in metres from a dense, two-sided sampled check.

    The hand's dense surface samples are tested against the object's exact SDF, and ~30k points
    on the object surface against the hand's capsules and palm hull. The second direction catches
    object edges and thin parts poking into the hand between hand samples. Both sample sets are
    independent of the ones the optimiser sees, but this is still a sampled check: features much
    smaller than the sample spacing (about 3 mm on the hand, 1-2 mm on the object) can be missed.
    """
    with torch.no_grad():
        fk = hand.forward(pose)
        into_object = torch.relu(-obj(hand.dense_surface_points(fk))).max(-1).values
        q = obj.surface_points().to(hand.dtype)
        into_hand = torch.zeros_like(into_object)
        for b in range(len(into_hand)):
            fb = {k: v[b : b + 1] for k, v in fk.items()}
            for part in q.split(chunk):
                into_hand[b] = torch.maximum(into_hand[b], torch.relu(-hand.signed_distance(fb, part[None])).max())
    return torch.maximum(into_object, into_hand)


def evaluate(hand: HandModel, obj: SDF, pose: HandPose, contacts, mu: float = 0.5, touch: float = 0.002):
    """Physical quality of each grasp in a batch.

    Returns a list of dicts with the Ferrari-Canny epsilon (contacts within
    ``touch`` metres of the surface, projected onto it), the maximum
    penetration depth from :func:`penetration_depth` and the mean contact
    distance, both in millimetres.
    """
    with torch.no_grad():
        fk = hand.forward(pose)
        c, _ = hand.contacts(fk, contacts)
        d_c = obj(c)
    pen = penetration_depth(hand, obj, pose)
    n = obj.normal(c).detach()
    out = []
    for b in range(len(c)):
        on = d_c[b].abs() < touch
        p = (c[b] - d_c[b, :, None] * n[b])[on].numpy()
        eps = epsilon_metric(p, n[b][on].numpy(), mu, center=obj.center, torque_scale=obj.radius) if on.sum() >= 3 else 0.0
        out.append({
            "epsilon": eps,
            "n_contacts": int(on.sum()),
            "penetration_mm": 1e3 * float(pen[b]),
            "contact_distance_mm": 1e3 * float(d_c[b].abs().mean()),
            # valid: force closure with >= 3 touching contacts and < 1 mm penetration
            "valid": bool(eps > 0 and on.sum() >= 3 and pen[b] < 1e-3),
        })
    return out
