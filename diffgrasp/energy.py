"""Differentiable grasp energy.

For a batch of hand poses the energy combines

* contact      -- chosen contact points (finger pads, palm) lie on the surface,
* alignment    -- their pad normals oppose the object normal,
* penetration  -- no sample on the hand surface lies inside the object,
* closure      -- small disturbance wrenches lie inside the friction-cone
                  wrench hull (a differentiable Ferrari-Canny surrogate, see
                  :mod:`diffgrasp.wrench`),
* self         -- fingers do not intersect each other or the palm,
* prior        -- joint angles stay near a relaxed posture.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .hand import HandModel, HandPose
from .objects import SDF
from .wrench import closure_energy

__all__ = ["GRASP_TYPES", "EnergyWeights", "grasp_energy"]

GRASP_TYPES = {
    "precision": ["thumb_distal", "index_distal", "middle_distal"],
    "fingertips": ["thumb_distal", "index_distal", "middle_distal", "ring_distal", "little_distal"],
    "power": [
        "thumb_distal", "thumb_middle",
        "index_distal", "index_middle", "index_proximal",
        "middle_distal", "middle_middle", "middle_proximal",
        "ring_distal", "ring_middle",
        "little_distal",
        "palm",
    ],
}  # fmt: skip


@dataclass
class EnergyWeights:
    contact: float = 3.0
    align: float = 0.5
    penetration: float = 20.0
    closure: float = 1.0
    self_collision: float = 20.0
    prior: float = 0.01
    scale: float = 0.01  # metres; distances are measured in units of 1 cm
    mu: float = 0.5  # friction coefficient assumed by the closure term
    contact_softness: float = 0.05  # smooth-L1 corner, in units of ``scale`` (0.5 mm)
    delta: float = 0.05  # radius of the disturbance-wrench set that must be resisted
    touch_radius: float = 0.003  # metres; friction cones fade out beyond this distance
    # "hull": friction-cone wrench hull (default). "normals": |G c|, the net wrench of equal
    # normal forces from Liu et al. (2021), kept as a baseline for ablations.
    closure_model: str = "hull"


def _self_collision_mask(hand: HandModel):
    _, rad, fid, kid = hand.collision_centres(hand.forward(hand.rest_pose(1)))
    diff = fid[:, None] != fid[None, :]
    knuckle = (kid == 0) & (fid > 0)  # spheres on proximal phalanges of fingers sit at the knuckles
    keep = diff & ~(knuckle[:, None] & knuckle[None, :])
    palm = ((kid > 0) & (fid > 0)) | ((kid == 2) & (fid == 0))
    return keep, palm


def grasp_energy(hand: HandModel, obj: SDF, pose: HandPose, contacts, w: EnergyWeights | None = None):
    """Return ``(total, terms)`` with ``total`` of shape ``(B,)``."""
    w = w or EnergyWeights()
    s = w.scale
    fk = hand.forward(pose)
    c, n_hand = hand.contacts(fk, contacts)
    d_c = obj(c)
    n_obj = obj.normal(c)
    terms = {}
    # smooth L1: an almost constant pull until the last millimetre or so, rounding off to zero within
    # about contact_softness * scale (0.5 mm) so that the gradient stays smooth
    eta = w.contact_softness
    terms["contact"] = (torch.sqrt((d_c / s) ** 2 + eta**2) - eta).sum(-1)
    terms["align"] = (1.0 + (n_hand * n_obj).sum(-1)).sum(-1)
    d_p = obj(hand.surface_points(fk))
    depth = torch.relu(-d_p) / s
    terms["penetration"] = (hand.sample_weights * (depth + depth**2)).sum(-1)  # area-weighted samples
    centre = torch.as_tensor(obj.center, dtype=c.dtype)
    if w.closure_model == "normals":
        # |G c| (Liu et al. 2021, Eq. 6): net wrench of equal normal forces, friction left out
        f = -n_obj
        net = torch.cat([f, torch.cross(c - centre, f, dim=-1) / obj.radius], -1).sum(-2)
        terms["closure"] = torch.sqrt((net**2).sum(-1) + 1e-12)
    else:
        # only contacts that actually touch may resist wrenches: cones fade with distance
        touching = torch.exp(-((d_c / w.touch_radius) ** 2))
        terms["closure"] = closure_energy(c, n_obj, centre, obj.radius, mu=w.mu, delta=w.delta, strength=touching) / w.delta**2
    if not hasattr(hand, "_collision_masks"):
        hand._collision_masks = _self_collision_mask(hand)
    keep, palm = hand._collision_masks
    cen, rad, _, _ = hand.collision_centres(fk)
    rad = 0.85 * rad
    D = torch.cdist(cen, cen)
    pen = torch.relu(rad[:, None] + rad[None, :] - D) * keep
    sd_palm = hand.palm_signed_distance(fk, cen[:, palm])
    terms["self"] = 0.5 * ((pen / s) ** 2).sum((-1, -2)) + ((torch.relu(rad[palm] - sd_palm) / s) ** 2).sum(-1)
    terms["prior"] = ((fk["angles"] - hand.rest_angles) ** 2).sum(-1)
    total = (
        w.contact * terms["contact"]
        + w.align * terms["align"]
        + w.penetration * terms["penetration"]
        + w.closure * terms["closure"]
        + w.self_collision * terms["self"]
        + w.prior * terms["prior"]
    )
    return total, terms
