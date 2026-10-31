"""A procedural, anatomically proportioned articulated hand.

The hand is a kinematic tree with 21 joint angles: for every finger
abduction + flexion at the MCP joint and flexion at PIP and DIP; for the
thumb three degrees of freedom at the CMC joint (abduction, flexion and the
axial rotation that lets the pad face the fingers in opposition) plus MCP
and IP flexion.
Each phalanx is a capsule; the palm is the convex hull of spheres placed at
the knuckles, the wrist and the thenar eminence. Everything is plain PyTorch,
so poses can be optimised with gradients.

Wrist frame (right hand): +y points along the middle finger, +x towards the
thumb, +z out of the palm. Units are metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .rotation import rot_x, rot_y, rot_z, rotation_6d_to_matrix

__all__ = ["HandModel", "HandPose", "FINGERS", "JOINT_NAMES"]

FINGERS = ["thumb", "index", "middle", "ring", "little"]
_SEG = ["proximal", "middle", "distal"]  # for the thumb: metacarpal, proximal, distal
JOINT_NAMES = [
    "thumb_cmc_abd", "thumb_cmc_flex", "thumb_cmc_rot", "thumb_mcp", "thumb_ip",
    "index_abd", "index_mcp", "index_pip", "index_dip",
    "middle_abd", "middle_mcp", "middle_pip", "middle_dip",
    "ring_abd", "ring_mcp", "ring_pip", "ring_dip",
    "little_abd", "little_mcp", "little_pip", "little_dip",
]  # fmt: skip

# name: (base position, base direction, lengths, radii)
_FINGER_SPECS = {
    "index": ((0.025, 0.088, 0.0), -0.10, (0.040, 0.024, 0.019), (0.0098, 0.0089, 0.0081)),
    "middle": ((0.007, 0.092, 0.0), 0.0, (0.044, 0.027, 0.020), (0.0101, 0.0091, 0.0083)),
    "ring": ((-0.011, 0.088, 0.0), 0.08, (0.041, 0.026, 0.019), (0.0095, 0.0086, 0.0078)),
    "little": ((-0.027, 0.080, 0.0), 0.18, (0.033, 0.020, 0.018), (0.0085, 0.0077, 0.0071)),
}
_THUMB = dict(
    base=(0.021, 0.024, 0.010),
    direction=(0.78, 0.60, 0.14),
    pad=(-0.402, 0.322, 0.857),  # thumb pad at rest (palmar-ulnar)
    lengths=(0.043, 0.032, 0.027),
    radii=(0.0122, 0.0106, 0.0096),
)
# limits in degrees: fingers (abd, mcp, pip, dip); thumb (cmc_abd, cmc_flex, cmc_rot, mcp, ip)
_LIMITS_DEG = {
    "thumb": ((-30, 35), (-25, 60), (-90, 40), (-10, 60), (-15, 85)),
    "finger": ((-18, 18), (-20, 95), (0, 110), (-5, 85)),
}
_REST_DEG = {"thumb": (0, 5, 0, 10, 10), "finger": (0, 15, 20, 10)}


@dataclass
class HandPose:
    """Unconstrained pose parameters for a batch of hands."""

    rot6d: torch.Tensor  # (B, 6) wrist orientation
    trans: torch.Tensor  # (B, 3) wrist position
    logits: torch.Tensor  # (B, 21) joint angles before the sigmoid limit map

    def detach(self) -> "HandPose":
        return HandPose(self.rot6d.detach().clone(), self.trans.detach().clone(), self.logits.detach().clone())

    def __getitem__(self, idx) -> "HandPose":
        return HandPose(self.rot6d[idx], self.trans[idx], self.logits[idx])


def _capsule_points(length, radius, n_rings=6, n_around=12, n_cap=2):
    """Points on a capsule along +y from 0 to ``length`` (local frame)."""
    phi = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    pts = []
    for y in np.linspace(0.0, length, n_rings):
        pts.append(np.stack([radius * np.cos(phi), np.full_like(phi, y), radius * np.sin(phi)], 1))
    for k in range(1, n_cap + 1):  # distal cap
        a = 0.5 * np.pi * k / (n_cap + 1)
        r = radius * np.cos(a)
        pts.append(np.stack([r * np.cos(phi), np.full_like(phi, length + radius * np.sin(a)), r * np.sin(phi)], 1))
    pts.append(np.array([[0.0, length + radius, 0.0]]))
    return np.concatenate(pts)


def _capsule_mesh(length, radius, n_around=20, n_lat=6, spacing=0.003):
    """Closed triangle mesh of a capsule along +y (both ends rounded).

    The cylinder is split into rings ``spacing`` apart so that all triangles
    are small, which keeps matplotlib's depth sorting reliable.
    """
    rings = []
    for k in range(n_lat, 0, -1):  # proximal cap, from pole to equator
        a = 0.5 * np.pi * k / n_lat
        rings.append((-radius * np.sin(a), radius * np.cos(a)))
    n_cyl = max(1, int(np.ceil(length / spacing)))
    rings += [(y, radius) for y in np.linspace(0.0, length, n_cyl + 1)]
    for k in range(1, n_lat + 1):
        a = 0.5 * np.pi * k / n_lat
        rings.append((length + radius * np.sin(a), radius * np.cos(a)))
    phi = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    V = [np.array([[0.0, -radius, 0.0]])]
    for y, r in rings:
        if r < 1e-9:
            continue
        V.append(np.stack([r * np.cos(phi), np.full_like(phi, y), r * np.sin(phi)], 1))
    V.append(np.array([[0.0, length + radius, 0.0]]))
    V = np.concatenate(V)
    n_r = (len(V) - 2) // n_around
    F = []
    for j in range(n_around):  # proximal pole fan
        F.append([0, 1 + (j + 1) % n_around, 1 + j])
    for i in range(n_r - 1):
        a0, b0 = 1 + i * n_around, 1 + (i + 1) * n_around
        for j in range(n_around):
            j1 = (j + 1) % n_around
            F.append([a0 + j, a0 + j1, b0 + j1])
            F.append([a0 + j, b0 + j1, b0 + j])
    top, last = len(V) - 1, 1 + (n_r - 1) * n_around
    for j in range(n_around):
        F.append([top, last + j, last + (j + 1) % n_around])
    F = np.array(F)
    # orient outward
    c = V[F].mean(1)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    axis = np.clip(c[:, 1], 0.0, length)
    out = c - np.stack([np.zeros_like(axis), axis, np.zeros_like(axis)], 1)
    flip = (n * out).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, F


def _smooth_hull_mesh(equations, lo, hi, sharpness=1500.0, res=40):
    """Fine, slightly rounded mesh of a convex hull for rendering.

    The hull's signed distance ``max_i (n_i . x + b_i)`` is replaced by a
    log-sum-exp soft maximum, which rounds the edges by ~``1/sharpness`` metres,
    and its zero level set is extracted with marching cubes.
    """
    from skimage.measure import marching_cubes

    pad = 0.004
    axes = [np.linspace(a - pad, b + pad, res) for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    z = sharpness * (g @ equations[:, :3].T + equations[:, 3])
    zmax = z.max(1, keepdims=True)
    d = (zmax[:, 0] + np.log(np.exp(z - zmax).sum(1))) / sharpness
    spacing = tuple(a[1] - a[0] for a in axes)
    V, F, _, _ = marching_cubes(d.reshape(res, res, res), 0.0, spacing=spacing)
    return V + np.array([a[0] for a in axes]), F


def _sphere_points(center, radius, n=60):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    return np.asarray(center) + radius * d


class HandModel:
    """Kinematics and geometry of the procedural hand.

    Example:
        >>> hand = HandModel()
        >>> pose = hand.rest_pose(batch=4)
        >>> fk = hand.forward(pose)
        >>> fk["tips"].shape
        torch.Size([4, 5, 3])
    """

    n_joints = 21

    def __init__(self, dtype=torch.float32, device="cpu"):
        self.dtype, self.device = dtype, device
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731

        base_R, base_p, lengths, radii = [], [], [], []
        d = np.asarray(_THUMB["direction"], float)
        d /= np.linalg.norm(d)
        z = np.asarray(_THUMB["pad"], float)
        z -= z @ d * d
        z /= np.linalg.norm(z)
        base_R.append(np.stack([np.cross(d, z), d, z], 1))
        base_p.append(_THUMB["base"])
        lengths.append(_THUMB["lengths"])
        radii.append(_THUMB["radii"])
        for name in FINGERS[1:]:
            p, splay, L, r = _FINGER_SPECS[name]
            c, s = np.cos(splay), np.sin(splay)
            base_R.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
            base_p.append(p)
            lengths.append(L)
            radii.append(r)
        self.base_R, self.base_p = t(np.array(base_R)), t(np.array(base_p))
        self.lengths, self.radii = t(np.array(lengths)), t(np.array(radii))  # (5, 3)

        lim = np.concatenate([np.array(_LIMITS_DEG["thumb"], float), np.array(_LIMITS_DEG["finger"] * 4, float)])
        self.lower, self.upper = t(np.radians(lim[:, 0])), t(np.radians(lim[:, 1]))
        rest = np.concatenate([_REST_DEG["thumb"], _REST_DEG["finger"] * 4]).astype(float)
        self.rest_angles = t(np.radians(rest))

        # palm: convex hull of spheres at knuckles, wrist and thenar eminence
        from scipy.spatial import ConvexHull

        spheres = [(p, r[0] + 0.0012) for p, (_, _, _, r) in zip(base_p[1:], [_FINGER_SPECS[f] for f in FINGERS[1:]])]
        spheres += [((0.020, 0.004, 0.0), 0.0135), ((-0.020, 0.004, 0.0), 0.0135), ((0.0, 0.0, 0.0), 0.014),
                    ((0.019, 0.034, 0.003), 0.0145), ((-0.022, 0.045, 0.0), 0.0125)]  # fmt: skip
        cloud = np.concatenate([_sphere_points(c, r, 80) for c, r in spheres])
        hull = ConvexHull(cloud)
        verts = cloud[np.unique(hull.simplices)]
        hull = ConvexHull(verts)
        F = hull.simplices.copy()
        n = np.cross(verts[F[:, 1]] - verts[F[:, 0]], verts[F[:, 2]] - verts[F[:, 0]])
        flip = (n * (verts[F].mean(1) - verts.mean(0))).sum(1) < 0
        F[flip] = F[flip][:, [0, 2, 1]]
        self.palm_vertices, self.palm_faces = t(verts), F
        self.palm_planes = t(hull.equations)  # rows (n, b): n . x + b <= 0 inside
        self.palm_render = _smooth_hull_mesh(hull.equations, cloud.min(0), cloud.max(0))
        palm_samples = verts[np.linspace(0, len(verts) - 1, min(len(verts), 160)).astype(int)]
        self.palm_samples = t(palm_samples)
        self.palm_front = float(verts[:, 2].max())

        # per-segment surface samples and render meshes (segment-local frames)
        self.seg_samples, self.seg_meshes = [], []
        for f in range(5):
            for k in range(3):
                L, r = lengths[f][k], radii[f][k]
                self.seg_samples.append(t(_capsule_points(L, r)))
                self.seg_meshes.append(_capsule_mesh(L, r))

        # contact candidates: (segment index or -1 for palm, local point, local normal)
        cand = {}
        for f, name in enumerate(FINGERS):
            for k, seg in enumerate(_SEG):
                L, r = lengths[f][k], radii[f][k]
                along = 0.62 if k == 2 else 0.5
                cand[f"{name}_{seg}"] = (3 * f + k, (0.0, along * L, r), (0.0, 0.0, 1.0))
        cand["palm"] = (-1, (0.004, 0.052, self.palm_front), (0.0, 0.0, 1.0))
        cand["palm_ulnar"] = (-1, (-0.016, 0.060, self.palm_front - 0.001), (0.0, 0.0, 1.0))
        self.contact_candidates = cand

        # spheres used for finger-finger and finger-palm collision (segment-local)
        self.collision_spheres = []  # (segment, local centre, radius)
        for f in range(5):
            for k in range(3):
                L, r = lengths[f][k], radii[f][k]
                for a in (0.25, 0.6, 1.0):
                    self.collision_spheres.append((3 * f + k, (0.0, a * L, 0.0), r))

    # ---------------------------------------------------------------- poses
    def angles(self, logits: torch.Tensor) -> torch.Tensor:
        """Map unconstrained logits into the anatomical joint limits."""
        return self.lower + (self.upper - self.lower) * torch.sigmoid(logits)

    def logits(self, angles: torch.Tensor) -> torch.Tensor:
        s = ((angles - self.lower) / (self.upper - self.lower)).clamp(1e-4, 1 - 1e-4)
        return torch.log(s) - torch.log1p(-s)

    def rest_pose(self, batch: int = 1) -> HandPose:
        rot6d = torch.tensor([1.0, 0, 0, 0, 1, 0], dtype=self.dtype, device=self.device).repeat(batch, 1)
        trans = torch.zeros(batch, 3, dtype=self.dtype, device=self.device)
        return HandPose(rot6d, trans, self.logits(self.rest_angles).repeat(batch, 1))

    # ----------------------------------------------------------- kinematics
    def forward(self, pose: HandPose) -> dict:
        """Forward kinematics.

        Returns a dict with world-space ``R`` (B, 15, 3, 3) and ``p`` (B, 15, 3)
        for the 15 phalanx frames (segment ``3 f + k``), the wrist rotation and
        translation, the joint angles and the fingertip positions ``tips``.
        """
        angles = self.angles(pose.logits)
        B = angles.shape[0]
        fingers = angles[:, 5:].reshape(B, 4, 4)
        zero = torch.zeros(B, 4, dtype=angles.dtype, device=angles.device)
        # per chain: abduction, flexion, axial rotation (thumb only), flexion, flexion
        q = torch.stack(
            [
                torch.cat([angles[:, :1], fingers[..., 0]], 1),
                torch.cat([angles[:, 1:2], fingers[..., 1]], 1),
                torch.cat([angles[:, 2:3], zero], 1),
                torch.cat([angles[:, 3:4], fingers[..., 2]], 1),
                torch.cat([angles[:, 4:5], fingers[..., 3]], 1),
            ],
            dim=-1,
        )  # (B, 5, 5)
        Rg = rotation_6d_to_matrix(pose.rot6d)
        tg = pose.trans
        R1 = self.base_R @ rot_z(q[..., 0]) @ rot_x(q[..., 1]) @ rot_y(q[..., 2])  # (B, 5, 3, 3)
        p1 = self.base_p.expand_as(R1[..., 0])
        R2 = R1 @ rot_x(q[..., 3])
        p2 = p1 + R1[..., :, 1] * self.lengths[:, 0, None]
        R3 = R2 @ rot_x(q[..., 4])
        p3 = p2 + R2[..., :, 1] * self.lengths[:, 1, None]
        tip = p3 + R3[..., :, 1] * (self.lengths[:, 2, None] + self.radii[:, 2, None])
        R = torch.stack([R1, R2, R3], dim=2).flatten(1, 2)  # (B, 15, 3, 3)
        p = torch.stack([p1, p2, p3], dim=2).flatten(1, 2)
        Rw = Rg[:, None] @ R
        pw = (Rg[:, None] @ p[..., None])[..., 0] + tg[:, None]
        tips = (Rg[:, None] @ tip[..., None])[..., 0] + tg[:, None]
        return {"R": Rw, "p": pw, "Rg": Rg, "t": tg, "angles": angles, "tips": tips}

    def to_world(self, fk: dict, segment: int, local: torch.Tensor) -> torch.Tensor:
        """Transform ``(n, 3)`` points from a segment frame (``-1`` = palm/wrist)."""
        if segment < 0:
            return local @ fk["Rg"].transpose(1, 2) + fk["t"][:, None]
        return local @ fk["R"][:, segment].transpose(1, 2) + fk["p"][:, segment, None]

    def surface_points(self, fk: dict) -> torch.Tensor:
        """``(B, P, 3)`` samples on all capsules and the palm, for penetration tests."""
        pts = [self.to_world(fk, s, loc) for s, loc in enumerate(self.seg_samples)]
        pts.append(self.to_world(fk, -1, self.palm_samples))
        return torch.cat(pts, dim=1)

    def contacts(self, fk: dict, names) -> tuple[torch.Tensor, torch.Tensor]:
        """World positions and outward normals ``(B, K, 3)`` of named contact candidates."""
        pts, nrm = [], []
        for name in names:
            seg, p, n = self.contact_candidates[name]
            p = torch.as_tensor(p, dtype=self.dtype, device=self.device)[None]
            n = torch.as_tensor(n, dtype=self.dtype, device=self.device)
            pts.append(self.to_world(fk, seg, p)[:, 0])
            R = fk["Rg"] if seg < 0 else fk["R"][:, seg]
            nrm.append(R @ n)
        return torch.stack(pts, 1), torch.stack(nrm, 1)

    def collision_centres(self, fk: dict):
        """World centres ``(B, S, 3)``, radii ``(S,)`` and finger ids ``(S,)`` of collision spheres."""
        cs, rs, fs, ks = [], [], [], []
        for seg, c, r in self.collision_spheres:
            cs.append(self.to_world(fk, seg, torch.as_tensor(c, dtype=self.dtype, device=self.device)[None])[:, 0])
            rs.append(r)
            fs.append(seg // 3)
            ks.append(seg % 3)
        t = lambda a, dt=self.dtype: torch.as_tensor(a, dtype=dt, device=self.device)  # noqa: E731
        return torch.stack(cs, 1), t(rs), t(fs, torch.long), t(ks, torch.long)

    def palm_signed_distance(self, fk: dict, x: torch.Tensor) -> torch.Tensor:
        """Approximate signed distance of world points ``(B, n, 3)`` to the palm hull."""
        local = (x - fk["t"][:, None]) @ fk["Rg"]  # R^T (x - t), written for row vectors
        d = local @ self.palm_planes[:, :3].T + self.palm_planes[:, 3]
        return d.max(dim=-1).values

    # --------------------------------------------------------------- meshes
    def mesh(self, fk: dict, b: int = 0):
        """Triangle mesh of hand ``b`` as numpy arrays ``(V, F, part)``."""
        Vs, Fs, parts, off = [], [], [], 0
        R, p = fk["R"][b].detach().cpu().numpy(), fk["p"][b].detach().cpu().numpy()
        for s, (V, F) in enumerate(self.seg_meshes):
            Vs.append(V @ R[s].T + p[s])
            Fs.append(F + off)
            parts.append(np.full(len(F), s))
            off += len(V)
        Rg, tg = fk["Rg"][b].detach().cpu().numpy(), fk["t"][b].detach().cpu().numpy()
        PV, PF = self.palm_render
        Vs.append(PV @ Rg.T + tg)
        Fs.append(PF + off)
        parts.append(np.full(len(PF), -1))
        return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(parts)
