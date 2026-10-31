"""Watch a power grasp on a mug being optimised (assets/grasp.gif).

    python examples/animate.py
"""
from __future__ import annotations

import numpy as np

import diffgrasp as dg
from common import ASSETS, framing, sdf_numpy, synthesize
from diffgrasp.render import INK, MUTED, OBJECT, Renderer, contact_colors, save_gif


def main():
    every = 10
    hand, obj, opt, res, b, stats = synthesize("mug", "power", n_seeds=32, record_every=every)
    OV, OF = obj.mesh(64)
    sdf = sdf_numpy(obj)
    fk_final = hand.forward(res.pose[b : b + 1])
    Vf, _, _ = hand.mesh(fk_final)
    centre, radius = framing(Vf, OV, scale=0.62)
    r = Renderer(center=centre, radius=radius, size=(520, 420), elev=18, zoom=1.15)
    state = {}

    def overlay(fig):
        fig.text(0.04, 0.92, "diffgrasp", color=INK, fontsize=14, fontweight="bold")
        fig.text(0.04, 0.875, "gradient-based grasp synthesis, power grasp on a mug", color=MUTED, fontsize=8.5)
        fig.text(0.96, 0.92, state["label"], color=MUTED, fontsize=9, ha="right", family="monospace")
        fig.text(0.04, 0.05, state["metrics"], color=MUTED, fontsize=8.5, family="monospace")

    r.overlays = [overlay]
    frames = []
    n = len(res.history)
    for k, snap in enumerate(res.history):
        pose = snap[b : b + 1]
        e = dg.evaluate(hand, obj, pose, opt.contacts, touch=0.002)[0]
        V, F, _ = hand.mesh(hand.forward(pose))
        r.azim = -80 + 70 * k / max(n - 1, 1)
        state["label"] = f"iteration {min(k * every, 800):4d}"
        state["metrics"] = f"contacts {e['n_contacts']:2d}/{len(opt.contacts)}   penetration {e['penetration_mm']:5.2f} mm   ε = {e['epsilon']:.3f}"
        frames.append(r.render([{"V": OV, "F": OF, "color": OBJECT}, {"V": V, "F": F, "color": contact_colors(V, F, sdf)}]))
    Vf, Ff, _ = hand.mesh(fk_final)
    cols = contact_colors(Vf, Ff, sdf)
    for k in range(48):  # orbit around the final grasp
        r.azim = -10 + 360 * k / 48
        state["label"] = "final grasp"
        frames.append(r.render([{"V": OV, "F": OF, "color": OBJECT}, {"V": Vf, "F": Ff, "color": cols}]))
    frames += [frames[-1]] * 6
    save_gif(frames, ASSETS / "grasp.gif", fps=20)
    print("wrote", ASSETS / "grasp.gif", len(frames), "frames;", stats["best"])


if __name__ == "__main__":
    main()
