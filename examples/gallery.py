"""Best grasp per object (assets/gallery.png) and a results table (assets/results.json).

    python examples/gallery.py
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from common import ASSETS, framing, sdf_numpy, synthesize
from diffgrasp.render import INK, MUTED, OBJECT, Renderer, contact_colors

CASES = [
    ("ball", "precision"), ("cube", "precision"), ("donut", "power"),
    ("can", "power"), ("mug", "power"), ("bottle", "power"),
]  # fmt: skip
VIEWS = {"ball": (22, -40), "cube": (25, -60), "donut": (35, -60), "can": (15, 120), "mug": (15, 125), "bottle": (10, 120)}


def main():
    tiles, table = [], []
    for name, grasp in CASES:
        hand, obj, opt, res, b, stats = synthesize(name, grasp)
        fk = hand.forward(res.pose[b : b + 1])
        V, F, _ = hand.mesh(fk)
        OV, OF = obj.mesh(72)
        centre, radius = framing(V, OV)
        elev, azim = VIEWS[name]
        r = Renderer(center=centre, radius=radius, size=(400, 400), elev=elev, azim=azim, zoom=1.2)
        e = stats["best"]

        def overlay(fig, name=name, grasp=grasp, e=e):
            fig.text(0.05, 0.92, f"{name}", color=INK, fontsize=13, fontweight="bold")
            fig.text(0.05, 0.875, f"{grasp} grasp", color=MUTED, fontsize=9)
            fig.text(0.95, 0.05, f"ε = {e['epsilon']:.3f}   contacts {e['n_contacts']}   pen. {e['penetration_mm']:.2f} mm",
                     color=MUTED, fontsize=8, ha="right", family="monospace")

        r.overlays = [overlay]
        tiles.append(r.render([{"V": OV, "F": OF, "color": OBJECT}, {"V": V, "F": F, "color": contact_colors(V, F, sdf_numpy(obj))}]))
        table.append(stats)
        print(f"{name:7s} {grasp:9s} valid {stats['valid_fraction']:.0%}  eps {e['epsilon']:.3f}  "
              f"contacts {e['n_contacts']}  pen {e['penetration_mm']:.2f} mm  ({stats['seconds']:.0f} s)", flush=True)
    grid = np.concatenate([np.concatenate(tiles[i : i + 3], 1) for i in (0, 3)], 0)
    ASSETS.mkdir(exist_ok=True)
    Image.fromarray(grid).save(ASSETS / "gallery.png", optimize=True)
    (ASSETS / "results.json").write_text(json.dumps(table, indent=2))
    print("wrote", ASSETS / "gallery.png")


if __name__ == "__main__":
    main()
