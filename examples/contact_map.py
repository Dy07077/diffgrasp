"""Where does the hand touch the object? (assets/contact_maps.png)

Each column shows a grasp and, below it, the same hand seen from the palm
with the contact map: skin within ~2 mm of the object surface is coloured.

    python examples/contact_map.py
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from common import ASSETS, framing, sdf_numpy, synthesize
from diffgrasp.render import INK, MUTED, OBJECT, Renderer, contact_colors

CASES = [("ball", "precision", (20, -40)), ("mug", "power", (15, 125)), ("donut", "power", (35, -60))]


def main():
    cols = []
    for name, grasp, (elev, azim) in CASES:
        hand, obj, opt, res, b, stats = synthesize(name, grasp)
        pose = res.pose[b : b + 1]
        fk = hand.forward(pose)
        V, F, _ = hand.mesh(fk)
        OV, OF = obj.mesh(64)
        colours = contact_colors(V, F, sdf_numpy(obj), width=0.0025)
        centre, radius = framing(V, OV)
        top = Renderer(center=centre, radius=radius, size=(360, 360), elev=elev, azim=azim, zoom=1.2)
        top.overlays = [lambda fig, n=name, g=grasp: (fig.text(0.05, 0.92, n, color=INK, fontsize=12, fontweight="bold"),
                                                      fig.text(0.05, 0.875, f"{g} grasp", color=MUTED, fontsize=9))]
        a = top.render([{"V": OV, "F": OF, "color": OBJECT}, {"V": V, "F": F, "color": colours}])
        # view the hand from the palm side: look along the palm normal
        palm_normal = fk["Rg"][0, :, 2].detach().numpy()
        el = np.degrees(np.arcsin(np.clip(palm_normal[2], -1, 1)))
        az = np.degrees(np.arctan2(palm_normal[1], palm_normal[0]))
        hc, hr = framing(V, V, scale=0.6)
        bottom = Renderer(center=hc, radius=hr, size=(360, 360), elev=el, azim=az, zoom=1.2)
        bottom.overlays = [lambda fig: fig.text(0.05, 0.92, "contact map (palm view)", color=MUTED, fontsize=9)]
        c = bottom.render([{"V": V, "F": F, "color": colours}])
        cols.append(np.concatenate([a, c], 0))
        print(name, grasp, stats["best"], flush=True)
    Image.fromarray(np.concatenate(cols, 1)).save(ASSETS / "contact_maps.png", optimize=True)
    print("wrote", ASSETS / "contact_maps.png")


if __name__ == "__main__":
    main()
