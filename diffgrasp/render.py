"""Offline matplotlib rendering of hands, objects and contacts."""
from __future__ import annotations

import numpy as np

__all__ = ["Renderer", "save_gif", "contact_colors", "SKY", "INK", "MUTED", "SKIN", "OBJECT", "CONTACT"]

SKY = "#0d1117"
INK = "#e6edf3"
MUTED = "#8b949e"
SKIN = "#e8b796"
OBJECT = "#58a6ff"
CONTACT = "#ff5f56"


def contact_colors(V, F, sdf, base=SKIN, hot=CONTACT, width=0.002):
    """Per-face colours that blend ``hot`` into ``base`` where the hand touches.

    ``sdf`` maps ``(n, 3)`` numpy points to signed distances (metres).
    """
    from matplotlib.colors import to_rgb

    d = np.asarray(sdf(V[F].mean(1)))
    w = np.exp(-((np.clip(d, 0.0, None) / width) ** 2))[:, None]
    return (1 - w) * np.array(to_rgb(base)) + w * np.array(to_rgb(hot))


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class Renderer:
    """Shaded triangle-soup renderer with back-face culling (matplotlib Agg)."""

    def __init__(self, center, radius, size=(480, 480), elev=20.0, azim=-60.0, background=SKY,
                 light=(0.3, -0.5, 1.0), zoom=1.0, focal_length=0.45, rect=(0, 0, 1, 1), dpi=100):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.elev, self.azim = float(elev), float(azim)
        self.background = background
        self.light = _unit(light)
        self.zoom, self.focal_length, self.rect = zoom, focal_length, rect
        self.fig = Figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=background)
        self.canvas = FigureCanvasAgg(self.fig)
        self.overlays = []

    def eye_direction(self):
        e, a = np.radians(self.elev), np.radians(self.azim)
        return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def axes(self, rect=None):
        ax = self.fig.add_axes(rect or self.rect, projection="3d", facecolor=self.background)
        ax.set_proj_type("persp", focal_length=self.focal_length)
        ax.computed_zorder = False
        c, r = self.center, self.radius
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1), zoom=self.zoom)
        ax.view_init(self.elev, self.azim)
        ax.set_axis_off()
        return ax

    def _shade(self, tris, base):
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        eye = self.eye_direction()
        key = np.clip(n @ self.light, 0, 1)
        fill = np.clip(n @ eye, 0, 1)
        spec = np.clip(n @ _unit(self.light + eye), 0, 1) ** 40
        rgb = base * (0.28 + 0.5 * key + 0.3 * fill)[:, None] + 0.22 * spec[:, None]
        return np.clip(rgb, 0, 1), n

    def draw_mesh(self, ax, V, F, color, alpha=1.0, edges=0.0, zorder=2):
        from matplotlib.colors import to_rgb
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        tris = np.asarray(V)[np.asarray(F)]
        base = np.array(to_rgb(color)) if isinstance(color, str) else np.asarray(color, float)
        base = np.broadcast_to(base, (len(tris), 3))
        rgb, n = self._shade(tris, base)
        eye = self.center + 6 * self.radius * self.eye_direction()
        vis = np.einsum("ij,ij->i", n, eye - tris.mean(1)) > 0
        fc = np.concatenate([rgb[vis], np.full((vis.sum(), 1), alpha)], 1)
        ec = np.concatenate([rgb[vis] * 0.6, np.full((vis.sum(), 1), edges)], 1)
        pc = Poly3DCollection(tris[vis], facecolors=fc, edgecolors=ec, linewidths=0.2, zorder=zorder)
        pc.set_clip_on(False)
        ax.add_collection3d(pc)
        return pc

    def render(self, meshes, points=None, segments=None) -> np.ndarray:
        """Draw ``meshes`` (dicts with V, F, color[, alpha, edges]) and markers.

        All meshes are merged into one collection so that triangles of
        different meshes are depth-sorted together.
        """
        from matplotlib.colors import to_rgb
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.fig.clear()
        ax = self.axes()
        tris, fcs, ecs = [], [], []
        eye = self.center + 6 * self.radius * self.eye_direction()
        for m in meshes:
            t = np.asarray(m["V"])[np.asarray(m["F"])]
            col = m["color"]
            if isinstance(col, str):
                base = np.broadcast_to(np.array(to_rgb(col)), (len(t), 3))
            else:
                base = np.broadcast_to(np.asarray(col, float), (len(t), 3))
            rgb, n = self._shade(t, base)
            vis = np.einsum("ij,ij->i", n, eye - t.mean(1)) > 0
            a = m.get("alpha", 1.0)
            tris.append(t[vis])
            fc = np.concatenate([rgb[vis], np.full((vis.sum(), 1), a)], 1)
            fcs.append(fc)
            # outline every face in its own colour: hides anti-aliasing seams between tiny triangles
            e = m.get("edges", 0.0)
            ecs.append(fc if e == 0 else np.concatenate([rgb[vis] * 0.6, np.full((vis.sum(), 1), e)], 1))
        pc = Poly3DCollection(np.concatenate(tris), facecolors=np.concatenate(fcs), edgecolors=np.concatenate(ecs),
                              linewidths=0.45, zorder=2)
        pc.set_clip_on(False)
        ax.add_collection3d(pc)
        if segments is not None:
            for seg in segments:
                a, b = np.asarray(seg["a"]), np.asarray(seg["b"])
                for p, q in zip(a, b):
                    ax.plot(*np.stack([p, q], 1), color=seg.get("color", INK), lw=seg.get("lw", 1.2),
                            alpha=seg.get("alpha", 1.0), zorder=3, clip_on=False)
        if points is not None:
            for pts in points:
                P = np.asarray(pts["x"])
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=pts.get("s", 18), color=pts.get("color", "#ff7b72"),
                           edgecolors=pts.get("edge", "white"), linewidths=0.6, depthshade=False, zorder=4)
        for draw in self.overlays:
            draw(self.fig)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames, path, fps=25.0, colors=200):
    from PIL import Image

    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    step = max(1, len(frames) // 12)
    palette = Image.fromarray(np.concatenate(frames[::step], 0)).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps)), loop=0, optimize=True, disposal=1)
