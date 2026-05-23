"""Animation + snapshot export from a trajectory dict.

``make_movie`` writes an mp4 (ffmpeg) or falls back to gif (pillow).
``save_snapshots`` writes PNG stills at chosen frames for slides. Cells are
colored by sensitivity; the chemokine field is drawn as a background.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove in Colab to show inline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def _field_at(traj, t):
    """Chemokine field for frame t (handles static vs. dynamic storage)."""
    cf = traj["chemokine_field"]
    return cf[t] if cf.ndim == 3 else cf


def _setup_axes(traj):
    L = traj["box_size"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def _draw_frame(ax, traj, t):
    L = traj["box_size"]
    ax.clear()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(_field_at(traj, t), origin="lower", extent=[0, L, 0, L],
              cmap="Greys", alpha=0.6, zorder=0)
    pos = traj["positions"][t]
    th = traj["orientations"][t]
    sens = traj["sensitivity"][t]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=sens, cmap="viridis",
                    vmin=0, vmax=1, s=18, zorder=2)
    ax.quiver(pos[:, 0], pos[:, 1], np.cos(th), np.sin(th),
              color="k", alpha=0.35, scale=40, width=0.003, zorder=1)
    sx, sy = traj["source_position"]
    ax.plot(sx, sy, "r*", ms=14, zorder=3)
    ax.set_title(f"step {t}", fontsize=10)
    return sc


def make_movie(traj, path="swarm.mp4", stride=10, fps=20, dpi=110):
    """Render the trajectory to a movie. Returns the written path."""
    n_steps = traj["positions"].shape[0]
    frames = range(0, n_steps, stride)
    fig, ax = _setup_axes(traj)
    sc = _draw_frame(ax, traj, 0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="sensitivity")

    def update(t):
        _draw_frame(ax, traj, t)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    try:
        anim.save(path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    except Exception:
        path = path.rsplit(".", 1)[0] + ".gif"
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def save_snapshots(traj, frames, prefix="snapshot", dpi=130):
    """Write PNG stills at the given frame indices. Returns list of paths."""
    paths = []
    for t in frames:
        fig, ax = _setup_axes(traj)
        _draw_frame(ax, traj, t)
        out = f"{prefix}_{t:05d}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths
