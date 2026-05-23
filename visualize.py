"""Run a simulation and render snapshots so we can SEE what the engine does.
All axes are labeled in PHYSICAL units (microns, minutes) via the anchors in
out["params"] -- see to_minutes / to_microns in simulate.py.

Saves:
  swarm_snapshots.png   cell positions over time on the chemokine field
  aggregation_curve.png mean distance-to-source & mean sensitivity vs time
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend (no display needed)
import matplotlib.pyplot as plt
from simulate import simulate, _min_image, to_minutes, to_microns


def snapshot_panels(out, frames=(0.0, 0.25, 0.5, 1.0), fname="swarm_snapshots.png"):
    p = out["params"]
    L = out["box_size"]
    L_um = to_microns(L, p)                                     # box size in microns
    srcs_um = to_microns(np.array(out.get("source_positions", [out["source_position"]])), p)
    field = out["chemokine_field"]
    n_steps = out["positions"].shape[0]

    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4.5))
    for ax, frac in zip(axes, frames):
        t = min(int(frac * n_steps), n_steps - 1)
        pos_um = to_microns(out["positions"][t], p)             # cells in microns
        sens = out["sensitivity"][t]
        field_t = field[t] if field.ndim == 3 else field

        # chemokine field as background heatmap (extent in microns)
        ax.imshow(field_t, extent=[0, L_um, 0, L_um], origin="lower",
                  cmap="Greys", alpha=0.6)
        # cells colored by sensitivity (1 = fresh receptors, 0 = desensitized)
        sc = ax.scatter(pos_um[:, 0], pos_um[:, 1], c=sens, cmap="viridis",
                        vmin=0, vmax=1, s=14, edgecolors="none")
        ax.plot(srcs_um[:, 0], srcs_um[:, 1], "r*", ms=16)       # chemokine sources
        ax.set_xlim(0, L_um); ax.set_ylim(0, L_um)
        ax.set_title(f"t = {to_minutes(t, p):.0f} min")
        ax.set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")

    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("sensitivity s")
    fig.suptitle("T-cell swarm: cells (dots) chasing chemokine source (red star)")
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    print("wrote", fname)


def aggregation_curve(out, fname="aggregation_curve.png"):
    """Mean distance-to-source vs time -- the swarm forming and (if alpha>0) dispersing."""
    p = out["params"]
    srcs = np.array(out.get("source_positions", [out["source_position"]]))
    L = out["box_size"]
    d = _min_image(out["positions"][:, :, None, :] - srcs[None, None, :, :], L)
    nearest = np.linalg.norm(d, axis=3).min(axis=2)
    mean_d_um = to_microns(nearest.mean(axis=1), p)            # microns
    mean_s = out["sensitivity"].mean(axis=1)
    t_min = to_minutes(np.arange(out["positions"].shape[0]), p)        # minutes

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t_min, mean_d_um, "b-", label="mean dist to source")
    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("mean distance to source (µm)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(t_min, mean_s, "g--", label="mean sensitivity")
    ax2.set_ylabel("mean sensitivity", color="g"); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="g")
    fig.suptitle("Swarm aggregation & sensitivity over time")
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print("wrote", fname)


if __name__ == "__main__":
    # A run that shows the full story: swarm in, desensitize, leave, recover, return.
    out = simulate({
        "n_steps": 2600,
        "alpha": 0.45,
        "beta": 0.25,
        "recovery_K": 0.6,
        "w_escape": 3.0,
        "k_rep": 6.0,
    }, seed=7)
    print(f"run spans {to_minutes(out['positions'].shape[0], out['params']):.0f} min "
          f"in a {to_microns(out['box_size'], out['params']):.0f} µm box")
    snapshot_panels(out)
    aggregation_curve(out)
