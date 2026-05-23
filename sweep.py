"""The (alpha, k_rep) phase diagram sweep -- biochemical vs. mechanical
self-limitation of the swarm.

Runs ``simulate`` across a grid of (desensitization rate, repulsion strength),
records headline observables, and renders heatmaps. Built to be embarrassingly
parallel-friendly (each cell is independent) but kept serial here for clarity.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import simulate, DEFAULTS
import observables as obs


def run_sweep(alphas, k_reps, base=None, seed=0, n_steps=None, N=None,
              chemokine_mode="static", progress=True):
    """Sweep (alpha, k_rep). Returns a dict of (len_a, len_k) observable grids."""
    base = dict(base or DEFAULTS)
    if n_steps is not None:
        base["n_steps"] = n_steps
    if N is not None:
        base["N"] = N
    base["dynamic_chemokine"] = (chemokine_mode == "dynamic")
    base["v_drift"] = base.get("v_drift", 0.0)

    na, nk = len(alphas), len(k_reps)
    grids = {k: np.zeros((na, nk)) for k in
             ("persistence", "internal_polar_order", "density_at_source",
              "core_density", "late_polar_order")}
    mode = np.empty((na, nk), dtype=object)

    for i, a in enumerate(alphas):
        for j, kr in enumerate(k_reps):
            p = dict(base, alpha=float(a), k_rep=float(kr), record_field=False)
            traj = simulate(p, seed=seed)
            s = obs.summary(traj)
            for key in grids:
                grids[key][i, j] = s[key]
            mode[i, j] = s["dispersal_mode"]
            if progress:
                print(f"  alpha={a:.3f} k_rep={kr:.2f} -> "
                      f"persist={s['persistence']:.2f} dens={s['density_at_source']:.1f} "
                      f"{s['dispersal_mode']}")
    grids["dispersal_mode"] = mode
    grids["alphas"] = np.asarray(alphas)
    grids["k_reps"] = np.asarray(k_reps)
    return grids


def plot_phase_diagram(grids, field="persistence", path="phase_diagram.png"):
    """Heatmap of one observable over the (alpha, k_rep) grid."""
    a, k = grids["alphas"], grids["k_reps"]
    Z = grids[field]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="magma",
                   extent=[k.min(), k.max(), a.min(), a.max()])
    ax.set_xlabel(r"$k_{rep}$  (repulsion strength, mechanical)")
    ax.set_ylabel(r"$\alpha$  (desensitization rate, biochemical)")
    ax.set_title(f"Swarm self-limitation: {field}")
    fig.colorbar(im, ax=ax, label=field)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    # Informative sweep: wide alpha to drive dispersal, long enough runs.
    # persistence/retention responds to alpha (biochemical); core_density
    # responds to k_rep (mechanical anti-collapse). Together they map which
    # mechanism limits the swarm.
    alphas = np.linspace(0.0, 0.6, 6)
    k_reps = np.linspace(0.0, 12.0, 6)
    grids = run_sweep(alphas, k_reps, n_steps=1200, N=150)
    # Headline figures: core_density shows BOTH mechanisms quench the
    # Keller-Segel pile-up; density_at_source shows the biochemical (alpha)
    # axis controlling overall retention at the infection site.
    print("saved:", plot_phase_diagram(grids, "core_density",
                                        path="phase_diagram_core.png"))
    print("saved:", plot_phase_diagram(grids, "density_at_source",
                                        path="phase_diagram_density.png"))
