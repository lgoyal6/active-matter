"""
(alpha, k_rep) phase diagram sweep — the primary scientific deliverable.

Question: what drives T-cell swarm self-limitation — biochemical desensitization
(alpha) or mechanical repulsion (k_rep)?

For every (alpha, k_rep) point on an 8×8 grid we run one simulation and extract:
  swarm_fraction  : fraction of cells near the source at steady state (↓ = self-limited)
  rebound_height  : peak re-dispersal distance after aggregation (↑ = stronger self-limit)
  mean_sensitivity: how desensitized cells are at the end

Then plot a 2D heatmap of swarm_fraction with region labels + 4 example
trajectory curves (one per quadrant) for biological interpretation.

Run:  python phase_diagram.py
Output: phase_diagram.png  (slide-ready figure)
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from simulate import simulate, DEFAULTS, _min_image, to_minutes, to_microns

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
N_ALPHA  = 8
N_KREP   = 8
ALPHAS   = np.linspace(0.0, 0.70, N_ALPHA)   # biochemical axis
K_REPS   = np.linspace(0.0, 22.0, N_KREP)    # mechanical axis

# Simulation settings for each sweep point (fast but biologically adequate)
SWEEP_PARAMS = dict(
    N=150,         # fewer particles → faster per run
    n_steps=1200,  # 60 min biological time
    eta=0.3,
    w_a=0.5,
    w_c=1.0,
    beta=0.01,
    r_excl=1.0,
    v_drift=0.0,
)
R_SWARM  = 10.0    # sim units; cells within this of source count as "in swarm" (=100 µm)
TAIL_FRAC = 0.40   # average observables over the last 40% of the run (steady state)

# Four representative parameter points (one per quadrant of the diagram) for example curves
EXAMPLES = [
    ("persistent swarm\n(low α, low k)",      0.05, 0.0,  "#4e9af1"),
    ("biochemical dispersal\n(high α, low k)", 0.60, 0.0,  "#e63946"),
    ("mechanical dispersal\n(low α, high k)",  0.05, 20.0, "#f4a261"),
    ("combined\n(high α, high k)",             0.60, 20.0, "#2a9d8f"),
]


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------
def _observables(out):
    """Extract swarm_fraction, rebound_height, mean_sensitivity from a trajectory."""
    p   = out["params"]
    src = np.array(out["source_position"])
    L   = out["box_size"]
    T   = out["positions"].shape[0]
    tail_start = int(T * (1 - TAIL_FRAC))

    # distance of every cell from the source at every step
    disp = _min_image(out["positions"] - src, L)         # (T, N, dim)
    dist = np.linalg.norm(disp, axis=2)                  # (T, N)

    # fraction of cells within R_SWARM of source (0=all dispersed, 1=all at source)
    in_swarm     = dist < R_SWARM                        # (T, N) bool
    swarm_frac   = in_swarm[tail_start:].mean()          # scalar

    # mean distance to source per step -> rebound height (self-limiting signature)
    mean_dist    = dist.mean(axis=1)                     # (T,)
    # find the minimum (aggregation peak) then measure how much it bounces back
    t_min        = int(np.argmin(mean_dist[:T//2 + 1]))
    post_min     = mean_dist[t_min:]
    rebound      = post_min.max() - post_min[0] if len(post_min) > 1 else 0.0

    # mean sensitivity at end
    mean_sens    = out["sensitivity"][tail_start:].mean()

    return float(swarm_frac), float(rebound), float(mean_sens)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep():
    n_total = N_ALPHA * N_KREP
    swarm_map   = np.zeros((N_KREP, N_ALPHA))
    rebound_map = np.zeros((N_KREP, N_ALPHA))
    sens_map    = np.zeros((N_KREP, N_ALPHA))

    print(f"Running {N_ALPHA}×{N_KREP} = {n_total} simulations …")
    t0 = time.time()
    for j, kr in enumerate(K_REPS):
        for i, al in enumerate(ALPHAS):
            out = simulate(dict(SWEEP_PARAMS, alpha=al, k_rep=kr), seed=0)
            sf, rb, ms = _observables(out)
            swarm_map[j, i]   = sf
            rebound_map[j, i] = rb
            sens_map[j, i]    = ms
            done = j * N_ALPHA + i + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (n_total - done)
            print(f"  [{done:2d}/{n_total}]  α={al:.2f}  k_rep={kr:.1f}  "
                  f"swarm={sf:.2f}  rebound={rb:.1f}  sens={ms:.2f}  "
                  f"ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Sweep done in {elapsed:.1f}s")
    return swarm_map, rebound_map, sens_map


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(swarm_map, rebound_map, sens_map):
    p = dict(SWEEP_PARAMS)
    src = np.array([p.get("L", DEFAULTS["L"]) / 2] * 2)
    L   = DEFAULTS["L"]

    # custom diverging cmap: red (persistent) -> white -> teal (self-limited)
    cmap_phase = LinearSegmentedColormap.from_list(
        "swarm", ["#2a9d8f", "#f4f1ec", "#e63946"], N=256)
    cmap_phase = cmap_phase.reversed()   # red=persists, teal=self-limited

    fig = plt.figure(figsize=(14, 10), facecolor="#f4f1ec")
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.10,
                            hspace=0.45, wspace=0.40)

    # ---- main heatmap (spans 2 cols) ----
    ax_main = fig.add_subplot(gs[0, :2])
    im = ax_main.imshow(
        swarm_map, origin="lower", aspect="auto",
        extent=[ALPHAS[0], ALPHAS[-1], K_REPS[0], K_REPS[-1]],
        cmap=cmap_phase, vmin=0, vmax=1, interpolation="bicubic")
    cb = fig.colorbar(im, ax=ax_main, fraction=0.045, pad=0.02)
    cb.set_label("steady-state swarm fraction\n(1=all cells at source, 0=dispersed)",
                 fontsize=8)
    cb.ax.tick_params(labelsize=8)

    # contour at 0.35: "half self-limited" boundary
    ax_main.contour(
        np.linspace(ALPHAS[0], ALPHAS[-1], swarm_map.shape[1]),
        np.linspace(K_REPS[0],  K_REPS[-1],  swarm_map.shape[0]),
        swarm_map, levels=[0.35], colors="white", linewidths=1.5, linestyles="--")
    ax_main.text(0.42, 10, "boundary", color="white", fontsize=8, rotation=-20)

    # regime labels
    ax_main.text(0.04, 1.5, "PERSISTENT\nSWARM", color="white",
                 fontsize=8.5, weight="bold", alpha=0.9)
    ax_main.text(0.40, 1.5, "BIOCHEMICAL\nDISPERSAL", color="white",
                 fontsize=8.5, weight="bold", alpha=0.9)
    ax_main.text(0.04, 16,  "MECHANICAL\nDISPERSAL", color="white",
                 fontsize=8.5, weight="bold", alpha=0.9)
    ax_main.text(0.40, 16,  "COMBINED\nSELF-LIMIT", color="white",
                 fontsize=8.5, weight="bold", alpha=0.9)

    # mark example points
    for label, al, kr, col in EXAMPLES:
        ax_main.plot(al, kr, "o", ms=9, mfc=col, mec="white", mew=1.5, zorder=5)

    ax_main.set_xlabel("desensitization rate  α  (biochemical)", fontsize=10)
    ax_main.set_ylabel("repulsion strength  k_rep  (mechanical)", fontsize=10)
    ax_main.set_title("Phase diagram: what self-limits the T-cell swarm?",
                      fontsize=11, weight="bold", pad=8)

    # ---- rebound height heatmap ----
    ax_rb = fig.add_subplot(gs[0, 2])
    im2 = ax_rb.imshow(
        rebound_map, origin="lower", aspect="auto",
        extent=[ALPHAS[0], ALPHAS[-1], K_REPS[0], K_REPS[-1]],
        cmap="viridis", interpolation="bicubic")
    fig.colorbar(im2, ax=ax_rb, fraction=0.06, pad=0.02).set_label("rebound (µm)", fontsize=7)
    ax_rb.set_xlabel("α", fontsize=9); ax_rb.set_ylabel("k_rep", fontsize=9)
    ax_rb.set_title("dispersal rebound height", fontsize=9)
    ax_rb.tick_params(labelsize=7)

    # ---- mean sensitivity heatmap ----
    ax_s = fig.add_subplot(gs[0, 3])
    im3 = ax_s.imshow(
        sens_map, origin="lower", aspect="auto",
        extent=[ALPHAS[0], ALPHAS[-1], K_REPS[0], K_REPS[-1]],
        cmap="RdYlGn", vmin=0, vmax=1, interpolation="bicubic")
    fig.colorbar(im3, ax=ax_s, fraction=0.06, pad=0.02).set_label("mean sensitivity", fontsize=7)
    ax_s.set_xlabel("α", fontsize=9); ax_s.set_ylabel("k_rep", fontsize=9)
    ax_s.set_title("mean receptor sensitivity", fontsize=9)
    ax_s.tick_params(labelsize=7)

    # ---- 4 example aggregation curves ----
    for idx, (label, al, kr, col) in enumerate(EXAMPLES):
        ax_ex = fig.add_subplot(gs[1, idx])
        out = simulate(dict(SWEEP_PARAMS, alpha=al, k_rep=kr, n_steps=1400), seed=0)
        src_v = np.array(out["source_position"])
        disp  = _min_image(out["positions"] - src_v, out["box_size"])
        mean_d_um = to_microns(np.linalg.norm(disp, axis=2).mean(axis=1), out["params"])
        mean_s    = out["sensitivity"].mean(axis=1)
        t_min_arr = to_minutes(np.arange(out["positions"].shape[0]), out["params"])

        ax_ex.fill_between(t_min_arr, mean_d_um, color=col, alpha=0.15)
        ax_ex.plot(t_min_arr, mean_d_um, color=col, lw=1.8, label="dist to source")
        ax2 = ax_ex.twinx()
        ax2.plot(t_min_arr, mean_s, color=col, lw=1.2, linestyle="--", alpha=0.7)
        ax2.set_ylim(0, 1.1); ax2.set_ylabel("sensitivity", fontsize=7, color=col)
        ax2.tick_params(labelsize=6, colors=col)
        ax_ex.set_xlabel("time (min)", fontsize=8)
        ax_ex.set_ylabel("dist to source (µm)", fontsize=8)
        ax_ex.set_title(label, fontsize=8.5, weight="bold", color=col)
        ax_ex.tick_params(labelsize=7)

    fig.text(0.5, 0.965,
             "T-cell Swarm Self-Limitation: Biochemical vs Mechanical Mechanisms",
             ha="center", fontsize=13, weight="bold", color="#2b2b2b")
    fig.text(0.5, 0.945,
             "Red star ★ = chemokine source   ·   Dashed line = sensitivity   ·   "
             "Contour = 35% swarm fraction boundary",
             ha="center", fontsize=9, color="#666")

    fname = "phase_diagram.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    print(f"saved {fname}")
    return fname


# ---------------------------------------------------------------------------
# Second phase diagram: (alpha, omega_deform) — Wiebke 2025 inspired
# Biologically realistic Poisson reorientation on; sweep biochemistry × active deformation.
# ---------------------------------------------------------------------------
N_OMEGA = 8
OMEGAS  = np.linspace(0.0, 2.0, N_OMEGA)    # deformation frequency axis

WIEBKE_BASE = dict(
    SWEEP_PARAMS,
    k_rep=8.0,          # repulsion on so deformation effects are visible
    deform_A=0.3,       # moderate amplitude (0 = fixed, stays on throughout)
    eta=0.0,            # pure Poisson reorientation
    lambda_r=0.5,       # ~1 min persistence time — biologically calibrated
)


def run_sweep_wiebke():
    n_total = N_ALPHA * N_OMEGA
    swarm_map   = np.zeros((N_OMEGA, N_ALPHA))
    rebound_map = np.zeros((N_OMEGA, N_ALPHA))

    print(f"\nRunning Wiebke-inspired sweep {N_ALPHA}×{N_OMEGA} = {n_total} sims …")
    t0 = time.time()
    for j, om in enumerate(OMEGAS):
        for i, al in enumerate(ALPHAS):
            out = simulate(dict(WIEBKE_BASE, alpha=al, omega_deform=om), seed=0)
            sf, rb, _ = _observables(out)
            swarm_map[j, i]   = sf
            rebound_map[j, i] = rb
            done = j * N_ALPHA + i + 1
            elapsed = time.time() - t0
            eta_t = elapsed / done * (n_total - done)
            print(f"  [{done:2d}/{n_total}]  α={al:.2f}  ω={om:.2f}  "
                  f"swarm={sf:.2f}  rebound={rb:.1f}  ETA {eta_t:.0f}s")

    print(f"Wiebke sweep done in {time.time()-t0:.1f}s")
    return swarm_map, rebound_map


def make_wiebke_figure(swarm_map, rebound_map):
    cmap_phase = LinearSegmentedColormap.from_list(
        "swarm", ["#2a9d8f", "#f4f1ec", "#e63946"], N=256).reversed()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#f4f1ec")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.16, wspace=0.38)

    for ax, data, title, cmap, vmin, vmax, cblabel in [
        (axes[0], swarm_map, "Steady-state swarm fraction",
         cmap_phase, 0, 1, "swarm fraction (1=persists, 0=self-limited)"),
        (axes[1], rebound_map, "Dispersal rebound height",
         "viridis", None, None, "rebound distance"),
    ]:
        im = ax.imshow(data, origin="lower", aspect="auto",
                       extent=[ALPHAS[0], ALPHAS[-1], OMEGAS[0], OMEGAS[-1]],
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cblabel, fontsize=8); cb.ax.tick_params(labelsize=8)
        ax.set_xlabel("desensitization rate  α  (biochemical)", fontsize=10)
        ax.set_ylabel("deformation frequency  ω  (active mechanics, Wiebke 2025)", fontsize=9)
        ax.set_title(title, fontsize=10, weight="bold")

    # mark optimal omega from Wiebke (≈1.0 rad/sim-time ≈ 2 cycles/min)
    for ax in axes:
        ax.axhline(1.0, color="white", lw=1.4, ls="--", alpha=0.8)
        ax.text(0.02, 1.05, "Wiebke optimal ω ≈ 1.0", color="white",
                fontsize=8, alpha=0.9)

    fig.suptitle(
        "T-cell Swarm: Biochemical Desensitization × Active Radius Oscillation\n"
        "(Poisson reorientation  λ=0.5,  deformation A=0.3,  k_rep=8 — Wiebke 2025 framework)",
        fontsize=11, weight="bold", color="#2b2b2b")

    fname = "phase_diagram_wiebke.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    print(f"saved {fname}")
    return fname


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Original (alpha, k_rep) diagram
    swarm_map, rebound_map, sens_map = run_sweep()
    make_figure(swarm_map, rebound_map, sens_map)

    # Wiebke-inspired (alpha, omega_deform) diagram
    w_swarm, w_rebound = run_sweep_wiebke()
    make_wiebke_figure(w_swarm, w_rebound)
