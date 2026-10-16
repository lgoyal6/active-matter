"""Validation against Wiebke 2025 (ITBA, adv. Parisi) — *not* a sanity/limit test.

sanity_checks.py verifies the engine behaves correctly (turn an ingredient off,
recover the simpler model). This file asks the harder question: does the engine
reproduce the *published findings* of the closest prior work?

The relevant model is the thesis's Chapter 6 (Part III): the minimal 2D
T cell-APC scanning problem — self-propelled particles with Poisson reorientation
and active radius oscillation, steric repulsion, periodic box, one fixed APC at
the centre, measuring cumulative unique contacts. That regime is:
    w_c=0, w_a=0 (no chemotaxis, no alignment), alpha=0, beta=0 (no
    desensitization), eta=0 (Poisson reorientation only via lambda_r).

Each check is tied to a specific thesis figure. Trends are asserted, not exact
numbers: these legacy checks use APC-off mode, where contacts are counted after
simulation and do not affect cell motion. The engine also differs from Wiebke's
in force form, and omega_deform is in rad/sim-time rather than oscillations/min.
Absolute optima can differ, so these checks assess qualitative trends only.
"""

import numpy as np

from simulate import simulate, DEFAULTS, _min_image
import observables as obs

# ---- Wiebke Chapter 6 regime: pure exploration + steric + deformation --------
WIEBKE = dict(
    DEFAULTS,
    dim=2,
    L=24.0,
    n_steps=600,
    v0=1.0,            # Wiebke v0 = 1
    w_c=0.0,           # no chemotaxis  (Ch.6 has no chemokine attraction)
    w_a=0.0,           # no Vicsek alignment
    alpha=0.0, beta=0.0,   # no desensitization
    eta=0.0,           # no continuous noise -> Poisson reorientation only
    k_rep=50.0,        # steric repulsion (Wiebke Frep = 50)
    r_excl=1.0,        # collision distance at rest (cell radius 0.5)
    deform_A=0.15,     # Wiebke A = 0.15
    omega_deform=2.0,  # moderate oscillation for the non-frequency checks
    lambda_r=0.2,      # default reorientation rate
)
SEEDS = (0, 1)
CONTACT_R = 1.5        # central "area of contact" radius (for statistics)


def rate(seeds=SEEDS, **overrides):
    """Mean contact rate over seeds for a Wiebke-regime run with overrides."""
    vals = []
    for s in seeds:
        traj = simulate(dict(WIEBKE, **overrides), seed=s)
        vals.append(obs.contact_rate(traj, contact_radius=CONTACT_R))
    return float(np.mean(vals))


def mean_speed_um_min(seeds=SEEDS, **overrides):
    """Median per-step speed converted to um/min under Wiebke's calibration
    (free speed v0 == 5 um/min when r0 = 5 um). Median (not mean) and a dilute
    system are used so the free-migration speed isn't swamped by stiff-repulsion
    velocity spikes at contacts."""
    sims = []
    for s in seeds:
        traj = simulate(dict(WIEBKE, **overrides), seed=s)
        pos, L, dt = traj["positions"], traj["box_size"], traj["params"]["dt"]
        disp = _min_image(pos[1:] - pos[:-1], L)         # periodic min image
        sims.append(np.median(np.sqrt((disp ** 2).sum(-1)) / dt))
    v0 = WIEBKE["v0"]
    return float(np.mean(sims)) * (5.0 / v0)             # v0 sim -> 5 um/min


# ---------------------------------------------------------------------------
def check_persistence():
    """Fig 6.3 / §6.4.1: lower persistence (higher lambda) -> FEWER contacts;
    highest contact rate at lambda = 0."""
    r0 = rate(lambda_r=0.0, rho=0.55)
    r1 = rate(lambda_r=1.0, rho=0.55)
    ok = r0 > r1
    print(f"[1] Persistence (Fig 6.3): rate(lambda=0)={r0:.3f} > rate(lambda=1)={r1:.3f}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_packing_fraction():
    """Fig 6.5 / §6.4.1: contact rate is non-monotonic in rho — peaks at an
    intermediate packing fraction.

    KNOWN GAP: this fails with the current engine and the failure is understood.
    Wiebke's high-density drop comes from the APC being a *physical obstacle*
    that gets caged by a static shell of cells (his Fig 6.1). Our engine has no
    physical APC; contacts are counted post-hoc against a freely-flowing central
    region, so higher density just streams more cells through -> monotonic rise.
    Closing this needs a fixed APC obstacle added to simulate().
    """
    lo = rate(lambda_r=0.1, rho=0.10)
    mid = rate(lambda_r=0.1, rho=0.50)
    hi = rate(lambda_r=0.1, rho=0.85)
    ok = mid > lo and mid > hi
    tag = "PASS" if ok else "FAIL (known gap: no physical APC obstacle -> no high-density caging)"
    print(f"[2] Packing fraction (Fig 6.5): rho 0.10={lo:.3f}  0.50={mid:.3f}  0.85={hi:.3f}  (mid highest) -> {tag}")
    return ok


def check_active_deformation():
    """Fig 6.5 / §6.4.1: active deformation yields more contacts than fixed
    radius (more pronounced at low lambda), in a crowded system."""
    on = rate(lambda_r=0.1, rho=0.70, deform_A=0.15)
    off = rate(lambda_r=0.1, rho=0.70, deform_A=0.0)
    ok = on > off
    print(f"[3] Active deformation (Fig 6.5): oscillating={on:.3f} > fixed-radius={off:.3f}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_optimal_frequency():
    """Fig 6.4 / §6.4.1: contact rate peaks at an INTERMEDIATE oscillation
    frequency (too slow = little fluidization; too fast = radius averages out)."""
    slow = rate(lambda_r=0.1, rho=0.70, omega_deform=0.0)
    mid = rate(lambda_r=0.1, rho=0.70, omega_deform=2.0)
    fast = rate(lambda_r=0.1, rho=0.70, omega_deform=8.0)
    ok = mid > slow and mid > fast
    print(f"[4] Optimal frequency (Fig 6.4): omega 0={slow:.3f}  2={mid:.3f}  8={fast:.3f}  (mid highest) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_speed_calibration():
    """§6.4.2: with r0 = 5 um the mean speed is ~5 um/min — the LOW end of the
    in-vivo 10-12 um/min range (§1.2). Assert a plausible few-um/min band."""
    v = mean_speed_um_min(lambda_r=0.2, rho=0.10)
    ok = 1.0 < v < 12.0
    print(f"[5] Speed calibration (§6.4.2): mean speed = {v:.1f} um/min (in-vivo 10-12; thesis ~5, low end) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_apc_obstacle_packing_fraction(raw_dir):
    """Explicit heavyweight APC study; caller supplies private raw-output path."""
    from pathlib import Path
    from evaluation.apc_study import build_protocol, run

    rhos, seeds, steps, dt = [0.2, 0.5, 0.71, 0.85], [11, 23, 37, 53], 500_000, 0.001
    laws = ["thesis_eq5_5_normalized", "zenodo_ch6_source_unnormalized"]
    rows = run(rhos, seeds, steps, laws, dt, Path(raw_dir),
               build_protocol(rhos, seeds, steps, laws, dt))
    by_law = {}
    for row in rows:
        by_law.setdefault(row["law"], {}).setdefault(row["requested_rho"], []).append(row["contact_rate"])
    for law, curve in by_law.items():
        means = {rho: float(np.mean(values)) for rho, values in curve.items()}
        print(f"APC density diagnostics ({law}): {means}; intermediate_peak={means[0.5] > means[0.2] and means[0.5] > means[0.85]}")
    return rows


def run_all():
    print("Validating engine against Wiebke 2025 thesis, Chapter 6 regime...\n")
    results = [
        check_persistence(),
        check_packing_fraction(),
        check_active_deformation(),
        check_optimal_frequency(),
        check_speed_calibration(),
    ]
    n = sum(results)
    print(f"\n{n}/{len(results)} thesis findings reproduced.")
    if not results[1]:
        print("Note: check [2] is a documented gap, not an engine bug — it needs a "
              "fixed APC obstacle in simulate() to reproduce high-density caging.")
    return n


if __name__ == "__main__":
    run_all()
