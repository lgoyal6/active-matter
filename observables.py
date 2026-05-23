"""Lightweight observables for Track A's phase diagram.

This is the minimal set the simulation track needs to label the (alpha, k_rep)
sweep. Harshatha owns the full ``swarm_analysis.py`` library (vorticity,
cluster stats, residence time, dispersal-mode classifier, ...). Everything here
takes the ``simulate()`` hand-off dict so it stays compatible with hers.
"""

import numpy as np


def _dist_to_source(traj):
    """(n_steps, N) minimum-image distance of every cell to the source."""
    src = np.array(traj["source_position"])
    L = traj["box_size"]
    d = traj["positions"] - src[None, None, :]
    d = d - L * np.round(d / L)
    return np.sqrt((d * d).sum(axis=-1))


def unique_contacts(traj, contact_radius=None):
    """Cumulative count of *distinct* cells that have touched the APC vs. time.

    Wiebke 2025 Chapter 6's primary observable. The APC is the fixed particle at
    the box centre (== source_position here). A cell "contacts" it the first time
    its centre comes within ``contact_radius`` (default = r_excl, i.e. one cell
    radius + one APC radius at rest). Each cell is counted at most once.

    Note: our engine has no physical APC obstacle, so this counts entries into the
    central contact region from the trajectory — a post-hoc approximation of
    Wiebke's setup that preserves the qualitative trends (the APC is one obstacle
    among hundreds). Returns a (n_steps,) cumulative-count array.
    """
    if contact_radius is None:
        contact_radius = traj["params"].get("r_excl", 1.0)
    d = _dist_to_source(traj)                       # (n_steps, N)
    inside = d < contact_radius                      # (n_steps, N)
    ever = inside.any(axis=0)                        # (N,) cells that ever touch
    first = inside.argmax(axis=0)                    # (N,) first contact step
    n_steps = d.shape[0]
    new_per_step = np.bincount(first[ever], minlength=n_steps)
    return np.cumsum(new_per_step)


def contact_rate(traj, contact_radius=None, transient_frac=0.05):
    """Slope of cumulative unique contacts vs. time (contacts per sim-time unit).

    Fit is taken after a short transient (Wiebke starts at t=25 min) so the
    initial relaxation from the packed lattice doesn't bias the rate.
    """
    counts = unique_contacts(traj, contact_radius)
    n = len(counts)
    start = max(1, int(n * transient_frac))
    t = np.arange(start, n) * traj["params"]["dt"]
    if len(t) < 2:
        return 0.0
    slope = np.polyfit(t, counts[start:], 1)[0]
    return float(slope)


def swarm_size(traj, R_swarm=None):
    """Fraction of cells within R_swarm of the source, per timestep.

    Default R_swarm is the source width sigma -- the scale of the chemokine
    blob, i.e. the actual swarm core, not the whole basin.
    """
    if R_swarm is None:
        R_swarm = traj["params"]["sigma"]
    return (_dist_to_source(traj) < R_swarm).mean(axis=1)


def core_density(traj, r_core=2.0):
    """Mean cell count inside a tight core radius over the last 20% of the run.

    Sensitive to Keller-Segel pile-up: with no repulsion, cells stack into a
    point and this blows up; with repulsion they spread into a finite blob.
    """
    d = _dist_to_source(traj)
    cnt = (d < r_core).sum(axis=1)
    tail = max(1, len(cnt) // 5)
    return float(cnt[-tail:].mean())


def polar_order(traj, within_swarm=True, R_swarm=None):
    """Polar order |<e^{i theta}>|, optionally restricted to the swarm cluster."""
    th = traj["orientations"]
    vx, vy = np.cos(th), np.sin(th)
    if within_swarm:
        if R_swarm is None:
            R_swarm = traj["params"]["sigma"]
        mask = _dist_to_source(traj) < R_swarm
        cnt = mask.sum(axis=1)
        sx = (vx * mask).sum(axis=1)
        sy = (vy * mask).sum(axis=1)
        return np.where(cnt > 0, np.sqrt(sx * sx + sy * sy) / np.maximum(cnt, 1), 0.0)
    return np.sqrt(vx.mean(axis=1) ** 2 + vy.mean(axis=1) ** 2)


def density_at_source(traj, R_swarm=None):
    """Mean cell count inside R_swarm over the last 20% of the run."""
    s = swarm_size(traj, R_swarm) * traj["params"]["N"]
    tail = max(1, len(s) // 5)
    return float(s[-tail:].mean())


def persistence(traj, R_swarm=None):
    """Swarm persistence: final swarm size relative to its peak (1 = no dispersal)."""
    s = swarm_size(traj, R_swarm)
    peak = s.max()
    if peak <= 0:
        return 0.0
    tail = max(1, len(s) // 10)
    return float(s[-tail:].mean() / peak)


def dispersal_mode(traj):
    """Classify late-stage dispersal: streaming (ordered) vs. diffuse.

    Returns (label, late_polar_order). High polar order during dispersal means
    cells leave coherently (streaming); low means they scatter (diffuse).
    """
    po = polar_order(traj, within_swarm=False)
    tail = max(1, len(po) // 5)
    late = float(po[-tail:].mean())
    return ("streaming" if late > 0.5 else "diffuse"), late


def summary(traj):
    """One-line dict of headline observables for a run."""
    label, late_po = dispersal_mode(traj)
    return {
        "persistence": persistence(traj),
        "internal_polar_order": float(polar_order(traj)[-len(traj["orientations"]) // 5:].mean()),
        "density_at_source": density_at_source(traj),
        "core_density": core_density(traj),
        "dispersal_mode": label,
        "late_polar_order": late_po,
    }
