"""Sanity-check protocol: a new ingredient is correct only if turning it off
recovers the previous model. Run before any phase-diagram sweep."""

import numpy as np
from simulate import simulate, _min_image


def _dists(out):
    src = np.array(out["source_position"]); L = out["box_size"]
    p = out["positions"]
    d0 = np.mean(np.linalg.norm(_min_image(p[0] - src, L), axis=1))
    d1 = np.mean(np.linalg.norm(_min_image(p[-1] - src, L), axis=1))
    return d0, d1


def _polar_order(out):
    th = out["orientations"][-1]
    return float(np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th))))


def _median_nn(out):
    """Median nearest-neighbor distance -- rises toward r_excl under repulsion."""
    p = out["positions"][-1]; L = out["box_size"]
    d = _min_image(p[:, None, :] - p[None, :, :], L)
    dist = np.sqrt(np.sum(d**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(1)))


def main():
    NS = 800
    print("LIMIT 1  w_c=0 (pure Vicsek) -> no aggregation; order high at low noise")
    o = simulate({"n_steps": NS, "w_c": 0.0, "eta": 0.1}, seed=2)
    d0, d1 = _dists(o)
    print(f"   dist {d0:.2f} -> {d1:.2f}   polar order = {_polar_order(o):.2f}")

    print("LIMIT 1b w_c=0, eta=1.2 (high noise) -> disordered, low order")
    o = simulate({"n_steps": NS, "w_c": 0.0, "eta": 1.2}, seed=2)
    print(f"   polar order = {_polar_order(o):.2f}")

    print("LIMIT 2  w_a=0 (chemotaxis-ABP) -> swarm forms at source")
    o = simulate({"n_steps": NS, "w_a": 0.0}, seed=2)
    d0, d1 = _dists(o)
    print(f"   dist {d0:.2f} -> {d1:.2f}   ({'aggregates' if d1 < d0 else 'NO'})")

    print("LIMIT 3  desensitization: alpha>0 should re-disperse vs alpha=0")
    e0 = _dists(simulate({"n_steps": 1500, "alpha": 0.0}, seed=2))[1]
    ea = _dists(simulate({"n_steps": 1500, "alpha": 0.6}, seed=2))[1]
    print(f"   end-dist  alpha=0: {e0:.2f}   alpha=0.6: {ea:.2f}   "
          f"({'disperses' if ea > e0 else 'no effect'})")

    print("LIMIT 4  excluded volume: k_rep>0 lifts nearest-neighbor spacing -> r_excl")
    s0 = _median_nn(simulate({"n_steps": NS, "k_rep": 0.0}, seed=2))
    sk = _median_nn(simulate({"n_steps": NS, "k_rep": 20.0, "r_excl": 1.0}, seed=2))
    print(f"   median NN dist  k_rep=0: {s0:.2f}   k_rep=20: {sk:.2f}  (r_excl=1.0)   "
          f"({'no stacking' if sk > s0 else 'no effect'})")


if __name__ == "__main__":
    main()
