"""
Canonical T-cell swarming simulation (Track A, Tijil).

2D self-propelled particles (ABP) in a periodic box with:
  - Vicsek alignment             (w_a)
  - Chemotaxis toward a source   (w_c, gated by per-cell sensitivity)
  - Angular noise                (eta)
and four creative ingredients, each OFF at its default so the model
reduces to the base layer:
  - Desensitization              (alpha>0 turns on; beta = recovery)
  - Excluded volume / repulsion  (k_rep>0 turns on)
  - Chemokine secretion field    (dynamic_chemokine=True turns on)
  - Directional drift w/ memory  (v_drift>0 turns on)

NumPy + (optional) Matplotlib only. Google-Colab friendly.
Periodic boundary conditions everywhere; gradients use the minimum image.

Hand-off contract -- simulate() returns:
{
    "positions":      (n_steps, N, 2)
    "orientations":   (n_steps, N)
    "sensitivity":    (n_steps, N)
    "chemokine_field":(grid, grid)  static, or (n_steps, grid, grid) dynamic
    "params":         dict
    "box_size":       float L
    "source_position":(x, y)
}
"""

import numpy as np

# ---------------------------------------------------------------------------
# Default parameters (single source of truth). simulate() copies & overrides.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # --- system / integration ---
    N=200,
    L=50.0,
    dt=0.05,
    n_steps=2000,
    grid=64,                 # chemokine field resolution (grid x grid)
    # --- physical-unit anchors (dimensionless model -> real biology) ---
    # The model is unitless; these two numbers map it to T-cell biology.
    #   length: r_excl=1 sim-length := one T-cell diameter ~ 10 um  -> L=50 is a 500 um field
    #   speed : v0=0.5 sim-len/sim-time matched to ~10 um/min crawling -> 1 sim-time = 0.5 min
    # Change ONLY these to re-scale every axis label; the physics is unchanged.
    um_per_length=10.0,      # microns per 1 simulation length unit
    min_per_time=0.5,        # minutes  per 1 simulation time unit
    # --- motility ---
    v0=0.5,
    eta=0.3,                 # angular noise (rad, Gaussian std per step)
    persist=1.0,             # weight of a cell's own heading (ABP persistence)
    # --- base interactions ---
    R=2.0,                   # alignment + interaction radius
    w_a=0.5,                 # alignment weight
    w_c=1.0,                 # chemotaxis weight
    # --- chemokine source (static Gaussian by default) ---
    sigma=8.0,               # Gaussian width of the source field
    dynamic_chemokine=False,
    D_chem=0.2,
    lambda_chem=0.02,
    S_chem=0.1,
    # --- creative ingredients (all inert at these defaults) ---
    alpha=0.0,               # desensitization rate   (0 -> sensitivity frozen)
    beta=0.01,               # sensitivity recovery
    k_rep=0.0,               # repulsion stiffness     (0 -> no excluded volume)
    r_excl=1.0,
    v_drift=0.0,             # drift magnitude         (0 -> no memory drift)
    gamma=0.05,              # drift memory rate
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _min_image(disp, L):
    """Wrap displacement components into [-L/2, L/2] for periodic distances."""
    return disp - L * np.round(disp / L)


# ---------------------------------------------------------------------------
# Unit conversion: dimensionless model -> physical biology units.
# These read the anchors from a params dict (defaults if absent), so any
# trajectory's `out["params"]` is enough to label its plots in real units.
# ---------------------------------------------------------------------------
def to_minutes(step, params=None):
    """Step index (or array of indices) -> elapsed minutes."""
    p = params or DEFAULTS
    return np.asarray(step) * p["dt"] * p["min_per_time"]


def to_microns(x, params=None):
    """Simulation length(s) -> microns."""
    p = params or DEFAULTS
    return np.asarray(x) * p["um_per_length"]


def _angles_to_unit(theta):
    """(N,) angles -> (N,2) unit vectors."""
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


# ---------------------------------------------------------------------------
# Chemokine: static analytic Gaussian
# ---------------------------------------------------------------------------
def _chemokine_static(pos, source, sigma, L):
    """
    Returns (c, grad) for a static Gaussian centred on `source`.
      c    : (N,)   concentration in [0, 1]
      grad : (N,2)  gradient vector (points toward the source)
    """
    disp = _min_image(source[None, :] - pos, L)        # (N,2) toward source
    r2 = np.sum(disp**2, axis=1)
    c = np.exp(-r2 / (2.0 * sigma**2))
    grad = (c / sigma**2)[:, None] * disp              # d/dx exp(...) -> toward source
    return c, grad


def _sample_field(field, pos, L):
    """Nearest-cell sample of concentration + central-difference gradient
    from a dynamic grid `field` (grid,grid). Returns (c, grad)."""
    grid = field.shape[0]
    h = L / grid
    ix = np.mod((pos[:, 0] / h).astype(int), grid)
    iy = np.mod((pos[:, 1] / h).astype(int), grid)
    c = field[iy, ix]
    gx = (field[iy, np.mod(ix + 1, grid)] - field[iy, np.mod(ix - 1, grid)]) / (2 * h)
    gy = (field[np.mod(iy + 1, grid), ix] - field[np.mod(iy - 1, grid), ix]) / (2 * h)
    return c, np.stack([gx, gy], axis=-1)


def _diffuse(field, rho, D, lam, S, dt, h):
    """One explicit step of  dc/dt = D*lap(c) - lam*c + S*rho  (periodic)."""
    lap = (
        np.roll(field, 1, 0) + np.roll(field, -1, 0)
        + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        - 4.0 * field
    ) / (h * h)
    return field + dt * (D * lap - lam * field + S * rho)


def _density_grid(pos, grid, L):
    """Cloud-in-cell-free nearest-grid histogram of cell density (grid,grid)."""
    h = L / grid
    ix = np.mod((pos[:, 0] / h).astype(int), grid)
    iy = np.mod((pos[:, 1] / h).astype(int), grid)
    rho = np.zeros((grid, grid))
    np.add.at(rho, (iy, ix), 1.0)
    return rho


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def simulate(params=None, seed=0):
    """
    Run the simulation. `params` overrides DEFAULTS. Returns the contract dict.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)

    N, L, dt, n_steps = p["N"], p["L"], p["dt"], p["n_steps"]
    grid = p["grid"]
    source = np.array([L / 2.0, L / 2.0])

    # --- initial state: uniform random positions, random headings, fully sensitive ---
    pos = rng.uniform(0, L, size=(N, 2))
    theta = rng.uniform(-np.pi, np.pi, size=N)
    s = np.ones(N)                                    # sensitivity starts at 1
    d_vec = _angles_to_unit(theta).copy()             # drift memory direction

    dynamic = p["dynamic_chemokine"]
    h = L / grid
    if dynamic:
        field = np.zeros((grid, grid))                # secretion builds it up

    # --- trajectory buffers ---
    pos_hist = np.empty((n_steps, N, 2))
    ori_hist = np.empty((n_steps, N))
    sen_hist = np.empty((n_steps, N))
    if dynamic:
        fld_hist = np.empty((n_steps, grid, grid))

    # =====================================================================
    # Time integration. Each step: (1) read the chemokine field, (2) decide
    # how every cell wants to turn, (3) move, (4) update internal sensitivity.
    # Everything is vectorized over the N cells -- no per-particle Python loop.
    # =====================================================================
    for t in range(n_steps):
        # ---- (1) Chemokine concentration c and gradient grad at each cell ----
        # `c` (N,)   = how much chemokine cell i currently sits in   -> drives desensitization
        # `grad`(N,2)= local uphill direction of the field           -> drives chemotaxis
        if dynamic:
            # Secretion mode: the field is a real PDE the cells build up themselves.
            rho = _density_grid(pos, grid, L)                     # where the cells are
            field = _diffuse(field, rho, p["D_chem"], p["lambda_chem"],
                             p["S_chem"], dt, h)                  # one diffusion+decay+source step
            c, grad = _sample_field(field, pos, L)                # interpolate field -> cells
        else:
            # Default: fixed Gaussian bump at the source, computed analytically.
            c, grad = _chemokine_static(pos, source, p["sigma"], L)

        # ---- (2a) Pairwise geometry: who is near whom (with periodic wrap) ----
        # diff[i,j] is the vector from cell j to cell i under the minimum image
        # convention, so neighbors across a wall count. dist is its length.
        diff = _min_image(pos[:, None, :] - pos[None, :, :], L)   # (N,N,2): i - j
        dist = np.sqrt(np.sum(diff**2, axis=2))                   # (N,N) distances
        np.fill_diagonal(dist, np.inf)                            # a cell is not its own neighbor

        # ---- (2b) Vicsek alignment: copy the average heading of nearby cells ----
        # Sum the unit heading vectors of all neighbors within radius R. Summing
        # vectors (not averaging angles) is the correct, wrap-around-safe way to
        # average directions. Self is excluded (diagonal is inf -> never < R).
        neigh = dist < p["R"]                                     # (N,N) bool neighbor mask
        head = _angles_to_unit(theta)                             # (N,2) current headings
        align_vec = neigh.astype(float) @ head                    # (N,2) sum of neighbor headings

        # ---- (2c) Pick the new heading by weighting the three steering urges ----
        # We add three direction vectors and take the angle of the result:
        #   persist  : keep going the way I'm already facing (ABP self-propulsion)
        #   w_a      : turn toward my neighbors' average heading (flocking)
        #   w_c * s  : turn up the chemokine gradient, GATED by my sensitivity s
        #              (a desensitized cell, s~0, ignores the gradient entirely)
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_hat = np.divide(grad, grad_norm, out=np.zeros_like(grad),
                             where=grad_norm > 1e-12)              # unit gradient (safe at c~0)
        combined = (
            p["persist"] * head
            + p["w_a"] * align_vec
            + p["w_c"] * s[:, None] * grad_hat
        )
        theta = np.arctan2(combined[:, 1], combined[:, 0])        # angle of the summed urge
        theta += p["eta"] * rng.standard_normal(N)                # (2d) angular noise (Brownian wobble)
        head = _angles_to_unit(theta)                             # refreshed heading after the turn

        # ---- (2e) Excluded-volume repulsion -- soft push-apart (off if k_rep==0) ----
        # For every overlapping pair (closer than r_excl) add a Hookean spring
        # force F = k_rep*(r_excl - dist) pointing from j to i. Summing over j
        # gives each cell's net shove. This is what stops the swarm collapsing to
        # a single point (the Keller-Segel pathology) and sets the packing spacing.
        if p["k_rep"] > 0.0:
            close = (dist < p["r_excl"])                          # (N,N) overlapping pairs
            overlap = np.where(close, p["r_excl"] - dist, 0.0)    # penetration depth (>=0)
            dir_unit = np.divide(diff, dist[:, :, None],          # unit vector j -> i
                                 out=np.zeros_like(diff),
                                 where=np.isfinite(dist)[:, :, None])
            f_rep = p["k_rep"] * np.sum(overlap[:, :, None] * dir_unit, axis=1)  # (N,2) net force
        else:
            f_rep = 0.0

        # ---- (2f) Directional drift with memory -- coherent streaming (off if v_drift==0) ----
        # Each cell keeps a low-pass-filtered "preferred direction" d_vec that
        # leans toward its recent motion. Adding v_drift*d_vec makes dispersing
        # cells stream together instead of scattering randomly.
        if p["v_drift"] > 0.0:
            d_vec = (1 - p["gamma"]) * d_vec + p["gamma"] * head  # blend old memory with new motion
            d_vec += 0.01 * rng.standard_normal((N, 2))           # small reorientation noise
            d_norm = np.linalg.norm(d_vec, axis=1, keepdims=True)
            d_vec = np.divide(d_vec, d_norm, out=d_vec, where=d_norm > 1e-12)  # keep it a unit vector
            drift = p["v_drift"] * d_vec
        else:
            drift = 0.0

        # ---- (3) Integrate position (overdamped: velocity = sum of urges) and wrap ----
        # Overdamped/Brownian-dynamics update: no inertia, displacement = velocity*dt.
        pos = pos + dt * (p["v0"] * head + drift + f_rep)
        pos = np.mod(pos, L)                                      # periodic box

        # ---- (4) Desensitization ODE: ds/dt = -alpha*c*s + beta*(1-s) ----
        # First term: lose receptors in proportion to chemokine exposure (alpha*c).
        # Second term: slowly recover toward s=1 when chemokine is low (beta).
        # This is the biochemical "off switch" that lets the swarm self-terminate.
        if p["alpha"] > 0.0 or p["beta"] > 0.0:
            s = s + dt * (-p["alpha"] * c * s + p["beta"] * (1.0 - s))
            np.clip(s, 0.0, 1.0, out=s)                           # keep sensitivity in [0,1]

        # ---- record this frame into the trajectory buffers ----
        pos_hist[t] = pos
        ori_hist[t] = theta
        sen_hist[t] = s
        if dynamic:
            fld_hist[t] = field

    # static field rendered once for analysis/plots
    if dynamic:
        chem_out = fld_hist
    else:
        xs = (np.arange(grid) + 0.5) * h
        gx, gy = np.meshgrid(xs, xs, indexing="xy")
        dx = _min_image(gx - source[0], L)
        dy = _min_image(gy - source[1], L)
        chem_out = np.exp(-(dx**2 + dy**2) / (2.0 * p["sigma"]**2))

    return {
        "positions": pos_hist,
        "orientations": ori_hist,
        "sensitivity": sen_hist,
        "chemokine_field": chem_out,
        "params": p,
        "box_size": L,
        "source_position": (float(source[0]), float(source[1])),
    }


if __name__ == "__main__":
    # smoke test: base layer only
    out = simulate({"n_steps": 200}, seed=1)
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"{k:16s} {v.shape} {v.dtype}")
        else:
            print(f"{k:16s} {v}")
    # quick swarm check: mean distance to source should drop over the run
    src = np.array(out["source_position"])
    L = out["box_size"]
    d0 = np.mean(np.linalg.norm(_min_image(out["positions"][0] - src, L), axis=1))
    d1 = np.mean(np.linalg.norm(_min_image(out["positions"][-1] - src, L), axis=1))
    print(f"\nmean dist to source: start={d0:.2f}  end={d1:.2f}  "
          f"({'aggregating' if d1 < d0 else 'NOT aggregating'})")
