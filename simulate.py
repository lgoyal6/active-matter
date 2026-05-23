"""
Canonical T-cell swarming simulation (Track A, Tijil).
Grounded in Wiebke 2025 doctoral thesis (ITBA / Parisi).

Dimension-general self-propelled particles (ABP) in a periodic box, dim = 2 or 3:
  - Vicsek alignment             (w_a)
  - Chemotaxis toward a source   (w_c, gated by per-cell sensitivity)
  - Poisson reorientation        (lambda_r>0) OR continuous noise (eta>0)
and creative ingredients, each OFF at its default so the model
reduces to the base layer:
  - Desensitization              (alpha>0 turns on; beta = recovery)
  - Active radius oscillation    (deform_A>0) — Wiebke 2025: r_i(t)=r0+A·sin(ω·t+φ_i)
  - Excluded volume / repulsion  (k_rep>0 turns on)
  - Chemokine secretion field    (dynamic_chemokine=True turns on; 2D only)
  - Directional drift w/ memory  (v_drift>0 turns on)

Density can be set via packing fraction rho = N·π·r₀²/L² (Wiebke convention).
Poisson reorientation (lambda_r) replaces continuous noise for biological realism:
  each cell independently picks a new random direction at rate lambda_r events/sim-time.
  Persistence time = 1/lambda_r. Biologically: ~0.3–1.0 (1–3 min real time).

Headings are stored as UNIT VECTORS (N, dim) -- this is what makes the same
code run in 2D and 3D. Noise is applied by perturbing the heading vector and
renormalizing, i.e. rotational diffusion, which is the physically correct way
an ABP randomizes its direction (and the right generalization to 3D).

NumPy + (optional) Matplotlib only. Google-Colab friendly.
Periodic boundary conditions everywhere; gradients use the minimum image.

Hand-off contract -- simulate() returns:
{
    "positions":      (n_steps, N, dim)
    "headings":       (n_steps, N, dim)   unit heading vectors
    "orientations":   (n_steps, N)        2D heading angle; None when dim==3
    "sensitivity":    (n_steps, N)
    "chemokine_field":(grid,)*dim static, or (n_steps, grid, grid) dynamic (2D)
    "params":         dict                (includes "dim")
    "box_size":       float L
    "source_position":(x, y[, z])
}
"""

import numpy as np

# ---------------------------------------------------------------------------
# Default parameters (single source of truth). simulate() copies & overrides.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # --- system / integration ---
    dim=2,                   # 2 or 3 spatial dimensions
    N=200,
    L=50.0,
    dt=0.05,
    n_steps=2000,
    grid=64,                 # chemokine field resolution per axis (3D uses <=32)
    # --- physical-unit anchors (dimensionless model -> real biology) ---
    # The model is unitless; these two numbers map it to T-cell biology.
    #   length: r_excl=1 sim-length := one T-cell diameter ~ 10 um  -> L=50 is a 500 um field
    #   speed : v0=0.5 sim-len/sim-time matched to ~10 um/min crawling -> 1 sim-time = 0.5 min
    # Change ONLY these to re-scale every axis label; the physics is unchanged.
    um_per_length=10.0,      # microns per 1 simulation length unit
    min_per_time=0.5,        # minutes  per 1 simulation time unit
    # --- motility ---
    v0=0.5,
    eta=0.3,                 # rotational-diffusion noise (per step)
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
    alpha=0.0,               # desensitization rate        (0 -> sensitivity frozen)
    beta=0.01,               # sensitivity recovery rate
    k_rep=0.0,               # repulsion stiffness          (0 -> no excluded volume)
    r_excl=1.0,              # base cell radius * 2 (collision distance at rest)
    v_drift=0.0,             # drift magnitude              (0 -> no memory drift)
    gamma=0.05,              # drift memory rate
    # --- Wiebke 2025 upgrades ---
    # Packing fraction: rho = N·π·(r_excl/2)²/L²  (Wiebke convention).
    # If set, overrides N. Biologically realistic range: 0.2–0.85.
    rho=None,
    # Poisson reorientation (replaces or supplements eta noise).
    # Each cell picks a new random direction with prob 1-exp(-lambda_r·dt) per step.
    # lambda_r=0 → off (use eta); lambda_r=0.5 → ~1 min persistence (realistic).
    lambda_r=0.0,
    # Active radius oscillation: r_i(t) = r_excl/2 + deform_A·sin(omega_deform·t + φ_i)
    # Collision threshold = r_i + r_j. deform_A=0 → fixed radius (off).
    # Optimal omega_deform ≈ 0.5–1.0 rad/sim-time (~1–2 cycles/min).
    deform_A=0.0,
    omega_deform=0.5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _min_image(disp, L):
    """Wrap displacement components into [-L/2, L/2] for periodic distances."""
    return disp - L * np.round(disp / L)


def _normalize(v, eps=1e-12):
    """Row-wise normalize an (N, dim) array to unit vectors (safe near zero)."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > eps)


def _random_unit(n, dim, rng):
    """n random unit vectors in `dim` dimensions (uniform on the sphere/circle)."""
    v = rng.standard_normal((n, dim))
    return _normalize(v)


# ---------------------------------------------------------------------------
# Unit conversion: dimensionless model -> physical biology units.
# ---------------------------------------------------------------------------
def to_minutes(step, params=None):
    """Step index (or array of indices) -> elapsed minutes."""
    p = params or DEFAULTS
    return np.asarray(step) * p["dt"] * p["min_per_time"]


def to_microns(x, params=None):
    """Simulation length(s) -> microns."""
    p = params or DEFAULTS
    return np.asarray(x) * p["um_per_length"]


# ---------------------------------------------------------------------------
# Chemokine: static analytic Gaussian (any dimension)
# ---------------------------------------------------------------------------
def _chemokine_static(pos, source, sigma, L):
    """
    Static Gaussian centred on `source`, valid in 2D or 3D.
      c    : (N,)     concentration in [0, 1]
      grad : (N,dim)  gradient vector (points toward the source)
    """
    disp = _min_image(source[None, :] - pos, L)        # (N,dim) toward source
    r2 = np.sum(disp**2, axis=1)
    c = np.exp(-r2 / (2.0 * sigma**2))
    grad = (c / sigma**2)[:, None] * disp
    return c, grad


def _static_field_grid(dim, grid, L, source, sigma):
    """Render the static Gaussian on a (grid,)*dim lattice for plotting/analysis."""
    xs = (np.arange(grid) + 0.5) * (L / grid)
    mesh = np.meshgrid(*([xs] * dim), indexing="ij")
    r2 = np.zeros_like(mesh[0])
    for ax, m in enumerate(mesh):
        r2 = r2 + _min_image(m - source[ax], L) ** 2
    return np.exp(-r2 / (2.0 * sigma**2))


# ---- dynamic secretion field (2D only) ----
def _density_grid(pos, grid, L):
    h = L / grid
    ix = np.mod((pos[:, 0] / h).astype(int), grid)
    iy = np.mod((pos[:, 1] / h).astype(int), grid)
    rho = np.zeros((grid, grid))
    np.add.at(rho, (iy, ix), 1.0)
    return rho


def _sample_field(field, pos, L):
    grid = field.shape[0]
    h = L / grid
    ix = np.mod((pos[:, 0] / h).astype(int), grid)
    iy = np.mod((pos[:, 1] / h).astype(int), grid)
    c = field[iy, ix]
    gx = (field[iy, np.mod(ix + 1, grid)] - field[iy, np.mod(ix - 1, grid)]) / (2 * h)
    gy = (field[np.mod(iy + 1, grid), ix] - field[np.mod(iy - 1, grid), ix]) / (2 * h)
    return c, np.stack([gx, gy], axis=-1)


def _diffuse(field, rho, D, lam, S, dt, h):
    lap = (
        np.roll(field, 1, 0) + np.roll(field, -1, 0)
        + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        - 4.0 * field
    ) / (h * h)
    return field + dt * (D * lap - lam * field + S * rho)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def simulate(params=None, seed=0):
    """Run the simulation. `params` overrides DEFAULTS. Returns the contract dict."""
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)

    dim = int(p["dim"])
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    # Packing fraction → N override (Wiebke 2025 convention: rho = N·π·(r/2)²/L²)
    if p["rho"] is not None:
        r_cell = p["r_excl"] / 2.0
        p["N"] = max(1, int(round(p["rho"] * p["L"]**2 / (np.pi * r_cell**2))))
    N, L, dt, n_steps = p["N"], p["L"], p["dt"], p["n_steps"]
    source = np.full(dim, L / 2.0)                    # source at box centre

    # secretion field PDE is only implemented in 2D; fall back to static in 3D
    dynamic = bool(p["dynamic_chemokine"]) and dim == 2
    if p["dynamic_chemokine"] and dim == 3:
        print("[simulate] dynamic_chemokine not supported in 3D -> using static field")
    grid = p["grid"] if dim == 2 else min(p["grid"], 32)
    h = L / grid

    # --- initial state ---
    pos  = rng.uniform(0, L, size=(N, dim))           # uniform random positions
    head = _random_unit(N, dim, rng)                  # random unit heading vectors
    s    = np.ones(N)                                 # sensitivity starts at 1
    d_vec = head.copy()                               # drift-memory direction
    # Per-cell random phase for active deformation — desynchronizes oscillations
    # so they don't all expand/contract at once, which would cause waves.
    phi_deform = rng.uniform(0, 2 * np.pi, N)
    if dynamic:
        field = np.zeros((grid, grid))

    # --- trajectory buffers ---
    pos_hist = np.empty((n_steps, N, dim))
    head_hist = np.empty((n_steps, N, dim))
    sen_hist = np.empty((n_steps, N))
    if dynamic:
        fld_hist = np.empty((n_steps, grid, grid))

    # =====================================================================
    # Time integration. Each step: (1) read the chemokine field, (2) decide
    # how every cell wants to turn, (3) move, (4) update internal sensitivity.
    # Vectorized over the N cells; works identically for dim = 2 or 3.
    # =====================================================================
    for t in range(n_steps):
        # ---- (1) Chemokine concentration c and gradient grad at each cell ----
        if dynamic:
            rho = _density_grid(pos, grid, L)
            field = _diffuse(field, rho, p["D_chem"], p["lambda_chem"],
                             p["S_chem"], dt, h)
            c, grad = _sample_field(field, pos, L)
        else:
            c, grad = _chemokine_static(pos, source, p["sigma"], L)

        # ---- (2a) Pairwise geometry: who is near whom (periodic min image) ----
        diff = _min_image(pos[:, None, :] - pos[None, :, :], L)   # (N,N,dim): i - j
        dist = np.sqrt(np.sum(diff**2, axis=2))                   # (N,N)
        np.fill_diagonal(dist, np.inf)                            # exclude self

        # ---- (2b) Vicsek alignment: sum neighbor heading vectors within R ----
        neigh = dist < p["R"]                                     # (N,N) bool
        align_vec = neigh.astype(float) @ head                    # (N,dim)

        # ---- (2c) Combine steering urges and set the new heading ----
        #   persist  : keep current heading (ABP self-propulsion)
        #   w_a      : turn toward neighbors' mean heading (flocking)
        #   w_c * s  : climb the chemokine gradient, GATED by sensitivity s
        grad_hat = _normalize(grad)
        combined = (
            p["persist"] * head
            + p["w_a"] * align_vec
            + p["w_c"] * s[:, None] * grad_hat
        )
        head = _normalize(combined)

        # ---- (2d) Reorientation: Poisson events (Wiebke 2025) and/or continuous noise ----
        # Poisson: each cell independently picks a uniformly random new direction at
        # rate lambda_r. Probability of reorienting in one step = 1 - exp(-lambda_r·dt).
        # This is the biologically correct model — cells tumble occasionally, not constantly.
        if p["lambda_r"] > 0.0:
            tumble = rng.uniform(size=N) < (1.0 - np.exp(-p["lambda_r"] * dt))
            if tumble.any():
                head[tumble] = _random_unit(int(tumble.sum()), dim, rng)
        # Continuous rotational-diffusion noise (keeps backward compat; can layer on top).
        if p["eta"] > 0.0:
            head = _normalize(head + p["eta"] * rng.standard_normal((N, dim)))

        # ---- (2e) Excluded-volume repulsion with active radius oscillation ----
        # Active deformation (Wiebke 2025): each cell's radius oscillates sinusoidally
        # with a random per-cell phase phi_deform. This "fluidizes" the crowd —
        # cells periodically shrink, creating gaps that let others squeeze through.
        # Collision threshold between cells i and j = r_i + r_j.
        if p["k_rep"] > 0.0:
            t_now = t * dt                                         # current sim time
            if p["deform_A"] > 0.0:
                r_cell = (p["r_excl"] / 2.0
                          + p["deform_A"] * np.sin(p["omega_deform"] * t_now + phi_deform))
                r_cell = np.maximum(r_cell, 0.01)                 # never negative radius
            else:
                r_cell = np.full(N, p["r_excl"] / 2.0)           # fixed radius
            # pairwise collision threshold: r_i + r_j  (N,N)
            threshold = r_cell[:, None] + r_cell[None, :]
            close = dist < threshold
            overlap = np.where(close, threshold - dist, 0.0)
            dir_unit = np.divide(diff, dist[:, :, None],
                                 out=np.zeros_like(diff),
                                 where=np.isfinite(dist)[:, :, None])
            f_rep = p["k_rep"] * np.sum(overlap[:, :, None] * dir_unit, axis=1)
        else:
            f_rep = 0.0

        # ---- (2f) Directional drift with memory (off when v_drift == 0) ----
        if p["v_drift"] > 0.0:
            d_vec = (1 - p["gamma"]) * d_vec + p["gamma"] * head
            d_vec = _normalize(d_vec + 0.01 * rng.standard_normal((N, dim)))
            drift = p["v_drift"] * d_vec
        else:
            drift = 0.0

        # ---- (3) Integrate position (overdamped) and wrap periodically ----
        pos = pos + dt * (p["v0"] * head + drift + f_rep)
        pos = np.mod(pos, L)

        # ---- (4) Desensitization ODE: ds/dt = -alpha*c*s + beta*(1-s) ----
        if p["alpha"] > 0.0 or p["beta"] > 0.0:
            s = s + dt * (-p["alpha"] * c * s + p["beta"] * (1.0 - s))
            np.clip(s, 0.0, 1.0, out=s)

        # ---- record this frame ----
        pos_hist[t] = pos
        head_hist[t] = head
        sen_hist[t] = s
        if dynamic:
            fld_hist[t] = field

    # static field rendered once for analysis/plots
    chem_out = fld_hist if dynamic else _static_field_grid(
        dim, grid, L, source, p["sigma"])

    # 2D heading angle for backward-compatible analysis; None in 3D
    ori_out = (np.arctan2(head_hist[:, :, 1], head_hist[:, :, 0])
               if dim == 2 else None)

    return {
        "positions": pos_hist,
        "headings": head_hist,
        "orientations": ori_out,
        "sensitivity": sen_hist,
        "chemokine_field": chem_out,
        "params": p,
        "box_size": L,
        "source_position": tuple(float(x) for x in source),
    }


if __name__ == "__main__":
    for dim in (2, 3):
        out = simulate({"dim": dim, "n_steps": 300}, seed=1)
        src = np.array(out["source_position"]); L = out["box_size"]
        d0 = np.mean(np.linalg.norm(_min_image(out["positions"][0] - src, L), axis=1))
        d1 = np.mean(np.linalg.norm(_min_image(out["positions"][-1] - src, L), axis=1))
        print(f"dim={dim}: positions {out['positions'].shape}  "
              f"field {np.asarray(out['chemokine_field']).shape}  "
              f"mean dist {d0:.2f} -> {d1:.2f} "
              f"({'aggregating' if d1 < d0 else 'NOT'})")
