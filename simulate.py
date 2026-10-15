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
    n_sources=1,             # number of chemokine source sites (1, 2, 3, or 4)
    source_distance=15.0,    # source spacing; overlap if comparable to sigma
    source_strength=1.0,     # source emission/field amplitude
    source_1_strength=1.0,   # per-source amplitudes; active sources use first n values
    source_2_strength=1.0,
    source_3_strength=1.0,
    source_4_strength=1.0,
    dynamic_chemokine=False,
    D_chem=0.2,
    lambda_chem=0.02,
    S_chem=0.1,
    # --- creative ingredients (all inert at these defaults) ---
    alpha=0.0,               # desensitization rate        (0 -> sensitivity frozen)
    beta=0.05,               # sensitivity recovery rate
    recovery_K=0.5,          # recovery quiet-zone scale; high total chemokine suppresses beta
    w_escape=1.5,            # desensitized cells bias away from sources to find recovery space
    alpha_1=None,            # optional per-source desensitization rates
    alpha_2=None,
    alpha_3=None,
    alpha_4=None,
    beta_1=None,             # optional per-source recovery rates
    beta_2=None,
    beta_3=None,
    beta_4=None,
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
    # Thesis-style APC is opt-in so the legacy default remains inert.
    apc_mode="off",
    apc_force_law="thesis_eq5_5_normalized",
    apc_r0=1.0,
    apc_record_trajectory=True,
    apc_neighbor_skin=0.2,
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


def _source_positions(dim, L, n_sources, source_distance):
    """Deterministic centered source layouts controlled by one spacing knob."""
    n = max(1, min(int(round(n_sources)), 4))
    center = np.full(dim, L / 2.0)
    d = float(np.clip(source_distance, 0.0, 0.80 * L))
    offsets = np.zeros((n, dim), dtype=float)

    if n == 2:
        offsets[:, 0] = [-d / 2.0, d / 2.0]
    elif n == 3:
        radius = d / np.sqrt(3.0) if d > 0.0 else 0.0
        angles = np.deg2rad([90.0, 210.0, 330.0])
        offsets[:, 0] = radius * np.cos(angles)
        offsets[:, 1] = radius * np.sin(angles)
    elif n == 4:
        half = d / 2.0
        offsets[:, :2] = [(-half, -half), (half, -half), (-half, half), (half, half)]

    return np.mod(center[None, :] + offsets, L)


APC_FORCE_LAWS = ("thesis_eq5_5_normalized", "zenodo_ch6_source_unnormalized")


def _apc_hcp_positions(n_mobile, L, r0):
    """Place a center APC and mobile cells on a nonoverlapping periodic HCP grid."""
    n_total = n_mobile + 1
    candidates = []
    for nx in range(2, int(L / (2 * r0)) + 1):
        dx = L / nx
        for ny in range(4, int(L / (np.sqrt(3) * r0)) + 1, 4):
            dy = L / ny
            min_sep = min(dx, 2 * dy, np.hypot(dx / 2, dy))
            if min_sep + 1e-12 < 2 * r0 or nx * ny < n_total:
                continue
            aspect_error = abs(dy / (np.sqrt(3) * dx / 2) - 1.0)
            candidates.append((nx * ny - n_total, aspect_error, nx, ny))
    if not candidates:
        raise ValueError("requested APC density does not fit a nonoverlapping periodic HCP lattice")
    _, _, nx, ny = min(candidates)
    dx, dy = L / nx, L / ny
    center = np.array([L / 2, L / 2], dtype=float)
    points = np.asarray([
        [col * dx + (row % 2) * dx / 2, row * dy]
        for row in range(ny) for col in range(nx)
    ])
    center_idx = (ny // 2) * nx + nx // 2
    points = np.mod(points + np.mod(center - points[center_idx], L), L)
    available = np.asarray([i for i in range(len(points)) if i != center_idx])
    selection = np.rint(np.linspace(0, len(available) - 1, n_mobile)).astype(int)
    mobile_indices = available[selection]
    mobile = points[mobile_indices]
    all_points = np.concatenate([center[None, :], mobile], axis=0)
    pair_diff = _min_image(all_points[:, None, :] - all_points[None, :, :], L)
    pair_dist = np.linalg.norm(pair_diff, axis=2)
    np.fill_diagonal(pair_dist, np.inf)
    min_distance = float(np.min(pair_dist))
    if min_distance + 1e-10 < 2 * r0:
        raise RuntimeError("periodic HCP initialization contains an overlap")
    return mobile, (nx, ny), min_distance


def _apc_cell_radii(t, phi, r0, amplitude, omega):
    """Begin radius oscillations at each phase's next mean-radius crossing."""
    if amplitude <= 0.0 or omega <= 0.0:
        return np.full(len(phi), r0)
    start_time = (np.ceil(phi / np.pi) * np.pi - phi) / omega
    active = t >= start_time
    radii = np.full(len(phi), r0, dtype=float)
    radii[active] += amplitude * np.sin(omega * t + phi[active])
    return np.maximum(radii, 0.01)


def _apc_pair_force(displacement, radius_i, radius_j, strength, law):
    """Linear overlap repulsion with an explicit thesis/source-code convention."""
    distance = float(np.linalg.norm(displacement))
    threshold = float(radius_i + radius_j)
    if distance <= 1e-12 or distance >= threshold or strength <= 0.0:
        return np.zeros_like(displacement, dtype=float)
    overlap = threshold - distance
    scale = overlap / threshold if law == "thesis_eq5_5_normalized" else overlap
    return float(strength) * scale * np.asarray(displacement, dtype=float) / distance


def _apc_neighbor_pairs(pos, L, cutoff):
    """Periodic cell-list candidate pairs, including every pair within cutoff."""
    bins_per_axis = max(1, int(np.floor(L / cutoff)))
    width = L / bins_per_axis
    buckets = {}
    indices = np.floor(np.mod(pos, L) / width).astype(int) % bins_per_axis
    for i, cell in enumerate(indices):
        buckets.setdefault((int(cell[0]), int(cell[1])), []).append(i)
    pairs = set()
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
               (0, 1), (1, -1), (1, 0), (1, 1))
    for (bx, by), left in buckets.items():
        for ox, oy in offsets:
            right = buckets.get(((bx + ox) % bins_per_axis,
                                 (by + oy) % bins_per_axis), ())
            for i in left:
                for j in right:
                    if i < j:
                        pairs.add((i, j))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def _apc_cell_forces(pos, radii, center, apc_radius, strength, law, L, pairs=None):
    """Production sparse pair-force path; coincident centers get zero direction."""
    force = np.zeros_like(pos, dtype=float)
    if pairs is None:
        i, j = np.triu_indices(len(pos), 1)
    else:
        i, j = pairs[:, 0], pairs[:, 1]
    pair_disp = _min_image(pos[i] - pos[j], L)
    distance = np.linalg.norm(pair_disp, axis=1)
    threshold = radii[i] + radii[j]
    overlap = np.maximum(threshold - distance, 0.0)
    direction = np.divide(pair_disp, distance[:, None], out=np.zeros_like(pair_disp),
                          where=(distance > 1e-12)[:, None])
    scale = overlap / threshold if law == "thesis_eq5_5_normalized" else overlap
    pair_force = float(strength) * scale[:, None] * direction
    np.add.at(force, i, pair_force)
    np.add.at(force, j, -pair_force)
    target_displacement = _min_image(pos - center, L)
    target_distance = np.linalg.norm(target_displacement, axis=1)
    target_threshold = radii + float(apc_radius)
    target_overlap = np.maximum(target_threshold - target_distance, 0.0)
    target_direction = np.divide(target_displacement, target_distance[:, None],
                                 out=np.zeros_like(target_displacement),
                                 where=(target_distance > 1e-12)[:, None])
    target_scale = (target_overlap / target_threshold
                    if law == "thesis_eq5_5_normalized" else target_overlap)
    force += float(strength) * target_scale[:, None] * target_direction
    return force


def _source_strengths(params, n_sources):
    """Per-source amplitudes, with source_strength kept as a global fallback."""
    base = float(params.get("source_strength", 1.0))
    vals = [
        params.get("source_1_strength", base),
        params.get("source_2_strength", base),
        params.get("source_3_strength", base),
        params.get("source_4_strength", base),
    ]
    strengths = np.array(vals[:n_sources], dtype=float)
    return np.clip(strengths, 0.0, None)


def _source_rates(params, base_name, n_sources):
    """Per-source alpha/beta rates, falling back to the global slider value."""
    base = float(params[base_name])
    vals = [params.get(f"{base_name}_{i}", None) for i in range(1, 5)]
    rates = np.array([base if v is None else v for v in vals[:n_sources]], dtype=float)
    return np.clip(rates, 0.0, None)


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
def _chemokine_static(pos, sources, sigma, L, strengths):
    """
    Static Gaussian sources, valid in 2D or 3D.
      c_each    : (N,S)     concentration from each source
      grad_each : (N,S,dim) gradient contribution from each source
    """
    disp = _min_image(sources[:, None, :] - pos[None, :, :], L)  # (S,N,dim)
    r2 = np.sum(disp**2, axis=2)
    c_each = strengths[:, None] * np.exp(-r2 / (2.0 * sigma**2))
    grad_each = (c_each / sigma**2)[:, :, None] * disp
    return c_each.T, np.transpose(grad_each, (1, 0, 2))


def _static_field_grid(dim, grid, L, sources, sigma, strengths):
    """Render the static Gaussian on a (grid,)*dim lattice for plotting/analysis."""
    xs = (np.arange(grid) + 0.5) * (L / grid)
    if dim == 2:
        yy, xx = np.meshgrid(xs, xs, indexing="ij")
        field = np.zeros((grid, grid), dtype=float)
        for source, strength in zip(sources, strengths):
            r2 = (_min_image(xx - source[0], L) ** 2
                  + _min_image(yy - source[1], L) ** 2)
            field = field + strength * np.exp(-r2 / (2.0 * sigma**2))
        return field

    mesh = np.meshgrid(*([xs] * dim), indexing="ij")
    field = np.zeros_like(mesh[0], dtype=float)
    for source, strength in zip(sources, strengths):
        r2 = np.zeros_like(mesh[0], dtype=float)
        for ax, m in enumerate(mesh):
            r2 = r2 + _min_image(m - source[ax], L) ** 2
        field = field + strength * np.exp(-r2 / (2.0 * sigma**2))
    return field


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


def _sample_fields(fields, pos, L):
    """Sample a stack of source-specific 2D fields: returns (N,S), (N,S,2)."""
    samples = [_sample_field(field, pos, L) for field in fields]
    c_each = np.stack([s[0] for s in samples], axis=1)
    grad_each = np.stack([s[1] for s in samples], axis=1)
    return c_each, grad_each


def _diffuse(field, rho, D, lam, S, source_field, dt, h):
    lap = (
        np.roll(field, 1, -2) + np.roll(field, -1, -2)
        + np.roll(field, 1, -1) + np.roll(field, -1, -1)
        - 4.0 * field
    ) / (h * h)
    source_count = field.shape[0] if field.ndim == 3 else 1
    secretion = ((S / source_count) * rho[None, :, :]
                 if field.ndim == 3 else S * rho)
    return field + dt * (D * lap - lam * field + secretion + source_field)


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

    apc_mode = p["apc_mode"]
    if apc_mode not in ("off", "thesis"):
        raise ValueError("apc_mode must be 'off' or 'thesis'")
    apc_enabled = apc_mode == "thesis"
    if apc_enabled:
        if dim != 2:
            raise ValueError("thesis APC mode is defined only for dim=2")
        if p["apc_force_law"] not in APC_FORCE_LAWS:
            raise ValueError(f"apc_force_law must be one of {APC_FORCE_LAWS}")
        if p["apc_r0"] <= 0.0 or p["deform_A"] < 0.0:
            raise ValueError("apc_r0 must be positive and deform_A nonnegative")
        if p["deform_A"] >= p["apc_r0"]:
            raise ValueError("thesis APC mode requires deform_A < apc_r0 to keep radii positive")
        if p["apc_neighbor_skin"] < 0.0:
            raise ValueError("apc_neighbor_skin must be nonnegative")
        if p["rho"] is not None and not 0.0 < float(p["rho"]) < 0.9069:
            raise ValueError("thesis APC rho must be between 0 and hexagonal close packing")

    # The thesis density counts the fixed APC among total HCP slots; legacy rho is untouched.
    if apc_enabled and p["rho"] is not None:
        total_slots = max(2, int(round(p["rho"] * p["L"]**2 / (np.pi * p["apc_r0"]**2))))
        p["N"] = total_slots - 1
    elif p["rho"] is not None:
        r_cell = p["r_excl"] / 2.0
        p["N"] = max(1, int(round(p["rho"] * p["L"]**2 / (np.pi * r_cell**2))))
    N, L, dt, n_steps = p["N"], p["L"], p["dt"], p["n_steps"]
    if apc_enabled:
        apc_center = np.full(2, L / 2.0)
        pos, apc_lattice_shape, apc_initial_min_distance = _apc_hcp_positions(
            N, L, float(p["apc_r0"]))
        p["N_mobile"] = int(N)
        p["apc_total_slots"] = int(N + 1)
        p["apc_achieved_rho"] = float((N + 1) * np.pi * p["apc_r0"]**2 / L**2)
        p["apc_requested_rho"] = None if p["rho"] is None else float(p["rho"])
        p["apc_lattice_shape"] = list(apc_lattice_shape)
        p["apc_initial_min_center_distance"] = apc_initial_min_distance
    else:
        apc_center = None
    sources = _source_positions(dim, L, p["n_sources"], p["source_distance"])
    strengths = _source_strengths(p, len(sources))
    alpha_src = _source_rates(p, "alpha", len(sources))
    beta_src = _source_rates(p, "beta", len(sources))
    source = sources[0]                               # compatibility anchor

    # secretion field PDE is only implemented in 2D; fall back to static in 3D
    dynamic = bool(p["dynamic_chemokine"]) and dim == 2
    if p["dynamic_chemokine"] and dim == 3:
        print("[simulate] dynamic_chemokine not supported in 3D -> using static field")
    grid = p["grid"] if dim == 2 else min(p["grid"], 32)
    h = L / grid

    # --- initial state ---
    if not apc_enabled:
        pos = rng.uniform(0, L, size=(N, dim))        # preserve legacy RNG draw order
    head = _random_unit(N, dim, rng)                  # random unit heading vectors
    s_src = np.ones((N, len(sources)))                # sensitivity per source starts at 1
    s    = s_src.mean(axis=1)                         # scalar summary for plotting
    d_vec = head.copy()                               # drift-memory direction
    # Per-cell random phase for active deformation — desynchronizes oscillations
    # so they don't all expand/contract at once, which would cause waves.
    phi_deform = rng.uniform(0, 2 * np.pi, N)
    if apc_enabled:
        apc_ids = np.arange(1, N + 1, dtype=np.int64)
        next_apc_id = N + 1
        record_apc_trajectory = bool(p["apc_record_trajectory"])
        radius_hist = np.empty((n_steps, N), dtype=np.float32) if record_apc_trajectory else None
        id_hist = np.empty((n_steps, N), dtype=np.int64) if record_apc_trajectory else None
        contact_event_hist = np.empty((n_steps, N), dtype=bool) if record_apc_trajectory else None
        contact_id_hist = np.empty((n_steps, N), dtype=np.int64) if record_apc_trajectory else None
        contact_curve = np.empty(n_steps, dtype=np.int32)
        encountered_apc_ids = set()
        contact_count = 0
        all_finite = True
        all_unit_headings = True
        all_in_box = True
    if dynamic:
        field = np.zeros((len(sources), grid, grid))
        source_field = np.stack([
            _static_field_grid(dim, grid, L, sources[i:i + 1], p["sigma"], strengths[i:i + 1])
            for i in range(len(sources))
        ])
    if apc_enabled and p["k_rep"] > 0.0:
        apc_cutoff = 2.0 * (float(p["apc_r0"]) + float(p["deform_A"]))
        neighbor_reference = pos.copy()
        neighbor_pairs = _apc_neighbor_pairs(pos, L, apc_cutoff + float(p["apc_neighbor_skin"]))

    # --- trajectory buffers ---
    store_frames = not apc_enabled or record_apc_trajectory
    pos_hist = np.empty((n_steps, N, dim)) if store_frames else None
    head_hist = np.empty((n_steps, N, dim)) if store_frames else None
    sen_hist = np.empty((n_steps, N)) if store_frames else None
    src_sen_hist = np.empty((n_steps, N, len(sources))) if store_frames else None
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
                             p["S_chem"], source_field, dt, h)
            c_each, grad_each = _sample_fields(field, pos, L)
        else:
            c_each, grad_each = _chemokine_static(
                pos, sources, p["sigma"], L, strengths)
        c = c_each.sum(axis=1)

        # ---- (2a) Pairwise geometry: who is near whom (periodic min image) ----
        # Preserve the legacy dense path exactly. APC studies set w_a=0, so no
        # alignment geometry is needed in that opt-in case.
        if apc_enabled and p["w_a"] == 0.0:
            align_vec = np.zeros_like(head)
        else:
            diff = _min_image(pos[:, None, :] - pos[None, :, :], L)   # (N,N,dim): i - j
            dist = np.sqrt(np.sum(diff**2, axis=2))                   # (N,N)
            np.fill_diagonal(dist, np.inf)                            # exclude self

            # ---- (2b) Vicsek alignment: sum neighbor heading vectors within R ----
            neigh = dist < p["R"]                                     # (N,N) bool
            align_vec = neigh.astype(float) @ head                    # (N,dim)

        # ---- (2c) Combine steering urges and set the new heading ----
        #   persist  : keep current heading (ABP self-propulsion)
        #   w_a      : turn toward neighbors' mean heading (flocking)
        #   w_c * s_src : climb each source's gradient, gated by that source's
        #                 receptor state. Cells can desensitize near one source
        #                 while staying responsive to another.
        chem_vec = (s_src[:, :, None] * grad_each).sum(axis=1)
        grad_hat = _normalize(chem_vec)
        chem_gate = np.divide(
            (c_each * s_src).sum(axis=1),
            c_each.sum(axis=1),
            out=np.zeros(N),
            where=c_each.sum(axis=1) > 1e-12,
        )
        recovery_K = max(float(p["recovery_K"]), 1e-12)
        quiet = np.exp(-c / recovery_K)
        source_disp = _min_image(pos[:, None, :] - sources[None, :, :], L)
        source_dist = np.linalg.norm(source_disp, axis=2, keepdims=True)
        away_unit = np.divide(
            source_disp, source_dist,
            out=np.zeros_like(source_disp),
            where=source_dist > 1e-12,
        )
        escape_weight = c_each * (1.0 - s_src)
        escape_hat = _normalize((escape_weight[:, :, None] * away_unit).sum(axis=1))
        escape_gate = np.divide(
            escape_weight.sum(axis=1),
            c_each.sum(axis=1),
            out=np.zeros(N),
            where=c_each.sum(axis=1) > 1e-12,
        ) * (1.0 - quiet)
        escape_vec = p["w_escape"] * escape_gate[:, None] * escape_hat
        escape_drift = p["v0"] * escape_vec

        combined = (
            p["persist"] * head
            + p["w_a"] * align_vec
            + p["w_c"] * chem_gate[:, None] * grad_hat
            + escape_vec
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
        if apc_enabled:
            t_now = t * dt
            r_cell = _apc_cell_radii(
                t_now, phi_deform, float(p["apc_r0"]),
                float(p["deform_A"]), float(p["omega_deform"]))
            target_distance = np.linalg.norm(_min_image(pos - apc_center, L), axis=1)
            contact_now = target_distance < r_cell + float(p["apc_r0"])
            if record_apc_trajectory:
                contact_event_hist[t] = contact_now
                contact_id_hist[t] = apc_ids
            for cell in np.flatnonzero(contact_now):
                identity = int(apc_ids[cell])
                if identity not in encountered_apc_ids:
                    encountered_apc_ids.add(identity)
                    contact_count += 1
            contact_curve[t] = contact_count
            if p["k_rep"] > 0.0:
                displacement_from_reference = _min_image(pos - neighbor_reference, L)
                if (p["apc_neighbor_skin"] == 0.0 or
                        np.max(np.linalg.norm(displacement_from_reference, axis=1)) > p["apc_neighbor_skin"] / 2.0):
                    neighbor_reference = pos.copy()
                    neighbor_pairs = _apc_neighbor_pairs(
                        pos, L, apc_cutoff + float(p["apc_neighbor_skin"]))
                f_rep = _apc_cell_forces(
                    pos, r_cell, apc_center, float(p["apc_r0"]), float(p["k_rep"]),
                    p["apc_force_law"], L, neighbor_pairs)
            else:
                f_rep = np.zeros_like(pos)
            if record_apc_trajectory:
                radius_hist[t] = _apc_cell_radii(
                    t_now + dt, phi_deform, float(p["apc_r0"]),
                    float(p["deform_A"]), float(p["omega_deform"]))
        elif p["k_rep"] > 0.0:
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
        pos_unwrapped = pos + dt * (p["v0"] * head + drift + escape_drift + f_rep)
        if apc_enabled:
            wrapped = np.any((pos_unwrapped < 0.0) | (pos_unwrapped >= L), axis=1)
            for cell in np.flatnonzero(wrapped):
                apc_ids[cell] = next_apc_id
                next_apc_id += 1
        pos = np.mod(pos_unwrapped, L)

        # ---- (4) Source-aware receptor state ----
        # ds_k/dt = -alpha_k*c_k*s_k + beta_k*quiet(c_total)*(1-s_k)
        # A cell can go temporarily blind to the source it has been bathing in,
        # while high total chemokine suppresses recovery by erasing quiet zones.
        if np.any(alpha_src > 0.0) or np.any(beta_src > 0.0):
            s_src = s_src + dt * (
                -alpha_src[None, :] * c_each * s_src
                + beta_src[None, :] * quiet[:, None] * (1.0 - s_src))
            np.clip(s_src, 0.0, 1.0, out=s_src)
        s = s_src.mean(axis=1)

        # ---- record this frame ----
        if store_frames:
            pos_hist[t] = pos
            head_hist[t] = head
            sen_hist[t] = s
            src_sen_hist[t] = s_src
        if apc_enabled:
            if record_apc_trajectory:
                id_hist[t] = apc_ids
            all_finite = all_finite and bool(np.isfinite(pos).all() and np.isfinite(head).all())
            all_unit_headings = all_unit_headings and bool(np.allclose(np.linalg.norm(head, axis=1), 1.0, atol=1e-8))
            all_in_box = all_in_box and bool((pos >= 0.0).all() and (pos < L).all())
        if dynamic:
            fld_hist[t] = field.sum(axis=0)

    # static field rendered once for analysis/plots
    chem_out = fld_hist if dynamic else _static_field_grid(
        dim, grid, L, sources, p["sigma"], strengths)

    # 2D heading angle for backward-compatible analysis; None in 3D
    ori_out = (np.arctan2(head_hist[:, :, 1], head_hist[:, :, 0])
               if dim == 2 and store_frames else None)

    result = {
        "positions": pos_hist,
        "headings": head_hist,
        "orientations": ori_out,
        "sensitivity": sen_hist,
        "source_sensitivity": src_sen_hist,
        "chemokine_field": chem_out,
        "params": p,
        "box_size": L,
        "source_position": tuple(float(x) for x in source),
        "source_positions": [tuple(float(x) for x in src) for src in sources],
        "source_strengths": [float(x) for x in strengths],
        "source_alpha": [float(x) for x in alpha_src],
        "source_beta": [float(x) for x in beta_src],
    }
    if apc_enabled:
        result.update({
            "apc_center": tuple(float(x) for x in apc_center),
            "apc_radius": float(p["apc_r0"]),
            # Contact checks occur before movement; stored state frames are after it.
            "contact_event_time_offset": 0.0,
            "position_frame_time_offset": float(dt),
        })
        if record_apc_trajectory:
            result.update({"cell_radii": radius_hist, "trajectory_ids": id_hist,
                           "contact_events": contact_event_hist,
                           "contact_event_ids": contact_id_hist})
        else:
            result.update({"contact_curve": contact_curve, "all_finite": all_finite,
                           "all_unit_headings": all_unit_headings, "all_in_box": all_in_box,
                           "final_positions": pos.copy(), "final_headings": head.copy(),
                           "final_ids": apc_ids.copy()})
    return result


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
