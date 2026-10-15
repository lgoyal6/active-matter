# Optional fixed-APC study mode

`apc_mode="thesis"` adds one immobile circular APC at the center of a periodic
2D box. The APC is radius `apc_r0` (default 1), is excluded from all mobile-cell
updates, and occupies one slot in the total-particle density. The requested
packing fraction maps to `N_mobile = round(rho * L^2 / (pi * apc_r0^2)) - 1`;
the returned parameters include requested and achieved density, mobile count,
total slots, HCP grid shape, and minimum periodic initial center separation.
Legacy `r_excl=1` means cell diameter 1 in the existing engine. APC mode uses
the explicit thesis scale `apc_r0=1`, and the study runner sets `r_excl=2` so
the legacy radius knob does not silently rescale the thesis-sized mobile cells.

Two named linear-overlap force variants are required because the thesis Eq. 5.5
and its Chapter 6 supplementary implementation disagree on normalization:

- `thesis_eq5_5_normalized`: magnitude `F_rep * overlap / (r_i + r_j)`.
- `zenodo_ch6_source_unnormalized`: magnitude `F_rep * overlap`.

Both apply to mobile-cell pairs and the fixed APC. The target itself never moves,
oscillates, reorients, contributes chemokine, or joins mobile arrays. Initial
positions use a periodic staggered HCP grid centered on the APC slot and reject
any grid whose minimum-image pair spacing is below `2*r0`. All mobile cells
start at mean radius; each begins oscillating at its next phase crossing of the
mean radius. This follows thesis prose; the archived C++ phase-start helper has
a discrepant start-time expression for some phases.

The optional APC contact observable checks surface distance
`d < r_cell(t) + r_APC` at the force-evaluation instant before each integration
step (time `t`), counts each then-current trajectory ID once, and preserves
the source convention of reserving ID 0 for the APC, assigning mobile IDs from 1,
and assigning a new ID after periodic re-entry. APC contact IDs are therefore
mobile-identity IDs. A custom `contact_radius` override is unsupported in APC
mode; the observable uses the time-varying cell-radius plus APC-radius surface
contact rule. The legacy post-hoc central-region observable remains unchanged
when APC mode is off. Contact samples therefore have times
`0, dt, ..., (n_steps-1)*dt`, while saved positions, radii, and trajectory IDs describe the state after each step at
`dt, 2*dt, ..., n_steps*dt`; result metadata exposes both offsets. The archived
executable passes `omega` directly to `sin(omega*t + phase)`, so the runner's
`omega_deform=2` is in radians per simulated minute. The thesis calls its
frequency axis oscillations per minute; the runner value follows the source-code
convention and is not claimed to match an exact thesis frequency tick.

The public thesis is Lucas Erwin Wiebke, *Exploring T Cell Crowd Dynamics via
Active Matter Modelling*, ITBA, August 2025, handle
<https://hdl.handle.net/20.500.14769/5252>. Its supplementary source record is
<https://doi.org/10.5281/zenodo.16740231> (CC BY 4.0). The thesis Chapter 6
reports `L=43`, `r0=1`, `A=.15`, `v0=1`, `F_rep=50`, `F_adh=0`, density range
approximately `.2-.85`, contact-rate fit from 25 to 500 minutes, and at least
four realizations/configuration. The Python runner interprets density as the
written physical packing fraction `N*pi*r0^2/L^2`. The archived `set_particulas.h`
instead uses `d=(2*r0)/densidad` as an HCP spacing scale and fills floored
lattice rows/columns; its asymptotic packing fraction is about
`(pi/(2*sqrt(3))) * densidad^2`, so the source-code density mapping differs.
No archived-run outputs were found to establish a matching density conversion.
The runner uses an explicit `dt=.001` and `min_per_time=1.0`, so one
simulation-time unit is one minute and 500,000 steps span 500 minutes. The
engine default `min_per_time=.5` would instead make this a 250-minute run, so
the study overrides and records that default. The 25-minute transient is
excluded from slope fitting, not from the simulated horizon. The achieved
density is recorded after integer rounding. These comparisons are diagnostics, not an exact reproduction claim:
the archived source and written force equation differ, and this Python model's
other dynamics are not yet established as equivalent.

The periodic cell list uses maximum cell radius plus a Verlet skin and rebuilds
when a cell moves more than half the skin. A dense pair kernel remains available
as the oracle for small-input equivalence tests. Compact recording retains the
cumulative contact curve and final state without allocating full trajectories.
The short pilot runs 1000 steps per density, one seed, both force laws. Its
one-minute horizon is below the 25-minute fitting transient, so pilot contact
counts and timings are warmup-only and it reports no contact-rate estimate.
The full planned comparison runs four densities, four seeds and 500,000 steps
per case per law. This is a 500-minute total horizon including the initial 25-minute
transient, which is excluded only from the slope fit. Before the full run, use
the shared `scripts/run_heavy.py` lock and confirm at least 20% free memory.
Retain one compressed cumulative contact curve per run, its hash, and the
protocol/source hashes. The runner appends a JSONL receipt only after each raw
curve is atomically written and hashed. `--max-cases` supports bounded chunks;
`--resume` validates source and protocol digests, unique case keys, raw-file
hashes, and sample lengths, stopping on malformed or truncated evidence.
Report every law and density, raw outcome rows, and no post-hoc winner selection. A density
response not showing an intermediate peak is a null/negative result, not a
reason to tune parameters.

This diagnostic uses four physical-packing-fraction densities, `lambda_r=0.2`,
and oscillatory radii `A=.15`. Thesis Fig. 6.5 sweeps five reorientation rates
and compares both fixed and oscillatory radii; this runner samples one slice
of that design and should not be described as the full figure reproduction.

For each density and force law, retain all four per-seed slopes and report the
mean, sample standard deviation, and standard error across the four independent
seed runs. Any interval is descriptive with `n=4`; timepoints within a run are
not treated as independent samples and are not bootstrapped. A comparison
between force laws may show per-seed paired slope differences descriptively.
