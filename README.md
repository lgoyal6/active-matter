# T-cell Swarming — Active Matter Simulation

2D self-propelled-particle model of why T-cell swarms self-limit in vivo:
ABP + Vicsek alignment + chemotaxis, plus desensitization, excluded volume,
chemokine secretion, and drift-with-memory. See `SKILL.md` for the science.

## Setup (one time)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is the human-readable spec; `requirements-lock.txt` is the
exact pinned set (`pip freeze`) for byte-for-byte reproducibility.

## Run things

Always `source .venv/bin/activate` first.

| Command | What it does |
|---|---|
| `python simulate.py` | Smoke test: prints output shapes, checks aggregation. |
| `python sanity_checks.py` | The limit-test protocol (turn each ingredient off → recover simpler model). |
| `python visualize.py` | Renders `swarm_snapshots.png` + `aggregation_curve.png` (µm / min units). |
| `python interactive_sim.py` | Live window: parameter sliders → **Run** → drag **Time** to replay. |

## Files

- `simulate.py` — the canonical engine. `simulate(params, seed)` returns the
  hand-off contract dict. `DEFAULTS` is the single source of truth for all
  parameters and the µm / minute unit anchors.
- `sanity_checks.py` — runnable sanity protocol.
- `visualize.py` — static figures.
- `interactive_sim.py` — interactive desktop explorer (needs a GUI backend).

## Analysis & sweep pipeline (Track A additions)

Built on top of the engine; everything consumes the same hand-off dict.

| File | What it does |
|---|---|
| `observables.py` | swarm size, internal polar order, source density, tight-core density, persistence, dispersal-mode classifier |
| `sweep.py` | `(alpha, k_rep)` phase-diagram sweep → `phase_diagram_*.png` heatmaps |
| `animate.py` | `make_movie` (mp4→gif fallback) + `save_snapshots` for slides |
| `io_utils.py` | `save_trajectory` / `load_trajectory` (.npz, params ride along as JSON) |

```python
from simulate import simulate, DEFAULTS
from sweep import run_sweep, plot_phase_diagram
from animate import make_movie
import numpy as np

traj = simulate(dict(DEFAULTS, dynamic_chemokine=True, alpha=0.1, k_rep=5.0, v_drift=0.4), seed=0)
make_movie(traj, "swarm.mp4")
grids = run_sweep(np.linspace(0, 0.6, 6), np.linspace(0, 12, 6), n_steps=1200)
plot_phase_diagram(grids, "core_density")
```

**Headline result:** in the `(alpha, k_rep)` sweep, the Keller-Segel pile-up
(`core_density`) survives *only* at low-alpha / low-k_rep — **either** biochemical
desensitization or mechanical repulsion quenches the singular collapse, while
`density_at_source` shows `alpha` setting overall retention at the infection site.
