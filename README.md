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
