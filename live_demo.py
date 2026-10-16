"""Run bounded public experiments through the existing NumPy simulator.
The web app calls this adapter; simulate.py remains the model implementation.
Results describe a small synthetic run, not the saved APC study or biology.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from simulate import simulate


LIMITS = {
    "seed": (0, 2**31 - 1, 1, int),
    "particles": (8, 96, 48, int),
    "steps": (20, 400, 200, int),
    "alignment": (0, 2, 0.5, float),
    "chemotaxis": (0, 2, 1.0, float),
    "noise": (0, 1, 0.3, float),
}
SOURCE_SHA256 = hashlib.sha256(Path(__file__).with_name("simulate.py").read_bytes()).hexdigest()


def run_experiment(payload):
    if not isinstance(payload, dict) or set(payload) - LIMITS.keys():
        raise ValueError("Use only seed, particles, steps, alignment, chemotaxis and noise.")
    values = {}
    for name, (low, high, default, kind) in LIMITS.items():
        value = payload.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not low <= value <= high or not math.isfinite(value)
                or (kind is int and value != int(value))):
            raise ValueError(f"{name} must be {'an integer' if kind is int else 'a number'} from {low} to {high}.")
        values[name] = kind(value)
    # Fix expensive model options so public input cannot expand the workload.
    trajectory = simulate({
        "dim": 2, "N": values["particles"], "n_steps": values["steps"],
        "L": 30.0, "grid": 24, "dt": 0.05, "rho": None,
        "w_a": values["alignment"], "w_c": values["chemotaxis"],
        "eta": values["noise"], "apc_mode": "off", "dynamic_chemokine": False,
    }, seed=values["seed"])
    positions = trajectory["positions"]
    headings = trajectory["headings"]
    if not np.isfinite(positions).all() or not np.isfinite(headings).all():
        raise RuntimeError("The simulation produced non-finite coordinates.")
    # Return at most 101 frames, including the final one, for a small response.
    indices = np.unique(np.linspace(0, len(positions) - 1, min(101, len(positions))).astype(int))
    frames = np.round(positions[indices], 5).tolist()
    source = np.asarray(trajectory["source_position"])
    delta = (positions[-1] - source + 15.0) % 30.0 - 15.0
    digest = hashlib.sha256(json.dumps(frames, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "mode": "live_cpu_simulation", "inputs": values,
        "engine_source_sha256": SOURCE_SHA256, "trajectory_sha256": digest,
        "box_size": 30.0, "source_position": source.tolist(),
        "frame_steps": indices.tolist(), "frames": frames,
        "metrics": {
            "final_mean_distance": float(np.linalg.norm(delta, axis=1).mean()),
            "final_alignment": float(np.linalg.norm(headings[-1].mean(axis=0))),
            "simulated_time": values["steps"] * 0.05,
        },
        "limits": "Small 2D ABP/Vicsek/chemotaxis run. Not the frozen APC sweep or biological validation.",
    }
