"""Drive the APC simulator and observable for the bounded density study.
Each raw curve is written before its JSONL receipt; resume accepts only the exact frozen case grid.
The protocol is a diagnostic slice, not an exact thesis reproduction claim.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observables import unique_contacts
from simulate import DEFAULTS, simulate


SOURCE_NAMES = ("simulate.py", "observables.py", "evaluation/apc_study.py", "tests/test_apc_mode.py")
LAWS = ("thesis_eq5_5_normalized", "zenodo_ch6_source_unnormalized")


def _digest_sources(repo):
    return hashlib.sha256(b"".join(
        (repo / name).name.encode() + b"\0" + (repo / name).read_bytes()
        for name in SOURCE_NAMES)).hexdigest()


def _load_average():
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


def _case_key(law, rho, seed, steps, dt, protocol_hash):
    payload = json.dumps([law, float(rho), int(seed), int(steps), float(dt), protocol_hash])
    return hashlib.sha256(payload.encode()).hexdigest()


def build_protocol(rhos, seeds, steps, laws, dt, pilot=False):
    return {"python_version": platform.python_version(), "numpy_version": np.__version__,
            "platform": platform.platform(), "rhos": list(map(float, rhos)), "seeds": list(map(int, seeds)),
            "steps": int(steps), "laws": list(laws), "dt": float(dt), "L": 43.0,
            "A": 0.15, "omega_deform_rad_per_min": 2.0, "F_rep": 50.0,
            "F_adh": 0.0, "apc_r0": 1.0, "r_excl": 2.0, "v0": 1.0,
            "lambda_r": 0.2, "w_a": 0.0, "w_c": 0.0, "alpha": 0.0,
            "beta": 0.0, "eta": 0.0, "neighbor_skin": 0.3,
            "min_per_time": 1.0, "dynamic_chemokine": False, "record_trajectory": False,
            "density_mapping": "physical_rho=N_total*pi*r0^2/L^2; rounded integer slots",
            "initialization": "periodic_staggered_HCP_with_uniform_index_thinning",
            "phase_start": "thesis_prose_next_mean_radius_crossing",
            "contact_rule": "pre_step_t_less_radius_cell_plus_apc",
            "contact_event_time_offset": 0.0, "position_frame_time_offset": dt,
            "random_generator": "numpy.default_rng (PCG64), seeded per run",
            "transient_minutes": 25.0,
            "fit": "ordinary_least_squares_cumulative_contacts_vs_time",
            "pilot_warmup_only": bool(pilot),
            "engine_defaults": {k: (v.item() if isinstance(v, np.generic) else v)
                               for k, v in DEFAULTS.items()},
            "summary": {"per_density_law": "retain four per-seed slopes; report mean, sample SD, and SE",
                        "independent_units": "seed runs, not correlated timesteps",
                        "paired_law_contrast": "descriptive per-seed paired slope differences",
                        "intervals": "descriptive n=4", "timepoint_bootstrap": False}}


def _load_manifest(path, raw_dir, protocol_hash, source_hash, resume, expected_keys):
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError(f"manifest exists; pass --resume to continue: {path}")
    rows, keys = [], set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed or truncated manifest line {line_number}") from exc
            if not line.endswith("\n"):
                raise ValueError(f"truncated manifest line {line_number}")
            required = ("case_key", "law", "requested_rho", "seed", "steps", "dt",
                        "protocol_sha256", "simulator_sha256", "raw_curve_path", "raw_curve_sha256")
            if any(key not in row for key in required):
                raise ValueError(f"manifest line {line_number} is missing required case evidence")
            if row["protocol_sha256"] != protocol_hash or row["simulator_sha256"] != source_hash:
                raise ValueError(f"manifest line {line_number} belongs to different protocol or source")
            expected_key = _case_key(row["law"], row["requested_rho"], row["seed"],
                                     row["steps"], row["dt"], protocol_hash)
            if row["case_key"] != expected_key or expected_key not in expected_keys or expected_key in keys:
                raise ValueError(f"manifest line {line_number} has a mismatched, unexpected, or duplicate case key")
            raw_path = Path(row["raw_curve_path"])
            if not raw_path.is_absolute():
                raw_path = raw_dir / raw_path
            if not raw_path.is_file() or hashlib.sha256(raw_path.read_bytes()).hexdigest() != row["raw_curve_sha256"]:
                raise ValueError(f"manifest line {line_number} has missing or hash-mismatched raw data")
            try:
                with np.load(raw_path) as raw:
                    for field in ("contact_curve", "contact_times", "position_frame_times"):
                        if field not in raw or len(raw[field]) != row["steps"]:
                            raise ValueError
            except Exception as exc:
                raise ValueError(f"manifest line {line_number} has malformed raw data") from exc
            keys.add(expected_key)
            rows.append(row)
    return rows


def run(rhos, seeds, steps, laws, dt, raw_dir, protocol=None,
        manifest_path=None, resume=False, max_cases=None):
    if protocol is None:
        protocol = build_protocol(rhos, seeds, steps, laws, dt)
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    source_hash = _digest_sources(repo)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (list(map(float, protocol["rhos"])) != list(map(float, rhos)) or
            list(map(int, protocol["seeds"])) != list(map(int, seeds)) or
            list(protocol["laws"]) != list(laws) or int(protocol["steps"]) != int(steps) or
            float(protocol["dt"]) != float(dt)):
        raise ValueError("runner arguments do not match the frozen protocol grid")
    manifest_path = manifest_path or raw_dir.parent / "apc-study-manifest.jsonl"
    expected_keys = {_case_key(law, rho, seed, steps, dt, protocol_hash)
                     for law in laws for rho in rhos for seed in seeds}
    rows = _load_manifest(manifest_path, raw_dir, protocol_hash, source_hash, resume, expected_keys)
    completed = {row["case_key"] for row in rows}
    new_count = 0
    case_order_index = 0
    case_total = len(laws) * len(rhos) * len(seeds)
    for law in laws:
        for rho in rhos:
            for seed in seeds:
                case_order_index += 1
                key = _case_key(law, rho, seed, steps, dt, protocol_hash)
                if key in completed:
                    continue
                if max_cases is not None and new_count >= max_cases:
                    return rows
                load_start = _load_average()
                started = time.perf_counter()
                sim_params = {"dim": 2, "L": protocol["L"], "rho": rho, "n_steps": steps, "dt": dt,
                    "apc_mode": "thesis", "apc_force_law": law, "apc_r0": protocol["apc_r0"],
                    "r_excl": protocol["r_excl"], "k_rep": protocol["F_rep"], "v0": protocol["v0"],
                    "w_a": protocol["w_a"], "w_c": protocol["w_c"], "alpha": protocol["alpha"], "beta": protocol["beta"],
                    "eta": protocol["eta"], "lambda_r": protocol["lambda_r"], "deform_A": protocol["A"],
                    "omega_deform": protocol["omega_deform_rad_per_min"], "min_per_time": protocol["min_per_time"],
                    "dynamic_chemokine": False, "apc_neighbor_skin": protocol["neighbor_skin"],
                    "apc_record_trajectory": False}
                traj = simulate(sim_params, seed=seed)
                elapsed = time.perf_counter() - started
                counts = unique_contacts(traj)
                horizon = steps * dt
                start = int(protocol["transient_minutes"] / dt)
                if horizon <= protocol["transient_minutes"]:
                    fit_rate, fit_status = None, "warmup_only_below_fit_window"
                else:
                    fit_t = np.arange(start, len(counts)) * dt
                    fit_rate = float(np.polyfit(fit_t, counts[start:], 1)[0]) if len(fit_t) > 1 else None
                    fit_status = "fit_available" if fit_rate is not None else "insufficient_fit_samples"
                raw_path = raw_dir / f"{law}-rho{rho:.2f}-seed{seed}.npz"
                temp_path = raw_path.with_name(raw_path.name + ".tmp.npz")
                np.savez_compressed(temp_path, contact_curve=counts,
                                    contact_times=np.arange(steps) * dt,
                                    position_frame_times=(np.arange(steps) + 1) * dt)
                os.replace(temp_path, raw_path)
                raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                row = {
                    "case_key": key, "case_order_index": case_order_index,
                    "case_total": case_total, "load_average_start": load_start,
                    "load_average_end": _load_average(), "law": law, "requested_rho": float(rho), "achieved_rho": traj["params"]["apc_achieved_rho"],
                    "seed": int(seed), "steps": int(steps), "dt": float(dt),
                    "simulation_parameters": sim_params,
                    "effective_parameters": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in traj["params"].items()},
                    "random_generator": protocol["random_generator"],
                    "mobile_cells": traj["params"]["N_mobile"], "total_slots": traj["params"]["apc_total_slots"],
                    "lattice_shape": traj["params"]["apc_lattice_shape"],
                    "initial_min_distance": traj["params"]["apc_initial_min_center_distance"],
                    "contacts_final": int(counts[-1]), "contact_rate": fit_rate, "fit_status": fit_status,
                    "fit_start_time": protocol["transient_minutes"], "total_horizon": horizon,
                    "finite": traj["all_finite"], "in_box": traj["all_in_box"], "unit_headings": traj["all_unit_headings"],
                    "raw_curve_path": str(raw_path), "raw_curve_sha256": raw_hash,
                    "simulator_sha256": source_hash, "protocol_sha256": protocol_hash,
                    "contact_event_time_offset": traj["contact_event_time_offset"],
                    "position_frame_time_offset": traj["position_frame_time_offset"], "elapsed_seconds": elapsed}
                with manifest_path.open("a") as manifest:
                    manifest.write(json.dumps(row, sort_keys=True) + "\n")
                    manifest.flush()
                    os.fsync(manifest.fileno())
                rows.append(row)
                completed.add(key)
                new_count += 1
                print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.pilot:
        rhos, seeds, steps, dt = [0.2, 0.5, 0.85], [11], 1000, 0.001
    else:
        rhos, seeds, steps, dt = [0.2, 0.5, 0.71, 0.85], [11, 23, 37, 53], 500_000, 0.001
    laws = list(LAWS)
    protocol = build_protocol(rhos, seeds, steps, laws, dt, pilot=args.pilot)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    raw_dir = args.raw_dir or args.output.with_suffix("").with_name(args.output.stem + "-raw")
    manifest_path = args.manifest or args.output.with_suffix(".jsonl")
    rows = run(rhos, seeds, steps, laws, dt, raw_dir, protocol,
               manifest_path, args.resume, args.max_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_keys = {_case_key(law, rho, seed, steps, dt, protocol_hash)
                     for law in laws for rho in rhos for seed in seeds}
    actual_keys = {row["case_key"] for row in rows}
    complete = actual_keys == expected_keys
    args.output.write_text(json.dumps({
        "source": "Wiebke (2025), Chapter 6; Zenodo supplementary Source_code.zip",
        "runner_version": 3, "raw_dir": str(raw_dir), "manifest": str(manifest_path),
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "pilot": args.pilot, "complete": complete, "source_sha256": _digest_sources(Path(__file__).resolve().parents[1]),
        "requested_laws": laws, "protocol": protocol, "protocol_sha256": protocol_hash,
        "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
