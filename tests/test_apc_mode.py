import numpy as np
import pytest

from simulate import (_apc_cell_forces, _apc_cell_radii, _apc_hcp_positions,
                     _apc_neighbor_pairs, _apc_pair_force, simulate)
from observables import unique_contacts


def _params(**overrides):
    p = dict(dim=2, L=12.0, N=12, n_steps=30, dt=0.001, k_rep=0.0,
             w_a=0.0, w_c=0.0, eta=0.0, lambda_r=0.0)
    p.update(overrides)
    return p


def test_legacy_off_mode_preserves_default_outputs():
    old = simulate({"n_steps": 24}, seed=19)
    explicit = simulate({"n_steps": 24, "apc_mode": "off"}, seed=19)
    for key in ("positions", "headings", "sensitivity", "source_sensitivity", "chemokine_field"):
        assert np.array_equal(old[key], explicit[key])


@pytest.mark.parametrize("law", ["thesis_eq5_5_normalized", "zenodo_ch6_source_unnormalized"])
def test_opt_in_geometry_and_force_law_metadata(law):
    traj = simulate(_params(apc_mode="thesis", apc_force_law=law, rho=0.2,
                            apc_r0=1.0, deform_A=0.15, omega_deform=1.0), seed=3)
    assert traj["apc_center"] == (6.0, 6.0)
    assert traj["params"]["apc_force_law"] == law
    assert traj["params"]["apc_total_slots"] == traj["params"]["N_mobile"] + 1
    assert abs(traj["params"]["apc_achieved_rho"] - 0.2) < 0.01
    assert traj["params"]["apc_initial_min_center_distance"] >= 2.0
    assert np.allclose(traj["cell_radii"][0], _apc_cell_radii(0.001, np.random.default_rng(3).uniform(0, 2*np.pi, traj["params"]["N_mobile"]), 1.0, 0.15, 1.0))
    assert len(set(traj["trajectory_ids"][0].tolist())) == traj["params"]["N_mobile"]


def test_hcp_has_no_periodic_seam_overlap():
    cells, _, minimum = _apc_hcp_positions(40, 20.0, 1.0)
    all_points = np.vstack(([10.0, 10.0], cells))
    d = all_points[:, None, :] - all_points[None, :, :]
    d -= 20.0 * np.round(d / 20.0)
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    assert minimum >= 2.0 - 1e-10
    assert dist.min() >= 2.0 - 1e-10


def test_hcp_mobile_population_is_centered_across_periodic_box():
    cells, _, _ = _apc_hcp_positions(117, 43.0, 1.0)
    assert cells.min(axis=0).min() < 2.0
    assert cells.max(axis=0).max() > 39.0
    quadrants = np.histogram2d(cells[:, 0], cells[:, 1], bins=2, range=[[0, 43], [0, 43]])[0]
    assert quadrants.min() >= 20
    assert quadrants.max() <= 40


def test_apc_contact_uses_surface_radius_and_counts_identity_once():
    traj = {
        "params": {"apc_mode": "thesis"}, "box_size": 10.0,
        "apc_center": (5.0, 5.0), "apc_radius": 1.0,
        "positions": np.array([[[7.0, 5.0]], [[6.9, 5.0]], [[5.0, 5.0]]]),
        "cell_radii": np.ones((3, 1)),
        "trajectory_ids": np.array([[0], [0], [1]]),
        "contact_events": np.array([[False], [True], [True]]),
        "contact_event_ids": np.array([[0], [0], [1]]),
    }
    assert np.array_equal(unique_contacts(traj), [0, 1, 2])
    with pytest.raises(ValueError, match="contact_radius override"):
        unique_contacts(traj, contact_radius=1.0)


@pytest.mark.parametrize("phase", [0.1, 2.0, 4.0, 6.0])
def test_phase_oscillation_starts_at_next_mean_radius_crossing(phase):
    radii = _apc_cell_radii(0.0, np.array([phase]), 1.0, 0.15, 1.0)
    assert radii[0] == 1.0


@pytest.mark.parametrize("law,expected", [
    ("thesis_eq5_5_normalized", 5.0),
    ("zenodo_ch6_source_unnormalized", 10.0),
])
def test_force_laws_have_explicit_overlap_magnitudes_and_outward_direction(law, expected):
    force = _apc_pair_force(np.array([1.5, 0.0]), 1.0, 1.0, 20.0, law)
    assert np.allclose(force, [expected, 0.0])
    assert np.array_equal(_apc_pair_force(np.array([2.0, 0.0]), 1, 1, 20, law), [0.0, 0.0])
    assert np.array_equal(_apc_pair_force(np.array([3.0, 0.0]), 1, 1, 20, law), [0.0, 0.0])


@pytest.mark.parametrize("law,expected", [
    ("thesis_eq5_5_normalized", 5.0),
    ("zenodo_ch6_source_unnormalized", 10.0),
])
def test_production_force_path_handles_apc_mobile_overlap_and_coincident_cells(law, expected):
    pos = np.array([[5.5, 5.0], [5.5, 5.0], [6.8, 5.0]])
    force = _apc_cell_forces(pos, np.ones(3), np.array([5.0, 5.0]), 1.0, 20.0, law, 10.0)
    assert np.isfinite(force).all()
    assert force[0, 0] > expected
    assert np.allclose(force[0] - force[1], [0.0, 0.0])
    assert force[2, 0] > 0.0


@pytest.mark.parametrize("law", ["thesis_eq5_5_normalized", "zenodo_ch6_source_unnormalized"])
def test_periodic_cell_list_force_matches_dense_pair_oracle(law):
    rng = np.random.default_rng(29)
    pos = rng.uniform(0, 12, size=(35, 2))
    radii = rng.uniform(0.85, 1.15, size=35)
    center = np.array([6.0, 6.0])
    pairs = _apc_neighbor_pairs(pos, 12.0, 2.5)
    sparse = _apc_cell_forces(pos, radii, center, 1.0, 20.0, law, 12.0, pairs)
    dense = _apc_cell_forces(pos, radii, center, 1.0, 20.0, law, 12.0, None)
    assert np.allclose(sparse, dense, rtol=1e-12, atol=1e-12)


def test_compact_recording_matches_full_state_and_contact_curve():
    params = _params(apc_mode="thesis", rho=0.2, L=12.0, n_steps=20,
                     k_rep=20.0, deform_A=0.1, omega_deform=1.0,
                     apc_force_law="thesis_eq5_5_normalized")
    full = simulate(params, seed=41)
    compact = simulate(dict(params, apc_record_trajectory=False), seed=41)
    assert np.array_equal(full["positions"][-1], compact["final_positions"])
    assert np.array_equal(full["headings"][-1], compact["final_headings"])
    assert np.array_equal(full["trajectory_ids"][-1], compact["final_ids"])
    assert np.array_equal(unique_contacts(full), unique_contacts(compact))


def test_apc_mode_rejects_3d_and_invalid_force_law():
    with pytest.raises(ValueError, match="dim=2"):
        simulate(_params(dim=3, apc_mode="thesis"), seed=0)
    with pytest.raises(ValueError, match="apc_force_law"):
        simulate(_params(apc_mode="thesis", apc_force_law="unknown"), seed=0)


def test_manifest_rejects_truncated_and_malformed_rows(tmp_path):
    from evaluation.apc_study import _load_manifest

    manifest = tmp_path / "study.jsonl"
    manifest.write_text('{"case_key":')
    with pytest.raises(ValueError, match="malformed or truncated"):
        _load_manifest(manifest, tmp_path, "protocol", "source", True, set())
    manifest.write_text('{"case_key": "a"}')
    with pytest.raises(ValueError, match="truncated"):
        _load_manifest(manifest, tmp_path, "protocol", "source", True, set())


def test_manifest_rejects_case_protocol_and_raw_mismatches(tmp_path):
    import hashlib
    import json
    from evaluation.apc_study import _case_key, _load_manifest

    raw = tmp_path / "curve.npz"
    np.savez(raw, contact_curve=np.array([0, 1]), contact_times=np.array([0.0, 0.1]),
             position_frame_times=np.array([0.1, 0.2]))
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    protocol_hash = "protocol"
    row = {"case_key": _case_key("law", 0.2, 1, 2, 0.1, protocol_hash),
           "law": "law", "requested_rho": 0.2, "seed": 1, "steps": 2, "dt": 0.1,
           "protocol_sha256": protocol_hash, "simulator_sha256": "source",
           "raw_curve_path": str(raw), "raw_curve_sha256": raw_hash}
    manifest = tmp_path / "study.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    expected = {_case_key("law", 0.2, 1, 2, 0.1, protocol_hash)}
    assert len(_load_manifest(manifest, tmp_path, protocol_hash, "source", True, expected)) == 1
    off_grid = dict(row, requested_rho=0.3)
    off_grid["case_key"] = _case_key("law", 0.3, 1, 2, 0.1, protocol_hash)
    manifest.write_text(json.dumps(off_grid) + "\n")
    with pytest.raises(ValueError, match="unexpected"):
        _load_manifest(manifest, tmp_path, protocol_hash, "source", True, expected)
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="different protocol or source"):
        _load_manifest(manifest, tmp_path, "different", "source", True, expected)
    row["case_key"] = "mismatched-case-key"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="mismatched, unexpected, or duplicate case key"):
        _load_manifest(manifest, tmp_path, protocol_hash, "source", True, expected)
    row["case_key"] = _case_key("law", 0.2, 1, 2, 0.1, protocol_hash)
    row["raw_curve_sha256"] = "wrong"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash-mismatched raw data"):
        _load_manifest(manifest, tmp_path, protocol_hash, "source", True, expected)


def test_manifest_rejects_wrong_raw_sample_lengths(tmp_path):
    import hashlib
    import json
    from evaluation.apc_study import _case_key, _load_manifest

    raw = tmp_path / "short.npz"
    np.savez(raw, contact_curve=np.array([0]), contact_times=np.array([0.0]),
             position_frame_times=np.array([0.1]))
    protocol_hash = "p"
    row = {"case_key": _case_key("law", 0.2, 1, 2, 0.1, protocol_hash),
           "law": "law", "requested_rho": 0.2, "seed": 1, "steps": 2, "dt": 0.1,
           "protocol_sha256": protocol_hash, "simulator_sha256": "s",
           "raw_curve_path": str(raw), "raw_curve_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
    manifest = tmp_path / "study.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="malformed raw data"):
        _load_manifest(manifest, tmp_path, protocol_hash, "s", True,
                       {_case_key("law", 0.2, 1, 2, 0.1, protocol_hash)})


def test_runner_rejects_arguments_outside_protocol_grid(tmp_path):
    from evaluation.apc_study import build_protocol, run

    protocol = build_protocol([0.2], [7], 2, ["law"], 0.1)
    with pytest.raises(ValueError, match="do not match the frozen protocol grid"):
        run([0.3], [7], 2, ["law"], 0.1, tmp_path, protocol)


def test_study_protocol_binds_seed_level_summary_plan():
    from evaluation.apc_study import build_protocol

    protocol = build_protocol([0.2], [11, 23, 37, 53], 500_000,
                              ["law-a", "law-b"], 0.001)
    assert protocol["min_per_time"] == 1.0
    assert protocol["summary"]["independent_units"] == "seed runs, not correlated timesteps"
    assert protocol["summary"]["timepoint_bootstrap"] is False
