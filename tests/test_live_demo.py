"""Check the web adapter against the real simulator and its resource bounds.
The adapter keeps input and output sizes bounded.
No external providers, saved results or biological data are used.
"""
import json

import numpy as np
import pytest

from live_demo import run_experiment
from simulate import simulate


def test_adapter_matches_direct_engine_and_keeps_last_frame():
    result = run_experiment({"particles": 8, "steps": 120, "seed": 13})
    direct = simulate({"dim": 2, "N": 8, "n_steps": 120, "L": 30.0,
                       "grid": 24, "dt": 0.05, "rho": None, "w_a": 0.5,
                       "w_c": 1.0, "eta": 0.3, "apc_mode": "off",
                       "dynamic_chemokine": False}, seed=13)
    expected = np.round(direct["positions"][result["frame_steps"]], 5)
    np.testing.assert_array_equal(result["frames"], expected)
    assert result["frame_steps"][0] == 0
    assert result["frame_steps"][-1] == 119
    assert len(result["frames"]) == 101
    assert result["mode"] == "live_cpu_simulation"


def test_reproducible_seed_and_changed_inputs():
    # Use a populated neighborhood so changing alignment can affect the run.
    inputs = {"particles": 48, "steps": 20, "seed": 13}
    original = run_experiment(inputs)
    assert original == run_experiment(inputs)
    for change in [{"seed": 14}, {"noise": 0.9}, {"chemotaxis": 0}, {"alignment": 2}]:
        assert run_experiment(inputs | change)["trajectory_sha256"] != original["trajectory_sha256"]


@pytest.mark.parametrize("payload", [None, [], {"extra": 1}, {"seed": True},
    {"seed": 10**1000}, {"particles": 7}, {"particles": 96.5}, {"steps": 401},
    {"seed": -1}, {"noise": float("nan")}, {"noise": float("inf")},
    {"alignment": "1"}, {"chemotaxis": 3}, {"noise": -0.1}])
def test_rejects_unsupported_or_unbounded_inputs(payload):
    with pytest.raises(ValueError):
        run_experiment(payload)


def test_largest_request_stays_bounded():
    result = run_experiment({"particles": 96, "steps": 400})
    assert np.asarray(result["frames"]).shape == (101, 96, 2)
    assert len(json.dumps(result)) < 250_000
    assert 0 <= result["metrics"]["final_alignment"] <= 1.0000001


