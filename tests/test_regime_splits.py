from evaluation.heldout_control import split_regimes


def test_regimes_are_disjoint_and_order_independent():
    rows = [{"trajectory_id": f"{g}-{i}", "regime": g}
            for g in ("low_noise", "high_noise", "obstacles", "sparse")
            for i in range(3)]
    a = split_regimes(rows)
    b = split_regimes(reversed(rows))
    assert set(a["train_groups"]).isdisjoint(a["heldout_groups"])
    assert a["train_groups"] == b["train_groups"]
    assert a["heldout_groups"] == b["heldout_groups"]
    assert {r["regime"] for r in a["train"]} == set(a["train_groups"])


def test_missing_or_duplicate_identity_rejected():
    try:
        split_regimes([{"trajectory_id": "a", "regime": "x"},
                       {"trajectory_id": "a", "regime": "y"}])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate trajectory accepted")
