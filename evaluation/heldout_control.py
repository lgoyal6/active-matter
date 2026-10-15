"""Deterministic, group-disjoint regime assignment for control trajectories."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping


def split_regimes(rows: Iterable[Mapping], *, group_key: str = "regime",
                  train_fraction: float = 0.7) -> dict[str, list[dict]]:
    """Split whole regimes; reject missing/duplicate IDs and invalid fractions.

    Stable hashing keeps assignments independent of row ordering and prevents
    trajectories from one regime leaking across train and held-out sets.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    by_group: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for source in rows:
        row = dict(source)
        if group_key not in row or row[group_key] in (None, ""):
            raise ValueError(f"missing group key: {group_key}")
        if "trajectory_id" not in row or row["trajectory_id"] in (None, ""):
            raise ValueError("every row requires trajectory_id")
        tid = str(row["trajectory_id"])
        if tid in seen_ids:
            raise ValueError(f"duplicate trajectory_id: {tid}")
        seen_ids.add(tid)
        by_group[str(row[group_key])].append(row)
    if len(by_group) < 2:
        raise ValueError("at least two distinct regimes are required")

    ordered = sorted(by_group, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    n_train = min(len(ordered) - 1, max(1, round(len(ordered) * train_fraction)))
    train_groups, heldout_groups = set(ordered[:n_train]), set(ordered[n_train:])
    return {
        "train": [r for g in ordered if g in train_groups for r in by_group[g]],
        "heldout": [r for g in ordered if g in heldout_groups for r in by_group[g]],
        "train_groups": sorted(train_groups),
        "heldout_groups": sorted(heldout_groups),
    }
