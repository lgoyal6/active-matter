"""Save / load the simulation hand-off dict.

Trajectories are stored as compressed .npz; the params dict rides along as a
JSON string so the file is fully self-describing. ``load_trajectory`` returns
the exact dict shape that ``simulate()`` produces and the analysis track
consumes.
"""

import json
import numpy as np


def save_trajectory(traj, path):
    """Write a trajectory dict to ``path`` (.npz)."""
    np.savez_compressed(
        path,
        positions=traj["positions"],
        orientations=traj["orientations"],
        sensitivity=traj["sensitivity"],
        chemokine_field=traj["chemokine_field"],
        box_size=np.array(traj["box_size"]),
        source_position=np.array(traj["source_position"]),
        params_json=np.array(json.dumps(traj["params"])),
    )
    return path


def load_trajectory(path):
    """Read a trajectory dict back from ``path`` (.npz)."""
    z = np.load(path, allow_pickle=False)
    return {
        "positions": z["positions"],
        "orientations": z["orientations"],
        "sensitivity": z["sensitivity"],
        "chemokine_field": z["chemokine_field"],
        "box_size": float(z["box_size"]),
        "source_position": tuple(z["source_position"].tolist()),
        "params": json.loads(str(z["params_json"])),
    }
