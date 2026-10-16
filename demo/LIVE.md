# Live swarm experiment

The public Flask app imports `live_demo.py`, which calls `simulate.py` directly.
It runs a new seeded NumPy simulation for each request. The browser displays
the returned trajectory; it does not substitute saved results on failure.

Run locally with `pip install -r requirements.txt` and `flask --app app run`.
Run `python -m pytest -q` for simulator and HTTP contract checks. Vercel detects
the top-level Flask `app.py` and installs the same runtime requirements.

Requests accept 8-96 particles and 20-400 steps in the existing 2D model.
Input is limited to 4 KiB. At most 101 trajectory frames are returned.
The server stores no requests and makes no external calls. Client cancellation
stops waiting for a result; the bounded server computation may still finish.

The `/study` route preserves the separate recorded APC sweep. This small live
ABP/Vicsek/chemotaxis run is not a rerun of that sweep or biological validation.
The result includes the engine source hash and a trajectory hash for comparison.
