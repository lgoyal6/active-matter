"""Claude vision service for T-cell swarm biological analysis.

Extracts key frames from a simulate() output dict, sends them to
claude-opus-4-7 as base64 PNG images, and returns a structured
biological interpretation: infection status, inflammation level,
swarm phase, and specific observable phenomena.

Usage:
    from analysis_service import analyze_simulation, print_analysis
    result = analyze_simulation(out, n_frames=6)
    print_analysis(result)
"""

import io
import base64
import re
import anthropic
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import simulate, _min_image, to_minutes, to_microns

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_frame_png(out, t: int, fig_size=(5, 5)) -> str:
    """Render simulation frame t as a base64-encoded PNG string."""
    p = out["params"]
    L = out["box_size"]
    L_um = to_microns(L, p)

    srcs = np.array(out.get("source_positions", [out["source_position"]]))
    srcs_um = to_microns(srcs, p)

    field = out["chemokine_field"]
    field_t = field[t] if field.ndim == 3 else field

    pos_um = to_microns(out["positions"][t], p)
    sens = out["sensitivity"][t]
    t_min = to_minutes(t, p)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(field_t, extent=[0, L_um, 0, L_um], origin="lower",
              cmap="YlOrRd", alpha=0.55)
    sc = ax.scatter(pos_um[:, 0], pos_um[:, 1], c=sens, cmap="viridis",
                    vmin=0, vmax=1, s=18, edgecolors="none")
    ax.plot(srcs_um[:, 0], srcs_um[:, 1], "r*", ms=18, label="infection source")
    ax.set_xlim(0, L_um); ax.set_ylim(0, L_um)
    ax.set_title(f"t = {t_min:.0f} min", fontsize=11)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("receptor sensitivity")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert computational biologist analyzing T-cell swarm dynamics from active matter simulations.

IMAGE LEGEND:
- Each colored dot = one T cell (N~200 cells in a periodic 2D domain)
- Dot color (viridis scale, yellow→green→blue→purple):
    • Yellow/green  = high receptor sensitivity (fresh, responsive receptors)
    • Blue/purple   = low sensitivity (desensitized receptors, recently in high chemokine)
- Orange/red background heatmap = chemokine concentration field (the T cells' attractant)
- Red star = infection/chemokine source (could represent a pathogen, tumor, or inflamed tissue)
- Frame title = elapsed simulation time in minutes

BIOLOGICAL MAPPINGS (use these to ground your analysis):
- Heavy clustering at the source → acute inflammation / active immune response
- Cells accumulating then rapidly purple-ing → receptor desensitization (β-arrestin internalization of CCR5/CXCR3)
- Swarm forms, persists, then disperses → healthy self-limiting immune response
- Swarm never disperses / grows denser over time → chronic inflammation risk
- Purple wave radiating outward from source → desensitization-driven dispersal (biochemical off-switch)
- Coherent streams leaving the cluster → streaming dispersal (ordered, high polar order)
- Cells scattering diffusely → diffuse dispersal (disordered, low polar order)
- Rotating ring or vortex pattern → collective rotational motion (common in confined swarms)
- All cells yellow/green throughout → no desensitization, swarm stuck at source (immune exhaustion risk)
- Cells uniformly distributed, no cluster → no effective chemotaxis or infection signal too weak

TASK:
You will receive a sequence of simulation frames in chronological order.
Provide a structured biological analysis covering:
1. Is there an active infection/immune response? (Yes/No + confidence)
2. Inflammation level: None / Mild / Moderate / Severe / Chronic
3. Current swarm phase at the final frame: Aggregating / Stable-Swarm / Dispersing / Dispersed / Oscillating
4. Dispersal mode (if dispersal is occurring): Streaming / Diffuse / None
5. Key observable phenomena (bulleted, 3–6 points)
6. Biological interpretation and clinical relevance (2–4 sentences)

Be specific — reference timing (from the frame titles), spatial patterns, color distributions, and dynamics you actually observe across the frames. Distinguish clearly between what you see and what you infer.
"""


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_simulation(out: dict, n_frames: int = 6) -> dict:
    """Analyze a simulate() output dict using Claude vision.

    Args:
        out:      Return value of simulate().
        n_frames: Number of evenly-spaced frames to sample (default 6).

    Returns:
        dict with keys:
            infection_detected  bool
            inflammation_level  str   ("None"/"Mild"/"Moderate"/"Severe"/"Chronic")
            phase               str   ("Aggregating"/"Stable-Swarm"/"Dispersing"/...)
            dispersal_mode      str   ("Streaming"/"Diffuse"/"None")
            observations        list  of str
            full_analysis       str   (raw Claude response text)
    """
    n_steps = out["positions"].shape[0]
    indices = [int(i * (n_steps - 1) / (n_frames - 1)) for i in range(n_frames)]

    print(f"Rendering {n_frames} frames ...", flush=True)
    images = []
    for t in indices:
        b64 = _render_frame_png(out, t)
        images.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
        t_min = to_minutes(t, out["params"])
        print(f"  frame t={t} ({t_min:.0f} min) rendered", flush=True)

    # Build message: all images first, then the analysis request
    content = images + [{
        "type": "text",
        "text": (
            f"Here are {n_frames} chronological frames from a T-cell swarming simulation "
            f"(frames at t ≈ "
            + ", ".join(f"{to_minutes(t, out['params']):.0f}" for t in indices)
            + " min). Analyze the swarm dynamics and provide your structured biological assessment."
        ),
    }]

    print("Sending frames to Claude for analysis ...", flush=True)
    response = _client().messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    full_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    return {**_parse_response(full_text), "full_analysis": full_text}


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    """Extract structured fields from Claude's free-text response."""

    def _find(pattern, default="Unknown"):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    # Infection detected
    infect_str = _find(r"infection[^:]*:\s*(yes|no)", "Unknown")
    infection_detected = infect_str.lower() == "yes"

    inflammation_level = _find(
        r"inflammation[^:]*:\s*(None|Mild|Moderate|Severe|Chronic)", "Unknown"
    )
    phase = _find(
        r"(?:swarm\s+)?phase[^:]*:\s*(Aggregating|Stable-Swarm|Stable Swarm|Dispersing|Dispersed|Oscillating)",
        "Unknown",
    )
    phase = phase.replace("Stable Swarm", "Stable-Swarm")

    dispersal_mode = _find(
        r"dispersal mode[^:]*:\s*(Streaming|Diffuse|None)", "None"
    )

    # Extract bullet-point observations
    obs_lines = re.findall(r"[-•*]\s+(.+)", text)
    observations = [line.strip() for line in obs_lines if len(line.strip()) > 10]

    return {
        "infection_detected": infection_detected,
        "inflammation_level": inflammation_level,
        "phase": phase,
        "dispersal_mode": dispersal_mode,
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_analysis(result: dict) -> None:
    """Print structured analysis result to stdout."""
    sep = "─" * 60
    print(sep)
    print("  T-CELL SWARM BIOLOGICAL ANALYSIS")
    print(sep)
    print(f"  Infection detected : {'YES' if result['infection_detected'] else 'NO'}")
    print(f"  Inflammation level : {result['inflammation_level']}")
    print(f"  Swarm phase        : {result['phase']}")
    print(f"  Dispersal mode     : {result['dispersal_mode']}")
    print(sep)
    print("  KEY OBSERVATIONS")
    for obs in result["observations"]:
        print(f"    • {obs}")
    print(sep)
    print("  FULL ANALYSIS")
    print()
    # Indent for readability
    for line in result["full_analysis"].splitlines():
        print(f"  {line}")
    print(sep)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SCENARIO 1: Persistent inflammation (no desensitization)")
    print("=" * 60)
    out1 = simulate({"n_steps": 1800, "alpha": 0.0, "k_rep": 5.0}, seed=1)
    result1 = analyze_simulation(out1, n_frames=5)
    print_analysis(result1)

    print()
    print("=" * 60)
    print("SCENARIO 2: Healthy self-limiting swarm")
    print("=" * 60)
    out2 = simulate({"n_steps": 2400, "alpha": 0.4, "beta": 0.2,
                     "k_rep": 5.0, "recovery_K": 0.5}, seed=2)
    result2 = analyze_simulation(out2, n_frames=6)
    print_analysis(result2)

    print()
    print("=" * 60)
    print("SCENARIO 3: Immune exhaustion (high desensitization, low recovery)")
    print("=" * 60)
    out3 = simulate({"n_steps": 2000, "alpha": 0.8, "beta": 0.02,
                     "k_rep": 2.0}, seed=3)
    result3 = analyze_simulation(out3, n_frames=5)
    print_analysis(result3)
