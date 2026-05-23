"""
Interactive T-cell swarm explorer (local desktop window), 2D and 3D.

Controls
--------
  * Sliders (right)  : knobs to turn before a run, grouped by role. Moving one
                       shows what it does + that a re-Run is needed.
  * 2D / 3D radio    : pick the spatial dimension (re-runs on change).
  * Run              : run ONE simulation with the current slider values.
  * Play / Pause     : auto-advance the time dragger so the run animates.
  * Reset            : restore slider defaults.
  * Time dragger     : scrub to any frame instantly (works while paused).
  * Status line      : confirms parameter changes and what each Run used.
  * Mini curve       : mean distance-to-source vs time, with a moving marker.

Run:  python interactive_sim.py   (best from the project venv)
"""

import sys
import platform
import matplotlib

# --- pick a working interactive backend regardless of how we're launched ----
# (avoids "non-GUI backend" / missing-GUI-module errors when run outside venv)
_CANDIDATES = (["macosx", "qtagg", "tkagg"] if platform.system() == "Darwin"
               else ["qtagg", "tkagg", "gtk3agg"])
import matplotlib.pyplot as plt
_BACKEND = None
for _bk in _CANDIDATES:
    try:
        plt.switch_backend(_bk)
        _BACKEND = _bk
        break
    except Exception:
        continue
if _BACKEND is None:
    sys.exit("No interactive Matplotlib backend available.\n"
             "Activate the project venv first:\n"
             "    source .venv/bin/activate && python interactive_sim.py")
print(f"[interactive] using backend: {_BACKEND}")

import threading
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

from simulate import simulate, DEFAULTS, _min_image, to_minutes, to_microns

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
BG       = "#f4f1ec"
PANEL    = "#ffffff"
ACCENT   = "#2a9d8f"
ACCENT_D = "#21867a"
DIRTY    = "#e07a3f"     # "changed, needs re-run" orange
OKGREEN  = "#2f8f4e"
RUN_CLR  = "#43aa8b"
PLAY_CLR = "#577590"
RESET_CLR  = "#d9d5cc"
ANALYZE_CLR= "#7b5ea7"
TEXT     = "#2b2b2b"
FIELD_CMAP = "magma"
MAIN_RECT = [0.055, 0.405, 0.50, 0.455]
CURVE_RECT = [0.055, 0.105, 0.50, 0.085]
TIME_RECT = [0.055, 0.235, 0.50, 0.025]
SCENE_Y = 0.275

# Slider spec per group:  name -> (label, min, max, what-it-does)
GROUPS = [
    ("STEERING — how cells decide where to go", {
        "w_a":     ("Alignment (w_a)",    0.0, 2.0,
                    "how strongly a cell copies its neighbors' direction (flocking)"),
        "w_c":     ("Chemotaxis (w_c)",   0.0, 3.0,
                    "how strongly cells are pulled UP the chemokine gradient toward the source"),
        "eta":     ("Noise (η)",          0.0, 1.5,
                    "random wobble in each cell's turning — high noise destroys order"),
    }),
    ("BIOCHEMISTRY — receptor desensitization", {
        "alpha":   ("Desensitization (α)", 0.0, 1.0,
                    "how fast cells go 'blind' to chemokine in high concentration"),
        "beta":    ("Recovery (β)",        0.0, 0.6,
                    "how fast a cell's sensitivity returns when chemokine is low"),
        "recovery_K": ("Quiet-zone scale", 0.05, 5.0,
                       "how much background chemokine suppresses receptor recovery"),
        "w_escape": ("Escape drive", 0.0, 5.0,
                     "how strongly desensitized cells move down-gradient toward quiet recovery space"),
    }),
    ("CHEMOKINES — sources, secretion, spread", {
        "n_sources":       ("Sources",          1, 4,
                            "how many infection/chemokine source sites are present"),
        "source_distance":  ("Source distance", 0.0, 40.0,
                            "spacing between source sites — close fields merge; far fields act independently"),
        "source_1_strength": ("Source 1 strength", 0.0, 2.0,
                              "chemokine amplitude for source 1"),
        "source_2_strength": ("Source 2 strength", 0.0, 2.0,
                              "chemokine amplitude for source 2"),
        "source_3_strength": ("Source 3 strength", 0.0, 2.0,
                              "chemokine amplitude for source 3"),
        "source_4_strength": ("Source 4 strength", 0.0, 2.0,
                              "chemokine amplitude for source 4"),
        "D_chem":          ("Diffusion",        0.0, 2.0,
                            "how quickly the chemokine field spreads through tissue"),
        "S_chem":          ("Cell secretion",   0.0, 1.0,
                            "how much chemokine the moving cells add to the field"),
        "lambda_chem":     ("Decay",            0.0, 0.2,
                            "how quickly chemokine disappears from the field"),
    }),
    ("MECHANICS — physical forces", {
        "k_rep":   ("Repulsion (k_rep)",   0.0, 30.0,
                    "soft push-apart so cells don't pile up; prevents collapse to a point"),
        "v_drift": ("Drift (v_drift)",     0.0, 1.0,
                    "self-driven motion along a remembered direction — coherent streaming"),
    }),
    ("RUN — simulation length", {
        "n_steps": ("Duration (# steps)",  200, 3000,
                    "how many time steps to simulate (longer = more biological time)"),
    }),
]
# flat lookup: name -> (clean_label, description)
INFO = {n: (lab, desc) for _, items in GROUPS for n, (lab, _lo, _hi, desc) in items.items()}

SCENES = {
    "Base": {
        "desc": "Base swarm: chemotaxis aggregation without desensitization",
        "params": dict(w_a=0.8, w_c=1.0, eta=0.3, alpha=0.0, beta=0.0,
                       w_escape=0.0, n_sources=1, k_rep=5.0, n_steps=2000),
    },
    "Ring": {
        "desc": "Containment ring: desensitization plus escape makes a cordon",
        "params": dict(w_a=0.5, w_c=1.0, eta=0.3, alpha=0.45, beta=0.25,
                       w_escape=2.2, recovery_K=0.63, n_sources=1,
                       k_rep=0.0, n_steps=2000),
    },
    "Osc": {
        "desc": "Healthy oscillation: swarm, self-limit, recover, return",
        "params": dict(w_a=0.8, w_c=1.0, eta=0.3, alpha=0.45, beta=0.25,
                       w_escape=0.5, recovery_K=0.8, n_sources=1,
                       k_rep=8.0, n_steps=2000),
    },
    "Vortex": {
        "desc": "Rotating vortex: high alignment turns the ring into a patrol",
        "params": dict(w_a=2.0, w_c=1.0, eta=0.2, alpha=0.45, beta=0.25,
                       w_escape=2.2, recovery_K=0.63, n_sources=1,
                       k_rep=0.0, n_steps=2000),
    },
    "Exhaust": {
        "desc": "Immune exhaustion: many sources erase quiet recovery zones",
        "params": dict(w_a=0.8, w_c=1.0, eta=0.3, alpha=0.8, beta=0.1,
                       w_escape=0.5, recovery_K=0.5, n_sources=4,
                       source_distance=15.0, k_rep=5.0, n_steps=2000),
    },
    "Rescue": {
        "desc": "Box-size rescue: same immune settings, larger tissue restores quiet space",
        "params": dict(w_a=0.8, w_c=1.0, eta=0.3, alpha=0.8, beta=0.1,
                       w_escape=0.5, recovery_K=0.5, n_sources=4,
                       source_distance=40.0, k_rep=5.0, n_steps=2000),
        "extra": dict(L=100.0),
    },
}


class SwarmExplorer:
    def __init__(self):
        self.out = None
        self.playing = False
        self.dim = int(DEFAULTS["dim"])
        self.ax = None
        self._cbar = None
        self.run_count = 0
        self.dynamic_chemokine = bool(DEFAULTS["dynamic_chemokine"])
        self.extra_params = {}
        self.scene_name = None
        self.field_img = None
        self._build_figure()
        self._build_timer()
        self.run()

    # ------------------------------------------------------------------ UI
    def _build_figure(self):
        self.fig = plt.figure(figsize=(13.8, 7.6), facecolor=BG)
        self.fig.canvas.manager.set_window_title("T-cell Swarm Explorer")
        self.fig.text(0.04, 0.955, "T-cell Swarm Explorer", fontsize=18,
                      weight="bold", color=TEXT)
        self.fig.text(0.04, 0.93, "pick 2D/3D · set knobs · Run · Play",
                      fontsize=10.5, color="#777")

        # ---- demo scene presets ----
        self.fig.text(0.055, SCENE_Y + 0.043, "SCENES", fontsize=8.5,
                      weight="bold", color=ACCENT_D)
        self.scene_buttons = {}
        bx, bw, gap = 0.055, 0.074, 0.009
        for i, name in enumerate(SCENES):
            b = self._button([bx + i * (bw + gap), SCENE_Y, bw, 0.036],
                             name, RESET_CLR, text_color=TEXT)
            b.label.set_fontsize(8.5)
            b.on_clicked(lambda _e, scene=name: self.apply_scene(scene))
            self.scene_buttons[name] = b

        # mini aggregation curve, kept away from the control stack
        self.ax_curve = self.fig.add_axes(CURVE_RECT, facecolor=PANEL)
        self._style_axis(self.ax_curve)

        # ---- sliders, grouped, with live descriptions on change ----
        self.widgets = {}
        self.group_headers = []
        self.source_strength_names = {
            "source_1_strength", "source_2_strength",
            "source_3_strength", "source_4_strength",
        }
        x, w = 0.685, 0.235
        self.ctrl_x, self.ctrl_w = x, w
        y = 0.92
        for header, items in GROUPS:
            txt = self.fig.text(x, y + 0.012, header, fontsize=8.5, weight="bold",
                                color=ACCENT_D)
            self.group_headers.append((txt, list(items.keys())))
            y -= 0.026
            for name, (label, lo, hi, _desc) in items.items():
                ax_s = self.fig.add_axes([x, y, w, 0.019], facecolor="#eceae4")
                step = 1 if name in ("n_steps", "n_sources") else None
                s = Slider(ax_s, label, lo, hi, valinit=float(DEFAULTS[name]),
                           valstep=step, color=ACCENT)
                s.label.set_fontsize(8.5); s.label.set_color(TEXT)
                s.valtext.set_fontsize(8.5)
                s.on_changed(self._make_change_cb(name))
                self.widgets[name] = s
                y -= 0.031
            y -= 0.008

        # ---- buttons row ----
        by = y - 0.025
        self.btn_run   = self._button([0.685, by, 0.072, 0.05], "Run",   RUN_CLR)
        self.btn_play  = self._button([0.765, by, 0.085, 0.05], "▶ Play", PLAY_CLR)
        self.btn_reset = self._button([0.858, by, 0.062, 0.05], "Reset", RESET_CLR,
                                      text_color=TEXT)
        self.btn_analyze = self._button([0.685, by - 0.062, 0.235, 0.048],
                                        "Analyze with Claude", ANALYZE_CLR)
        self.btn_analyze.label.set_fontsize(9.5)
        selector_y = by - 0.132
        self.btn_chem = self._button([0.800, selector_y, 0.120, 0.072],
                                     "Chem: static", RESET_CLR, text_color=TEXT)
        self.btn_run.on_clicked(lambda _e: self.run())
        self.btn_play.on_clicked(lambda _e: self.toggle_play())
        self.btn_reset.on_clicked(lambda _e: self.reset())
        self.btn_analyze.on_clicked(lambda _e: self.analyze())
        self.btn_chem.on_clicked(lambda _e: self.toggle_dynamic_chemokine())

        # ---- dimension radio ----
        self.view_label = self.fig.text(0.685, selector_y + 0.080, "VIEW", fontsize=8.5,
                                        weight="bold", color=ACCENT_D)
        self.ax_radio = self.fig.add_axes([0.685, selector_y, 0.105, 0.075], facecolor="#eceae4")
        self.radio = RadioButtons(self.ax_radio, ("2D", "3D"),
                                  active=0 if self.dim == 2 else 1, activecolor=ACCENT)
        for lbl in self.radio.labels:
            lbl.set_fontsize(10)
        self.radio.on_clicked(self._set_dim)

        # ---- time dragger ----
        ax_t = self.fig.add_axes(TIME_RECT, facecolor="#eceae4")
        self.time_slider = Slider(ax_t, "time", 0, 1, valinit=0, valstep=1, color=PLAY_CLR)
        self.time_slider.label.set_color(TEXT)
        self.time_slider.on_changed(self._draw_frame)

        # ---- status line (parameter-change + run feedback) ----
        self.status = self.fig.text(0.055, 0.055, "", fontsize=10, color=TEXT)
        self.hint = self.fig.text(0.055, 0.028, "", fontsize=9.5, color=DIRTY,
                                  style="italic")
        self._refresh_source_controls(draw=False)

    def _style_axis(self, ax):
        for sp in ax.spines.values():
            sp.set_color("#cfcabd")
        ax.tick_params(colors="#777", labelsize=8)

    def _button(self, rect, label, color, text_color="white"):
        b = Button(self.fig.add_axes(rect), label, color=color, hovercolor=ACCENT_D)
        b.label.set_fontsize(10.5); b.label.set_weight("bold")
        b.label.set_color(text_color)
        return b

    def _refresh_source_controls(self, draw=True):
        """Show only source-strength sliders that correspond to active sources."""
        if not hasattr(self, "widgets") or "n_sources" not in self.widgets:
            return
        active = int(round(self.widgets["n_sources"].val))
        x, w = self.ctrl_x, self.ctrl_w
        y = 0.92
        for header_text, names in self.group_headers:
            header_text.set_position((x, y + 0.012))
            y -= 0.026
            for name in names:
                slider = self.widgets[name]
                source_idx = int(name.split("_")[1]) if name in self.source_strength_names else None
                visible = source_idx is None or source_idx <= active
                slider.ax.set_visible(visible)
                if visible:
                    slider.ax.set_position([x, y, w, 0.019])
                    y -= 0.031
            y -= 0.008

        by = y - 0.025
        self.btn_run.ax.set_position([0.685, by, 0.072, 0.05])
        self.btn_play.ax.set_position([0.765, by, 0.085, 0.05])
        self.btn_reset.ax.set_position([0.858, by, 0.062, 0.05])
        self.btn_analyze.ax.set_position([0.685, by - 0.062, 0.235, 0.048])
        selector_y = by - 0.132
        self.btn_chem.ax.set_position([0.800, selector_y, 0.120, 0.072])
        self.view_label.set_position((0.685, selector_y + 0.080))
        self.ax_radio.set_position([0.685, selector_y, 0.105, 0.075])
        if draw:
            self.fig.canvas.draw_idle()

    def _build_timer(self):
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self._advance)

    # --------------------------------------------------------------- feedback
    def _make_change_cb(self, name):
        def _cb(val):
            label, desc = INFO[name]
            shown = f"{int(val)}" if name in ("n_steps", "n_sources") else f"{val:.2f}"
            # status: what this knob does + its new value
            self.status.set_text(f"{label} = {shown}")
            self.status.set_color(ACCENT_D)
            self.status.set_weight("bold")
            self._set_desc(desc)
            if name == "n_sources":
                self._refresh_source_controls(draw=False)
            # hint: changes are pending until Run
            self.hint.set_text("● parameters changed — press  Run  to apply")
            self.hint.set_color(DIRTY)
            self.fig.canvas.draw_idle()
        return _cb

    def _set_desc(self, desc):
        if not hasattr(self, "_desc_txt"):
            self._desc_txt = self.fig.text(0.055, 0.008, "", fontsize=9.0,
                                           color="#666", style="italic")
        self._desc_txt.set_text(f"—  {desc}")

    # --------------------------------------------------------------- actions
    def _set_dim(self, label):
        self.dim = 2 if label == "2D" else 3
        self.extra_params = {}
        self.scene_name = None
        self.run()

    def _params_from_sliders(self):
        p = {name: w.val for name, w in self.widgets.items()}
        p["n_steps"] = int(p["n_steps"])
        p["n_sources"] = int(p["n_sources"])
        p["dim"] = self.dim
        p["dynamic_chemokine"] = self.dynamic_chemokine
        p.update(self.extra_params)
        return p

    def apply_scene(self, name):
        self._pause()
        spec = SCENES[name]
        params = spec["params"]
        self.scene_name = name
        self.extra_params = dict(spec.get("extra", {}))
        self.dynamic_chemokine = bool(spec.get("dynamic_chemokine", False))
        self._update_chem_button()

        for key, slider in self.widgets.items():
            target = params.get(key, DEFAULTS[key])
            slider.set_val(target)
        self._refresh_source_controls(draw=False)
        self._set_desc(spec["desc"])
        self.run()

    def run(self):
        self._pause()
        params = self._params_from_sliders()
        self.out = simulate(params, seed=0)
        self.run_count += 1
        self._draw_background()
        self._draw_curve()
        n = self.out["positions"].shape[0]
        self.time_slider.valmax = n - 1
        self.time_slider.ax.set_xlim(0, n - 1)
        self.time_slider.set_val(0)
        self._draw_frame(0)
        # status: confirm the run + the exact params it used
        self.status.set_text(
            f"✓ Run #{self.run_count} applied   ·   {self.dim}D"
            + (f"   ·   {self.scene_name}" if self.scene_name else ""))
        self.status.set_color(OKGREEN); self.status.set_weight("bold")
        if hasattr(self, "_desc_txt"):
            self._desc_txt.set_text(
                f"—  w_a={params['w_a']:.2f}  w_c={params['w_c']:.2f}  "
                f"η={params['eta']:.2f}  α={params['alpha']:.2f}  β={params['beta']:.3f}  "
                f"K={params['recovery_K']:.2f}  escape={params['w_escape']:.2f}  "
                f"sources={params['n_sources']}  d={params['source_distance']:.1f}  "
                f"k_rep={params['k_rep']:.0f}  v_drift={params['v_drift']:.2f}  "
                f"· {params['n_steps']} steps")
        self.hint.set_text("")
        self.fig.canvas.draw_idle()

    def reset(self):
        self._pause()
        for w in self.widgets.values():
            w.reset()
        self.dynamic_chemokine = bool(DEFAULTS["dynamic_chemokine"])
        self.extra_params = {}
        self.scene_name = None
        self._update_chem_button()
        self._refresh_source_controls(draw=False)
        self.status.set_text("↺ defaults restored — press Run to apply")
        self.status.set_color(TEXT); self.status.set_weight("normal")
        self.hint.set_text("● parameters changed — press  Run  to apply")
        self.hint.set_color(DIRTY)
        self.fig.canvas.draw_idle()

    def toggle_dynamic_chemokine(self):
        self.dynamic_chemokine = not self.dynamic_chemokine
        self.extra_params = {}
        self.scene_name = None
        self._update_chem_button()
        self.status.set_text("Chemokine field mode changed")
        self.status.set_color(ACCENT_D); self.status.set_weight("bold")
        self._set_desc("dynamic mode diffuses, decays, and is secreted by cells; static mode is analytic")
        self.hint.set_text("● parameters changed — press  Run  to apply")
        self.hint.set_color(DIRTY)
        self.fig.canvas.draw_idle()

    def analyze(self):
        if self.out is None:
            self.status.set_text("Run a simulation first, then Analyze")
            self.status.set_color(DIRTY)
            self.fig.canvas.draw_idle()
            return
        self.btn_analyze.label.set_text("Analyzing...")
        self.btn_analyze.ax.set_facecolor(RESET_CLR)
        self.fig.canvas.draw_idle()
        out_snapshot = self.out

        def _run():
            try:
                from analysis_service import analyze_simulation, print_analysis
                result = analyze_simulation(out_snapshot, n_frames=6)
                print_analysis(result)
                infect = "INFECTION" if result["infection_detected"] else "no infection"
                self.status.set_text(
                    f"Claude: {infect}  ·  {result['inflammation_level']} inflammation"
                    f"  ·  {result['phase']}")
                self.status.set_color(ANALYZE_CLR)
                self.status.set_weight("bold")
            except Exception as exc:
                print(f"[analysis] error: {exc}")
                self.status.set_text(f"Analysis error — check terminal")
                self.status.set_color(DIRTY)
            finally:
                self.btn_analyze.label.set_text("Analyze with Claude")
                self.btn_analyze.ax.set_facecolor(ANALYZE_CLR)
                self.fig.canvas.draw_idle()

        threading.Thread(target=_run, daemon=True).start()

    def _update_chem_button(self):
        label = "Chem: dynamic" if self.dynamic_chemokine else "Chem: static"
        self.btn_chem.label.set_text(label)

    def toggle_play(self):
        self._pause() if self.playing else self._play()

    def _play(self):
        if self.out is None:
            return
        if int(self.time_slider.val) >= self.time_slider.valmax:
            self.time_slider.set_val(0)
        self.playing = True
        self.btn_play.label.set_text("❚❚ Pause")
        self.timer.start()
        self.fig.canvas.draw_idle()

    def _pause(self):
        self.playing = False
        self.timer.stop()
        self.btn_play.label.set_text("▶ Play")
        self.fig.canvas.draw_idle()

    def _advance(self):
        if not self.playing:   # timer can fire once after stop() on macOS
            return
        t = int(self.time_slider.val)
        nxt = t + max(1, int(self.out["positions"].shape[0] / 400))
        if nxt > self.time_slider.valmax:
            nxt = 0
        self.time_slider.set_val(nxt)

    # --------------------------------------------------------------- drawing
    def _draw_background(self):
        p = self.out["params"]
        L_um = to_microns(self.out["box_size"], p)

        if self._cbar is not None:
            self._cbar.remove(); self._cbar = None
        if self.ax is not None:
            self.ax.remove()
        strengths = np.array(
            self.out.get("source_strengths", [1.0] * len(self.out.get("source_positions", []))))
        source_sizes = 90.0 + 190.0 * np.clip(strengths, 0.0, 2.0) / 2.0

        if self.dim == 2:
            self.ax = self.fig.add_axes(MAIN_RECT, facecolor=PANEL)
            self._style_axis(self.ax)
            field = self.out["chemokine_field"]
            field0 = field[0] if field.ndim == 3 else field
            self.field_img = self.ax.imshow(field0, extent=[0, L_um, 0, L_um],
                                            origin="lower", cmap=FIELD_CMAP, alpha=0.85)
            self.scat = self.ax.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=1,
                                        s=20, edgecolors="white", linewidths=0.3)
            srcs_um = to_microns(
                np.array(self.out.get("source_positions", [self.out["source_position"]])), p)
            self.ax.scatter(srcs_um[:, 0], srcs_um[:, 1], marker="*", s=source_sizes,
                            c="#e63946",
                            edgecolors="white", linewidths=1.2, zorder=5)
            self.ax.set_xlim(0, L_um); self.ax.set_ylim(0, L_um)
            self.ax.set_xlabel("x (µm)", color=TEXT); self.ax.set_ylabel("y (µm)", color=TEXT)
        else:
            self.ax = self.fig.add_axes(MAIN_RECT, projection="3d")
            self.ax.set_facecolor(PANEL)
            self.scat = self.ax.scatter(
                np.zeros(p["N"]), np.zeros(p["N"]), np.zeros(p["N"]),
                c=np.ones(p["N"]), cmap="viridis", vmin=0, vmax=1,
                s=18, edgecolors="white", linewidths=0.2, depthshade=True)
            srcs_um = to_microns(
                np.array(self.out.get("source_positions", [self.out["source_position"]])), p)
            self.ax.scatter(srcs_um[:, 0], srcs_um[:, 1], srcs_um[:, 2],
                            marker="*", s=source_sizes, c="#e63946",
                            edgecolors="white", linewidths=1.2)
            self.ax.set_xlim(0, L_um); self.ax.set_ylim(0, L_um); self.ax.set_zlim(0, L_um)
            self.ax.set_xlabel("x (µm)", fontsize=8); self.ax.set_ylabel("y (µm)", fontsize=8)
            self.ax.set_zlabel("z (µm)", fontsize=8)
            try:
                self.ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass

        self._cbar = self.fig.colorbar(self.scat, ax=self.ax, fraction=0.045, pad=0.08)
        self._cbar.set_label("sensitivity  s", color=TEXT)
        self._cbar.ax.tick_params(labelsize=8, colors="#777")

    def _draw_curve(self):
        p = self.out["params"]
        srcs = np.array(self.out.get("source_positions", [self.out["source_position"]]))
        L = self.out["box_size"]
        d = _min_image(self.out["positions"][:, :, None, :] - srcs[None, None, :, :], L)
        nearest = np.linalg.norm(d, axis=3).min(axis=2)
        self._mean_d_um = to_microns(nearest.mean(axis=1), p)
        self._t_min = to_minutes(np.arange(self.out["positions"].shape[0]), p)

        self.ax_curve.clear()
        self._style_axis(self.ax_curve)
        self.ax_curve.fill_between(self._t_min, self._mean_d_um, color=ACCENT, alpha=0.15)
        self.ax_curve.plot(self._t_min, self._mean_d_um, color=ACCENT_D, lw=1.5)
        self.ax_curve.set_xlabel("time (min)", fontsize=7.5, color=TEXT, labelpad=1)
        self.ax_curve.set_ylabel("dist (µm)", fontsize=7.5, color=TEXT, labelpad=1)
        self.ax_curve.set_title(f"swarm aggregation  ({self.dim}D)", fontsize=8.5, color=TEXT, pad=2)
        self.ax_curve.tick_params(labelsize=7)
        self._time_marker = self.ax_curve.axvline(0, color="#e63946", lw=1.4)

    def _draw_frame(self, val):
        if self.out is None:
            return
        t = int(val)
        p = self.out["params"]
        pos_um = to_microns(self.out["positions"][t], p)
        sens = self.out["sensitivity"][t]
        if self.dim == 2:
            field = self.out["chemokine_field"]
            if field.ndim == 3 and self.field_img is not None:
                self.field_img.set_data(field[t])
            self.scat.set_offsets(pos_um)
            self.scat.set_array(sens)
        else:
            self.scat._offsets3d = (pos_um[:, 0], pos_um[:, 1], pos_um[:, 2])
            self.scat.set_array(sens)
        self.ax.set_title(f"t = {to_minutes(t, p):.1f} min      "
                          f"step {t} / {self.out['positions'].shape[0] - 1}",
                          color=TEXT, fontsize=11)
        if hasattr(self, "_time_marker"):
            self._time_marker.set_xdata([self._t_min[t], self._t_min[t]])
        self.fig.canvas.draw_idle()


def launch():
    app = SwarmExplorer()
    plt.show()
    return app


if __name__ == "__main__":
    launch()
