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
RESET_CLR= "#d9d5cc"
TEXT     = "#2b2b2b"
FIELD_CMAP = "magma"
MAIN_RECT = [0.05, 0.32, 0.50, 0.54]

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
        "beta":    ("Recovery (β)",        0.0, 0.2,
                    "how fast a cell's sensitivity returns when chemokine is low"),
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


class SwarmExplorer:
    def __init__(self):
        self.out = None
        self.playing = False
        self.dim = int(DEFAULTS["dim"])
        self.ax = None
        self._cbar = None
        self.run_count = 0
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

        # mini aggregation curve
        self.ax_curve = self.fig.add_axes([0.68, 0.10, 0.27, 0.15], facecolor=PANEL)
        self._style_axis(self.ax_curve)

        # ---- sliders, grouped, with live descriptions on change ----
        self.widgets = {}
        x, w = 0.685, 0.235
        y = 0.92
        for header, items in GROUPS:
            self.fig.text(x, y + 0.012, header, fontsize=8.5, weight="bold",
                          color=ACCENT_D)
            y -= 0.030
            for name, (label, lo, hi, _desc) in items.items():
                ax_s = self.fig.add_axes([x, y, w, 0.022], facecolor="#eceae4")
                step = 1 if name == "n_steps" else None
                s = Slider(ax_s, label, lo, hi, valinit=float(DEFAULTS[name]),
                           valstep=step, color=ACCENT)
                s.label.set_fontsize(8.5); s.label.set_color(TEXT)
                s.valtext.set_fontsize(8.5)
                s.on_changed(self._make_change_cb(name))
                self.widgets[name] = s
                y -= 0.040
            y -= 0.016

        # ---- buttons row ----
        by = y - 0.005
        self.btn_run   = self._button([0.685, by, 0.072, 0.05], "Run",   RUN_CLR)
        self.btn_play  = self._button([0.765, by, 0.085, 0.05], "▶ Play", PLAY_CLR)
        self.btn_reset = self._button([0.858, by, 0.062, 0.05], "Reset", RESET_CLR,
                                      text_color=TEXT)
        self.btn_run.on_clicked(lambda _e: self.run())
        self.btn_play.on_clicked(lambda _e: self.toggle_play())
        self.btn_reset.on_clicked(lambda _e: self.reset())

        # ---- dimension radio ----
        self.fig.text(0.685, by - 0.03, "VIEW", fontsize=8.5, weight="bold", color=ACCENT_D)
        ax_radio = self.fig.add_axes([0.685, by - 0.115, 0.10, 0.075], facecolor="#eceae4")
        self.radio = RadioButtons(ax_radio, ("2D", "3D"),
                                  active=0 if self.dim == 2 else 1, activecolor=ACCENT)
        for lbl in self.radio.labels:
            lbl.set_fontsize(10)
        self.radio.on_clicked(self._set_dim)

        # ---- time dragger ----
        ax_t = self.fig.add_axes([0.05, 0.19, 0.50, 0.025], facecolor="#eceae4")
        self.time_slider = Slider(ax_t, "time", 0, 1, valinit=0, valstep=1, color=PLAY_CLR)
        self.time_slider.label.set_color(TEXT)
        self.time_slider.on_changed(self._draw_frame)

        # ---- status line (parameter-change + run feedback) ----
        self.status = self.fig.text(0.05, 0.085, "", fontsize=10, color=TEXT)
        self.hint = self.fig.text(0.05, 0.045, "", fontsize=9.5, color=DIRTY,
                                  style="italic")

    def _style_axis(self, ax):
        for sp in ax.spines.values():
            sp.set_color("#cfcabd")
        ax.tick_params(colors="#777", labelsize=8)

    def _button(self, rect, label, color, text_color="white"):
        b = Button(self.fig.add_axes(rect), label, color=color, hovercolor=ACCENT_D)
        b.label.set_fontsize(10.5); b.label.set_weight("bold")
        b.label.set_color(text_color)
        return b

    def _build_timer(self):
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self._advance)

    # --------------------------------------------------------------- feedback
    def _make_change_cb(self, name):
        def _cb(val):
            label, desc = INFO[name]
            shown = f"{int(val)}" if name == "n_steps" else f"{val:.2f}"
            # status: what this knob does + its new value
            self.status.set_text(f"{label} = {shown}")
            self.status.set_color(ACCENT_D)
            self.status.set_weight("bold")
            self._set_desc(desc)
            # hint: changes are pending until Run
            self.hint.set_text("● parameters changed — press  Run  to apply")
            self.hint.set_color(DIRTY)
            self.fig.canvas.draw_idle()
        return _cb

    def _set_desc(self, desc):
        if not hasattr(self, "_desc_txt"):
            self._desc_txt = self.fig.text(0.20, 0.085, "", fontsize=9.5,
                                           color="#666", style="italic")
        self._desc_txt.set_text(f"—  {desc}")

    # --------------------------------------------------------------- actions
    def _set_dim(self, label):
        self.dim = 2 if label == "2D" else 3
        self.run()

    def _params_from_sliders(self):
        p = {name: w.val for name, w in self.widgets.items()}
        p["n_steps"] = int(p["n_steps"])
        p["dim"] = self.dim
        return p

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
            f"✓ Run #{self.run_count} applied   ·   {self.dim}D")
        self.status.set_color(OKGREEN); self.status.set_weight("bold")
        if hasattr(self, "_desc_txt"):
            self._desc_txt.set_text(
                f"—  w_a={params['w_a']:.2f}  w_c={params['w_c']:.2f}  "
                f"η={params['eta']:.2f}  α={params['alpha']:.2f}  β={params['beta']:.3f}  "
                f"k_rep={params['k_rep']:.0f}  v_drift={params['v_drift']:.2f}  "
                f"· {params['n_steps']} steps")
        self.hint.set_text("")
        self.fig.canvas.draw_idle()

    def reset(self):
        self._pause()
        for w in self.widgets.values():
            w.reset()
        self.status.set_text("↺ defaults restored — press Run to apply")
        self.status.set_color(TEXT); self.status.set_weight("normal")
        self.hint.set_text("● parameters changed — press  Run  to apply")
        self.hint.set_color(DIRTY)
        self.fig.canvas.draw_idle()

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
        t = int(self.time_slider.val)
        nxt = t + max(1, int(self.out["positions"].shape[0] / 400))
        if nxt > self.time_slider.valmax:
            nxt = 0
        self.time_slider.set_val(nxt)

    # --------------------------------------------------------------- drawing
    def _draw_background(self):
        p = self.out["params"]
        L_um = to_microns(self.out["box_size"], p)
        src_um = to_microns(np.array(self.out["source_position"]), p)

        if self._cbar is not None:
            self._cbar.remove(); self._cbar = None
        if self.ax is not None:
            self.ax.remove()

        if self.dim == 2:
            self.ax = self.fig.add_axes(MAIN_RECT, facecolor=PANEL)
            self._style_axis(self.ax)
            self.ax.imshow(self.out["chemokine_field"], extent=[0, L_um, 0, L_um],
                           origin="lower", cmap=FIELD_CMAP, alpha=0.85)
            self.scat = self.ax.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=1,
                                        s=20, edgecolors="white", linewidths=0.3)
            self.ax.scatter(*src_um, marker="*", s=320, c="#e63946",
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
            self.ax.scatter(*src_um, marker="*", s=320, c="#e63946",
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
        src = np.array(self.out["source_position"]); L = self.out["box_size"]
        d = _min_image(self.out["positions"] - src, L)
        self._mean_d_um = to_microns(np.linalg.norm(d, axis=2).mean(axis=1), p)
        self._t_min = to_minutes(np.arange(self.out["positions"].shape[0]), p)

        self.ax_curve.clear()
        self._style_axis(self.ax_curve)
        self.ax_curve.fill_between(self._t_min, self._mean_d_um, color=ACCENT, alpha=0.15)
        self.ax_curve.plot(self._t_min, self._mean_d_um, color=ACCENT_D, lw=1.5)
        self.ax_curve.set_xlabel("time (min)", fontsize=8, color=TEXT)
        self.ax_curve.set_ylabel("dist (µm)", fontsize=8, color=TEXT)
        self.ax_curve.set_title(f"swarm aggregation  ({self.dim}D)", fontsize=9, color=TEXT)
        self._time_marker = self.ax_curve.axvline(0, color="#e63946", lw=1.4)

    def _draw_frame(self, val):
        if self.out is None:
            return
        t = int(val)
        p = self.out["params"]
        pos_um = to_microns(self.out["positions"][t], p)
        sens = self.out["sensitivity"][t]
        if self.dim == 2:
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
