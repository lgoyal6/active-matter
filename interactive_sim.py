"""
Interactive T-cell swarm explorer (local desktop window).

Controls
--------
  * Sliders (right)  : knobs to turn before a run, grouped by role.
  * Run              : run ONE simulation with the current slider values.
  * Play / Pause     : auto-advance the time dragger so the run animates.
  * Reset            : restore slider defaults.
  * Time dragger     : scrub to any frame instantly (works while paused).
  * Mini curve       : mean distance-to-source vs time, with a moving marker.

Workflow: set sliders -> Run -> Play (or drag Time) to watch it evolve.

Run:  python interactive_sim.py
Needs an interactive Matplotlib backend (macosx / TkAgg / QtAgg) -- a real
window, not headless. For Colab use ipywidgets instead.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from simulate import simulate, DEFAULTS, _min_image, to_minutes, to_microns

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
BG       = "#f4f1ec"     # window background (warm paper)
PANEL    = "#ffffff"     # plot background
ACCENT   = "#2a9d8f"     # slider fill / highlights (teal)
ACCENT_D = "#21867a"     # darker teal (hover)
RUN_CLR  = "#43aa8b"     # run button
PLAY_CLR = "#577590"     # play button
RESET_CLR= "#d9d5cc"     # reset button
TEXT     = "#2b2b2b"
FIELD_CMAP = "rocket" if "rocket" in plt.colormaps() else "magma"

# Sliders grouped by physical role:  name -> (label, min, max)
GROUPS = [
    ("STEERING", {
        "w_a":     ("alignment  w_a",   0.0, 2.0),
        "w_c":     ("chemotaxis w_c",   0.0, 3.0),
        "eta":     ("noise  η",         0.0, 1.5),
    }),
    ("BIOCHEMISTRY", {
        "alpha":   ("desensitize  α",   0.0, 1.0),
        "beta":    ("recovery  β",      0.0, 0.2),
    }),
    ("MECHANICS", {
        "k_rep":   ("repulsion  k_rep", 0.0, 30.0),
        "v_drift": ("drift  v_drift",   0.0, 1.0),
    }),
    ("RUN LENGTH", {
        "n_steps": ("# steps",          200, 3000),
    }),
]


class SwarmExplorer:
    def __init__(self):
        self.out = None
        self.playing = False
        self._build_figure()
        self._build_timer()
        self.run()                         # initial run so there's something to see

    # ------------------------------------------------------------------ UI
    def _build_figure(self):
        self.fig = plt.figure(figsize=(13.5, 7.4), facecolor=BG)
        self.fig.text(0.04, 0.95, "T-cell Swarm Explorer", fontsize=18,
                      weight="bold", color=TEXT)
        self.fig.text(0.04, 0.925, "set the knobs · Run · Play to animate",
                      fontsize=10.5, color="#777")

        # main scatter panel
        self.ax = self.fig.add_axes([0.05, 0.30, 0.50, 0.56], facecolor=PANEL)
        self._style_axis(self.ax)

        # mini aggregation curve
        self.ax_curve = self.fig.add_axes([0.68, 0.10, 0.27, 0.17], facecolor=PANEL)
        self._style_axis(self.ax_curve)

        # ---- sliders, grouped with section headers ----
        self.widgets = {}
        x, w = 0.685, 0.235
        y = 0.90
        for header, items in GROUPS:
            self.fig.text(x, y + 0.012, header, fontsize=9, weight="bold",
                          color=ACCENT_D)
            y -= 0.028
            for name, (label, lo, hi) in items.items():
                ax_s = self.fig.add_axes([x, y, w, 0.022], facecolor="#eceae4")
                step = 1 if name == "n_steps" else None
                s = Slider(ax_s, label, lo, hi, valinit=float(DEFAULTS[name]),
                           valstep=step, color=ACCENT)
                s.label.set_fontsize(8.5)
                s.label.set_color(TEXT)
                s.valtext.set_fontsize(8.5)
                self.widgets[name] = s
                y -= 0.042
            y -= 0.018

        # ---- buttons row ----
        by = y - 0.005
        self.btn_run   = self._button([0.685, by, 0.072, 0.05], "Run",   RUN_CLR)
        self.btn_play  = self._button([0.765, by, 0.085, 0.05], "▶ Play", PLAY_CLR)
        self.btn_reset = self._button([0.858, by, 0.062, 0.05], "Reset", RESET_CLR,
                                      text_color=TEXT)
        self.btn_run.on_clicked(lambda _e: self.run())
        self.btn_play.on_clicked(lambda _e: self.toggle_play())
        self.btn_reset.on_clicked(lambda _e: self.reset())

        # ---- time dragger ----
        ax_t = self.fig.add_axes([0.05, 0.17, 0.50, 0.025], facecolor="#eceae4")
        self.time_slider = Slider(ax_t, "time", 0, 1, valinit=0, valstep=1,
                                  color=PLAY_CLR)
        self.time_slider.label.set_color(TEXT)
        self.time_slider.on_changed(self._draw_frame)

    def _style_axis(self, ax):
        for sp in ax.spines.values():
            sp.set_color("#cfcabd")
        ax.tick_params(colors="#777", labelsize=8)

    def _button(self, rect, label, color, text_color="white"):
        b = Button(self.fig.add_axes(rect), label, color=color,
                   hovercolor=ACCENT_D)
        b.label.set_fontsize(10.5)
        b.label.set_weight("bold")
        b.label.set_color(text_color)
        return b

    def _build_timer(self):
        # drives Play: fires repeatedly, nudging the time dragger forward
        self.timer = self.fig.canvas.new_timer(interval=40)  # ~25 fps
        self.timer.add_callback(self._advance)

    # --------------------------------------------------------------- actions
    def _params_from_sliders(self):
        p = {name: w.val for name, w in self.widgets.items()}
        p["n_steps"] = int(p["n_steps"])
        return p

    def run(self):
        """Run one simulation with current slider values; reset the dragger."""
        self._pause()
        self.out = simulate(self._params_from_sliders(), seed=0)
        self._draw_background()
        self._draw_curve()
        n = self.out["positions"].shape[0]
        self.time_slider.valmax = n - 1
        self.time_slider.ax.set_xlim(0, n - 1)
        self.time_slider.set_val(0)
        self._draw_frame(0)
        self.fig.canvas.draw_idle()

    def reset(self):
        self._pause()
        for w in self.widgets.values():
            w.reset()

    def toggle_play(self):
        self._pause() if self.playing else self._play()

    def _play(self):
        if self.out is None:
            return
        # if we're at the end, rewind before playing
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
        """Timer callback: step the dragger forward, loop at the end."""
        t = int(self.time_slider.val)
        nxt = t + max(1, int(self.out["positions"].shape[0] / 400))  # ~400 frames/run
        if nxt > self.time_slider.valmax:
            nxt = 0
        self.time_slider.set_val(nxt)        # triggers _draw_frame

    # --------------------------------------------------------------- drawing
    def _draw_background(self):
        p = self.out["params"]
        L_um = to_microns(self.out["box_size"], p)
        src_um = to_microns(np.array(self.out["source_position"]), p)

        self.ax.clear()
        self._style_axis(self.ax)
        self.ax.imshow(self.out["chemokine_field"], extent=[0, L_um, 0, L_um],
                       origin="lower", cmap=FIELD_CMAP, alpha=0.85)
        self.scat = self.ax.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=1,
                                    s=20, edgecolors="white", linewidths=0.3)
        self.ax.scatter(*src_um, marker="*", s=320, c="#e63946",
                        edgecolors="white", linewidths=1.2, zorder=5)
        self.ax.set_xlim(0, L_um); self.ax.set_ylim(0, L_um)
        self.ax.set_xlabel("x (µm)", color=TEXT); self.ax.set_ylabel("y (µm)", color=TEXT)
        if not hasattr(self, "_cbar"):
            self._cbar = self.fig.colorbar(self.scat, ax=self.ax,
                                           fraction=0.045, pad=0.02)
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
        self.ax_curve.set_ylabel("dist to source (µm)", fontsize=8, color=TEXT)
        self.ax_curve.set_title("swarm aggregation", fontsize=9, color=TEXT)
        self._time_marker = self.ax_curve.axvline(0, color="#e63946", lw=1.4)

    def _draw_frame(self, val):
        if self.out is None:
            return
        t = int(val)
        p = self.out["params"]
        self.scat.set_offsets(to_microns(self.out["positions"][t], p))
        self.scat.set_array(self.out["sensitivity"][t])
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
