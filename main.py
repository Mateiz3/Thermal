import os

import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

# Prefer a GUI backend when display + Tk stack are actually available.
# If not, force Agg so the script falls back to saving PNGs.
if os.environ.get("DISPLAY"):
    try:
        import tkinter  # noqa: F401
        from PIL import ImageTk  # noqa: F401
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

def calculate_temperature(x, t_ambient, theta_0, alpha, P, chi, S):
    # the 'a' parameter
    a = np.sqrt((alpha * P) / (chi * S))

    # theta (temperature difference)
    theta = theta_0 * np.exp(-a * x)

    # temperature is theta + ambient temperature
    return theta + t_ambient

# --- Experimental Data ---

t_ambient = 24.0          # Ambient temperature in Celsius
theta_0 = 264.0           # Temperature difference at the hot end (x=0)

# Thermal conductivities (W/(m K))
MATERIALS = {
    "Copper": 384.1,
    "Aluminum": 205.0,
    "Brass": 109.0,
    "Steel": 50.0,
    "Silver": 429.0,
}
default_material = "Copper"
DEFAULT_CHI_STUDIED = 373.482  # Calculated average for the studied bar (reset target when field is cleared)
chi_studied = DEFAULT_CHI_STUDIED

# Assumptions for physical dimensions to make the visualization work realistically
radius = 0.01             # 1 cm radius
P = 2 * np.pi * radius    # Perimeter
S = np.pi * radius**2     # Cross-section area
alpha = 15.0              # convective heat transfer coefficient for air
default_length_cm = 30.0
heat_levels = 9
temp_min_fixed = 0.0
temp_max_fixed = 460.0
fixed_temp_edges = np.linspace(temp_min_fixed, temp_max_fixed, heat_levels + 1)


def band_colors(levels=heat_levels):
    values = np.linspace(0.0, 1.0, levels)
    colors = np.zeros((levels, 3))
    # Blue (cold) -> red (hot) gradient.
    colors[:, 0] = 0.10 + 0.90 * values
    colors[:, 1] = 0.05 + 0.10 * (1.0 - values)
    colors[:, 2] = 0.95 - 0.90 * values
    return colors


def red_heat_strip(temps, temp_edges):
    """Create stepped red strip using explicit temperature bins."""
    palette = band_colors(len(temp_edges) - 1)
    band_idx = np.digitize(temps, temp_edges[1:-1], right=False)
    band_idx = np.clip(band_idx, 0, len(palette) - 1)
    strip_row = palette[band_idx]
    strip = np.repeat(strip_row[np.newaxis, :, :], 20, axis=0)
    return strip


def draw_temp_legend_lists(ax, temp_edges_local, x0, title, y_top, swatches, labels):
    """Clear and redraw a temperature-band legend (swatches = Rectangle patches, labels = Text)."""
    for swatch in swatches:
        swatch.remove()
    for label in labels:
        label.remove()
    swatches.clear()
    labels.clear()

    colors = band_colors(len(temp_edges_local) - 1)
    title_text = ax.text(x0, y_top, title, fontsize=9, fontweight="bold", va="bottom")
    labels.append(title_text)
    for i in range(len(temp_edges_local) - 1):
        y = (y_top - 0.14) - i * 0.12
        swatch = Rectangle((x0, y), 0.9, 0.10, facecolor=colors[i], edgecolor="black", lw=0.5)
        ax.add_patch(swatch)
        label = ax.text(
            x0 + 1.05,
            y + 0.05,
            f"{temp_edges_local[i]:.1f}-{temp_edges_local[i + 1]:.1f} °C",
            va="center",
            fontsize=7.5,
        )
        swatches.append(swatch)
        labels.append(label)


backend = matplotlib.get_backend().lower()
non_gui_backends = {"agg", "pdf", "ps", "svg", "cairo", "template"}
script_dir = os.path.dirname(os.path.abspath(__file__))


def make_data(ambient, heat_delta, length_cm, chi_reference, chi_target):
    distances = np.linspace(0, length_cm / 100.0, 140)
    temps_reference = calculate_temperature(distances, ambient, heat_delta, alpha, P, chi_reference, S)
    temps_studied = calculate_temperature(distances, ambient, heat_delta, alpha, P, chi_target, S)
    return distances, temps_reference, temps_studied


def save_outputs(fig_graph, fig_bars):
    graph_output = os.path.join(script_dir, "temperature_distribution.png")
    bars_output = os.path.join(script_dir, "bar_visualization.png")
    fig_graph.savefig(graph_output, dpi=150)
    fig_bars.savefig(bars_output, dpi=150)
    print(f"Saved plots to:\n- {graph_output}\n- {bars_output}", flush=True)


if os.environ.get("DISPLAY") and backend not in non_gui_backends:
    # Prefer live interactive view when GUI is available.
    distances, temps_reference, temps_studied = make_data(
        t_ambient,
        theta_0,
        default_length_cm,
        MATERIALS[default_material],
        chi_studied,
    )
    x_cm = distances * 100

    # --- Graph window ---
    fig_graph, ax_graph = plt.subplots(figsize=(10, 6))
    line_copper, = ax_graph.plot(
        x_cm,
        temps_reference,
        label=fr"{default_material} Bar ($\chi$ = {MATERIALS[default_material]:.1f} W/m·K)",
        color="orange",
        linewidth=2,
    )
    line_studied, = ax_graph.plot(
        x_cm,
        temps_studied,
        label=fr'Studied Bar ($\chi$ = {chi_studied:.1f} W/m·K)',
        color="green",
        linestyle="--",
        linewidth=2,
    )
    ambient_line = ax_graph.axhline(y=t_ambient, color="gray", linestyle=":", label=f"Ambient Temp ({t_ambient:.1f}°C)")
    ax_graph.set_title("Steady-State Temperature Distribution Along Metal Bars", fontsize=14)
    ax_graph.set_xlabel("Distance from Hot Source (cm)", fontsize=12)
    ax_graph.set_ylabel("Temperature (°C)", fontsize=12)
    ax_graph.grid(True, alpha=0.5)
    ax_graph.legend()
    fig_graph.tight_layout()

    # --- Physical-style live window ---
    fig_bars, ax = plt.subplots(figsize=(12, 7))
    fig_bars.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.28)
    ax.set_facecolor("#f5f5f5")

    heater = Rectangle((-4.0, 0.35), 3.5, 1.3, facecolor="#555555", edgecolor="black", lw=1.5)
    ax.add_patch(heater)
    ax.text(-2.25, 1.0, "HEAT\nSOURCE", color="white", ha="center", va="center", fontsize=9, fontweight="bold")

    copper_img = ax.imshow(red_heat_strip(temps_reference, fixed_temp_edges), extent=[0, default_length_cm, 1.15, 1.55], aspect="auto", zorder=2)
    studied_img = ax.imshow(red_heat_strip(temps_studied, fixed_temp_edges), extent=[0, default_length_cm, 0.45, 0.85], aspect="auto", zorder=2)
    copper_outline = Rectangle((0, 1.15), default_length_cm, 0.40, fill=False, edgecolor="black", lw=1.2, zorder=3)
    studied_outline = Rectangle((0, 0.45), default_length_cm, 0.40, fill=False, edgecolor="black", lw=1.2, zorder=3)
    ax.add_patch(copper_outline)
    ax.add_patch(studied_outline)
    copper_label = ax.text(default_length_cm / 2, 1.08, f"{default_material} bar", va="top", ha="center", fontsize=10, fontweight="bold")
    studied_label = ax.text(default_length_cm / 2, 0.38, "Studied bar", va="top", ha="center", fontsize=10, fontweight="bold")
    heat_caption = ax.text(default_length_cm / 2, 1.78, "Red = hotter / Blue = colder", ha="center", fontsize=11, color="#8b0000", fontweight="bold")

    ruler_line, = ax.plot([0, default_length_cm], [0.15, 0.15], color="black", lw=1.3)
    ruler_ticks = []
    ruler_labels = []
    legend_swatches_ref = []
    legend_labels_ref = []

    def draw_ruler(length_cm):
        for tick_line in ruler_ticks:
            tick_line.remove()
        for label in ruler_labels:
            label.remove()
        ruler_ticks.clear()
        ruler_labels.clear()
        ruler_line.set_data([0, length_cm], [0.15, 0.15])
        for tick in np.arange(0, length_cm + 0.001, 1):
            if tick % 10 == 0:
                tick_size = 0.10
            elif tick % 5 == 0:
                tick_size = 0.07
            else:
                tick_size = 0.04
            line, = ax.plot([tick, tick], [0.15, 0.15 - tick_size], color="black", lw=1.0)
            ruler_ticks.append(line)
            if tick % 10 == 0:
                label = ax.text(tick, 0.02, f"{int(tick)} cm", ha="center", va="top", fontsize=8)
                ruler_labels.append(label)

    draw_ruler(default_length_cm)
    draw_temp_legend_lists(
        ax,
        fixed_temp_edges,
        default_length_cm + 1.3,
        "temperature ranges",
        1.72,
        legend_swatches_ref,
        legend_labels_ref,
    )
    ax.set_title("Live Physical Bar View: Heater, Temperature Bands, and Ruler", fontsize=13)
    ax.set_xlim(-4.5, default_length_cm + 12.0)
    ax.set_ylim(-0.05, 1.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Controls
    ax_ambient = fig_bars.add_axes([0.12, 0.18, 0.50, 0.03])
    ax_heat = fig_bars.add_axes([0.12, 0.12, 0.50, 0.03])
    ax_length = fig_bars.add_axes([0.12, 0.06, 0.50, 0.03])
    ambient_slider = Slider(ax_ambient, "Ambient (°C)", 0.0, 60.0, valinit=t_ambient, valstep=0.5)
    heat_slider = Slider(ax_heat, "Heat Δθ₀ (°C)", 50.0, 400.0, valinit=theta_0, valstep=1.0)
    length_slider = Slider(ax_length, "Bar length (cm)", 10.0, 100.0, valinit=default_length_cm, valstep=1.0)
    ax_material = fig_bars.add_axes([0.65, 0.05, 0.12, 0.16], facecolor="white")
    material_selector = RadioButtons(ax_material, list(MATERIALS.keys()), active=list(MATERIALS.keys()).index(default_material))
    ax_material.set_title("Material", fontsize=9, pad=4)
    ax_chi = fig_bars.add_axes([0.79, 0.10, 0.12, 0.05])
    chi_box = TextBox(ax_chi, "Studied χ", initial=f"{chi_studied:.3f}")
    ax_save = fig_bars.add_axes([0.91, 0.08, 0.08, 0.06])
    save_button = Button(ax_save, "Save PNGs")
    studied_chi_hint = ax.text(0.79, 0.055, "W/m·K (blank = default)", transform=fig_bars.transFigure, fontsize=7, color="#444444")

    state = {
        "x_cm": x_cm,
        "temps_copper": temps_reference,
        "temps_studied": temps_studied,
        "length_cm": default_length_cm,
        "material": default_material,
        "chi_studied": chi_studied,
    }
    cursor_state = {"x_cm": default_length_cm * 0.5}
    dragging = {"active": False}

    graph_cursor_line = ax_graph.axvline(cursor_state["x_cm"], color="#444444", linestyle=":", linewidth=1.4)
    bar_cursor_line = ax.axvline(cursor_state["x_cm"], color="#444444", linestyle=":", linewidth=1.4, zorder=4)
    graph_cursor_copper, = ax_graph.plot([cursor_state["x_cm"]], [np.interp(cursor_state["x_cm"], x_cm, temps_reference)], "o", color="orange", markersize=6)
    graph_cursor_studied, = ax_graph.plot([cursor_state["x_cm"]], [np.interp(cursor_state["x_cm"], x_cm, temps_studied)], "o", color="green", markersize=6)
    cursor_info = ax.text(
        0.01,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )

    def update_cursor(x_pos):
        x_pos = float(np.clip(x_pos, 0.0, state["length_cm"]))
        cursor_state["x_cm"] = x_pos
        tc = float(np.interp(x_pos, state["x_cm"], state["temps_copper"]))
        ts = float(np.interp(x_pos, state["x_cm"], state["temps_studied"]))
        graph_cursor_line.set_xdata([x_pos, x_pos])
        bar_cursor_line.set_xdata([x_pos, x_pos])
        graph_cursor_copper.set_data([x_pos], [tc])
        graph_cursor_studied.set_data([x_pos], [ts])
        cursor_info.set_text(f"x = {x_pos:.1f} cm\n{state['material']}: {tc:.1f} °C\nStudied: {ts:.1f} °C")

    def update(_):
        ambient = ambient_slider.val
        heat_delta = heat_slider.val
        length_cm = length_slider.val
        material_name = state["material"]
        distances_u, copper_u, studied_u = make_data(
            ambient,
            heat_delta,
            length_cm,
            MATERIALS[material_name],
            state["chi_studied"],
        )
        x_cm_u = distances_u * 100

        line_copper.set_data(x_cm_u, copper_u)
        line_copper.set_label(fr"{material_name} Bar ($\chi$ = {MATERIALS[material_name]:.1f} W/m·K)")
        line_studied.set_data(x_cm_u, studied_u)
        line_studied.set_label(fr'Studied Bar ($\chi$ = {state["chi_studied"]:.1f} W/m·K)')
        ambient_line.set_ydata([ambient, ambient])
        ambient_line.set_label(f"Ambient Temp ({ambient:.1f}°C)")
        ax_graph.set_xlim(0, length_cm)
        y_max = max(float(np.max(copper_u)), float(np.max(studied_u))) + 10.0
        ax_graph.set_ylim(ambient - 5.0, y_max)
        ax_graph.legend()

        copper_img.set_data(red_heat_strip(copper_u, fixed_temp_edges))
        studied_img.set_data(red_heat_strip(studied_u, fixed_temp_edges))
        copper_img.set_extent([0, length_cm, 1.15, 1.55])
        studied_img.set_extent([0, length_cm, 0.45, 0.85])
        copper_outline.set_width(length_cm)
        studied_outline.set_width(length_cm)
        copper_label.set_position((length_cm / 2, 1.08))
        copper_label.set_text(f"{material_name} bar")
        studied_label.set_position((length_cm / 2, 0.38))
        heat_caption.set_position((length_cm / 2, 1.78))
        draw_ruler(length_cm)
        draw_temp_legend_lists(
            ax,
            fixed_temp_edges,
            length_cm + 1.3,
            "temperature ranges",
            1.72,
            legend_swatches_ref,
            legend_labels_ref,
        )
        ax.set_xlim(-4.5, length_cm + 12.0)

        state["x_cm"] = x_cm_u
        state["temps_copper"] = copper_u
        state["temps_studied"] = studied_u
        state["length_cm"] = length_cm
        update_cursor(cursor_state["x_cm"])

        fig_graph.canvas.draw_idle()
        fig_bars.canvas.draw_idle()

    ambient_slider.on_changed(update)
    heat_slider.on_changed(update)
    length_slider.on_changed(update)

    def on_material_change(label):
        state["material"] = label
        update(None)

    material_selector.on_clicked(on_material_change)

    chi_submit_guard = {"ignore": False}

    def set_chi_box_text(text):
        """Update the TextBox without re-triggering on_submit side effects."""
        chi_submit_guard["ignore"] = True
        chi_box.set_val(text)
        chi_submit_guard["ignore"] = False

    def on_studied_chi_submit(text):
        if chi_submit_guard["ignore"]:
            return
        cleaned = text.strip()
        if cleaned == "":
            state["chi_studied"] = DEFAULT_CHI_STUDIED
            update(None)
            set_chi_box_text(f"{DEFAULT_CHI_STUDIED:.3f}")
            return
        try:
            value = float(cleaned)
            if value <= 0:
                raise ValueError
            state["chi_studied"] = value
            update(None)
            set_chi_box_text(f"{value:.3f}")
        except ValueError:
            print("Invalid studied thermal conductivity. Enter a positive number or leave blank.", flush=True)
            set_chi_box_text(f"{state['chi_studied']:.3f}")

    chi_box.on_submit(on_studied_chi_submit)

    def on_save(_event):
        save_outputs(fig_graph, fig_bars)

    save_button.on_clicked(on_save)

    def cursor_x_from_event(event):
        """Distance along the bar (cm) from a mouse event on either interactive axes."""
        if event.xdata is None:
            return None
        if event.inaxes == ax_graph or event.inaxes == ax:
            return float(event.xdata)
        return None

    def on_press(event):
        x_cm_click = cursor_x_from_event(event)
        if x_cm_click is not None:
            dragging["active"] = True
            update_cursor(x_cm_click)
            fig_graph.canvas.draw_idle()
            fig_bars.canvas.draw_idle()

    def on_motion(event):
        if not dragging["active"]:
            return
        x_cm_move = cursor_x_from_event(event)
        if x_cm_move is not None:
            update_cursor(x_cm_move)
            fig_graph.canvas.draw_idle()
            fig_bars.canvas.draw_idle()

    def on_release(_event):
        dragging["active"] = False

    for _canvas in (fig_graph.canvas, fig_bars.canvas):
        _canvas.mpl_connect("button_press_event", on_press)
        _canvas.mpl_connect("motion_notify_event", on_motion)
        _canvas.mpl_connect("button_release_event", on_release)
    update_cursor(cursor_state["x_cm"])

    plt.show()
else:
    distances, temps_copper, temps_studied = make_data(
        t_ambient,
        theta_0,
        default_length_cm,
        MATERIALS[default_material],
        chi_studied,
    )
    x_cm = distances * 100

    fig_graph, ax_graph = plt.subplots(figsize=(10, 6))
    ax_graph.plot(x_cm, temps_copper, label=fr"{default_material} Bar ($\chi$ = {MATERIALS[default_material]:.1f} W/m·K)", color="orange", linewidth=2)
    ax_graph.plot(
        x_cm,
        temps_studied,
        label=fr'Studied Bar ($\chi$ = {chi_studied:.1f} W/m·K)',
        color="green",
        linestyle="--",
        linewidth=2,
    )
    ax_graph.axhline(y=t_ambient, color="gray", linestyle=":", label=f"Ambient Temp ({t_ambient:.1f}°C)")
    ax_graph.set_title("Steady-State Temperature Distribution Along Metal Bars", fontsize=14)
    ax_graph.set_xlabel("Distance from Hot Source (cm)", fontsize=12)
    ax_graph.set_ylabel("Temperature (°C)", fontsize=12)
    ax_graph.grid(True, alpha=0.5)
    ax_graph.legend()
    fig_graph.tight_layout()

    fig_bars, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#f5f5f5")
    heater = Rectangle((-4.0, 0.35), 3.5, 1.3, facecolor="#555555", edgecolor="black", lw=1.5)
    ax.add_patch(heater)
    ax.text(-2.25, 1.0, "HEAT\nSOURCE", color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.imshow(red_heat_strip(temps_copper, fixed_temp_edges), extent=[0, default_length_cm, 1.15, 1.55], aspect="auto", zorder=2)
    ax.imshow(red_heat_strip(temps_studied, fixed_temp_edges), extent=[0, default_length_cm, 0.45, 0.85], aspect="auto", zorder=2)
    ax.add_patch(Rectangle((0, 1.15), default_length_cm, 0.40, fill=False, edgecolor="black", lw=1.2, zorder=3))
    ax.add_patch(Rectangle((0, 0.45), default_length_cm, 0.40, fill=False, edgecolor="black", lw=1.2, zorder=3))
    ax.plot([0, default_length_cm], [0.15, 0.15], color="black", lw=1.3)
    for tick in np.arange(0, default_length_cm + 0.01, 5):
        tick_size = 0.08 if tick % 10 == 0 else 0.05
        ax.plot([tick, tick], [0.15, 0.15 - tick_size], color="black", lw=1.0)
        if tick % 10 == 0:
            ax.text(tick, 0.02, f"{int(tick)} cm", ha="center", va="top", fontsize=8)
    ax.text(default_length_cm / 2, 1.78, "Red = hotter / Blue = colder", ha="center", fontsize=11, color="#8b0000", fontweight="bold")
    ax.set_title("Physical Bar View: Heater, Temperature Bands, and Ruler", fontsize=13)
    _headless_legend_swatches = []
    _headless_legend_labels = []
    draw_temp_legend_lists(
        ax,
        fixed_temp_edges,
        default_length_cm + 1.3,
        "temperature ranges",
        1.72,
        _headless_legend_swatches,
        _headless_legend_labels,
    )
    ax.set_xlim(-4.5, default_length_cm + 12.0)
    ax.set_ylim(-0.05, 1.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig_bars.tight_layout()

    save_outputs(fig_graph, fig_bars)
    print("No GUI display/backend available. Saved PNG files instead of live windows.", flush=True)