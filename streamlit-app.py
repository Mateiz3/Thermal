import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Thermal Heat Transfer", layout="wide")
st.title("Stationary Heat Transfer Visualisation")

# --- Constants (aligned with main.py) ---
MATERIALS = {
    "Copper": 384.1,
    "Aluminum": 205.0,
    "Brass": 109.0,
    "Steel": 50.0,
    "Silver": 429.0,
}
default_material = "Copper"
DEFAULT_CHI_STUDIED = 373.482

radius = 0.01
P = 2 * np.pi * radius
S = np.pi * radius**2
alpha = 15.0
heat_levels = 9
temp_min_fixed = 0.0
temp_max_fixed = 460.0
fixed_temp_edges = np.linspace(temp_min_fixed, temp_max_fixed, heat_levels + 1)


def calculate_temperature(x, t_ambient, theta_0, alpha, P, chi, S):
    a = np.sqrt((alpha * P) / (chi * S))
    theta = theta_0 * np.exp(-a * x)
    return theta + t_ambient


def make_data(ambient, heat_delta, length_cm, chi_reference, chi_target):
    distances = np.linspace(0, length_cm / 100.0, 140)
    temps_reference = calculate_temperature(distances, ambient, heat_delta, alpha, P, chi_reference, S)
    temps_studied = calculate_temperature(distances, ambient, heat_delta, alpha, P, chi_target, S)
    return distances, temps_reference, temps_studied


def band_colors(levels=heat_levels):
    values = np.linspace(0.0, 1.0, levels)
    colors = np.zeros((levels, 3))
    # Blue (cold) -> red (hot) gradient (same as main.py).
    colors[:, 0] = 0.10 + 0.90 * values
    colors[:, 1] = 0.05 + 0.10 * (1.0 - values)
    colors[:, 2] = 0.95 - 0.90 * values
    return colors


def red_heat_strip(temps, temp_edges):
    palette = band_colors(len(temp_edges) - 1)
    band_idx = np.digitize(temps, temp_edges[1:-1], right=False)
    band_idx = np.clip(band_idx, 0, len(palette) - 1)
    strip_row = palette[band_idx]
    return np.repeat(strip_row[np.newaxis, :, :], 20, axis=0)


def draw_temp_legend(ax, temp_edges_local, x0, title, y_top):
    colors = band_colors(len(temp_edges_local) - 1)
    ax.text(x0, y_top, title, fontsize=9, fontweight="bold", va="bottom")
    for i in range(len(temp_edges_local) - 1):
        y = (y_top - 0.14) - i * 0.12
        swatch = Rectangle((x0, y), 0.9, 0.10, facecolor=colors[i], edgecolor="black", lw=0.5)
        ax.add_patch(swatch)
        ax.text(x0 + 1.05, y + 0.05, f"{temp_edges_local[i]:.1f}-{temp_edges_local[i + 1]:.1f} °C", va="center", fontsize=7.5)


def fig_to_png_bytes(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


# --- Streamlit Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
t_ambient = st.sidebar.slider("Ambient (°C)", 0.0, 60.0, 24.0, 0.5)
theta_0 = st.sidebar.slider("Heat Δθ₀ (°C)", 50.0, 400.0, 264.0, 1.0)
length_cm = st.sidebar.slider("Bar length (cm)", 10.0, 100.0, 30.0, 1.0)
material_name = st.sidebar.radio("Reference Material", list(MATERIALS.keys()), index=list(MATERIALS.keys()).index(default_material))

chi_input = st.sidebar.text_input("Studied χ (W/m·K)", "")
if chi_input.strip() == "":
    chi_studied = DEFAULT_CHI_STUDIED
else:
    try:
        chi_studied = float(chi_input)
        if chi_studied <= 0:
            raise ValueError
    except ValueError:
        st.sidebar.error("Invalid thermal conductivity. Using default.")
        chi_studied = DEFAULT_CHI_STUDIED

st.sidebar.divider()
st.sidebar.header("Data Inspector")
cursor_x = st.sidebar.slider("Cursor Distance (cm)", 0.0, float(length_cm), float(length_cm) / 2.0, 0.1)

# --- Calculations (same pipeline as main.py make_data) ---
distances, temps_reference, temps_studied = make_data(
    t_ambient, theta_0, length_cm, MATERIALS[material_name], chi_studied
)
x_cm = distances * 100

tc_cursor = float(np.interp(cursor_x, x_cm, temps_reference))
ts_cursor = float(np.interp(cursor_x, x_cm, temps_studied))

st.sidebar.info(
    f"**Temperatures at {cursor_x:.1f} cm:**\n\n"
    f"**{material_name}:** {tc_cursor:.1f} °C\n\n"
    f"**Studied:** {ts_cursor:.1f} °C"
)

# --- Plot 1: The Graph ---
st.subheader("Temperature Distribution ")
fig_graph, ax_graph = plt.subplots(figsize=(10, 6))
ax_graph.plot(x_cm, temps_reference, label=f"{material_name} Bar (χ = {MATERIALS[material_name]:.1f} W/m·K)", color="orange", linewidth=2)
ax_graph.plot(x_cm, temps_studied, label=f'Studied Bar (χ = {chi_studied:.1f} W/m·K)', color="green", linestyle="--", linewidth=2)

ax_graph.axvline(cursor_x, color="#444444", linestyle=":", linewidth=1.5, label="Cursor Position")
ax_graph.plot([cursor_x], [tc_cursor], "o", color="orange", markersize=8)
ax_graph.plot([cursor_x], [ts_cursor], "o", color="green", markersize=8)

ax_graph.axhline(y=t_ambient, color="gray", linestyle=":", label=f"Ambient Temp ({t_ambient:.1f}°C)")
ax_graph.set_title("Steady-State Temperature Distribution Along Metal Bars", fontsize=14)
ax_graph.set_xlabel("Distance from Hot Source (cm)", fontsize=12)
ax_graph.set_ylabel("Temperature (°C)", fontsize=12)
ax_graph.set_xlim(0, length_cm)
y_max = max(float(np.max(temps_reference)), float(np.max(temps_studied))) + 10.0
ax_graph.set_ylim(t_ambient - 5.0, y_max)
ax_graph.grid(True, alpha=0.5)
ax_graph.legend()
fig_graph.tight_layout()
graph_png = fig_to_png_bytes(fig_graph)
st.pyplot(fig_graph)
plt.close(fig_graph)

# --- Plot 2: Physical Bar View ---
st.subheader("Physical Bar View")
fig_bars, ax = plt.subplots(figsize=(12, 7))
fig_bars.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12)
ax.set_facecolor("#f5f5f5")

heater = Rectangle((-4.0, 0.35), 3.5, 1.3, facecolor="#555555", edgecolor="black", lw=1.5)
ax.add_patch(heater)
ax.text(-2.25, 1.0, "HEAT\nSOURCE", color="white", ha="center", va="center", fontsize=9, fontweight="bold")

ax.imshow(red_heat_strip(temps_reference, fixed_temp_edges), extent=[0, length_cm, 1.15, 1.55], aspect="auto", zorder=2)
ax.imshow(red_heat_strip(temps_studied, fixed_temp_edges), extent=[0, length_cm, 0.45, 0.85], aspect="auto", zorder=2)

ax.add_patch(Rectangle((0, 1.15), length_cm, 0.40, fill=False, edgecolor="black", lw=1.2, zorder=3))
ax.add_patch(Rectangle((0, 0.45), length_cm, 0.40, fill=False, edgecolor="black", lw=1.2, zorder=3))

ax.text(length_cm / 2, 1.08, f"{material_name} bar", va="top", ha="center", fontsize=10, fontweight="bold")
ax.text(length_cm / 2, 0.38, "Studied bar", va="top", ha="center", fontsize=10, fontweight="bold")

ax.plot([0, length_cm], [0.15, 0.15], color="black", lw=1.3)
for tick in np.arange(0, length_cm + 0.001, 1):
    tick_size = 0.10 if tick % 10 == 0 else (0.07 if tick % 5 == 0 else 0.04)
    ax.plot([tick, tick], [0.15, 0.15 - tick_size], color="black", lw=1.0)
    if tick % 10 == 0:
        ax.text(tick, 0.02, f"{int(tick)} cm", ha="center", va="top", fontsize=8)

ax.axvline(cursor_x, color="#444444", linestyle=":", linewidth=1.5, zorder=4)

draw_temp_legend(ax, fixed_temp_edges, length_cm + 1.3, "temperature ranges", 1.72)

ax.text(length_cm / 2, 1.78, "Red = hotter / Blue = colder", ha="center", fontsize=11, color="#8b0000", fontweight="bold")

ax.set_title("Physical Bar View: Heater, Temperature Bands, and Ruler", fontsize=13)
ax.set_xlim(-4.5, length_cm + 12.0)
ax.set_ylim(-0.05, 1.95)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

bars_png = fig_to_png_bytes(fig_bars)
st.pyplot(fig_bars)
plt.close(fig_bars)

st.download_button(
    "Download temperature_distribution.png",
    data=graph_png,
    file_name="temperature_distribution.png",
    mime="image/png",
)
st.download_button(
    "Download bar_visualization.png",
    data=bars_png,
    file_name="bar_visualization.png",
    mime="image/png",
)
