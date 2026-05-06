import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# Set the page configuration for the website
st.set_page_config(page_title="Thermal Heat Transfer", layout="wide")

st.title("Stationary Heat Transfer Visualisation")
st.markdown("This interactive tool visualises heat transfer through metal bars based on Fourier's law in a steady state.")

# --- Interactive Sidebar Controls ---
st.sidebar.header("Simulation Parameters")

# Create sliders that the user can drag to change values
t_ambient = st.sidebar.slider("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=24.0, step=1.0)
theta_0 = st.sidebar.slider("Hot Source Temp Difference (°C)", min_value=100.0, max_value=500.0, value=264.0, step=1.0)
chi_studied = st.sidebar.slider("Studied Bar Conductivity (W/m·K)", min_value=50.0, max_value=500.0, value=373.5, step=0.5)

# Fixed variables
t_hot_source = t_ambient + theta_0 
chi_copper = 384.1        
radius = 0.01             
P = 2 * np.pi * radius    
S = np.pi * radius**2     
alpha = 15.0              

# Distance array
bar_length_m = 0.3
distances_m = np.linspace(0, bar_length_m, 200)

# --- Physics Calculation Functions ---
def get_a_parameter(alpha_val, P_val, chi_val, S_val):
    return np.sqrt((alpha_val * P_val) / (chi_val * S_val))

def calculate_exact_temps(x_array, t_amb, theta0, a_param):
    theta_x = theta0 * np.exp(-a_param * x_array)
    return theta_x + t_amb

# Calculate current state based on sliders
a_cu = get_a_parameter(alpha, P, chi_copper, S)
a_std = get_a_parameter(alpha, P, chi_studied, S)

temps_cu = calculate_exact_temps(distances_m, t_ambient, theta_0, a_cu)
temps_std = calculate_exact_temps(distances_m, t_ambient, theta_0, a_std)

# --- Visualization Setup ---
fig, (ax_graph, ax_sim) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1.2]})
plt.subplots_adjust(hspace=0.4)

# A. Top Subplot: Precise Graph
ax_graph.set_title('Steady-State Temperature Distribution', fontsize=12, fontweight='bold')
ax_graph.plot(distances_m * 100, temps_cu, label=fr'Copper Bar ($\chi$ = {chi_copper} W/m·K)', color='tab:orange', linewidth=2.5)
ax_graph.plot(distances_m * 100, temps_std, label=fr'Studied Bar ($\chi$ $\approx$ {chi_studied:.1f} W/m·K)', color='tab:green', linestyle='--', linewidth=2.5)
ax_graph.axhline(y=t_ambient, color='gray', linestyle=':', label=f'Ambient Temp ({t_ambient}°C)')
ax_graph.set_ylabel('Temperature (°C)')
ax_graph.grid(True, alpha=0.3)
ax_graph.legend(loc='upper right')
ax_graph.set_xlim(0, 30)

# B. Bottom Subplot: Stylized Physical Simulation
ax_sim.set_title('Physical Visualisation', fontsize=12, fontweight='bold')
bar_height, bar_v_spacing = 10, 8
copper_y_start = 25
studied_y_start = copper_y_start - bar_height - bar_v_spacing
sim_width_cm = 30

cmap = plt.cm.get_cmap('YlOrRd')
norm = mcolors.Normalize(vmin=t_ambient, vmax=max(t_hot_source, 500))

# Heat Source Block
source_width = 1.5
ax_sim.add_patch(patches.Rectangle((-source_width, studied_y_start - 2), source_width, (bar_height*2 + bar_v_spacing + 4), facecolor='red', edgecolor='darkred', linewidth=2))
ax_sim.text(-source_width/2, copper_y_start + bar_height/2, 'HEAT', color='white', fontweight='bold', ha='center', va='center', rotation=90)

def draw_thermal_bar(ax, y_start, height, length_cm, temp_array, label_text):
    ax.add_patch(patches.Rectangle((0, y_start), length_cm, height, linewidth=1.5, edgecolor='black', facecolor='none', zorder=10))
    ax.text(-2, y_start + height/2, label_text, fontweight='bold', ha='right', va='center')
    ax.imshow(temp_array.reshape(1, -1), cmap=cmap, norm=norm, aspect='auto', extent=[0, length_cm, y_start, y_start + height], zorder=1)

draw_thermal_bar(ax_sim, copper_y_start, bar_height, sim_width_cm, temps_cu, 'Copper')
draw_thermal_bar(ax_sim, studied_y_start, bar_height, sim_width_cm, temps_std, 'Studied')

# Ruler
ruler_y = studied_y_start - 5
ax_sim.plot([0, sim_width_cm], [ruler_y, ruler_y], color='black', linewidth=1.5)
for cm in range(sim_width_cm + 1):
    if cm % 5 == 0:
        ax_sim.plot([cm, cm], [ruler_y, ruler_y - 2], color='black', linewidth=1.5)
        ax_sim.text(cm, ruler_y - 3, str(cm), ha='center', va='top', fontsize=10)
    else:
        ax_sim.plot([cm, cm], [ruler_y, ruler_y - 1], color='gray', linewidth=0.8)

ax_sim.set_xlabel('Distance from Hot Source (cm)')
ax_sim.set_yticks([])
ax_sim.set_xlim(-source_width - 3, sim_width_cm + 1)
ax_sim.set_ylim(ruler_y - 6, copper_y_start + bar_height + 3)
ax_sim.axis('off')

# Render the plot in Streamlit
st.pyplot(fig)
