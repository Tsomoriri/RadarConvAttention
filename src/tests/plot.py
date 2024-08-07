import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib.colors import LinearSegmentedColormap

# Load the JSON data
file_paths = [
    "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_standard_radar_movies.npy_comparison_results.json",
    "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_attention_standard_radar_movies.npy_comparison_results.json",
    "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_attention_physics_physics_radar_movies.npy_comparison_results.json",
    "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_attention_physics_physics_dynamic_grid_radar_movies.npy_comparison_results.json"
]

data = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data.append(json.load(file))

# Extract the relevant metrics
labels = ['Test Loss', 'MAE', 'SSIM']
model_names = [d['model'] for d in data]
values = [[d['test_loss'], d['mae'], d['ssim']] for d in data]

# New normalization function
def custom_normalize(data, min_value, max_value, lower_bound=0.2):
    range = max_value - min_value
    return lower_bound + (1 - lower_bound) * (data - min_value) / range

# Apply custom normalization
min_values = np.min(values, axis=0)
max_values = np.max(values, axis=0)

normalized_values = np.array([
    [custom_normalize(d[0], min_values[0], max_values[0]),
     custom_normalize(d[1], min_values[1], max_values[1]),
     1 - custom_normalize(d[2], min_values[2], max_values[2])]  # Invert SSIM
    for d in values
])

# Radar chart setup
num_vars = len(labels)

angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], labels, color='grey', size=8)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
plt.ylim(0, 1)

# Define a custom color palette
colors = ['#FF1493', '#00CED1', '#FFD700', '#32CD32', '#FF4500', '#1E90FF', '#8A2BE2', '#20B2AA']

# Plot data
for i, (model, value) in enumerate(zip(model_names, normalized_values)):
    values = value.tolist()
    values += values[:1]
    color = colors[i % len(colors)]  # Cycle through colors if more models than colors
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)
    
    # Add scatter points
    ax.scatter(angles, values, s=60, color=color, edgecolor='white', linewidth=1, zorder=10)

# Customize the chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.grid(True, color='gray', linestyle='--', alpha=0.5)

# Add a legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

# Adjust the layout and save the figure
plt.tight_layout()
fig.subplots_adjust(right=0.7)  # Make room for the legend
plt.savefig("./radar_chart_with_scatter.png", dpi=300, bbox_inches='tight')
plt.show()