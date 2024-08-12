import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
from matplotlib.colors import LinearSegmentedColormap

from math import pi

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Load the JSON data
file_paths = [
   "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_atn_phys_physics_dynamic_grid_radar_movies.npy_comparison_results.json",
   "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_atn_phys_physics_radar_movies.npy_comparison_results.json",
   "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_atn_standard_radar_movies.npy_comparison_results.json",
   "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_phys_physics_dynamic_grid_radar_movies.npy_comparison_results.json",
   "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_phys_physics_radar_movies.npy_comparison_results.json",
   "/home/tso/RadarConvAttention/Train_results/radar_movies_convlstm_standard_radar_movies.npy_comparison_results.json",


]

data = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data.append(json.load(file))

# Function to shorten model names
def shorten_model_name(name):
    name_map = {
        "convlstm": "CL",
        "convlstm_attention": "CL-Att",
        "convlstm_attention_physics": "CL-Att-Phy",
        "convlstm_attention_physics_dynamicgrid": "CL-Att-Phy-DG"
    }
     # Check for exact match or match at the beginning of the string
    for key, value in name_map.items():
        if key == name or name.startswith(key + "_"):  # Add underscore check
            return value
    return name  # Return original name if no match found

# Extract the relevant metrics
labels = ['RMSE', 'MAE', 'SSIM']
model_names = [d['model'] for d in data]
print("model names for radar scatter are:->>",model_names)
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

fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], labels, color='grey', size=10)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1)

# Define a custom color palette
colors = ['#FF1493', '#00CED1', '#FFD700', '#32CD32', '#FF4500', '#1E90FF', '#8A2BE2', '#20B2AA']

# Plot data
for i, (model, value) in enumerate(zip(model_names, normalized_values)):
    values = value.tolist()
    values += values[:1]
    color = colors[i % len(colors)]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)
    
    # Add scatter points
    ax.scatter(angles, values, s=60, color=color, edgecolor='white', linewidth=1, zorder=10)

# Customize the chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
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
##################################################################################################

def create_candlestick_plot(ax, x, open, high, low, close, width=0.6, up_color='g', down_color='r',rev=True):
    for i in range(len(x)):
        if rev == True:
            if close[i] < open[i]:
                color = up_color
            else:
                color = down_color
        else:
            if close[i] > open[i]:
                color = up_color
            else:
                color = down_color

        
        # Plot the wick
        ax.plot([x[i], x[i]], [low[i], high[i]], color='black', linewidth=1)
        
        # Plot the body
        body_low = max(open[i], close[i])
        body_high = min(open[i], close[i])
        body_height = body_high - body_low
        
        ax.bar(x[i], body_height, width, bottom=body_low, color=color, edgecolor='black', linewidth=1)

def plot_mse_comparison(save_path='mse_comparison_candlestick.png'):
    models = ['ConvLSTM', 'ConvLSTM-Physics', 'ConvLSTM-Physics (DG)', 
              'ConvLSTM-Attention', 'ConvLSTM-Attention-Physics', 'ConvLSTM-Attention-Physics (DG)']
    mse_3_particles = [0.0023, 0.0021, 0.0020, 0.0024, 0.0022, 0.0023]
    mse_11_particles = [0.0062, 0.0056, 0.0051, 0.0065, 0.0057, 0.0055]

    baseline_3 = mse_3_particles[0]
    baseline_11 = mse_11_particles[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=300)
    
    for ax, data, baseline, title in zip([ax1, ax2], 
                                         [mse_3_particles, mse_11_particles],
                                         [baseline_3, baseline_11],
                                         ['3 Particles', '11 Particles']):
        open_data = [baseline] * len(models)
        close_data = data
        high_data = [min(baseline, d) for d in data]
        low_data = [max(baseline, d) for d in data]
        
        create_candlestick_plot(ax, range(len(models)), open_data, high_data, low_data, close_data, width=0.4)
        
        ax.set_ylim(min(low_data) * 0.9, max(high_data) * 1.1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title(f'MSE Comparison - {title}', fontsize=14, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        for i, v in enumerate(data):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

        # Add a horizontal line for the baseline
        ax.axhline(y=baseline, color='blue', linestyle='--', alpha=0.7)
        ax.text(-0.5, baseline, f'Baseline: {baseline:.4f}', va='center', ha='right', fontsize=8, color='blue')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_real_radar_data(save_path='real_radar_data_candlestick.png'):
    models = ['ConvLSTM', 'ConvLSTM-Physics', 'ConvLSTM-Physics (DG)', 
              'ConvLSTM-Attention', 'ConvLSTM-Attention-Physics', 'ConvLSTM-Attention-Physics (DG)']
    mse_values = [941.0779, 940.7903, 935.9675, 847.1528, 840.4709, 999.1830]

    baseline = mse_values[0]
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    open_data = [baseline] * len(models)
    close_data = mse_values
    high_data = [max(baseline, d) for d in mse_values]
    low_data = [min(baseline, d) for d in mse_values]
    
    create_candlestick_plot(ax, range(len(models)), open_data, high_data, low_data, close_data, width=0.4)
    
    ax.set_ylim(min(low_data) * 0.9, max(high_data) * 1.1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_title('MSE Comparison on Real Radar Data', fontsize=14, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    for i, v in enumerate(mse_values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # Add a horizontal line for the baseline
    ax.axhline(y=baseline, color='blue', linestyle='--', alpha=0.7)
    ax.text(-0.5, baseline, f'Baseline: {baseline:.2f}', va='center', ha='right', fontsize=8, color='blue')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Call the functions to generate and save the plots
plot_mse_comparison()
plot_real_radar_data()

print("Plots have been saved as 'mse_comparison_candlestick.png' and 'real_radar_data_candlestick.png'.")

def plot_ssim_comparison(data, save_path='ssim_comparison_candlestick.png'):
    """
    Plots SSIM values as candlestick charts with ConvLSTM as the baseline.

    Args:
        data (list of dict): The JSON data containing model names and SSIM values.
        save_path (str): The path to save the plot.
    """

    models = [d['model'] for d in data]
    print(models)
    ssim_values = [d['ssim'] for d in data]

    baseline = ssim_values[5]  # ConvLSTM is the first model

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

    open_data = [baseline] * len(models)
    close_data = ssim_values
    high_data = [max(baseline, d) for d in ssim_values]
    low_data = [min(baseline, d) for d in ssim_values]

    # Use green for higher SSIM (better) and red for lower SSIM
    create_candlestick_plot(ax, range(len(models)), open_data, high_data, low_data, close_data,
                            up_color='g', down_color='r',rev=False)

    ax.set_ylim(min(low_data) * 0.9, max(high_data) * 1.1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_title('SSIM Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('SSIM', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    for i, v in enumerate(ssim_values):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    # Add a horizontal line for the baseline
    ax.axhline(y=baseline, color='blue', linestyle='--', alpha=0.7)
    ax.text(-0.5, baseline, f'Baseline: {baseline:.4f}', va='center', ha='right', fontsize=8, color='blue')

    fig.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Call the function to generate and save the plot
plot_ssim_comparison(data)

print("SSIM plot has been saved as 'ssim_comparison_candlestick.png'.")
