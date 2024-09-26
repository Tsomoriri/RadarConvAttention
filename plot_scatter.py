import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


def load_results_with_key_mapping(folder_path):
    """
    Loads JSON results files, restructures them into a dictionary, 
    and calculates RMSE from test_loss.

    Args:
        folder_path (str): Path to the folder containing JSON result files.

    Returns:
        dict: A dictionary containing the restructured results.
    """
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)

                    model_name = data["model"]
                    results[model_name] = {
                        "rmse": data["test_loss"],  # Calculate RMSE
                        "mae": data["mae"],
                        "ssim": data["ssim"]
                    }

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {filename}")
    return results


# Example Usage:
# Replace with your folder path
folder_path = "/home/tso/RadarConvAttention/Train_results"
# data = load_results_with_key_mapping(folder_path)


data = {
    'convlstm': {'rmse': 947.8107256208148, 'mae': 12993.7529296875, 'ssim': 0.6190339129110118, 'technique': 'baseline'},
    'convlstm_phys': {'rmse': 940.9223265355947, 'mae': 12957.1337890625, 'ssim': 0.6219235537664675, 'technique': 'physics'},
    'convlstm_atn_phys(DG)': {'rmse': 827.0492485202088, 'mae': 12896.12109375, 'ssim': 0.5914251768895028, 'technique': 'all'},
    'convlstm_atn_phys': {'rmse': 819.3426793935348, 'mae': 13883.7294921875, 'ssim': 0.5810395998977547, 'technique': 'attention+physics'},
    'convlstm_attention_physics_dynamicgrid': {'rmse': 880.7046714315609, 'mae': 13008.236328125, 'ssim': 0.5750093829773906, 'technique': 'all'},
    'convlstm_atn': {'rmse': 802.3219230807557, 'mae': 12294.8662109375, 'ssim': 0.6218325994595123, 'technique': 'attention'},
    'convlstm_attention': {'rmse': 856.7838570731027, 'mae': 12484.2861328125, 'ssim': 0.6079093572557905, 'technique': 'attention'},
    'convlstm_attention_physics': {'rmse': 845.8341874103157, 'mae': 14084.875, 'ssim': 0.6092352922809324, 'technique': 'attention+physics'},
    'convlstm_physics': {'rmse': 1214.1238677355707, 'mae': 16603.15234375, 'ssim': 0.4765177258526705, 'technique': 'physics'}
}


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


# Data
techniques = ['ConvLSTM', '+ Physics', '+ Attention',
              '+ Att.\n+ Phys.', '+ Att.\n+ Phys.\n+ DG']
rmse = [947.81, 940.92, 819.34, 802.32, 827.05]
mae = [12993.75, 12957.13, 1383.73, 12294.87, 12896.12]
ssim = [0.6190, 0.6219, 0.5810, 0.6218, 0.5914]

# Normalize the data to 0.1-0.8 range


def normalize(data):
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return 0.1 + normalized * 0.7


rmse_norm = 1 - normalize(rmse)  # Invert RMSE so higher is better
mae_norm = 1 - normalize(mae)    # Invert MAE so higher is better
ssim_norm = normalize(ssim)      # SSIM is already in the correct direction

# Selected Exeter colors for better visibility
colors = ['#005C3C', '#C84B31', '#8F3931']  # Dark Green, Orange, Dark Red


def create_smooth_overlay_plot(rmse_data, mae_data, ssim_data, filename):
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased width for label space

    metrics = [rmse_data, mae_data, ssim_data]
    labels = ['RMSE', 'MAE', 'SSIM']

    x = np.arange(len(techniques))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        # Add padding to prevent breaks at extremes
        x_padded = np.concatenate(([x[0] - 0.1], x, [x[-1] + 0.1]))
        y_padded = np.concatenate(([metric[0]], metric, [metric[-1]]))

        spl = make_interp_spline(x_padded, y_padded, k=3)
        y_smooth = spl(x_smooth)
        line, = ax.plot(x_smooth, y_smooth, color=colors[i], linewidth=2)
        ax.scatter(x, metric, color=colors[i], s=30, zorder=3)

        # Add label at the end of each line
        ax.text(x[-1] + 0.1, metric[-1], f'{label}',
                color=colors[i], va='center', ha='left', fontweight='bold')

    ax.set_xlabel('Model Complexity', fontsize=12, labelpad=10)
    ax.set_ylabel('Normalized Metric Value\n(Higher is Better)',
                  fontsize=12, labelpad=10)
    ax.set_xticks(range(len(techniques)))
    ax.set_xticklabels(techniques, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1)
    # Adjust x-axis to make room for labels
    ax.set_xlim(-0.1, len(techniques) - 0.5)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # Remove y-axis ticks
    ax.yaxis.set_ticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Create smooth overlay plot
create_smooth_overlay_plot(
    rmse_norm, mae_norm, ssim_norm, 'exeter_integrated_metric_overlay.png')

print("Integrated smooth overlay plot with Exeter colors saved as exeter_integrated_metric_overlay.png")
