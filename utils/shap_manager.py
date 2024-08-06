import shap
import numpy as np
import torch
import matplotlib.pyplot as plt

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

class SHAPExplainer:
    def __init__(self, model, background_data, device):
        self.model = model
        self.device = device
        self.explainer = shap.DeepExplainer(model, background_data.to(device))

    def explain(self, data):
        with torch.no_grad():
            shap_values = self.explainer.shap_values(data.to(self.device))
        return [to_numpy(sv) for sv in shap_values]

    @staticmethod
    def calculate_shap_values(model, data_loader, num_background=100, num_explain=100):
        device = next(model.parameters()).device
        model.eval()

        # Get background data
        background_data = next(iter(data_loader))[0][:num_background]

        # Initialize explainer
        explainer = SHAPExplainer(model, background_data, device)

        # Get data to explain
        explain_data = next(iter(data_loader))[0][:num_explain]

        # Calculate SHAP values
        shap_values = explainer.explain(explain_data)

        return shap_values, to_numpy(explain_data)

    @staticmethod
    def save_shap_values(shap_values, input_data, model_name, scheme, dataset_name, experiment_name, results_dir):
        # Create a directory for SHAP results
        shap_dir = os.path.join(results_dir, 'shap_values')
        os.makedirs(shap_dir, exist_ok=True)

        # Save SHAP values and input data
        np.save(os.path.join(shap_dir, f'{experiment_name}_{model_name}_{scheme}_{dataset_name}_shap_values.npy'), shap_values)
        np.save(os.path.join(shap_dir, f'{experiment_name}_{model_name}_{scheme}_{dataset_name}_input_data.npy'), input_data)

        # Visualize SHAP values
        SHAPExplainer.visualize_shap_values(shap_values, input_data, model_name, scheme, dataset_name, experiment_name, results_dir)

    @staticmethod
    def visualize_shap_values(shap_values, input_data, model_name, scheme, dataset_name, experiment_name, results_dir):
        # Create a directory for SHAP visualizations
        vis_dir = os.path.join(results_dir, 'shap_visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Summarize the SHAP values
        shap_summary = np.mean(np.abs(shap_values[0]), axis=0)

        # Plot SHAP summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[0], input_data, plot_type="bar", show=False)
        plt.title(f'SHAP Summary: {model_name} - {scheme} - {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{experiment_name}_{model_name}_{scheme}_{dataset_name}_shap_summary.png'))
        plt.close()

        # Plot SHAP values for a single example
        plt.figure(figsize=(12, 8))
        shap.image_plot(shap_values[0][0], input_data[0])
        plt.title(f'SHAP Values: {model_name} - {scheme} - {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{experiment_name}_{model_name}_{scheme}_{dataset_name}_shap_image.png'))
        plt.close()
