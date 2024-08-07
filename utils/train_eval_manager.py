import os
import time
import configparser
import argparse
import ast
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import your model classes here
from src.models.ConvLSTM import ConvLSTM
from src.models.ConvLSTM_Physics import ConvLSTM_iPINN as ConvLSTM_Physics
from src.models.AttentionConvLSTM import ConvLSTM as ConvLSTM_Attention
from src.models.AttentionConvLSTM_Physics import ConvLSTM as ConvLSTM_Attention_Physics
from utils.shap_manager import SHAPExplainer

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from lime.lime_image import LimeImageExplainer
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
# Import your ConvLSTM model
from src.models.ConvLSTM import ConvLSTM
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import io
from PIL import Image

class TrainEvalManager:
    def __init__(self, models_config, datasets_config, device='cuda', batch_size=32, num_epochs=50, learning_rate=0.001):
        self.models_config = models_config
        self.datasets_config = datasets_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Train_results')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self, path):
        data = np.load(path)

        x = data[:, :, :, :4]
        y = data[:, :, :, 4:5]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train_model(self, model, train_loader, optimizer, loss_fn, scheme):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            output, _ = model(batch_x)
            output = output.squeeze(1)

            if scheme == 'standard':
                loss = loss_fn(output, batch_y)
            elif scheme in ['physics', 'physics_dynamic_grid']:
                data_loss = loss_fn(output, batch_y)
                rin_physics = torch.zeros_like(batch_x, device=self.device, requires_grad=True)
                if scheme == 'physics_dynamic_grid':
                    rin_physics = self.update_grid(rin_physics.cpu().detach().numpy())
                    rin_physics = torch.tensor(rin_physics, dtype=torch.float32, device=self.device, requires_grad=True)
                physics_output, _ = model(rin_physics)
                physics_loss = model.advection_loss(rin_physics, physics_output)
                loss = 2 * data_loss + physics_loss
            else:
                raise ValueError(f"Unknown training scheme: {scheme}")

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        return total_loss / len(train_loader.dataset)

    def evaluate_model(self, model, test_loader):
        model.eval()
        total_loss = 0.0
        mae_sum = 0.0
        ssim_sum = 0.0
        ssim_count = 0  # Initialize ssim_count here
        n_samples = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output, _ = model(batch_x)
                output = output.squeeze(1)

                loss = nn.MSELoss()(output, batch_y)
                total_loss += loss.item() * batch_x.size(0)

                output_np = output.cpu().numpy()
                batch_y_np = batch_y.cpu().numpy()

                mae_sum += np.sum(np.abs(output_np - batch_y_np))
                for i in range(output_np.shape[0]):
                    try:
                        ssim_value = ssim(output_np[i].squeeze(), batch_y_np[i].squeeze(), data_range=batch_y_np[i].max() - batch_y_np[i].min())
                        if not np.isnan(ssim_value):
                            ssim_sum += ssim_value
                            ssim_count += 1
                    except Exception as e:
                        print(f"SSIM calculation failed: {e}")
                n_samples += batch_x.size(0)

                all_outputs.append(output_np)
                all_targets.append(batch_y_np)

        avg_loss = total_loss / n_samples
        avg_mae = mae_sum / n_samples
        avg_ssim = ssim_sum / n_samples

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        return avg_loss, avg_mae, avg_ssim, all_outputs, all_targets

    def update_grid(self, rin_physics):
        # Get the shape of the input tensor
        shape = rin_physics.shape
        # Create an empty tensor with the same shape
        updated_grid = np.zeros(shape)

        # Iterate through each element in the batch
        for i in range(shape[0]):
            # Extract the individual grid
            grid = rin_physics[i]

            # Find the max and min x, y values
            max_x, max_y = np.unravel_index(np.argmax(grid[:, :, 0]), grid[:, :, 0].shape)
            min_x, min_y = np.unravel_index(np.argmin(grid[:, :, 0]), grid[:, :, 0].shape)

            # Set the pattern
            updated_grid[i, max_x, max_y, :] = 1
            updated_grid[i, min_x, min_y, :] = 0

        return updated_grid
    def create_comparison_gif(self, outputs, targets, filename):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        def update(i):
            ax[0].clear()
            ax[1].clear()
            ax[0].imshow(outputs[i].squeeze(), cmap='viridis')
            ax[0].set_title('Model Output')
            ax[1].imshow(targets[i].squeeze(), cmap='viridis')
            ax[1].set_title('Ground Truth')

        anim = FuncAnimation(fig, update, frames=min(len(outputs), 100), interval=200)  # Limit to 100 frames
        anim.save(filename, writer='pillow', fps=5)
        print(f"Comparison GIF saved as {filename}")
        plt.close(fig)

    def run_experiment(self, model_config, train_dataset_path, test_dataset_paths, experiment_name):
        model_name, model_class, model_params, training_schemes = model_config
        train_loader,_ = self.load_data(train_dataset_path)

        results = []

        for scheme in training_schemes:
            print(f"Model Parameters: {model_params}")
            model = model_class(**model_params).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.MSELoss()

            print(f"Training {model_name} with {scheme} scheme on {os.path.basename(train_dataset_path)}")
            for epoch in range(self.num_epochs):
                train_loss = self.train_model(model, train_loader, optimizer, loss_fn, scheme)
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {train_loss:.4f}")

             # Evaluate on each test dataset
            if isinstance(test_dataset_paths,list):
                for test_dataset_path in test_dataset_paths:
                    _,test_loader = self.load_data(test_dataset_path)
                    test_loss, mae, ssim_value, outputs, targets = self.evaluate_model(model, test_loader)
                    print(f"Test on {os.path.basename(test_dataset_path)}:")
                    print(f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}")

                     # Generate LIME explanation
                    explanation = self.explain_predictions(model, test_loader)
                    
                    if explanation:
                        # Get all labels explained by LIME
                        labels = explanation.top_labels

                        # Create a figure with subplots for each label
                        fig, axes = plt.subplots(len(labels), 1, figsize=(10, 5*len(labels)))
                        if len(labels) == 1:
                            axes = [axes]  # Ensure axes is always a list

                        for idx, label in enumerate(labels):
                            temp, mask = explanation.get_image_and_mask(
                                label, positive_only=True, num_features=5, hide_rest=True
                            )
                            axes[idx].imshow(mask, cmap='RdBu_r', alpha=0.7)
                            axes[idx].set_title(f"LIME Explanation for Label {label}")
                            fig.colorbar(axes[idx].imshow(mask, cmap='RdBu_r', alpha=0.7), ax=axes[idx])

                        plt.tight_layout()
                        lime_filename = os.path.join(self.results_dir, f'lime_explanations_{model_name}_{scheme}.png')
                        plt.savefig(lime_filename)
                        print(f"LIME explanations saved as '{lime_filename}'")

                        # Print additional information about the explanation
                        for label in explanation.top_labels:
                            print(f"Label {label}:")
                            if hasattr(explanation, 'local_pred') and explanation.local_pred is not None:
                                if isinstance(explanation.local_pred, (list, np.ndarray)):
                                    print(f"  Local prediction: {explanation.local_pred[0]}")
                                else:
                                    print(f"  Local prediction: {explanation.local_pred}")
                            else:
                                print("  Local prediction: Not available")
                            
                            if hasattr(explanation, 'intercept') and explanation.intercept is not None:
                                if isinstance(explanation.intercept, dict):
                                    print(f"  Intercept: {explanation.intercept.get(label, 'Not available')}")
                                else:
                                    print(f"  Intercept: {explanation.intercept}")
                            else:
                                print("  Intercept: Not available")
                            
                            print("  Top features:")
                            if hasattr(explanation, 'local_exp') and explanation.local_exp is not None:
                                if isinstance(explanation.local_exp, dict) and label in explanation.local_exp:
                                    for feature, weight in sorted(explanation.local_exp[label], key=lambda x: abs(x[1]), reverse=True)[:5]:
                                        print(f"    Feature {feature}: {weight}")
                                else:
                                    print("    Not available")
                            else:
                                print("    Not available")
                            print()
                    else:
                        print("No explanation generated.")
                    print(f"Test on {os.path.basename(test_dataset_paths)}:")
                    print(f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}")

                        
                    results.append({
                        'model': model_name,
                        'scheme': scheme,
                        'train_dataset': os.path.basename(train_dataset_path),
                        'test_dataset': os.path.basename(test_dataset_path),
                        'test_loss': test_loss,
                        'mae': mae,
                        'ssim': ssim_value
                    })

                    # Create and save comparison GIF
                    gif_filename = os.path.join(self.results_dir,
                                                f'{experiment_name}_{model_name}_{scheme}_{os.path.basename(test_dataset_path)}_comparison.gif')
                    self.create_comparison_gif(outputs, targets, gif_filename)
                    print(f"Comparison GIF saved as {gif_filename}")

                    # Save results dictionary
                    results_filename = gif_filename.replace('.gif', '_results.json')
                    with open(results_filename, 'w') as f:
                        json.dump(results[-1], f, indent=4)
                    print(f"Results saved as {results_filename}")
            else:
                _,test_loader = self.load_data(test_dataset_paths)
                test_loss, mae, ssim_value, outputs, targets = self.evaluate_model(model, test_loader)
                # Generate LIME explanation
                explanation = self.explain_predictions(model, test_loader)
                
                if explanation:
                    # Get all labels explained by LIME
                    labels = explanation.top_labels

                    # Create a figure with subplots for each label
                    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 5*len(labels)))
                    if len(labels) == 1:
                        axes = [axes]  # Ensure axes is always a list

                    for idx, label in enumerate(labels):
                        temp, mask = explanation.get_image_and_mask(
                            label, positive_only=True, num_features=5, hide_rest=True
                        )
                        axes[idx].imshow(mask, cmap='RdBu_r', alpha=0.7)
                        axes[idx].set_title(f"LIME Explanation for Label {label}")
                        fig.colorbar(axes[idx].imshow(mask, cmap='RdBu_r', alpha=0.7), ax=axes[idx])

                    plt.tight_layout()
                    lime_filename = os.path.join(self.results_dir, f'lime_explanations_{model_name}_{scheme}.png')
                    plt.savefig(lime_filename)
                    print(f"LIME explanations saved as '{lime_filename}'")

                    # Print additional information about the explanation
                    for label in explanation.top_labels:
                        print(f"Label {label}:")
                        if hasattr(explanation, 'local_pred') and explanation.local_pred is not None:
                            if isinstance(explanation.local_pred, (list, np.ndarray)):
                                print(f"  Local prediction: {explanation.local_pred[0]}")
                            else:
                                print(f"  Local prediction: {explanation.local_pred}")
                        else:
                            print("  Local prediction: Not available")
                        
                        if hasattr(explanation, 'intercept') and explanation.intercept is not None:
                            if isinstance(explanation.intercept, dict):
                                print(f"  Intercept: {explanation.intercept.get(label, 'Not available')}")
                            else:
                                print(f"  Intercept: {explanation.intercept}")
                        else:
                            print("  Intercept: Not available")
                        
                        print("  Top features:")
                        if hasattr(explanation, 'local_exp') and explanation.local_exp is not None:
                            if isinstance(explanation.local_exp, dict) and label in explanation.local_exp:
                                for feature, weight in sorted(explanation.local_exp[label], key=lambda x: abs(x[1]), reverse=True)[:5]:
                                    print(f"    Feature {feature}: {weight}")
                            else:
                                print("    Not available")
                        else:
                            print("    Not available")
                        print()
                else:
                    print("No explanation generated.")
                print(f"Test on {os.path.basename(test_dataset_paths)}:")
                print(f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}")

                results.append({
                    'model': model_name,
                    'scheme': scheme,
                    'train_dataset': os.path.basename(train_dataset_path),
                    'test_dataset': os.path.basename(test_dataset_paths),
                    'test_loss': test_loss,
                    'mae': mae,
                    'ssim': ssim_value
                })

                # Create and save comparison GIF
                gif_filename = os.path.join(self.results_dir,
                                            f'{experiment_name}_{model_name}_{scheme}_{os.path.basename(test_dataset_paths)}_comparison.gif')
                self.create_comparison_gif(outputs, targets, gif_filename)
                print(f"Comparison GIF saved as {gif_filename}")

                # Save results dictionary
                results_filename = gif_filename.replace('.gif', '_results.json')
                with open(results_filename, 'w') as f:
                    json.dump(results[-1], f, indent=4)
                print(f"Results saved as {results_filename}")
                            
    
    def run_all_experiments(self):
        for dataset_config in self.datasets_config:
            dataset_path, experiment_name = dataset_config
            for model_config in self.models_config:
                self.run_experiment(model_config, dataset_path, dataset_path, experiment_name)

    def run_extrapolation_experiment(self):
        train_dataset_path = "/home/sushen/PhysNet-RadarNowcast/src/datasets/rect_movie.npy"
        test_dataset_paths = [
            "/home/sushen/PhysNet-RadarNowcast/src/datasets/3rect_movie.npy",
            "/home/sushen/PhysNet-RadarNowcast/src/datasets/11rect_movie.npy"
        ]
        experiment_name = "train_1rect_test_3_11rect"

        for model_config in self.models_config:
            self.run_experiment(model_config, train_dataset_path, test_dataset_paths, experiment_name)
    
    
    def explain_predictions(self,model, test_loader):
        model.eval()
        explainer = LimeImageExplainer()

        # Get a sample from the test loader
        for batch_x, batch_y in test_loader:
            sample_x = batch_x[0].to(self.device)  # Shape: [40, 40, 4]
            break

        print(f"Sample shape: {sample_x.shape}")

        def predict(input_array):
            with torch.no_grad():
                # input_array shape: [n_samples, height, width, channels]
                input_tensor = torch.from_numpy(input_array).float().to(self.device)
                print(f"Input tensor shape: {input_tensor.shape}")  # Debug print
                
                # Reshape to [n_samples, 1, channels, height, width]
                input_tensor = input_tensor.permute(0, 3, 1, 2).unsqueeze(1)
                
                output, _ = model(input_tensor)
                # output shape: [n_samples, 1, height, width]
                
                # Reshape to [n_samples, height * width]
                return output.view(output.size(0), -1).cpu().numpy()

        def custom_segmentation(image):
            # Use the mean of all channels for segmentation
            mean_image = np.mean(image, axis=2)
            return felzenszwalb(mean_image, scale=100, sigma=0.5, min_size=50)

        try:
            # Prepare the input for LIME
            lime_input = sample_x.cpu().numpy()  # [height, width, channels]
            
            explanation = explainer.explain_instance(
                lime_input,
                predict,
                top_labels=5,
                hide_color=0,
                num_samples=100,
                segmentation_fn=custom_segmentation
            )
            return explanation
        except Exception as e:
            print(f"Error generating LIME explanation: {e}")
            import traceback
            traceback.print_exc()
            return None
