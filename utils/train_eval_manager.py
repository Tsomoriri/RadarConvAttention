import os
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




from lime.lime_image import LimeImageExplainer
from skimage.segmentation import felzenszwalb




class TrainEvalManager:
    """
    A class to manage the training and evaluation of machine learning models.

    Attributes:
        models_config (list): Configuration for the models to be trained.
        datasets_config (list): Configuration for the datasets used in training and testing.
        device (torch.device): The device to run the model on (CPU or GPU).
        batch_size (int): The number of samples per batch.
        num_epochs (int): The number of epochs for training.
        learning_rate (float): The learning rate for the optimizer.
        results_dir (str): Directory to save training results and visualizations.

    Methods:
        load_data(path): Loads and splits the dataset into training and testing sets.
        train_model(model, train_loader, optimizer, loss_fn, scheme): Trains the model on the training data.
        evaluate_model(model, test_loader): Evaluates the model on the test data.
        update_grid(rin_physics): Updates the grid based on physics-based logic.
        create_comparison_gif(outputs, targets, filename): Creates a GIF comparing model outputs and ground truth.
        run_experiment(model_config, train_dataset_path, test_dataset_paths, experiment_name): Runs a training and evaluation experiment.
        run_all_experiments(): Runs experiments for all datasets and models.
        run_extrapolation_experiment(): Runs a specific extrapolation experiment.
        explain_predictions(model, test_loader): Generates LIME explanations for model predictions.
        visualize_lime_explanations(model, test_loader, explainer, num_labels=5): Visualizes LIME explanations for multiple labels.
    """
    def __init__(
        self,
        models_config,
        datasets_config,
        device="cuda",
        batch_size=32,
        num_epochs=50,
        learning_rate=0.001,
    ):
        self.models_config = models_config
        self.datasets_config = datasets_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.results_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "Train_results"
        )
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self, path):
        data = np.load(path)

        x = data[:, :, :, :4]
        y = data[:, :, :, 4:5]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        train_dataset = TensorDataset(
            torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, test_loader

    def train_model(self, model, train_loader, optimizer, loss_fn, scheme):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            output, _ = model(batch_x)
            output = output.squeeze(1)

            if scheme == "standard":
                loss = loss_fn(output, batch_y)
            elif scheme in ["physics", "physics_dynamic_grid"]:
                data_loss = loss_fn(output, batch_y)
                rin_physics = torch.zeros_like(
                    batch_x, device=self.device, requires_grad=True
                )
                if scheme == "physics_dynamic_grid":
                    rin_physics = self.update_grid(rin_physics.cpu().detach().numpy())
                    rin_physics = torch.tensor(
                        rin_physics,
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=True,
                    )
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
                        ssim_value = ssim(
                            output_np[i].squeeze(),
                            batch_y_np[i].squeeze(),
                            data_range=batch_y_np[i].max() - batch_y_np[i].min(),
                        )
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

        return (
            float(avg_loss),
            float(avg_mae),
            float(avg_ssim),
            all_outputs,
            all_targets,
        )

    def update_grid(self, rin_physics):
        batch_size, grid_height, grid_width, channels = rin_physics.shape
        updated_grid = np.zeros((batch_size, grid_height, grid_width, channels))

        # Define thresholds for dense and sparse grid regions
        dense_threshold = 0.7  # Adjust as needed
        sparse_threshold = 0.3  # Adjust as needed

        for i in range(batch_size):
            grid = rin_physics[i, :, :, 0]  # Process the first channel

            # Calculate gradients
            grad_x, grad_y = np.gradient(grid)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize the gradient magnitude
            normalized_grad = (grad_magnitude - grad_magnitude.min()) / (
                grad_magnitude.max() - grad_magnitude.min()
            )

            # Apply the density logic
            for x in range(grid_height):
                for y in range(grid_width):
                    if normalized_grad[x, y] > dense_threshold:
                        updated_grid[i, x, y, 0] = 1  # Dense grid point
                    elif normalized_grad[x, y] < sparse_threshold:
                        updated_grid[i, x, y, 0] = 0  # No grid point (sparse)
                    else:
                        # Interpolate between dense and sparse
                        density = (normalized_grad[x, y] - sparse_threshold) / (
                            dense_threshold - sparse_threshold
                        )
                        if np.random.rand() < density:
                            updated_grid[i, x, y, 0] = 1

        return updated_grid

    def create_comparison_gif(self, outputs, targets, filename):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        def update(i):
            ax[0].clear()
            ax[1].clear()
            ax[0].imshow(outputs[i].squeeze(), cmap="viridis")
            ax[0].set_title("Model Output")
            ax[1].imshow(targets[i].squeeze(), cmap="viridis")
            ax[1].set_title("Ground Truth")

        anim = FuncAnimation(
            fig, update, frames=min(len(outputs), 100), interval=200
        )  # Limit to 100 frames
        anim.save(filename, writer="pillow", fps=5)
        print(f"Comparison GIF saved as {filename}")
        plt.close(fig)

    def run_experiment(
        self, model_config, train_dataset_path, test_dataset_paths, experiment_name
    ):
        model_name, model_class, model_params, training_schemes = model_config
        train_loader, _ = self.load_data(train_dataset_path)

        results = []

        for scheme in training_schemes:
            print(f"Model Parameters: {model_params}")
            model = model_class(**model_params).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.MSELoss()

            print(
                f"Training {model_name} with {scheme} scheme on {os.path.basename(train_dataset_path)}"
            )
            for epoch in range(self.num_epochs):
                train_loss = self.train_model(
                    model, train_loader, optimizer, loss_fn, scheme
                )
                if (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {train_loss:.4f}"
                    )

            # Evaluate on each test dataset
            if isinstance(test_dataset_paths, list):
                for test_dataset_path in test_dataset_paths:
                    _, test_loader = self.load_data(test_dataset_path)
                    test_loss, mae, ssim_value, outputs, targets = self.evaluate_model(
                        model, test_loader
                    )
                    print(f"Test on {os.path.basename(test_dataset_path)}:")
                    print(
                        f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}"
                    )

                    # Generate LIME explanation
                    explanation = self.explain_predictions(model, test_loader)

                    if explanation:
                        # Get all labels explained by LIME
                        labels = explanation.top_labels

                        # Create a figure with subplots for each label
                        fig, axes = plt.subplots(
                            len(labels), 1, figsize=(10, 5 * len(labels))
                        )
                        if len(labels) == 1:
                            axes = [axes]  # Ensure axes is always a list

                        for idx, label in enumerate(labels):
                            temp, mask = explanation.get_image_and_mask(
                                label,
                                positive_only=True,
                                num_features=5,
                                hide_rest=True,
                            )
                            axes[idx].imshow(mask, cmap="RdBu_r", alpha=0.7)
                            axes[idx].set_title(f"LIME Explanation for Label {label}")
                            fig.colorbar(
                                axes[idx].imshow(mask, cmap="RdBu_r", alpha=0.7),
                                ax=axes[idx],
                            )

                        plt.tight_layout()
                        lime_filename = os.path.join(
                            self.results_dir,
                            f"lime_explanations_{model_name}_{scheme}.png",
                        )
                        plt.savefig(lime_filename)
                        print(f"LIME explanations saved as '{lime_filename}'")

                        # Print additional information about the explanation
                        for label in explanation.top_labels:
                            print(f"Label {label}:")
                            if (
                                hasattr(explanation, "local_pred")
                                and explanation.local_pred is not None
                            ):
                                if isinstance(
                                    explanation.local_pred, (list, np.ndarray)
                                ):
                                    print(
                                        f"  Local prediction: {explanation.local_pred[0]}"
                                    )
                                else:
                                    print(
                                        f"  Local prediction: {explanation.local_pred}"
                                    )
                            else:
                                print("  Local prediction: Not available")

                            if (
                                hasattr(explanation, "intercept")
                                and explanation.intercept is not None
                            ):
                                if isinstance(explanation.intercept, dict):
                                    print(
                                        f"  Intercept: {explanation.intercept.get(label, 'Not available')}"
                                    )
                                else:
                                    print(f"  Intercept: {explanation.intercept}")
                            else:
                                print("  Intercept: Not available")

                            print("  Top features:")
                            if (
                                hasattr(explanation, "local_exp")
                                and explanation.local_exp is not None
                            ):
                                if (
                                    isinstance(explanation.local_exp, dict)
                                    and label in explanation.local_exp
                                ):
                                    for feature, weight in sorted(
                                        explanation.local_exp[label],
                                        key=lambda x: abs(x[1]),
                                        reverse=True,
                                    )[:5]:
                                        print(f"    Feature {feature}: {weight}")
                                else:
                                    print("    Not available")
                            else:
                                print("    Not available")
                            print()
                    else:
                        print("No explanation generated.")
                    print(f"Test on {os.path.basename(test_dataset_paths)}:")
                    print(
                        f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}"
                    )

                    def generate_save_name(model_name, model_type, version=None):
                        """
                        Generates a unique and descriptive save name for a model.

                        Args:
                            model_name: The base name of the model (e.g., "convlstm").
                            model_type: A string indicating the type or architecture of the model (e.g., "2d", "3d").
                            version: An optional version number or identifier.

                        Returns:
                            A string representing the save name for the model.
                        """
                        save_name = (
                            model_name.upper()
                        )  # Convert to uppercase for consistency

                        if model_type:
                            save_name += (
                                f"_{model_type.upper()}"  # Add model type if available
                            )

                        if version:
                            save_name += f"_v{version}"  # Add version if available

                
                        return save_name

                    results.append(
                        {
                            "model": generate_save_name(model_name, scheme),
                            "scheme": scheme,
                            "train_dataset": os.path.basename(train_dataset_path),
                            "test_dataset": os.path.basename(test_dataset_path),
                            "test_loss": test_loss,
                            "mae": mae,
                            "ssim": ssim_value,
                        }
                    )

                    # Create and save comparison GIF
                    gif_filename = os.path.join(
                        self.results_dir,
                        f"{experiment_name}_{model_name}_{scheme}_{os.path.basename(test_dataset_path)}_comparison.gif",
                    )
                    self.create_comparison_gif(outputs, targets, gif_filename)
                    print(f"Comparison GIF saved as {gif_filename}")

                    # Save results dictionary
                    results_filename = gif_filename.replace(".gif", "_results.json")
                    with open(results_filename, "w") as f:
                        json.dump(results[-1], f, indent=4)
                    print(f"Results saved as {results_filename}")
            else:
                _, test_loader = self.load_data(test_dataset_paths)
                test_loss, mae, ssim_value, outputs, targets = self.evaluate_model(
                    model, test_loader
                )
                # Generate LIME explanation
                explanation = self.explain_predictions(model, test_loader)

                if explanation:
                    # Get all labels explained by LIME
                    labels = explanation.top_labels

                    # Create a figure with subplots for each label
                    fig, axes = plt.subplots(
                        len(labels), 1, figsize=(10, 5 * len(labels))
                    )
                    if len(labels) == 1:
                        axes = [axes]  # Ensure axes is always a list

                    for idx, label in enumerate(labels):
                        temp, mask = explanation.get_image_and_mask(
                            label, positive_only=True, num_features=5, hide_rest=True
                        )
                        axes[idx].imshow(mask, cmap="RdBu_r", alpha=0.7)
                        axes[idx].set_title(f"LIME Explanation for Label {label}")
                        fig.colorbar(
                            axes[idx].imshow(mask, cmap="RdBu_r", alpha=0.7),
                            ax=axes[idx],
                        )

                    plt.tight_layout()
                    lime_filename = os.path.join(
                        self.results_dir, f"lime_explanations_{model_name}_{scheme}.png"
                    )
                    plt.savefig(lime_filename)
                    print(f"LIME explanations saved as '{lime_filename}'")

                    # Print additional information about the explanation
                    for label in explanation.top_labels:
                        print(f"Label {label}:")
                        if (
                            hasattr(explanation, "local_pred")
                            and explanation.local_pred is not None
                        ):
                            if isinstance(explanation.local_pred, (list, np.ndarray)):
                                print(
                                    f"  Local prediction: {explanation.local_pred[0]}"
                                )
                            else:
                                print(f"  Local prediction: {explanation.local_pred}")
                        else:
                            print("  Local prediction: Not available")

                        if (
                            hasattr(explanation, "intercept")
                            and explanation.intercept is not None
                        ):
                            if isinstance(explanation.intercept, dict):
                                print(
                                    f"  Intercept: {explanation.intercept.get(label, 'Not available')}"
                                )
                            else:
                                print(f"  Intercept: {explanation.intercept}")
                        else:
                            print("  Intercept: Not available")

                        print("  Top features:")
                        if (
                            hasattr(explanation, "local_exp")
                            and explanation.local_exp is not None
                        ):
                            if (
                                isinstance(explanation.local_exp, dict)
                                and label in explanation.local_exp
                            ):
                                for feature, weight in sorted(
                                    explanation.local_exp[label],
                                    key=lambda x: abs(x[1]),
                                    reverse=True,
                                )[:5]:
                                    print(f"    Feature {feature}: {weight}")
                            else:
                                print("    Not available")
                        else:
                            print("    Not available")
                        print()

                    # Add the new detailed LIME visualization
                    explainer = LimeImageExplainer()
                    fig = self.visualize_lime_explanations(
                        model, test_loader, explainer
                    )
                    lime_filename = os.path.join(
                        self.results_dir,
                        f"detailed_lime_explanations_{model_name}_{scheme}.png",
                    )
                    fig.savefig(lime_filename)
                    plt.close(fig)
                    print(f"Detailed LIME explanations saved as '{lime_filename}'")
                else:
                    print("No explanation generated.")
                print(f"Test on {os.path.basename(test_dataset_paths)}:")
                print(
                    f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}"
                )

                results.append(
                    {
                        "model": model_name,
                        "scheme": scheme,
                        "train_dataset": os.path.basename(train_dataset_path),
                        "test_dataset": os.path.basename(test_dataset_paths),
                        "test_loss": test_loss,
                        "mae": mae,
                        "ssim": ssim_value,
                    }
                )

                # Create and save comparison GIF
                gif_filename = os.path.join(
                    self.results_dir,
                    f"{experiment_name}_{model_name}_{scheme}_{os.path.basename(test_dataset_paths)}_comparison.gif",
                )
                self.create_comparison_gif(outputs, targets, gif_filename)
                print(f"Comparison GIF saved as {gif_filename}")

                # Save results dictionary
                results_filename = gif_filename.replace(".gif", "_results.json")
                with open(results_filename, "w") as f:
                    json.dump(results[-1], f, indent=4)
                print(f"Results saved as {results_filename}")

    def run_all_experiments(self):
        for dataset_config in self.datasets_config:
            dataset_path, experiment_name = dataset_config
            for model_config in self.models_config:
                self.run_experiment(
                    model_config, dataset_path, dataset_path, experiment_name
                )

    def run_extrapolation_experiment(self):
        train_dataset_path = (
            "/home/sushen/PhysNet-RadarNowcast/src/datasets/rect_movie.npy"
        )
        test_dataset_paths = [
            "/home/sushen/PhysNet-RadarNowcast/src/datasets/3rect_movie.npy",
            "/home/sushen/PhysNet-RadarNowcast/src/datasets/11rect_movie.npy",
        ]
        experiment_name = "train_1rect_test_3_11rect"

        for model_config in self.models_config:
            self.run_experiment(
                model_config, train_dataset_path, test_dataset_paths, experiment_name
            )

    def explain_predictions(self, model, test_loader):
        model.eval()
        explainer = LimeImageExplainer()

        # Get a sample from the test loader
        for batch_x, batch_y in test_loader:
            sample_x = batch_x[0].to(self.device)  # Shape: [40, 40, 4]
            # Ensure sample is 4D: (batch_size, height, width, channels)
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
                if output.dim() == 5:
                    output = output[:, -1]  # Take the last time step if 5D

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
                segmentation_fn=custom_segmentation,
            )
            return explanation
        except Exception as e:
            print(f"Error generating LIME explanation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def visualize_lime_explanations(self, model, test_loader, explainer, num_labels=5):
        model.eval()

        # Get a sample from the test loader
        for batch_x, batch_y in test_loader:
            sample_x = batch_x[0].to(self.device)
            break

        lime_input = sample_x.cpu().numpy()

        def predict_fn(input_array):
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_array).float().to(self.device)
                input_tensor = input_tensor.permute(0, 3, 1, 2).unsqueeze(1)
                output, _ = model(input_tensor)
                return output.view(output.size(0), -1).cpu().numpy()

        def custom_segmentation(image):
            mean_image = np.mean(image, axis=2)
            return felzenszwalb(mean_image, scale=100, sigma=0.5, min_size=50)

        explanation = explainer.explain_instance(
            lime_input,
            predict_fn,
            top_labels=num_labels,
            hide_color=0,
            num_samples=100,
            segmentation_fn=custom_segmentation,
        )

        # Create a figure with subplots for each label
        fig, axes = plt.subplots(num_labels, 2, figsize=(20, 6 * num_labels))

        for idx, label in enumerate(explanation.top_labels):
            # Get the image and mask for this label
            temp, mask = explanation.get_image_and_mask(
                label, positive_only=True, num_features=10, hide_rest=False
            )

            # Plot original image
            axes[idx, 0].imshow(np.mean(lime_input, axis=2), cmap="gray")
            axes[idx, 0].set_title("Original Image")
            axes[idx, 0].axis("off")

            # Plot LIME explanation
            im = axes[idx, 1].imshow(
                mask, cmap="hot", alpha=0.7, interpolation="nearest"
            )
            axes[idx, 1].imshow(np.mean(lime_input, axis=2), cmap="gray", alpha=0.3)
            axes[idx, 1].set_title(f"LIME Explanation for Label {label}")
            axes[idx, 1].axis("off")

            # Add colorbar
            cbar = fig.colorbar(im, ax=axes[idx, 1])
            cbar.set_label("LIME importance")

            # Highlight the specific pixel/region for this label
            y, x = np.unravel_index(label, lime_input.shape[:2])
            axes[idx, 1].add_patch(
                plt.Circle((x, y), radius=3, color="blue", fill=False)
            )
            axes[idx, 1].text(
                x + 5,
                y + 5,
                f"Label {label}",
                color="blue",
                fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        plt.tight_layout()
        return fig
