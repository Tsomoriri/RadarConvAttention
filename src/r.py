import os
import numpy as np
import torch
import torch.nn as nn
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001

def load_data(path):
    data = np.load(path)
    x = data[:, :, :, :4]
    y = data[:, :, :, 4:5]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output, _ = model(batch_x)
        output = output.squeeze(1)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    mae_sum = 0.0
    ssim_sum = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
                    ssim_sum += ssim_value
                except Exception as e:
                    print(f"SSIM calculation error: {e}")
            n_samples += batch_x.size(0)
    avg_loss = total_loss / n_samples
    avg_mae = mae_sum / n_samples
    avg_ssim = ssim_sum / n_samples if n_samples > 0 else 0
    return avg_loss, avg_mae, avg_ssim

def explain_predictions(model, test_loader):
    model.eval()
    explainer = LimeImageExplainer()

    # Get a sample from the test loader
    for batch_x, batch_y in test_loader:
        sample_x = batch_x[0].to(device)  # Shape: [40, 40, 4]
        break

    print(f"Sample shape: {sample_x.shape}")

    def predict(input_array):
        with torch.no_grad():
            # input_array shape: [n_samples, height, width, channels]
            input_tensor = torch.from_numpy(input_array).float().to(device)
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

def visualize_evaluation(model, test_loader, num_batches=10, fps=1):
    model.eval()
    
    all_frames = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Get model prediction
            output, _ = model(batch_x)
            
            # Create frames for this batch
            for j in range(batch_x.size(0)):  # Iterate over samples in the batch
                fig = plt.figure(figsize=(20, 5))
                
                # Plot 4 input images
                for k in range(4):
                    ax = fig.add_subplot(1, 5, k+1, aspect='equal')
                    im = ax.imshow(batch_x[j, k].cpu().numpy(), cmap='RdBu_r')
                    ax.set_title(f'Input {k+1}')
                    ax.axis('off')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Plot predicted image
                ax = fig.add_subplot(1, 5, 5, aspect='equal')
                im = ax.imshow(output[j, 0].cpu().numpy(), cmap='RdBu_r')
                ax.set_title('Predicted')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                
                # Convert plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)
                all_frames.append(img)
                
                plt.close(fig)
    
    # Save as GIF
    all_frames[0].save('model_evaluation.gif', save_all=True, append_images=all_frames[1:], duration=1000//fps, loop=0)
    print("Evaluation visualization saved as 'model_evaluation.gif'")

def main():
    # Load data
    data_path = "/home/tso/RadarConvAttention/src/datasets/radar_movies.npy"
    train_loader, test_loader = load_data(data_path)

    # Initialize model
    model = ConvLSTM(input_dim=4, hidden_dim=40, kernel_size=(3,3), num_layers=2, physics_kernel_size=(3,3), output_dim=1, batch_first=True).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Evaluate model
    test_loss, mae, ssim_value = evaluate_model(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}")

    # Generate LIME explanation
    explanation = explain_predictions(model, test_loader)
    visualize_evaluation(model, test_loader)
    
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
        plt.savefig('lime_explanations.png')
        print("LIME explanations saved as 'lime_explanations.png'")

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

if __name__ == "__main__":
    main()