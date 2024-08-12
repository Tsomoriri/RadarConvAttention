import numpy as np
import matplotlib.pyplot as plt


def update_grid(rin_physics):
    shape = rin_physics.shape
    updated_grid = np.zeros(shape)

    # Calculate gradients to identify areas of change
    grad_x, grad_y = np.gradient(rin_physics)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize the gradient magnitude for easier thresholding
    normalized_grad = (grad_magnitude - grad_magnitude.min()) / (
        grad_magnitude.max() - grad_magnitude.min()
    )

    # Define thresholds for dense and sparse grid regions
    dense_threshold = 0.7  # Adjust as needed
    sparse_threshold = 0.3  # Adjust as needed

    # Create a grid of coordinates
    x, y = np.mgrid[0 : shape[0], 0 : shape[1]]

    # Apply the density logic
    for i in range(shape[0]):
        for j in range(shape[1]):
            if normalized_grad[i, j] > dense_threshold:
                updated_grid[i, j] = 1  # Dense grid point
            elif normalized_grad[i, j] < sparse_threshold:
                updated_grid[i, j] = 0  # No grid point (sparse)
            else:
                # Interpolate between dense and sparse based on gradient magnitude
                density = (normalized_grad[i, j] - sparse_threshold) / (
                    dense_threshold - sparse_threshold
                )
                if np.random.rand() < density:
                    updated_grid[i, j] = (
                        1  # Add a point with probability based on density
                    )

    return updated_grid


def load_radar_image(
    file_path="/home/tso/RadarConvAttention/src/datasets/radar_movies.npy",
):
    try:
        radar_data = np.load(file_path)
        if radar_data.ndim > 2:
            # If the loaded data has more than 2 dimensions, we'll use the first 2D slice
            radar_image = radar_data[
                0, :, :, 2
            ]  # Assuming the first channel is the radar reflectivity
        else:
            radar_image = radar_data
        return radar_image
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the radar image: {e}")
        return None


def visualize_grids(radar_image, updated_grid, grid_size=40):
    if radar_image is None:
        print("Radar image not available for visualization.")
        return

    # Ensure the grid size fits within the radar image dimensions
    radar_height, radar_width = radar_image.shape
    if grid_size > radar_height or grid_size > radar_width:
        print(
            f"Grid size {grid_size} is too large for the radar image dimensions {radar_height}x{radar_width}."
        )
        return

    # Create a figure and axes for plotting (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed
    normal_grid = np.zeros(radar_image.shape)
    # Display the original radar image with the uniform grid on the left
    axes[0].imshow(radar_image, cmap="viridis")
    axes[0].imshow(normal_grid, cmap="grey", alpha=0.5)
    axes[0].set_title("Radar Image with Uniform Grid")

    # Display the radar image with the updated grid on the right
    axes[1].imshow(radar_image, cmap="viridis")
    axes[1].imshow(updated_grid, cmap="grey", alpha=0.5)
    axes[1].set_title("Radar Image with Updated Grid")

    # Adjust layout and save/show the plot
    plt.tight_layout()
    plt.savefig("./radar_grid_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


# Example usage:
radar_image = load_radar_image()  # Load your radar image
print(radar_image.shape)
updated_grid = update_grid(radar_image)  # Update the grid based on your logic
visualize_grids(radar_image, updated_grid)
