import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def split_and_label_gifs(folder_path):
    image_data = []
    target_model_name = "convlstm_atn_phys_dynamic_grid"

    for filename in os.listdir(folder_path):
        if filename.endswith(".gif"):
            filepath = os.path.join(folder_path, filename)

            # Extract model name, prioritizing the target name
            if target_model_name in filename:
                model_name = target_model_name
            else:
                # Handle other variations
                model_name = next((name for name in 
                                   ["convlstm_atn_phys_dg","convlstm_phys_dg","convlstm_atn_phys", "convlstm_atn", "convlstm_phys", "convlstm" ] 
                                   if name in filename), "unknown") 

            gif = imageio.mimread(filepath)
            frames = [Image.fromarray(frame) for frame in gif[:5]] # Convert frames to list of Images
            image_data.append((frames, model_name)) # Append the list of frames


    return image_data

def create_grid_image(image_data):
    fig, axes = plt.subplots(nrows=len(image_data), ncols=5, figsize=(30, len(image_data) * 3))

    for i, (image, model_name) in enumerate(image_data):
        for j in range(5):
            axes[i, j].imshow(image[j])
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(model_name)

    plt.tight_layout()

    # Convert to PIL Image
    canvas = FigureCanvas(fig)
    canvas.draw()
    grid_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

    return grid_image

if __name__ == "__main__":
    folder_path = "/home/tso/RadarConvAttention/Train_results/gift"  # Replace with your folder path
    image_data = split_and_label_gifs(folder_path)
    grid_image = create_grid_image(image_data)

    # Save or display the grid image
    grid_image.save("grid_output.png",dpi=(900,900))
    # grid_image.show() 
