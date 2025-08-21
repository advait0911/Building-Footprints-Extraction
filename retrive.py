import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and display the image and its corresponding mask
def display_image_and_mask(image_path, folder_path='train'):
    # Extract the base filename (e.g., "n_image.tif" -> "n")
    base_name = os.path.basename(image_path)
    file_prefix = base_name.split('_image.tif')[0]  # Get the "n" part
    
    # Construct the corresponding label file name
    label_file_name = f"{file_prefix}_label.tif"
    label_path = os.path.join(folder_path, label_file_name)
    
    # Load the input image and the corresponding label using PIL
    try:
        input_image = Image.open(os.path.join(folder_path, base_name))
        true_mask = Image.open(label_path)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Convert images to numpy arrays for visualization
    input_image_np = np.array(input_image)
    true_mask_np = np.array(true_mask)
    
    # Display the original image and the true mask side-by-side
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # True Mask
    plt.subplot(1, 2, 2)
    plt.imshow(true_mask_np, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.show()

# Main function to input an image file and display its corresponding label
if __name__ == "__main__":
    # User input for the image file path
    image_file_path = input("Enter the path to the image file (e.g., train/1_image.tif): ")
    
    # Check if the input image exists in the train folder
    if not os.path.exists(image_file_path):
        print(f"Error: The file '{image_file_path}' does not exist.")
    else:
        # Display the image and its corresponding mask
        display_image_and_mask(image_file_path)
