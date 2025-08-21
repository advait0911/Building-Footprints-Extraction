import os
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Function to normalize the mask for better visibility
def normalize_mask(mask_image):
    mask_array = np.array(mask_image)
    
    # Scale mask to 0-255 for better visualization
    scaled_mask = (mask_array - mask_array.min()) / (mask_array.max() - mask_array.min())
    scaled_mask = (scaled_mask * 255).astype(np.uint8)
    
    # Convert back to an Image object
    return Image.fromarray(scaled_mask)

# Streamlit App
def main():
    st.title("Building Footprint Extraction")
    st.write("Upload an image file (in format `n_image.tif`).")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image file", type=["tif", "tiff"])
    
    if uploaded_file is not None:
        # Extract the filename
        file_name = uploaded_file.name
        
        # Send the image to the Flask API along with the filename
        try:
            response = requests.post(
                "http://127.0.0.1:5000/get-mask",  # Corrected URL for the Flask server
                files={'image': uploaded_file.getvalue()},
                data={'filename': file_name}
            )

            if response.status_code == 200:
                # Load the received mask image directly from the response content
                mask_image = Image.open(BytesIO(response.content))

                # Normalize the mask for visualization
                enhanced_mask_image = normalize_mask(mask_image)

                # Display the input image and received mask side-by-side
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # Display Original Image
                input_image = Image.open(uploaded_file)
                ax[0].imshow(input_image)
                ax[0].set_title('Original Image')
                ax[0].axis('off')
                
                # Display enhanced Mask
                ax[1].imshow(enhanced_mask_image, cmap='gray')
                ax[1].set_title('Enhanced Predicted Mask')
                ax[1].axis('off')
                
                # Display the plot in Streamlit
                st.pyplot(fig)
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
