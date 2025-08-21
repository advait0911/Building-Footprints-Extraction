import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('./my_modelv2.keras')

# Load the original image
image_cv = cv2.imread('./2_image.tif')

# Prepare the image for prediction
input_size = (128, 128)  # Adjust to the input size your model expects
resized_image = cv2.resize(image_cv, input_size)
normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
input_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

# Predict the mask using the model
predicted_mask = model.predict(input_image)

# Inspect the raw prediction output
print("Prediction value range:", predicted_mask.min(), "to", predicted_mask.max())

# Normalize prediction to [0, 1] range for visualization
normalized_predicted_mask = (predicted_mask[0, :, :, 0] - predicted_mask.min()) / (predicted_mask.max() - predicted_mask.min())

# Enhanced visualization by scaling up the contrast
# Scale to [0, 1] for visualization
scaled_mask = (normalized_predicted_mask - normalized_predicted_mask.min()) / (normalized_predicted_mask.max() - normalized_predicted_mask.min())

# Display the original image and the scaled predicted mask for enhanced contrast
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Scaled Predicted Mask with Enhanced Contrast
plt.subplot(1, 2, 2)
plt.imshow(scaled_mask, cmap='gray')
plt.title('Enhanced Contrast - Raw Predicted Mask')
plt.axis('off')

plt.show()

# Save the enhanced contrast predicted mask
cv2.imwrite('enhanced_contrast_predicted_mask.png', (scaled_mask * 255).astype(np.uint8))
print("Enhanced contrast predicted mask saved as 'enhanced_contrast_predicted_mask.png'.")

# Experiment with a much lower threshold (e.g., 0.02)
binary_mask = (predicted_mask[0, :, :, 0] > 0.02).astype(np.uint8)

# Resize the binary mask back to the original image size
original_size_mask = cv2.resize(binary_mask, (image_cv.shape[1], image_cv.shape[0]))

# Display the original image and the binary mask with a lower threshold
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Binary Mask Image with Lower Threshold
plt.subplot(1, 2, 2)
plt.imshow(original_size_mask, cmap='gray')
plt.title('Binary Mask (Threshold 0.02)')
plt.axis('off')

plt.show()

# Save the binary mask with lower threshold to a file
cv2.imwrite('lower_threshold_binary_mask.png', original_size_mask * 255)
print("Binary mask with lower threshold saved as 'lower_threshold_binary_mask.png'.")
