from flask import Flask, request, jsonify, send_file
import os
import io
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# Create Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = './my_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
input_size = (128, 128)  # Input size for the model

# Function to predict mask using the trained model
def predict_mask(image_cv):
    # Prepare the image for prediction
    resized_image = cv2.resize(image_cv, input_size)
    normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    input_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

    # Predict the mask using the model
    predicted_mask = model.predict(input_image)
    
    # Normalize the predicted mask for visualization
    scaled_mask = ((predicted_mask[0, :, :, 0] - predicted_mask.min()) /
                   (predicted_mask.max() - predicted_mask.min())) * 255
    scaled_mask = scaled_mask.astype(np.uint8)

    # Convert the numpy array to a PIL Image
    mask_image = Image.fromarray(scaled_mask)

    return mask_image

# Route to handle mask prediction
@app.route('/get-mask', methods=['POST'])
def get_mask_route():
    if 'image' not in request.files or 'filename' not in request.form:
        return jsonify({"error": "No image file or filename provided."}), 400
    
    image_file = request.files['image']
    image_name = request.form['filename']  # Retrieve the filename from the form data

    # Save the uploaded image temporarily
    image_path = os.path.join('temp', image_name)
    os.makedirs('temp', exist_ok=True)
    image_file.save(image_path)
    
    # Load the image using OpenCV
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        return jsonify({"error": "Invalid image file."}), 400

    try:
        # Predict the mask
        predicted_mask = predict_mask(image_cv)
        
        # Send the predicted mask back as a response
        img_io = io.BytesIO()
        predicted_mask.save(img_io, format='TIFF')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/tiff')
    except Exception as e:
        return jsonify({"error": f"Error predicting mask: {str(e)}"}), 500
    finally:
        # Clean up temporary files
        os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
