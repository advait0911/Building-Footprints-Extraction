import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the dataset
def load_data():
    # Adjust file paths as needed for your local environment
    x_train = np.load('train_xx.npy').astype('float32')
    y_train = np.load('train_yy.npy').astype('float32')
    x_test = np.load('test_xx.npy').astype('float32')
    y_test = np.load('test_yy.npy').astype('float32')

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    
    return x_train, y_train, x_test, y_test

# Build the U-Net model
def build_unet(input_shape=(128, 128, 3)):
    x_in = Input(shape=input_shape)
    
    ''' Encoder '''
    x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_in)
    x_temp = Dropout(0.25)(x_temp)
    x_skip1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = MaxPooling2D((2,2))(x_skip1)
    x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = Dropout(0.25)(x_temp)
    x_skip2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = MaxPooling2D((2,2))(x_skip2)
    x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = Dropout(0.25)(x_temp)
    x_skip3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = MaxPooling2D((2,2))(x_skip3)
    x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = Dropout(0.5)(x_temp)
    x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)

    ''' Decoder '''
    x_temp = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = Dropout(0.5)(x_temp)
    x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x_temp)
    x_temp = Concatenate()([x_temp, x_skip3])
    x_temp = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = Dropout(0.5)(x_temp)
    x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x_temp)
    x_temp = Concatenate()([x_temp, x_skip2])
    x_temp = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = Dropout(0.5)(x_temp)
    x_temp = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x_temp)
    x_temp = Concatenate()([x_temp, x_skip1])
    x_temp = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x_temp)
    x_temp = Dropout(0.5)(x_temp)
    x_temp = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x_temp)

    ''' Output Layer '''
    x_out = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x_temp)
    
    model = Model(inputs=x_in, outputs=x_out)
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=10, verbose=1)
    
    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# Predict and visualize results
def visualize_predictions(model, x_test, y_test, threshold=0.5):
    predictions = model.predict(x_test)
    predictions_binary = (predictions > threshold).astype(np.uint8)
    
    # Visualize predictions vs actual masks
    for i in range(5):
        plt.figure(figsize=(10, 5))
        
        # Prediction
        plt.subplot(1, 2, 1)
        plt.title("Prediction")
        plt.imshow(predictions_binary[i, :, :, 0], cmap='gray')
        
        # Actual Mask
        plt.subplot(1, 2, 2)
        plt.title("Actual Mask")
        plt.imshow(y_test[i, :, :, 0], cmap='gray')
        
        plt.show()

# Main script
if __name__ == "__main__":
    # Load the dataset
    x_train, y_train, x_test, y_test = load_data()

    # Build the model
    model = build_unet(input_shape=(128, 128, 3))

    # Train the model
    train_model(model, x_train, y_train, x_test, y_test)

    # Visualize predictions
    visualize_predictions(model, x_test, y_test)

    # Save the trained model
    model.save('my_modelv2.keras')
