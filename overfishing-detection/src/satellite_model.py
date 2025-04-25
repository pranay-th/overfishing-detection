import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

def train_satellite_model(X_train, X_test, y_train, y_test):
    """
    Train a CNN on satellite image data and evaluate its performance.
    Args:
        X_train (np.ndarray): Training images.
        X_test (np.ndarray): Testing images.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
    Returns:
        tf.keras.Model: Trained CNN model.
    """
    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Satellite Model Test Accuracy: {test_acc}")

    # Save the model
    model.save('satellite_model.h5')
    print("Satellite model saved as 'satellite_model.h5'.")

    # Plot training history
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return model

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("../data/satellite/X_train.npy")
    X_test = np.load("../data/satellite/X_test.npy")
    y_train = np.load("../data/satellite/y_train.npy")
    y_test = np.load("../data/satellite/y_test.npy")

    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

    # Train the CNN model
    train_satellite_model(X_train, X_test, y_train, y_test)