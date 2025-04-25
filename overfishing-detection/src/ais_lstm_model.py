import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_and_preprocess_ais_data  # Import the data loader function

def prepare_lstm_data(data, sequence_length=10):
    """
    Prepare AIS data for LSTM input by creating sequences.
    Args:
        data (pd.DataFrame): AIS data.
        sequence_length (int): Length of each sequence.
    Returns:
        np.ndarray, np.ndarray: Features and labels for LSTM input.
    """
    features = data.drop(columns=['is_fishing']).values
    labels = data['is_fishing'].values

    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(labels[i + sequence_length])

    return np.array(X), np.array(y)

def train_lstm_model(X_train, X_test, y_train, y_test):
    """
    Train an LSTM model on AIS data and evaluate its performance.
    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
    Returns:
        tf.keras.Model: Trained LSTM model.
    """
    # Define the LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"LSTM Model Test Accuracy: {test_acc}")

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

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fishing', 'Fishing'], yticklabels=['Not Fishing', 'Fishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fishing', 'Fishing']))

    # ROC Curve
    y_pred_proba = model.predict(X_test).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Visualize Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:50], label='True Labels', marker='o')
    plt.plot(y_pred[:50], label='Predicted Labels', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.title('True vs Predicted Labels (First 50 Samples)')
    plt.legend()
    plt.show()

    return model

if __name__ == "__main__":
    # Load preprocessed AIS data
    ais_data_dir = "../data/ais/ais-sample-data"
    ais_data = load_and_preprocess_ais_data(ais_data_dir)

    # Prepare data for LSTM
    sequence_length = 10
    X, y = prepare_lstm_data(ais_data, sequence_length=sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # Train the LSTM model
    train_lstm_model(X_train, X_test, y_train, y_test)