import joblib
import pandas as pd
import numpy as np
from data_loader import load_and_preprocess_ais_data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed AIS data
ais_data_dir = "../data/ais/ais-sample-data"
ais_data = load_and_preprocess_ais_data(ais_data_dir)

# Encode 'source' column if it exists
if 'source' in ais_data.columns:
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    ais_data['source'] = encoder.fit_transform(ais_data['source'])

# Separate features and labels
X = ais_data.drop(columns=['is_fishing'])
y = ais_data['is_fishing']

# Load the trained model (e.g., from the last fold)
model_filename = "ais_model_fold_5.pkl"  # Change the fold number if needed
model = joblib.load(model_filename)
print(f"Loaded model from '{model_filename}'.")

# Select random samples from the training data
random_samples = X.sample(10, random_state=42)  # Select 10 random samples
random_labels = y.loc[random_samples.index]  # Get the true labels for the samples

# Make predictions on the random samples
predictions = model.predict(random_samples)
predicted_probabilities = model.predict_proba(random_samples)[:, 1]

# Display the results
results = pd.DataFrame({
    "Sample Index": random_samples.index,
    "True Label": random_labels.values,
    "Predicted Label": predictions,
    "Prediction Probability": predicted_probabilities
})
print("Random Sample Predictions:")
print(results)

# Plot a confusion matrix for the random samples
cm = confusion_matrix(random_labels, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fishing', 'Fishing'], yticklabels=['Not Fishing', 'Fishing'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Random Samples")
plt.show()

# Display classification report for the random samples
print("\nClassification Report for Random Samples:")
print(classification_report(random_labels, predictions))