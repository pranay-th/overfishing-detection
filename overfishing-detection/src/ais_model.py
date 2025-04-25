import os
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_loader import load_and_preprocess_ais_data  # Import the data loader function

def train_and_evaluate_with_cross_validation(X, y, n_splits=5):
    """
    Train and evaluate the XGBoost model using k-fold cross-validation.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        n_splits (int): Number of folds for cross-validation.
    Returns:
        None
    """
    # Define the XGBoost model with optimized hyperparameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=300,  # Number of trees
        max_depth=8,  # Maximum depth of a tree
        learning_rate=0.05,  # Learning rate
        subsample=0.8,  # Subsample ratio of the training instances
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        reg_alpha=0.1,  # L1 regularization term on weights
        reg_lambda=1.0,  # L2 regularization term on weights
        random_state=42
    )

    # Perform stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1

    # Lists to store metrics for each fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []

    for train_index, test_index in skf.split(X, y):
        print(f"\nStarting Fold {fold}...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Save the model for the current fold
        model_filename = f'ais_model_fold_{fold}.pkl'
        joblib.dump(model, model_filename)
        print(f"Model for Fold {fold} saved as '{model_filename}'.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print(f"Fold {fold} Classification Report:")
        print(classification_report(y_test, y_pred))

        # Calculate and display metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Fold {fold} Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")

        # Append metrics to lists
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)

        # Plot and save the ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Fold {fold})")
        plt.legend()
        plt.savefig(f'roc_curve_fold_{fold}.png')  # Save the ROC curve
        plt.close()

        fold += 1

    # Plot metrics improvement across folds
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_splits + 1), accuracy_list, label='Accuracy', marker='o')
    plt.plot(range(1, n_splits + 1), precision_list, label='Precision', marker='o')
    plt.plot(range(1, n_splits + 1), recall_list, label='Recall', marker='o')
    plt.plot(range(1, n_splits + 1), f1_list, label='F1 Score', marker='o')
    plt.plot(range(1, n_splits + 1), roc_auc_list, label='ROC AUC', marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title('Metrics Improvement Across Folds')
    plt.legend()
    plt.grid()
    plt.savefig('metrics_improvement_across_folds.png')  # Save the metrics improvement plot
    plt.show()

    print("\nCross-validation completed.")

if __name__ == "__main__":
    # Load preprocessed AIS data
    ais_data_dir = "../data/ais/ais-sample-data"
    ais_data = load_and_preprocess_ais_data(ais_data_dir)

    # Encode 'source' column if it exists
    if 'source' in ais_data.columns:
        encoder = LabelEncoder()
        ais_data['source'] = encoder.fit_transform(ais_data['source'])

    # Separate features and labels
    X = ais_data.drop(columns=['is_fishing'])
    y = ais_data['is_fishing']

    # Train and evaluate the model using cross-validation
    train_and_evaluate_with_cross_validation(X, y, n_splits=5)