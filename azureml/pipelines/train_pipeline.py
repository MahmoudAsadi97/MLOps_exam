import argparse
import os
import joblib
import pandas as pd
import mlflow

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ready", type=str, required=True)
    parser.add_argument("--test_ready", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="house_affiliation")

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    # --------------------
    # Load prepared data
    # --------------------
    X_train = pd.read_csv(os.path.join(args.train_ready, "X_train.csv"))
    y_train = pd.read_csv(
        os.path.join(args.train_ready, "y_train.csv")
    )[args.target_col].astype(str)

    X_test = pd.read_csv(os.path.join(args.test_ready, "X_test.csv"))
    y_test = pd.read_csv(
        os.path.join(args.test_ready, "y_test.csv")
    )[args.target_col].astype(str)

    # --------------------
    # Train model
    # --------------------
    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)

    # --------------------
    # Evaluate
    # --------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    # --------------------
    # MLflow logging
    # IMPORTANT:
    # Azure ML manages the run automatically
    # DO NOT call mlflow.start_run()
    # --------------------
    mlflow.log_metric("accuracy", accuracy)

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)

    # --------------------
    # Save model artifact
    # --------------------
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.joblib")
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
