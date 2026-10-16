from __future__ import annotations

from backend.ml.train_random_forest import train_model


if __name__ == "__main__":
    report = train_model()
    print(f"Random Forest test MAE: {report['held_out_test']['random_forest']['mae']} hours")
    print(f"Artifact: {report['artifact_path']}")