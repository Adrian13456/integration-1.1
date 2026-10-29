"""Train the model and persist the resulting artifact."""

from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .model import (
    CAT,
    NUM,
    TARGET,
    load_training_data,
    train_model,
)


ARTIFACTS = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
)


def main() -> None:
    """Train the model on a train split, evaluate on a hold-out test set,
    report metrics, and persist the artifact trained on the full data
    (or train split — we train final model on full data after evaluation).
    """

    df = load_training_data()

    # Hold-out evaluation to obtain an unbiased estimate of performance.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    # Train on the training split only.
    model_eval = train_model(train_df)

    y_true = test_df[TARGET]
    y_pred = model_eval.predict(test_df[NUM + CAT])

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(
        f"[train] Ames hold-out (20% test): "
        f"R2={r2:.4f}, MAE={mae:.0f} USD"
    )

    # Also report previous-style in-sample metrics for comparison.
    y_train_pred = model_eval.predict(train_df[NUM + CAT])
    r2_in = r2_score(train_df[TARGET], y_train_pred)
    mae_in = mean_absolute_error(train_df[TARGET], y_train_pred)
    print(
        f"[train] Ames in-sample (train split): "
        f"R2={r2_in:.4f}, MAE={mae_in:.0f} USD"
    )

    # Final model trained on the entire dataset for production use.
    model = train_model(df)

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path = ARTIFACTS / "model.pkl"

    joblib.dump(
        model,
        artifact_path,
    )

    print(
        f"[train] artifact persisted: {artifact_path}"
    )


if __name__ == "__main__":
    main()
