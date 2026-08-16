"""Model utility helpers for local-only training and storage."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .settings import get_settings


MODEL_PATH = get_settings().model_dir / "best_revenue_model.joblib"


def save_model_locally(model, model_path: str | Path = MODEL_PATH) -> Path:
    """Save a trained model artifact locally under /models."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model_locally(model_path: str | Path = MODEL_PATH):
    """Load a local model artifact if it exists."""

    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def numeric_training_frame(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric X/y data for local scikit-learn compatible models."""

    y = pd.to_numeric(df[target_column], errors="coerce")
    X = df.drop(columns=[target_column]).select_dtypes(include=["number"]).copy()
    keep = y.notna()
    X = X.loc[keep].fillna(0)
    y = y.loc[keep]
    return X, y
