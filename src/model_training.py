"""Local-only model training for sales performance revenue prediction.

The model is trained only on the local Excel workbook after it has been loaded
into pandas DataFrames. No internet, external APIs, hosted LLMs, cloud ML
services, or pretrained online models are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model_utils import MODEL_PATH, save_model_locally


DEFAULT_HOLDOUT_MONTHS = 6


@dataclass
class TrainingResult:
    model_comparison: pd.DataFrame
    best_model: object | None
    feature_importance: pd.DataFrame
    model_path: str | None
    training_frame: pd.DataFrame


@dataclass
class RevenueModelAnalysis:
    selected_training: TrainingResult
    baseline_training: TrainingResult
    operational_training: TrainingResult
    feature_set_comparison: pd.DataFrame
    selected_feature_set: str


def _optional_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {}

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=120,
            learning_rate=0.05,
            depth=4,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    except Exception:
        pass

    return models


def candidate_regression_models(random_state: int = 42) -> dict[str, object]:
    """Return local candidate models, including optional libraries if present."""

    models: dict[str, object] = {
        "DummyRegressor baseline": DummyRegressor(strategy="mean"),
        "LinearRegression": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
    }
    models.update(_optional_models(random_state=random_state))
    return models


def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, predictions)
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(math.sqrt(mse)),
        "R2": float(r2_score(y_true, predictions)) if len(y_true) > 1 else float("nan"),
    }


def _extract_feature_importance(model: object, feature_names: list[str]) -> pd.DataFrame:
    fitted = model
    if isinstance(model, Pipeline):
        fitted = model.named_steps.get("model", model)

    values = None
    if hasattr(fitted, "feature_importances_"):
        values = fitted.feature_importances_
    elif hasattr(fitted, "coef_"):
        values = np.abs(np.ravel(fitted.coef_))

    if values is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def build_revenue_training_frame(
    data: dict[str, pd.DataFrame], include_operational_features: bool = True
) -> pd.DataFrame:
    """Build monthly revenue rows using only information available before the target month."""

    monthly = data.get("MonthlyPerformance", pd.DataFrame()).copy()
    if monthly.empty:
        return pd.DataFrame()
    monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
    monthly = monthly.dropna(subset=["Month", "SalespersonID", "Revenue"]).sort_values(
        ["SalespersonID", "Month"]
    )
    operational_columns: list[str] = []
    meetings = data.get("Meetings", pd.DataFrame()).copy() if include_operational_features else pd.DataFrame()
    if not meetings.empty and {"MeetingDate", "SalespersonID"}.issubset(meetings.columns):
        meetings["Month"] = pd.to_datetime(meetings["MeetingDate"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        meeting_type = meetings.get("MeetingType", pd.Series("", index=meetings.index)).astype(str).str.lower()
        meetings["OperationalMeetings"] = 1
        meetings["HealthCheckMeetings"] = meeting_type.str.contains("health check").astype(int)
        meetings["EscalationMeetings"] = meeting_type.str.contains("support escalation").astype(int)
        meetings["CriticalMeetingFindings"] = meetings.get(
            "CriticalFindingFlag", pd.Series(False, index=meetings.index)
        ).fillna(False).astype(bool).astype(int)
        meeting_monthly = meetings.groupby(["SalespersonID", "Month"], as_index=False)[
            ["OperationalMeetings", "HealthCheckMeetings", "EscalationMeetings", "CriticalMeetingFindings"]
        ].sum()
        monthly = monthly.merge(meeting_monthly, on=["SalespersonID", "Month"], how="left")
        operational_columns.extend(
            ["OperationalMeetings", "HealthCheckMeetings", "EscalationMeetings", "CriticalMeetingFindings"]
        )

    notes = data.get("OpportunityNotes", pd.DataFrame()).copy() if include_operational_features else pd.DataFrame()
    if not notes.empty and {"NoteDate", "SalespersonID"}.issubset(notes.columns):
        notes["Month"] = pd.to_datetime(notes["NoteDate"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        notes["OpportunityNotesReceived"] = 1
        notes["ResponseRequiredNotes"] = notes.get(
            "ResponseRequired", pd.Series(False, index=notes.index)
        ).fillna(False).astype(bool).astype(int)
        notes["CriticalOpportunityNotes"] = notes.get(
            "CriticalFindingFlag", pd.Series(False, index=notes.index)
        ).fillna(False).astype(bool).astype(int)
        note_monthly = notes.groupby(["SalespersonID", "Month"], as_index=False)[
            ["OpportunityNotesReceived", "ResponseRequiredNotes", "CriticalOpportunityNotes"]
        ].sum()
        monthly = monthly.merge(note_monthly, on=["SalespersonID", "Month"], how="left")
        operational_columns.extend(
            ["OpportunityNotesReceived", "ResponseRequiredNotes", "CriticalOpportunityNotes"]
        )

    for column in operational_columns:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce").fillna(0)
    lag_sources = [
        "CustomerReachouts",
        "Meetings",
        "OpportunitiesCreated",
        "OpportunitiesWon",
        "NewCustomers",
        "CrossSellOpportunities",
        "Revenue",
        "GrossProfit",
        "CrossSellRevenue",
        "RetentionRate",
        *operational_columns,
    ]
    lag_sources = [column for column in lag_sources if column in monthly.columns]
    grouped = monthly.groupby("SalespersonID", group_keys=False)
    for column in lag_sources:
        monthly[f"Previous{column}"] = grouped[column].shift(1)
        monthly[f"Rolling3{column}"] = grouped[column].transform(
            lambda values: values.shift(1).rolling(3, min_periods=1).mean()
        )

    monthly["MonthNumber"] = monthly["Month"].dt.month
    monthly["MonthSin"] = np.sin(2 * np.pi * monthly["MonthNumber"] / 12)
    monthly["MonthCos"] = np.cos(2 * np.pi * monthly["MonthNumber"] / 12)
    people = data.get("Salespeople", pd.DataFrame())
    category_columns = [column for column in ["Segment", "Region", "Seniority", "PrimarySpecialism"] if column in people]
    if category_columns:
        monthly = monthly.merge(
            people[["SalespersonID", *category_columns]], on="SalespersonID", how="left"
        )
        monthly = pd.get_dummies(monthly, columns=category_columns, dtype=int)

    previous_columns = [column for column in monthly.columns if column.startswith("Previous")]
    monthly = monthly.dropna(subset=previous_columns)
    keep = ["Month", "SalespersonID", "Revenue"] + [
        column
        for column in monthly.columns
        if column.startswith(("Previous", "Rolling3", "Segment_", "Region_", "Seniority_", "PrimarySpecialism_"))
        or column in ["MonthNumber", "MonthSin", "MonthCos"]
    ]
    return monthly[keep].reset_index(drop=True)


def _temporal_train_test(
    training_frame: pd.DataFrame,
    test_size: float,
    holdout_months: int | None = DEFAULT_HOLDOUT_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    months = sorted(training_frame["Month"].dropna().unique())
    if holdout_months is None:
        test_month_count = max(1, int(math.ceil(len(months) * test_size)))
    elif len(months) <= 1:
        test_month_count = 1
    else:
        test_month_count = min(max(1, holdout_months), len(months) - 1)
    test_months = set(months[-test_month_count:])
    test_mask = training_frame["Month"].isin(test_months)
    feature_columns = [
        column
        for column in training_frame.select_dtypes(include=["number"]).columns
        if column != "Revenue"
    ]
    X = training_frame[feature_columns].fillna(0)
    y = pd.to_numeric(training_frame["Revenue"], errors="coerce")
    label = f"Temporal holdout: final {test_month_count} month(s)"
    return X.loc[~test_mask], X.loc[test_mask], y.loc[~test_mask], y.loc[test_mask], label


def train_revenue_models(
    data: dict[str, pd.DataFrame],
    test_size: float = 0.3,
    random_state: int = 42,
    save_path=MODEL_PATH,
    include_operational_features: bool = True,
    save_best: bool = True,
    holdout_months: int | None = DEFAULT_HOLDOUT_MONTHS,
) -> TrainingResult:
    """Train local revenue prediction models and save the best local artifact."""

    training_frame = build_revenue_training_frame(
        data, include_operational_features=include_operational_features
    )
    if training_frame.empty or "Revenue" not in training_frame.columns:
        empty = pd.DataFrame(
            columns=[
                "Model", "MAE", "RMSE", "R2", "Rows", "Training Rows",
                "Test Rows", "Holdout Months", "Notes",
            ]
        )
        return TrainingResult(empty, None, pd.DataFrame(), None, training_frame)

    if len(training_frame) < 20 or training_frame["Revenue"].nunique() < 2:
        comparison = pd.DataFrame(
            [
                {
                    "Model": "Not trained",
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Rows": len(training_frame),
                    "Training Rows": 0,
                    "Test Rows": 0,
                    "Holdout Months": 0,
                    "Notes": "Need at least 20 lagged monthly rows and variation in Revenue.",
                }
            ]
        )
        return TrainingResult(comparison, None, pd.DataFrame(), None, training_frame)

    X_train, X_test, y_train, y_test, split_note = _temporal_train_test(
        training_frame, test_size, holdout_months=holdout_months
    )
    actual_holdout_months = training_frame.loc[X_test.index, "Month"].nunique()
    X_all = pd.concat([X_train, X_test]).sort_index()
    y_all = pd.concat([y_train, y_test]).sort_index()

    rows = []
    fitted_models: dict[str, object] = {}
    for name, model in candidate_regression_models(random_state=random_state).items():
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = _regression_metrics(y_test, predictions)
            rows.append(
                {
                    "Model": name,
                    **metrics,
                    "Rows": len(training_frame),
                    "Training Rows": len(X_train),
                    "Test Rows": len(X_test),
                    "Holdout Months": actual_holdout_months,
                    "Features": len(X_train.columns),
                    "Notes": split_note,
                }
            )
            fitted_models[name] = model
        except Exception as exc:
            rows.append(
                {
                    "Model": name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Rows": len(training_frame),
                    "Training Rows": len(X_train),
                    "Test Rows": len(X_test),
                    "Holdout Months": actual_holdout_months,
                    "Features": len(X_train.columns),
                    "Notes": f"Skipped: {exc}",
                }
            )

    comparison = pd.DataFrame(rows).sort_values(
        by=["RMSE", "MAE"], ascending=[True, True], na_position="last"
    )
    successful = comparison.dropna(subset=["RMSE"])
    if successful.empty:
        return TrainingResult(comparison, None, pd.DataFrame(), None, training_frame)

    best_name = successful.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_model.fit(X_all, y_all)
    model_path = save_model_locally(best_model, save_path) if save_best else None
    importance = _extract_feature_importance(best_model, list(X_train.columns))

    return TrainingResult(
        model_comparison=comparison,
        best_model=best_model,
        feature_importance=importance,
        model_path=str(model_path) if model_path is not None else None,
        training_frame=training_frame,
    )


def _best_result_row(result: TrainingResult) -> pd.Series | None:
    if result.model_comparison.empty or "RMSE" not in result.model_comparison:
        return None
    successful = result.model_comparison.dropna(subset=["RMSE"])
    return successful.iloc[0] if not successful.empty else None


def analyse_revenue_models(
    data: dict[str, pd.DataFrame],
    test_size: float = 0.3,
    random_state: int = 42,
    save_path=MODEL_PATH,
    save_best: bool = True,
    holdout_months: int | None = DEFAULT_HOLDOUT_MONTHS,
) -> RevenueModelAnalysis:
    """Compare feature sets and save only the best holdout-validated local model."""

    baseline = train_revenue_models(
        data,
        test_size=test_size,
        random_state=random_state,
        save_path=save_path,
        include_operational_features=False,
        save_best=False,
        holdout_months=holdout_months,
    )
    operational = train_revenue_models(
        data,
        test_size=test_size,
        random_state=random_state,
        save_path=save_path,
        include_operational_features=True,
        save_best=False,
        holdout_months=holdout_months,
    )
    baseline_best = _best_result_row(baseline)
    operational_best = _best_result_row(operational)

    candidates = [
        ("Lagged workbook baseline", baseline, baseline_best),
        ("Operational activity features", operational, operational_best),
    ]
    viable = [item for item in candidates if item[2] is not None]
    if viable:
        selected_name, selected, _ = min(viable, key=lambda item: float(item[2]["RMSE"]))
    else:
        selected_name, selected = "No viable feature set", baseline

    baseline_rmse = float(baseline_best["RMSE"]) if baseline_best is not None else np.nan
    comparison_rows = []
    for feature_set, result, best in candidates:
        rmse = float(best["RMSE"]) if best is not None else np.nan
        delta = rmse - baseline_rmse if np.isfinite(rmse) and np.isfinite(baseline_rmse) else np.nan
        delta_pct = delta / baseline_rmse if np.isfinite(delta) and baseline_rmse else np.nan
        if feature_set == "Lagged workbook baseline":
            decision = "Reference feature set"
        elif not np.isfinite(delta):
            decision = "Not comparable"
        elif delta < 0:
            decision = "Improved; eligible for selection"
        elif delta > 0:
            decision = "Worsened; exclude from saved model"
        else:
            decision = "No measurable change"
        comparison_rows.append(
            {
                "Feature Set": feature_set,
                "Best Model": str(best["Model"]) if best is not None else "Not trained",
                "MAE": float(best["MAE"]) if best is not None else np.nan,
                "RMSE": rmse,
                "R2": float(best["R2"]) if best is not None else np.nan,
                "Rows": len(result.training_frame),
                "Training Rows": int(best["Training Rows"]) if best is not None and "Training Rows" in best else 0,
                "Test Rows": int(best["Test Rows"]) if best is not None and "Test Rows" in best else 0,
                "Holdout Months": int(best["Holdout Months"]) if best is not None and "Holdout Months" in best else 0,
                "Features": int(best["Features"]) if best is not None and "Features" in best else 0,
                "RMSE Change": delta,
                "RMSE Change Pct": delta_pct,
                "Decision": decision,
                "Selected": feature_set == selected_name,
            }
        )

    if save_best and selected.best_model is not None:
        model_path = save_model_locally(selected.best_model, save_path)
        selected.model_path = str(model_path)

    return RevenueModelAnalysis(
        selected_training=selected,
        baseline_training=baseline,
        operational_training=operational,
        feature_set_comparison=pd.DataFrame(comparison_rows),
        selected_feature_set=selected_name,
    )
