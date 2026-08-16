"""Local opportunity pipeline forecasting and target-achievability guidance.

All calculations and model fitting use only the local workbook. No internet,
external API, hosted model, or off-machine inference is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import create_opportunity_features
from .settings import get_settings


PIPELINE_MODEL_PATH = get_settings().model_dir / "best_pipeline_model.joblib"


@dataclass
class PipelineForecastResult:
    opportunity_forecast: pd.DataFrame
    salesperson_summary: pd.DataFrame
    team_summary: dict[str, float | str]
    model_metrics: pd.DataFrame
    suggestions: pd.DataFrame
    model_path: str | None


def _snapshot_date(data: dict[str, pd.DataFrame]) -> pd.Timestamp:
    contracts = data.get("Contracts", pd.DataFrame())
    values = pd.to_datetime(contracts.get("SnapshotDate"), errors="coerce")
    if isinstance(values, pd.Series) and values.notna().any():
        return values.max().normalize()
    dates = pd.to_datetime(data.get("Meetings", pd.DataFrame()).get("MeetingDate"), errors="coerce")
    if isinstance(dates, pd.Series) and dates.notna().any():
        return dates.max().normalize()
    return pd.Timestamp.today().normalize()


def _opportunity_model() -> tuple[Pipeline, list[str], list[str]]:
    numeric = ["PipelineValue", "ExpectedGrossProfit", "SalesCycleDays"]
    categorical = ["Product", "OpportunityType", "SalespersonID"]
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=6,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return model, numeric, categorical


def train_opportunity_win_model(
    data: dict[str, pd.DataFrame], save_model: bool = True
) -> tuple[Pipeline | None, pd.DataFrame, str | None]:
    """Train and evaluate a local win classifier on closed opportunities."""

    opportunities = data.get("Opportunities", pd.DataFrame()).copy()
    if opportunities.empty:
        return None, pd.DataFrame(), None
    opportunities["CreatedDate"] = pd.to_datetime(opportunities["CreatedDate"], errors="coerce")
    closed = opportunities[
        opportunities["Stage"].astype(str).str.lower().isin(["won", "lost"])
    ].dropna(subset=["CreatedDate"]).copy()
    closed["Won"] = closed["Stage"].astype(str).str.lower().eq("won").astype(int)
    model, numeric, categorical = _opportunity_model()
    features = numeric + categorical
    if len(closed) < 40 or closed["Won"].nunique() < 2:
        return None, pd.DataFrame(), None

    latest_year = int(closed["CreatedDate"].dt.year.max())
    train = closed[closed["CreatedDate"].dt.year < latest_year]
    test = closed[closed["CreatedDate"].dt.year == latest_year]
    split_note = f"Temporal holdout: closed {latest_year} opportunities"
    if len(train) < 30 or len(test) < 10 or test["Won"].nunique() < 2:
        ordered = closed.sort_values("CreatedDate")
        split_at = max(1, int(len(ordered) * 0.8))
        train, test = ordered.iloc[:split_at], ordered.iloc[split_at:]
        split_note = "Chronological 80/20 opportunity holdout"

    model.fit(train[features], train["Won"])
    predictions = model.predict(test[features])
    probabilities = model.predict_proba(test[features])[:, 1]
    metrics = pd.DataFrame(
        [
            {
                "Model": "RandomForest opportunity win classifier",
                "Accuracy": accuracy_score(test["Won"], predictions),
                "Precision": precision_score(test["Won"], predictions, zero_division=0),
                "Recall": recall_score(test["Won"], predictions, zero_division=0),
                "F1": f1_score(test["Won"], predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(test["Won"], probabilities),
                "Training Rows": len(train),
                "Test Rows": len(test),
                "Notes": split_note,
            }
        ]
    )
    model.fit(closed[features], closed["Won"])
    model_path = None
    if save_model:
        PIPELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, PIPELINE_MODEL_PATH)
        model_path = str(PIPELINE_MODEL_PATH)
    return model, metrics, model_path


def _list_text(values: list[str]) -> str:
    values = [str(value) for value in values if value]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} and {values[-1]}"


def build_pipeline_forecast(
    data: dict[str, pd.DataFrame], save_model: bool = True
) -> PipelineForecastResult:
    """Forecast current-year pipeline revenue and assess annual target coverage."""

    snapshot = _snapshot_date(data)
    year = snapshot.year
    year_end = pd.Timestamp(year, 12, 31)
    model, model_metrics, model_path = train_opportunity_win_model(data, save_model=save_model)
    opportunity_view = create_opportunity_features(data)
    opportunity_view["CreatedDate"] = pd.to_datetime(opportunity_view["CreatedDate"], errors="coerce")
    opportunity_view["ExpectedCloseDate"] = pd.to_datetime(
        opportunity_view.get("ExpectedCloseDate"), errors="coerce"
    )
    open_pipeline = opportunity_view[
        opportunity_view["CreatedDate"].dt.year.eq(year)
        & opportunity_view["Stage"].astype(str).str.lower().eq("open")
        & opportunity_view["ExpectedCloseDate"].le(year_end)
    ].copy()

    model_features = [
        "PipelineValue", "ExpectedGrossProfit", "SalesCycleDays", "Product",
        "OpportunityType", "SalespersonID",
    ]
    if model is not None and not open_pipeline.empty:
        open_pipeline["ModelWinProbability"] = model.predict_proba(
            open_pipeline[model_features]
        )[:, 1]
    else:
        open_pipeline["ModelWinProbability"] = 0.5

    closed = opportunity_view[
        opportunity_view["Stage"].astype(str).str.lower().isin(["won", "lost"])
    ].copy()
    closed["_won"] = closed["Stage"].astype(str).str.lower().eq("won")
    salesperson_win_rate = closed.groupby("SalespersonID")["_won"].mean().to_dict()
    team_win_rate = float(closed["_won"].mean()) if not closed.empty else 0.5
    open_pipeline["HistoricalWinRate"] = open_pipeline["SalespersonID"].map(
        salesperson_win_rate
    ).fillna(team_win_rate)
    stage_probability = pd.to_numeric(
        open_pipeline.get("WinProbability"), errors="coerce"
    ).fillna(0.5)
    classifier_auc = float(model_metrics.iloc[0]["ROC-AUC"]) if not model_metrics.empty else 0
    classifier_weight = 0.15 if classifier_auc >= 0.55 else 0.0
    stage_weight = 0.80 - classifier_weight
    probability = (
        stage_probability * stage_weight
        + open_pipeline["HistoricalWinRate"] * 0.20
        + open_pipeline["ModelWinProbability"] * classifier_weight
    )
    risk_multiplier = open_pipeline.get(
        "PipelineRisk", pd.Series("Medium", index=open_pipeline.index)
    ).map({"Low": 1.0, "Medium": 0.90, "High": 0.78}).fillna(0.90)
    response_multiplier = np.where(open_pipeline["WaitingResponseCount"] > 0, 0.88, 1.0)
    critical_multiplier = np.where(
        (open_pipeline["CriticalMeetingFindings"] + open_pipeline["CriticalNoteCount"]) > 0,
        0.90,
        1.0,
    )
    open_pipeline["AdjustedWinProbability"] = (
        probability * risk_multiplier * response_multiplier * critical_multiplier
    ).clip(0.03, 0.97)
    open_pipeline["ForecastRevenue"] = (
        pd.to_numeric(open_pipeline["PipelineValue"], errors="coerce").fillna(0)
        * open_pipeline["AdjustedWinProbability"]
    )
    open_pipeline["ProbabilityAdjustment"] = (
        open_pipeline["AdjustedWinProbability"] - stage_probability
    )
    open_pipeline = open_pipeline.sort_values(
        ["ForecastCategory", "ForecastRevenue"], ascending=[True, False]
    ).reset_index(drop=True)

    monthly = data.get("MonthlyPerformance", pd.DataFrame()).copy()
    monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
    current_monthly = monthly[monthly["Month"].dt.year.eq(year)]
    ytd = current_monthly.groupby("SalespersonID")["Revenue"].sum()
    targets = data.get("Targets", pd.DataFrame()).set_index("SalespersonID")
    people = data.get("Salespeople", pd.DataFrame()).set_index("SalespersonID")
    pipeline_summary = open_pipeline.groupby("SalespersonID").agg(
        OpenOpportunities=("OpportunityID", "nunique"),
        OpenPipeline=("PipelineValue", "sum"),
        WeightedPipelineForecast=("ForecastRevenue", "sum"),
        CommitPipeline=("PipelineValue", lambda values: values[open_pipeline.loc[values.index, "ForecastCategory"].eq("Commit")].sum()),
        HighRiskPipeline=("PipelineValue", lambda values: values[open_pipeline.loc[values.index, "PipelineRisk"].eq("High")].sum()),
        WaitingResponseOpportunities=("WaitingResponseCount", lambda values: int((values > 0).sum())),
        StalledOpportunities=("DaysInStage", lambda values: int((pd.to_numeric(values, errors="coerce") > 45).sum())),
    )
    rows = []
    elapsed_fraction = snapshot.dayofyear / (366 if snapshot.is_leap_year else 365)
    for salesperson_id in people.index:
        annual_target = float(targets["AnnualRevenueTarget"].get(salesperson_id, 0)) if "AnnualRevenueTarget" in targets else 0
        actual = float(ytd.get(salesperson_id, 0))
        target_gap = max(annual_target - actual, 0)
        pipeline_row = pipeline_summary.loc[salesperson_id] if salesperson_id in pipeline_summary.index else pd.Series(dtype=float)
        raw_pipeline = float(pipeline_row.get("OpenPipeline", 0))
        weighted = float(pipeline_row.get("WeightedPipelineForecast", 0))
        forecast_year_end = actual + weighted
        run_rate = actual / max(annual_target * elapsed_fraction, 1)
        raw_coverage = raw_pipeline / max(target_gap, 1)
        weighted_coverage = weighted / max(target_gap, 1)
        achievability_score = float(np.clip(
            weighted_coverage * 55 + min(raw_coverage / 3, 1) * 20 + min(run_rate, 1.2) / 1.2 * 25,
            0,
            100,
        ))
        achievability = "Likely" if achievability_score >= 75 else "Possible" if achievability_score >= 50 else "At risk"
        rows.append(
            {
                "SalespersonID": salesperson_id,
                "Salesperson": people.loc[salesperson_id].get("Salesperson", salesperson_id),
                "YTDRevenue": actual,
                "AnnualTarget": annual_target,
                "TargetGap": target_gap,
                "OpenOpportunities": int(pipeline_row.get("OpenOpportunities", 0)),
                "OpenPipeline": raw_pipeline,
                "WeightedPipelineForecast": weighted,
                "ForecastYearEndRevenue": forecast_year_end,
                "ForecastGap": forecast_year_end - annual_target,
                "PipelineCoverage": raw_coverage,
                "WeightedCoverage": weighted_coverage,
                "RunRateVsYTDTarget": run_rate,
                "AchievabilityScore": round(achievability_score),
                "Achievability": achievability,
                "CommitPipeline": float(pipeline_row.get("CommitPipeline", 0)),
                "HighRiskPipeline": float(pipeline_row.get("HighRiskPipeline", 0)),
                "WaitingResponseOpportunities": int(pipeline_row.get("WaitingResponseOpportunities", 0)),
                "StalledOpportunities": int(pipeline_row.get("StalledOpportunities", 0)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("AchievabilityScore", ascending=False).reset_index(drop=True)

    suggestion_rows = []
    for row in summary.itertuples(index=False):
        shortfall = max(-float(row.ForecastGap), 0)
        if shortfall > 0:
            required_pipeline = shortfall / max(team_win_rate, 0.1)
            suggestion_rows.append(
                {
                    "SalespersonID": row.SalespersonID,
                    "Priority": 1,
                    "Action": f"Create or qualify approximately £{required_pipeline:,.0f} of additional pipeline",
                    "Evidence": f"The probability-adjusted year-end forecast is £{shortfall:,.0f} below target.",
                    "ActionType": "Coverage",
                }
            )
        if row.WaitingResponseOpportunities:
            suggestion_rows.append(
                {
                    "SalespersonID": row.SalespersonID,
                    "Priority": 2,
                    "Action": f"Respond to customer notes on {row.WaitingResponseOpportunities} open opportunities",
                    "Evidence": "Waiting responses reduce the adjusted close probability used by the local forecast.",
                    "ActionType": "Responsiveness",
                }
            )
        if row.StalledOpportunities:
            suggestion_rows.append(
                {
                    "SalespersonID": row.SalespersonID,
                    "Priority": 3,
                    "Action": f"Requalify or close out {row.StalledOpportunities} opportunities stalled over 45 days",
                    "Evidence": f"£{row.HighRiskPipeline:,.0f} of pipeline is currently marked high risk.",
                    "ActionType": "Pipeline hygiene",
                }
            )
        person_pipeline = open_pipeline[open_pipeline["SalespersonID"] == row.SalespersonID]
        best_case = person_pipeline.nlargest(2, "ForecastRevenue")
        if not best_case.empty:
            ids = _list_text(best_case["OpportunityID"].tolist())
            suggestion_rows.append(
                {
                    "SalespersonID": row.SalespersonID,
                    "Priority": 4,
                    "Action": f"Protect the next steps for {ids}",
                    "Evidence": f"These opportunities contribute £{best_case['ForecastRevenue'].sum():,.0f} to weighted pipeline.",
                    "ActionType": "Conversion",
                }
            )
    suggestions = pd.DataFrame(suggestion_rows).sort_values(
        ["SalespersonID", "Priority"]
    ).reset_index(drop=True)

    team_actual = float(summary["YTDRevenue"].sum())
    team_target = float(summary["AnnualTarget"].sum())
    team_pipeline = float(summary["OpenPipeline"].sum())
    team_weighted = float(summary["WeightedPipelineForecast"].sum())
    team_forecast = team_actual + team_weighted
    team_gap = team_forecast - team_target
    team_score = float(np.average(
        summary["AchievabilityScore"], weights=summary["AnnualTarget"].clip(lower=1)
    )) if not summary.empty else 0
    team_summary = {
        "ForecastYear": year,
        "SnapshotDate": snapshot.strftime("%d %b %Y"),
        "YTDRevenue": team_actual,
        "AnnualTarget": team_target,
        "TargetRemaining": max(team_target - team_actual, 0),
        "OpenPipeline": team_pipeline,
        "WeightedPipelineForecast": team_weighted,
        "ForecastYearEndRevenue": team_forecast,
        "ForecastGap": team_gap,
        "PipelineCoverage": team_pipeline / max(team_target - team_actual, 1),
        "WeightedCoverage": team_weighted / max(team_target - team_actual, 1),
        "AchievabilityScore": round(team_score),
        "Achievability": "Likely" if team_score >= 75 else "Possible" if team_score >= 50 else "At risk",
        "HistoricalWinRate": team_win_rate,
        "ClassifierWeight": classifier_weight,
        "ClassifierGuardrail": "Active" if classifier_weight > 0 else "Excluded: ROC-AUC below 0.55",
    }
    return PipelineForecastResult(
        opportunity_forecast=open_pipeline,
        salesperson_summary=summary,
        team_summary=team_summary,
        model_metrics=model_metrics,
        suggestions=suggestions,
        model_path=model_path,
    )
