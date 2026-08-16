"""Modern local model monitoring and data-science portal."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from src.features import (
    create_customer_features,
    create_opportunity_features,
    create_salesperson_features,
    feature_summary,
)
from src.model_training import analyse_revenue_models
from src.pipeline_forecasting import build_pipeline_forecast
from src.validation import data_consistency_flags, row_counts_by_sheet, validation_summary

from .manager_portal import (
    COLORS,
    READ_ONLY_GRAPH_CONFIG,
    _chart_panel,
    _chart_style,
    _data_table,
    _metric,
    clean_label,
)


def _missing_values_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sheet, frame in data.items():
        for column, count in frame.isna().sum().items():
            if count:
                rows.append(
                    {
                        "Sheet": clean_label(sheet),
                        "Field": clean_label(column),
                        "Missing Values": int(count),
                        "Missing Rate": float(count / max(len(frame), 1)),
                    }
                )
    return pd.DataFrame(rows)


def _model_figures(comparison: pd.DataFrame, importance: pd.DataFrame):
    successful = comparison.dropna(subset=["RMSE"]).copy()
    if successful.empty:
        successful = pd.DataFrame({"Model": ["Not trained"], "MAE": [0], "RMSE": [0], "R2": [0]})

    error_long = successful.melt(
        id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error"
    )
    error_fig = px.bar(
        error_long.sort_values("Error", ascending=False),
        x="Error",
        y="Model",
        color="Metric",
        barmode="group",
        orientation="h",
        title="Revenue Prediction Error",
        color_discrete_map={"MAE": COLORS[1], "RMSE": COLORS[2]},
    )
    error_fig.update_xaxes(tickprefix="£", tickformat="~s")
    error_fig.update_traces(hovertemplate="%{y}<br>%{fullData.name}: £%{x:,.0f}<extra></extra>")
    _chart_style(error_fig, 370)

    r2_fig = go.Figure()
    ordered = successful.sort_values("R2")
    for _, row in ordered.iterrows():
        color = COLORS[0] if row["R2"] >= 0.7 else COLORS[2] if row["R2"] >= 0 else COLORS[3]
        r2_fig.add_trace(
            go.Scatter(
                x=[row["R2"]],
                y=[row["Model"]],
                mode="markers",
                marker={"size": 14, "color": color, "line": {"color": "white", "width": 2}},
                hovertemplate=f"{row['Model']}<br>R2: {row['R2']:.3f}<extra></extra>",
                showlegend=False,
            )
        )
        r2_fig.add_shape(
            type="line", x0=min(0, row["R2"]), x1=row["R2"], y0=row["Model"], y1=row["Model"],
            line={"color": color, "width": 4},
        )
    r2_fig.add_vline(x=0, line_dash="dash", line_color="#83928c")
    r2_fig.update_layout(title="Explained Variance (R2)")
    r2_fig.update_xaxes(range=[min(-0.2, ordered["R2"].min() - 0.1), 1.0], tickformat=".1f")
    _chart_style(r2_fig, 370, False)

    if importance.empty:
        importance = pd.DataFrame({"Feature": ["Not available"], "Importance": [0]})
    importance_fig = px.bar(
        importance.head(12).sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Features Used Most by the Best Local Model",
        color_discrete_sequence=[COLORS[4]],
    )
    importance_fig.update_traces(hovertemplate="%{y}<br>Importance: %{x:,.3f}<extra></extra>")
    _chart_style(importance_fig, 410, False)
    return error_fig, r2_fig, importance_fig


def _data_quality_figures(row_counts: pd.DataFrame, missing: pd.DataFrame):
    rows = row_counts.copy()
    rows["Sheet"] = rows["Sheet"].map(clean_label)
    row_fig = px.bar(
        rows.sort_values("Rows"), x="Rows", y="Sheet", orientation="h",
        title="Workbook Coverage by Sheet", color_discrete_sequence=[COLORS[1]],
    )
    row_fig.update_traces(hovertemplate="%{y}<br>%{x:,} rows<extra></extra>")
    _chart_style(row_fig, 500, False)

    if missing.empty:
        missing = pd.DataFrame({"Field Label": ["No missing values"], "Missing Values": [0], "Missing Rate": [0]})
    else:
        missing = missing.sort_values("Missing Values", ascending=False).head(15).copy()
        missing["Field Label"] = missing["Sheet"] + " / " + missing["Field"]
    missing_fig = px.bar(
        missing.sort_values("Missing Values"), x="Missing Values", y="Field Label", orientation="h",
        title="Largest Missing-Value Gaps", color="Missing Rate",
        color_continuous_scale=[[0, "#d8e8e1"], [1, COLORS[3]]],
    )
    missing_fig.update_traces(hovertemplate="%{y}<br>%{x:,} missing values<extra></extra>")
    missing_fig.update_coloraxes(showscale=False)
    _chart_style(missing_fig, 500, False)
    return row_fig, missing_fig


def _contract_monitoring_figures(data: dict[str, pd.DataFrame]):
    contracts = data["Contracts"].copy()
    days_fig = px.histogram(
        contracts, x="DaysToRenewal", nbins=24, title="Renewal Timing Distribution",
        color_discrete_sequence=[COLORS[0]],
    )
    days_fig.add_vline(x=0, line_dash="dash", line_color=COLORS[3])
    days_fig.update_traces(hovertemplate="Days to renewal: %{x}<br>Contracts: %{y}<extra></extra>")
    _chart_style(days_fig, 320, False)

    audit_counts = contracts[["EndDateChangeCount", "RollbackCount"]].melt(
        var_name="Audit Indicator", value_name="Count"
    )
    audit_counts["Audit Indicator"] = audit_counts["Audit Indicator"].map(clean_label)
    audit_fig = px.histogram(
        audit_counts, x="Count", color="Audit Indicator", barmode="group",
        title="Contract Change and Rollback Profile",
        color_discrete_sequence=[COLORS[2], COLORS[3]],
    )
    audit_fig.update_traces(hovertemplate="%{fullData.name}<br>Count: %{x}<br>Contracts: %{y}<extra></extra>")
    _chart_style(audit_fig, 320)
    return days_fig, audit_fig


def build_data_scientist_layout(data: dict[str, pd.DataFrame]) -> html.Div:
    """Build the local data-quality and model-monitoring experience."""

    validation = validation_summary(data)
    consistency = data_consistency_flags(data)
    row_counts = row_counts_by_sheet(data)
    missing = _missing_values_frame(data)
    salesperson_features = create_salesperson_features(data)
    customer_features = create_customer_features(data)
    opportunity_features = create_opportunity_features(data)
    summaries = feature_summary(salesperson_features, customer_features, opportunity_features)
    revenue_analysis = analyse_revenue_models(data)
    training = revenue_analysis.selected_training
    baseline_training = revenue_analysis.baseline_training
    pipeline_forecast = build_pipeline_forecast(data)
    pipeline_metrics = pipeline_forecast.model_metrics.copy()
    comparison = training.model_comparison.copy()
    importance = training.feature_importance.head(20).copy()
    if "Feature" in importance:
        importance["Feature"] = importance["Feature"].map(clean_label)

    successful = comparison.dropna(subset=["RMSE"]) if "RMSE" in comparison else pd.DataFrame()
    best = successful.iloc[0] if not successful.empty else None
    total_rows = int(row_counts["Rows"].sum())
    pass_rate = float(validation["Status"].astype(str).str.lower().eq("pass").mean()) if not validation.empty else 0
    missing_total = int(missing["Missing Values"].sum()) if not missing.empty else 0
    consistency_total = len(consistency)
    consistency_errors = int(consistency["Severity"].eq("Error").sum()) if not consistency.empty else 0
    customer_name_flags = int(
        consistency["Issue"].str.contains("customer name", case=False, na=False).sum()
    ) if not consistency.empty else 0
    best_name = str(best["Model"]) if best is not None else "Not trained"
    best_rmse = float(best["RMSE"]) if best is not None else 0
    best_mae = float(best["MAE"]) if best is not None else 0
    best_r2 = float(best["R2"]) if best is not None else 0
    validation_note = str(best["Notes"]) if best is not None and "Notes" in best else "No validation split"
    holdout_rows = int(best["Test Rows"]) if best is not None and "Test Rows" in best else 0
    dummy = successful[
        successful["Model"].astype(str).str.contains("DummyRegressor", case=False)
    ] if not successful.empty else pd.DataFrame()
    dummy_rmse = float(dummy.iloc[0]["RMSE"]) if not dummy.empty else 0
    rmse_improvement = (dummy_rmse - best_rmse) / dummy_rmse if dummy_rmse else 0
    rmse_improvement_label = f"{rmse_improvement:.0%} lower" if rmse_improvement > 0 else "No improvement"
    if best_r2 >= 0.5 and rmse_improvement >= 0.1:
        model_readiness = "Promising; monitor"
        readiness_tone = "positive"
    elif best_r2 > 0 and rmse_improvement > 0:
        model_readiness = "Limited evidence"
        readiness_tone = "warning"
    else:
        model_readiness = "Not reliable"
        readiness_tone = "danger"

    row_fig, missing_fig = _data_quality_figures(row_counts, missing)
    error_fig, r2_fig, importance_fig = _model_figures(comparison, importance)
    days_fig, audit_fig = _contract_monitoring_figures(data)

    tracker = revenue_analysis.feature_set_comparison.copy()
    monthly = data.get("MonthlyPerformance", pd.DataFrame())
    synthetic_rows = int(
        monthly.get("DataOrigin", pd.Series("", index=monthly.index))
        .astype(str).str.contains("synthetic", case=False).sum()
    )
    selected_feature_short = (
        "Baseline lags"
        if revenue_analysis.selected_feature_set == "Lagged workbook baseline"
        else "Operational features"
        if revenue_analysis.selected_feature_set == "Operational activity features"
        else revenue_analysis.selected_feature_set
    )
    selected_feature_note = (
        "The lagged workbook baseline had the lowest holdout RMSE, so operational meeting and note features are excluded from the saved model."
        if revenue_analysis.selected_feature_set == "Lagged workbook baseline"
        else "The selected feature set had the lowest holdout RMSE and is the one saved locally."
    )
    evidence_note = (
        "The selected model beats the dummy baseline, but the limited history, synthetic rows, pooled metrics, and single holdout mean it still requires monitoring and more real observations."
        if rmse_improvement > 0
        else "The selected model does not improve on the dummy baseline, so the current local data does not support a useful forward estimate yet."
    )
    design_summary = pd.DataFrame(
        [
            {
                "Design Area": "Prediction target",
                "Implementation": "Monthly salesperson Revenue",
                "Why It Matters": "The model estimates next-month revenue; it is separate from the opportunity pipeline forecast.",
            },
            {
                "Design Area": "Feature timing",
                "Implementation": "Previous-month and rolling three-month values only",
                "Why It Matters": "Current-month outcomes are excluded so information from the target period cannot leak into predictors.",
            },
            {
                "Design Area": "Validation",
                "Implementation": f"{validation_note}; {holdout_rows:,} holdout rows",
                "Why It Matters": "The model is tested on later periods instead of randomly mixing future and past observations.",
            },
            {
                "Design Area": "Feature-set selection",
                "Implementation": revenue_analysis.selected_feature_set,
                "Why It Matters": "Only the feature set with the lowest holdout RMSE is saved; features that worsen accuracy are excluded.",
            },
            {
                "Design Area": "Interpretation",
                "Implementation": "Global feature importance",
                "Why It Matters": "Importance shows predictive reliance and association, not a causal effect on revenue.",
            },
            {
                "Design Area": "Current limitation",
                "Implementation": f"{len(training.training_frame):,} lagged rows; {synthetic_rows:,} synthetic source rows; one temporal holdout",
                "Why It Matters": "Results are demonstration evidence and should be monitored on more real history before operational decisions.",
            },
        ]
    )
    audit = data["ContractAuditLog"].head(30)
    metric_definitions = data.get("MetricDefinitions", pd.DataFrame())
    metric_report = data.get("MetricCalculationReport", pd.DataFrame())

    return html.Div(
        [
            html.Div(
                [html.Div("MODEL ANALYSIS", className="eyebrow"), html.H2("Data quality and model performance", className="page-title")],
                className="page-heading",
            ),
            html.Div(
                [
                    _metric("Workbook Sheets", f"{len(data):,}", f"{total_rows:,} loaded rows", "info"),
                    _metric("Validation Pass Rate", f"{pass_rate:.0%}", f"{missing_total:,} missing values", "positive" if pass_rate >= 0.9 else "warning"),
                    _metric("Selected Revenue Model", best_name, revenue_analysis.selected_feature_set, readiness_tone),
                    _metric("Best MAE", f"£{best_mae:,.0f}", "Mean absolute error"),
                    _metric("Best RMSE", f"£{best_rmse:,.0f}", "Lower is better", "warning"),
                    _metric("Best R2", f"{best_r2:.3f}", "Closer to 1 is better", "positive" if best_r2 >= 0.7 else "warning"),
                ],
                className="metric-grid six-up ds-metrics",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Div("REVENUE MODEL PURPOSE", className="eyebrow"),
                            html.H3("What the model predicts and how to interpret it", className="section-title compact-title"),
                        ]
                    ),
                    html.Div(
                        "The revenue model estimates a salesperson's next monthly revenue from information available before that month. It is designed for monitored planning experiments, not as a causal explanation, an employment score, or the probability-adjusted pipeline forecast shown in the manager portal.",
                        className="model-caveat metric-note",
                    ),
                    html.Div(
                        [
                            _metric(
                                "Prediction Unit", "Next month", "Revenue by salesperson", "info",
                                "The target is one salesperson's revenue for the next workbook month. It is not annual revenue or the value of open opportunities.",
                            ),
                            _metric(
                                "Selected Feature Set", selected_feature_short, revenue_analysis.selected_feature_set, "positive",
                                selected_feature_note,
                            ),
                            _metric(
                                "RMSE vs Dummy", rmse_improvement_label, f"Dummy RMSE £{dummy_rmse:,.0f}", readiness_tone,
                                "This compares the selected model with predicting the training-set mean. A lower RMSE indicates useful predictive signal on the held-out months.",
                            ),
                            _metric(
                                "Evidence Status", model_readiness, "Demonstration data", readiness_tone,
                                evidence_note,
                            ),
                        ],
                        className="metric-grid model-summary-metrics",
                    ),
                    html.H4("Model design and limitations", className="subsection-title"),
                    _data_table(
                        design_summary,
                        ["Design Area", "Implementation", "Why It Matters"],
                        8,
                        info="Explains the target, feature timing, validation, feature-set decision, interpretation boundary, and current evidence limitations. It matters for using the model within its intended scope.",
                    ),
                ],
                className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("MODEL ACCURACY", className="eyebrow"), html.H3("Candidate model leaderboard", className="section-title compact-title")]),
                    html.Div(
                        [
                            _chart_panel(
                                error_fig,
                                "Compares MAE and RMSE for each local revenue model. Lower errors matter because they mean predictions are closer to observed revenue on the chronological holdout.",
                            ),
                            _chart_panel(
                                r2_fig,
                                "Shows how much holdout revenue variation each model explains. Values near 1 are stronger; values below 0 mean the model performs worse than a simple mean prediction.",
                            ),
                        ],
                        className="dashboard-grid two-column",
                    ),
                    _data_table(
                        comparison.round(4), ["Model", "MAE", "RMSE", "R2", "Rows", "Training Rows", "Test Rows", "Holdout Months", "Features", "Notes"], 8,
                        info="Lists every local revenue-model candidate and its holdout errors. It matters for selecting the best model transparently rather than relying on a single accuracy number.",
                    ),
                    html.Div(
                        f"Metrics use {validation_note.lower()} over {len(training.training_frame):,} lagged monthly rows and are pooled across salespeople. The same holdout compares candidate models and feature sets, so results may be optimistic until confirmed on later unseen months. Treat weak or negative R2 as evidence that the current local data does not support a reliable forward estimate.",
                        className="model-caveat",
                    ),
                ],
                className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("PIPELINE CLASSIFICATION", className="eyebrow"), html.H3("Local opportunity win model", className="section-title compact-title")]),
                    html.Div(
                        "The current pipeline forecast blends recorded stage probability and historical salesperson conversion with a low-weight local classifier. Risk, unanswered notes, and critical findings reduce the final probability. The classifier is diagnostic and does not make autonomous sales decisions.",
                        className="model-caveat metric-note",
                    ),
                    _data_table(
                        pipeline_metrics.round(4),
                        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Training Rows", "Test Rows", "Notes"],
                        5,
                        info="Shows classification quality for opportunity-win prediction. Precision and recall expose different error types, while ROC-AUC measures ranking quality across thresholds.",
                    ),
                    html.H4("Current team forecast", className="subsection-title"),
                    _data_table(
                        pd.DataFrame([pipeline_forecast.team_summary]),
                        ["ForecastYear", "SnapshotDate", "YTDRevenue", "AnnualTarget", "TargetRemaining", "OpenPipeline", "WeightedPipelineForecast", "ForecastYearEndRevenue", "ForecastGap", "PipelineCoverage", "WeightedCoverage", "AchievabilityScore", "Achievability", "HistoricalWinRate", "ClassifierWeight", "ClassifierGuardrail"],
                        5,
                        info="Summarises the current local team forecast, target gap, pipeline coverage, and classifier guardrail. It matters for judging whether the forecast is both commercially useful and statistically defensible.",
                    ),
                ],
                className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("EXPLAINABILITY", className="eyebrow"), html.H3("Feature behaviour", className="section-title compact-title")]),
                    html.Div(
                        [
                            _chart_panel(
                                importance_fig,
                                "Ranks the features used most by the selected local model. It matters for understanding what drives predictions and detecting reliance on inappropriate or leaky fields.",
                            ),
                            html.Div(
                                [
                                    html.H4("Accuracy experiment tracker", className="subsection-title tracker-title"),
                                    _data_table(
                                        tracker.round(4), list(tracker.columns), 5,
                                        info="Compares the baseline feature set with operational additions on the same holdout. It matters because new features are retained only when they improve accuracy without unacceptable leakage.",
                                    ),
                                ],
                                className="experiment-panel",
                            ),
                        ],
                        className="dashboard-grid two-column",
                    ),
                    html.H4("Feature-set summaries", className="subsection-title"),
                    _data_table(
                        summaries, list(summaries.columns), 6,
                        info="Summarises the distribution and coverage of engineered salesperson, customer, and opportunity features. It matters for spotting implausible ranges before training.",
                    ),
                ],
                className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("METRIC GOVERNANCE", className="eyebrow"), html.H3("Canonical sales metrics", className="section-title compact-title")]),
                    html.Div(
                        f"{len(metric_definitions):,} workbook metric definitions are governed by the local sales_metrics.yaml file. Derived fields are recomputed from source columns when the workbook loads.",
                        className="model-caveat metric-note",
                    ),
                    html.H4("Metric catalogue", className="subsection-title"),
                    _data_table(
                        metric_definitions, ["MetricName", "Category", "Definition", "FormulaOrLogic", "PrimarySource", "Direction", "Availability", "ModelUse", "LeakageRisk"], 10,
                        info="Defines the governed meaning, formula, source, model use, and leakage risk of each sales metric. It matters for consistent reporting and reproducible modelling.",
                        filter_action="native",
                    ),
                    html.H4("Formula execution", className="subsection-title"),
                    _data_table(
                        metric_report, ["Metric", "Formula", "Status", "Message"], 8,
                        info="Reports whether governed metric formulas were recomputed successfully from local source columns. It matters for detecting broken calculations before analysis.",
                    ),
                ],
                className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("DATA QUALITY", className="eyebrow"), html.H3("Workbook coverage and validation", className="section-title compact-title")]),
                    html.Div(
                        [
                            _chart_panel(
                                row_fig,
                                "Shows row counts for every loaded workbook sheet. It matters for detecting missing, unexpectedly small, or disproportionately large data sources.",
                            ),
                            _chart_panel(
                                missing_fig,
                                "Ranks fields with the most missing values and their missing rates. It matters because incomplete inputs can bias features, forecasts, and manager conclusions.",
                            ),
                        ],
                        className="dashboard-grid two-column",
                    ),
                    html.Div(
                        [
                            _metric(
                                "Consistency Flags", f"{consistency_total:,}",
                                "Cross-sheet records requiring review",
                                "positive" if consistency_total == 0 else "warning",
                            ),
                            _metric(
                                "Data Conflicts", f"{consistency_errors:,}",
                                "Broken ownership or record relationships",
                                "positive" if consistency_errors == 0 else "warning",
                            ),
                            _metric(
                                "Customer Name Flags", f"{customer_name_flags:,}",
                                "Checked against the customer master",
                                "positive" if customer_name_flags == 0 else "warning",
                            ),
                        ],
                        className="metric-grid quality-metrics",
                    ),
                    html.H4("Cross-sheet consistency flags", className="subsection-title"),
                    _data_table(
                        consistency if not consistency.empty else pd.DataFrame(
                            [{
                                "Severity": "Pass", "Sheet": "All", "RecordID": "",
                                "Field": "", "Issue": "No consistency mismatches detected",
                                "ExpectedValue": "", "ActualValue": "", "SuggestedAction": "No action required",
                            }]
                        ),
                        ["Severity", "Sheet", "RecordID", "Field", "Issue", "ExpectedValue", "ActualValue", "SuggestedAction"],
                        12,
                        info="Lists record-level conflicts across customer, opportunity, project, ticket, and task relationships. It matters because mismatched ownership or names can produce incorrect reporting and model features.",
                        filter_action="native",
                    ),
                    html.H4("Validation checks", className="subsection-title"),
                    _data_table(
                        validation, ["Check", "Sheet", "Field", "Status", "Value", "Message"], 12,
                        info="Shows required-sheet, column, missing-value, duplicate, date, relationship, and metric checks. It matters for deciding whether the workbook is fit for analysis.",
                        filter_action="native",
                    ),
                ],
                className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("CONTRACT FEATURES", className="eyebrow"), html.H3("Renewal and audit monitoring", className="section-title compact-title")]),
                    html.Div(
                        [
                            _chart_panel(
                                days_fig,
                                "Shows the distribution of days until contract renewal. It matters for understanding workload concentration and identifying overdue or near-term renewal risk.",
                            ),
                            _chart_panel(
                                audit_fig,
                                "Shows how often contract end dates changed and were rolled back. It matters because repeated changes can indicate governance issues or unstable renewal planning.",
                            ),
                        ],
                        className="dashboard-grid two-column",
                    ),
                    html.H4("Contract audit sample", className="subsection-title"),
                    _data_table(
                        audit, ["AuditID", "ContractID", "CustomerID", "ChangeDate", "ChangeType", "RollbackFlag", "DaysChanged", "ApprovalStatus", "SalespersonLinked", "ChangeReason"], 10,
                        info="Shows sample contract changes, approvals, rollback flags, and reasons. It matters for tracing unusual renewal dates back to their recorded audit events.",
                    ),
                ],
                className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("FEATURE LAB", className="eyebrow"), html.H3("Operational and future local features", className="section-title compact-title")]),
                    html.Div(
                        [
                            html.Div([html.H4("Call transcriptions"), html.P("Future local text-derived behaviour signals.")], className="lab-item"),
                            html.Div([html.H4("Opportunity notes"), html.P("Active local response, escalation, and next-step signals.")], className="lab-item"),
                            html.Div([html.H4("CRM notes"), html.P("Future local opportunity and objection signals.")], className="lab-item"),
                            html.Div([html.H4("Meeting summaries"), html.P("Active local meeting-purpose and critical-finding signals.")], className="lab-item"),
                        ],
                        className="lab-grid",
                    ),
                    html.Div("A new feature is promoted only when the same local evaluation shows improved accuracy and business usefulness without unacceptable leakage or governance risk.", className="model-caveat"),
                ],
                className="content-section",
            ),
        ],
        className="portal data-scientist-portal",
    )
