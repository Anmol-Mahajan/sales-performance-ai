"""Modern, explainable Sales Manager Portal."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html

from src.insights import (
    answer_data_question,
    create_customer_whitespace,
    create_performance_profiles,
    create_synergy_summary,
    meeting_records_between,
    performance_drivers,
    recommendations_for_salesperson,
)
from src.features import create_opportunity_features
from src.local_llm import answer_with_local_llm, local_llm_status
from src.pipeline_forecasting import build_pipeline_forecast


COLORS = ["#236b55", "#406a8a", "#d68a3d", "#b94d55", "#6a5d84", "#4f7f85"]
SUGGESTED_QUESTIONS = [
    "Who are the top performers?",
    "Who needs coaching support?",
    "What metrics are used to compare performance?",
    "Which contracts need health checks?",
    "Show renewals within 60 days",
    "Where are the best customer whitespace opportunities?",
    "Which referral partnerships convert best?",
    "Who held the most meetings in the last 14 days?",
    "Which opportunity notes are waiting for a response?",
    "What are the upcoming cross-sell opportunities?",
    "Show critical findings from customer meetings",
    "What projects are in progress?",
    "Show blocked tickets",
    "Which projects have blocked tasks?",
    "Which opportunities have blocked projects or tasks?",
    "How achievable is the pipeline revenue target?",
    "What actions can help cover the pipeline gap?",
]
READ_ONLY_GRAPH_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
    "showTips": False,
}
TABLE_STYLE = {
    "style_table": {"overflowX": "auto", "border": "1px solid #dce3df", "borderRadius": "6px"},
    "style_cell": {
        "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "fontSize": 13,
        "padding": "11px 12px",
        "whiteSpace": "normal",
        "height": "auto",
        "minWidth": "110px",
        "maxWidth": "280px",
        "textAlign": "left",
        "border": "0",
        "borderBottom": "1px solid #edf0ee",
        "color": "#24332e",
    },
    "style_header": {
        "fontWeight": "700",
        "backgroundColor": "#f1f5f3",
        "border": "0",
        "borderBottom": "1px solid #cfd9d4",
        "color": "#3c5149",
        "textAlign": "left",
    },
    "style_data_conditional": [
        {"if": {"row_index": "odd"}, "backgroundColor": "#fafcfb"},
        {"if": {"state": "selected"}, "backgroundColor": "#e5f1ec", "border": "1px solid #7cae9b"},
    ],
}


def clean_label(value: str) -> str:
    """Convert workbook and feature names into readable display labels."""

    label = str(value).replace("_", " ")
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label).strip().title()
    for source, target in {
        "Id": "ID", "Mrr": "MRR", "Arr": "ARR", "Gbp": "£",
        "R2": "R2", "Rmse": "RMSE", "Mae": "MAE", "Roc Auc": "ROC-AUC",
    }.items():
        label = re.sub(rf"\b{re.escape(source)}\b", target, label)
    return label


def table_columns(columns) -> list[dict]:
    return [{"name": clean_label(column), "id": column} for column in columns]


def _info_icon(text: str) -> html.Span:
    """Return a keyboard-accessible information icon with a concise tooltip."""

    return html.Span(
        "i",
        className="info-icon",
        title=text,
        tabIndex=0,
        **{"aria-label": text, "data-tooltip": text},
    )


def _chart_panel(figure: go.Figure, info: str, class_name: str = "") -> html.Div:
    classes = " ".join(value for value in ["chart-panel", class_name] if value)
    return html.Div(
        [_info_icon(info), dcc.Graph(figure=figure, config=READ_ONLY_GRAPH_CONFIG)],
        className=classes,
    )


def _table_panel(table, info: str) -> html.Div:
    return html.Div([_info_icon(info), table], className="table-info-panel")


def _money(value: float, compact: bool = False) -> str:
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    if compact and absolute >= 1_000_000:
        return f"{sign}£{absolute / 1_000_000:.2f}m"
    if compact and absolute >= 1_000:
        return f"{sign}£{absolute / 1_000:.0f}k"
    return f"{sign}£{absolute:,.0f}"


def _metric(
    title: str,
    value: str,
    detail: str = "",
    tone: str = "neutral",
    info: str | None = None,
) -> html.Div:
    explanation = info or (
        f"{title} is a summary indicator. {detail or 'Use the supporting charts and tables for record-level context.'} "
        "It helps managers identify where closer review may be needed."
    )
    return html.Div(
        [
            _info_icon(explanation),
            html.Div(title, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className=f"metric-tile metric-{tone}",
    )


def _safe_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _is_date_field(column: str) -> bool:
    """Identify semantic date fields without confusing date-related counters."""

    compact = re.sub(r"[^a-z0-9]", "", str(column).lower())
    if "count" in compact or "number" in compact or compact.endswith("months"):
        return False
    return compact in {"month", "customersince"} or compact.endswith(("date", "month"))


def table_records(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    """Return Dash-friendly records, including readable local dates."""

    display = df.copy()
    if columns is not None:
        display = display[columns]
    for column in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[column]):
            display[column] = display[column].dt.strftime("%d %b %Y")
        elif _is_date_field(column):
            parsed = pd.to_datetime(display[column], errors="coerce")
            if parsed.notna().any():
                display[column] = parsed.dt.strftime("%d %b %Y").fillna(display[column].astype(str))
    return display.replace([float("inf"), float("-inf")], "").fillna("").to_dict("records")


def _data_table(
    df: pd.DataFrame,
    columns: list[str],
    page_size: int = 8,
    info: str | None = None,
    **kwargs,
) -> html.Div:
    columns = _safe_columns(df, columns)
    labels = ", ".join(clean_label(column) for column in columns[:4])
    explanation = info or (
        f"This table shows record-level {labels or 'information'}. It is important for checking the records "
        "behind the summary, comparing exceptions, and deciding where follow-up is required."
    )
    table = dash_table.DataTable(
        data=table_records(df, columns), columns=table_columns(columns), page_size=page_size,
        sort_action="native", **TABLE_STYLE, **kwargs,
    )
    return _table_panel(table, explanation)


def _chart_style(fig: go.Figure, height: int = 350, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", colorway=COLORS,
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "color": "#34483f", "size": 12},
        title={"font": {"size": 16, "color": "#18342b"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 48, "r": 24, "t": 58, "b": 44},
        legend={"title": "", "orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        showlegend=legend, hoverlabel={"bgcolor": "#18342b", "font_color": "white"},
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None, fixedrange=True)
    fig.update_yaxes(gridcolor="#edf1ef", zeroline=False, title=None, fixedrange=True)
    return fig


def _answer_visual_panel(visual) -> html.Div:
    """Render an answer-level visual recommendation as a read-only chart."""

    if visual is None or visual.data.empty:
        return html.Div()
    frame = visual.data.copy()
    if visual.x in frame and _is_date_field(visual.x):
        frame[visual.x] = pd.to_datetime(frame[visual.x], errors="coerce")
    if visual.chart_type == "bar":
        fig = px.bar(
            frame,
            x=visual.x,
            y=visual.y,
            color=visual.color if visual.color in frame else None,
            orientation=visual.orientation,
            barmode=visual.barmode,
            title=visual.title,
            color_discrete_sequence=COLORS,
        )
    else:
        fig = px.line(
            frame,
            x=visual.x,
            y=visual.y,
            color=visual.color if visual.color in frame else None,
            markers=True,
            title=visual.title,
            color_discrete_sequence=COLORS,
        )
    value_axis = "x" if visual.orientation == "h" else "y"
    axis_update = {}
    if visual.value_prefix:
        axis_update["tickprefix"] = visual.value_prefix
        axis_update["tickformat"] = "~s"
    if visual.value_suffix == "%":
        axis_update["tickformat"] = ".0%"
    elif visual.value_suffix:
        axis_update["ticksuffix"] = visual.value_suffix
    if axis_update:
        if value_axis == "x":
            fig.update_xaxes(**axis_update)
        else:
            fig.update_yaxes(**axis_update)
    _chart_style(fig, 340, legend=bool(visual.color))
    return _chart_panel(
        fig,
        visual.explanation or "This chart was selected from the verified local answer table.",
        "answer-visual",
    )


def _renewal_view(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    renewals = data["UpcomingRenewals"].copy()
    renewals = renewals.merge(
        data["Customers"][["CustomerID", "CustomerName", "Segment"]], on="CustomerID", how="left"
    ).merge(
        data["Salespeople"][["SalespersonID", "Salesperson"]],
        left_on="AccountOwnerID", right_on="SalespersonID", how="left",
    )
    services = data["ContractServices"].groupby("ContractID")["Service"].agg(
        lambda values: ", ".join(sorted(set(values.astype(str))))
    ).rename("Services")
    return renewals.merge(services, on="ContractID", how="left").sort_values("DaysToRenewal")


def _overview_figures(data, profiles, whitespace, synergy):
    monthly = data["MonthlyPerformance"].copy()
    monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
    latest_year = int(monthly["Month"].dt.year.max())
    monthly = monthly[monthly["Month"].dt.year.eq(latest_year)]
    trend = monthly.groupby("Month", as_index=False).agg(Revenue=("Revenue", "sum"), GrossProfit=("GrossProfit", "sum"))
    trend = trend.melt("Month", var_name="Metric", value_name="Value")
    revenue_fig = px.line(
        trend, x="Month", y="Value", color="Metric", markers=True, title="Revenue Momentum",
        color_discrete_map={"Revenue": COLORS[0], "GrossProfit": COLORS[2]},
    )
    revenue_fig.update_traces(line={"width": 3}, marker={"size": 6})
    revenue_fig.update_yaxes(tickprefix="£", tickformat="~s")
    _chart_style(revenue_fig, 370)

    top = profiles.head(7).sort_values("performance_score")
    top_fig = px.bar(
        top, x="performance_score", y="Salesperson", orientation="h", title="Top Performance Scores",
        color="support_status",
        color_discrete_map={"Leading": COLORS[0], "On track": COLORS[1], "Watch": COLORS[2], "Needs support": COLORS[3]},
    )
    top_fig.update_traces(hovertemplate="%{y}<br>Score: %{x}/100<extra></extra>")
    top_fig.update_xaxes(range=[0, 100])
    _chart_style(top_fig, 370, False)

    whitespace_fig = px.bar(
        whitespace.head(8).sort_values("Estimated Annual Potential"),
        x="Estimated Annual Potential", y="Customer", orientation="h",
        title="Highest Customer Whitespace Potential", color="Recommended Product",
        color_discrete_sequence=COLORS,
    ) if not whitespace.empty else go.Figure()
    whitespace_fig.update_xaxes(tickprefix="£", tickformat="~s")
    _chart_style(whitespace_fig, 360, False)

    synergy_fig = px.scatter(
        synergy, x="Referrals", y="Conversion Rate", size="Converted", color="From Salesperson",
        hover_name="To Salesperson", title="Referral Partnership Effectiveness",
        color_discrete_sequence=COLORS, size_max=28,
    ) if not synergy.empty else go.Figure()
    synergy_fig.update_yaxes(tickformat=".0%")
    _chart_style(synergy_fig, 360, False)
    return revenue_fig, top_fig, whitespace_fig, synergy_fig


def _pipeline_figures(forecast):
    summary = forecast.salesperson_summary.sort_values("AchievabilityScore")
    target_fig = go.Figure()
    target_fig.add_trace(
        go.Bar(
            x=summary["YTDRevenue"], y=summary["Salesperson"], orientation="h",
            name="YTD Revenue", marker_color=COLORS[0],
            hovertemplate="%{y}<br>YTD revenue: £%{x:,.0f}<extra></extra>",
        )
    )
    target_fig.add_trace(
        go.Bar(
            x=summary["WeightedPipelineForecast"], y=summary["Salesperson"], orientation="h",
            name="Weighted Pipeline", marker_color=COLORS[1],
            hovertemplate="%{y}<br>Weighted pipeline: £%{x:,.0f}<extra></extra>",
        )
    )
    target_fig.add_trace(
        go.Scatter(
            x=summary["AnnualTarget"], y=summary["Salesperson"], mode="markers",
            name="Annual Target", marker={"symbol": "diamond", "size": 10, "color": COLORS[3]},
            hovertemplate="%{y}<br>Annual target: £%{x:,.0f}<extra></extra>",
        )
    )
    target_fig.update_layout(title="Probability-Adjusted Forecast vs Annual Target", barmode="stack")
    target_fig.update_xaxes(tickprefix="£", tickformat="~s")
    _chart_style(target_fig, 430)

    opportunities = forecast.opportunity_forecast.copy()
    stage = opportunities.groupby("PipelineStage", as_index=False).agg(
        OpenPipeline=("PipelineValue", "sum"),
        WeightedForecast=("ForecastRevenue", "sum"),
    ).melt("PipelineStage", var_name="Measure", value_name="Value")
    stage_fig = px.bar(
        stage, x="PipelineStage", y="Value", color="Measure", barmode="group",
        title="Pipeline Value by Sales Stage",
        color_discrete_map={"OpenPipeline": COLORS[2], "WeightedForecast": COLORS[0]},
    )
    stage_fig.update_yaxes(tickprefix="£", tickformat="~s")
    stage_fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: £%{y:,.0f}<extra></extra>")
    _chart_style(stage_fig, 430)
    return target_fig, stage_fig


def build_manager_layout(data: dict[str, pd.DataFrame]) -> html.Div:
    """Build the manager experience from local workbook DataFrames."""

    monthly = data["MonthlyPerformance"].copy()
    monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
    current_year = int(monthly["Month"].dt.year.max())
    monthly = monthly[monthly["Month"].dt.year.eq(current_year)]
    opportunities = data["Opportunities"]
    contracts = data["Contracts"]
    profiles = create_performance_profiles(data)
    whitespace = create_customer_whitespace(data)
    synergy = create_synergy_summary(data)
    renewals = _renewal_view(data)
    pipeline_forecast = build_pipeline_forecast(data)
    pipeline_team = pipeline_forecast.team_summary
    revenue_fig, top_fig, whitespace_fig, synergy_fig = _overview_figures(data, profiles, whitespace, synergy)
    pipeline_target_fig, pipeline_stage_fig = _pipeline_figures(pipeline_forecast)
    llm_status = local_llm_status()

    total_revenue = float(monthly["Revenue"].sum())
    total_target = float(profiles["expected_revenue"].sum())
    total_gp = float(monthly["GrossProfit"].sum())
    won, created = float(monthly["OpportunitiesWon"].sum()), float(monthly["OpportunitiesCreated"].sum())
    open_pipeline = float(pipeline_team["OpenPipeline"])
    health_checks = int(pd.to_numeric(contracts["HealthCheckRequired"], errors="coerce").fillna(0).sum())
    meetings = data.get("Meetings", pd.DataFrame())
    notes = data.get("OpportunityNotes", pd.DataFrame())
    projects = data.get("Projects", pd.DataFrame())
    tickets = data.get("OpportunityTickets", pd.DataFrame())
    critical_meetings = int(meetings.get("CriticalFindingFlag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    waiting_notes = notes[notes.get("ResponseStatus", pd.Series("", index=notes.index)).astype(str).str.lower().eq("waiting response")] if not notes.empty else notes
    critical_waiting = int(waiting_notes.get("CriticalFindingFlag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not waiting_notes.empty else 0
    active_projects = int((~projects.get("ProjectStatus", pd.Series("", index=projects.index)).astype(str).str.lower().eq("complete")).sum()) if not projects.empty else 0
    blocked_projects = int((projects.get("ProjectStatus", pd.Series("", index=projects.index)).astype(str).str.lower().eq("on hold") | projects.get("DeliveryHealth", pd.Series("", index=projects.index)).astype(str).str.lower().eq("red")).sum()) if not projects.empty else 0
    escalated_tickets = int(tickets.get("EscalationFlag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not tickets.empty else 0
    salesperson_options = [
        {"label": f"{row.Salesperson}  |  {row.Segment}", "value": row.SalespersonID}
        for row in profiles.itertuples(index=False)
    ]
    default_salesperson = salesperson_options[0]["value"] if salesperson_options else None
    priority_options = [{"label": value, "value": value} for value in sorted(renewals["Priority"].dropna().unique())]
    service_options = [{"label": value, "value": value} for value in sorted(data["ContractServices"]["Service"].dropna().unique())]
    segment_options = [{"label": value, "value": value} for value in sorted(data["Customers"]["Segment"].dropna().unique())]
    renewal_columns = [
        "Priority", "CustomerName", "Salesperson", "ContractName", "ContractStatus", "Services",
        "CurrentEndDate", "DaysToRenewal", "ContractARR", "RenewalRisk", "EndDateChangeCount",
        "RollbackCount", "HealthCheckReason", "SuggestedAction",
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Div("MANAGER VIEW", className="eyebrow"), html.H2("Team performance and next actions", className="page-title")]),
                    html.Div([html.Label("Focus salesperson", className="control-label"), dcc.Dropdown(id="salesperson-dropdown", options=salesperson_options, value=default_salesperson, clearable=False)], className="heading-control"),
                ], className="page-heading manager-heading",
            ),
            html.Div(
                [
                    _metric(f"{current_year} YTD Revenue", _money(total_revenue, True), f"{total_revenue / max(total_target, 1):.0%} of YTD target", "positive"),
                    _metric("YTD Target", _money(total_target, True), f"Pace gap {_money(total_revenue - total_target, True)}"),
                    _metric("Open Pipeline", _money(open_pipeline, True), "Expected to close this year", "info"),
                    _metric("Gross Profit", _money(total_gp, True), f"{total_gp / max(total_revenue, 1):.1%} margin"),
                    _metric("Win Rate", f"{won / max(created, 1):.1%}", f"{won:,.0f} won of {created:,.0f} created", "warning"),
                    _metric("Health Checks", f"{health_checks:,}", "Contracts requiring attention", "danger"),
                ], className="metric-grid six-up",
            ),
            html.Div(
                [
                    _chart_panel(
                        revenue_fig,
                        "Shows monthly revenue and gross-profit movement. It matters because sustained growth and margin movement reveal whether commercial performance is improving or weakening.",
                        "span-two",
                    ),
                    _chart_panel(
                        top_fig,
                        "Compares salesperson performance and support status. It matters for recognising leaders and identifying people who may need focused coaching.",
                    ),
                ],
                className="dashboard-grid overview-grid",
            ),
            html.Section(
                [
                    html.Div([html.Div("PIPELINE FORECAST", className="eyebrow"), html.H3(f"{current_year} revenue achievability", className="section-title compact-title")]),
                    html.Div(
                        [
                            _metric("Annual Target", _money(pipeline_team["AnnualTarget"], True), "Full-year sales target"),
                            _metric("Target Remaining", _money(pipeline_team["TargetRemaining"], True), "After YTD recognised revenue", "warning"),
                            _metric("Weighted Pipeline", _money(pipeline_team["WeightedPipelineForecast"], True), f"{pipeline_team['WeightedCoverage']:.0%} of remaining target", "info"),
                            _metric("Year-End Forecast", _money(pipeline_team["ForecastYearEndRevenue"], True), "YTD revenue plus weighted pipeline", "positive"),
                            _metric("Forecast Gap", _money(pipeline_team["ForecastGap"], True), "Probability-adjusted scenario", "danger" if pipeline_team["ForecastGap"] < 0 else "positive"),
                            _metric("Achievability", str(pipeline_team["Achievability"]), f"{pipeline_team['AchievabilityScore']:.0f} / 100", "warning" if pipeline_team["Achievability"] == "Possible" else "danger" if pipeline_team["Achievability"] == "At risk" else "positive"),
                        ],
                        className="metric-grid six-up pipeline-metrics",
                    ),
                    html.Div(
                        [
                            _chart_panel(
                                pipeline_target_fig,
                                "Compares recognised revenue and weighted pipeline with annual targets. It matters because it shows whether each target is realistically covered at current probabilities.",
                            ),
                            _chart_panel(
                                pipeline_stage_fig,
                                "Shows raw and probability-weighted pipeline by sales stage. It matters because concentration in early stages makes the forecast less certain.",
                            ),
                        ],
                        className="dashboard-grid two-column",
                    ),
                    html.H4("Salesperson target coverage", className="subsection-title"),
                    _data_table(
                        pipeline_forecast.salesperson_summary,
                        ["Salesperson", "YTDRevenue", "AnnualTarget", "TargetGap", "OpenPipeline", "WeightedPipelineForecast", "ForecastYearEndRevenue", "ForecastGap", "PipelineCoverage", "AchievabilityScore", "Achievability", "WaitingResponseOpportunities", "StalledOpportunities"],
                        10,
                    ),
                    html.H4("Highest forecast contribution opportunities", className="subsection-title"),
                    _data_table(
                        pipeline_forecast.opportunity_forecast.sort_values("ForecastRevenue", ascending=False).head(30),
                        ["OpportunityID", "CustomerName", "Salesperson", "Product", "PipelineStage", "PipelineValue", "ExpectedCloseDate", "ForecastCategory", "PipelineRisk", "AdjustedWinProbability", "ForecastRevenue", "DaysInStage", "NextStep"],
                        8,
                    ),
                    html.Div(f"Forecasts are local probability-adjusted scenarios, not guaranteed revenue. Stage probability, historical conversion, customer responses, and recorded risks affect the result. Classifier guardrail: {pipeline_team['ClassifierGuardrail']}.", className="model-caveat"),
                ],
                className="content-section pipeline-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("CUSTOMER OPERATIONS", className="eyebrow"), html.H3("Engagement, responses, and delivery", className="section-title compact-title")]),
                    html.Div(
                        [
                            _metric("Customer Meetings", f"{len(meetings):,}", "Detailed local meeting records", "info"),
                            _metric("Critical Findings", f"{critical_meetings:,}", "Flagged meeting observations", "danger" if critical_meetings else "positive"),
                            _metric("Waiting Responses", f"{len(waiting_notes):,}", f"{critical_waiting:,} critical", "warning"),
                            _metric("Active Projects", f"{active_projects:,}", "Linked to opportunities", "positive"),
                            _metric("Blocked Projects", f"{blocked_projects:,}", "Red or on hold", "danger" if blocked_projects else "positive"),
                            _metric("Escalated Tickets", f"{escalated_tickets:,}", "High-priority delivery attention", "warning"),
                        ],
                        className="metric-grid six-up operational-metrics",
                    ),
                ],
                className="content-section operational-overview",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Div([html.Div("LOCAL QUESTION ENGINE", className="eyebrow"), html.H3("Ask your sales data", className="section-title compact-title")]),
                            html.Div(llm_status.message, className=f"llm-status {'llm-ready' if llm_status.model_installed else 'llm-offline'}"),
                        ],
                        className="ask-heading",
                    ),
                    dcc.RadioItems(
                        id="question-mode",
                        options=[
                            {"label": "Workbook engine", "value": "workbook"},
                            {"label": "Local LLM", "value": "llm", "disabled": not llm_status.model_installed},
                        ],
                        value="workbook",
                        inline=True,
                        className="question-mode",
                        inputClassName="question-mode-input",
                        labelClassName="question-mode-label",
                    ),
                    html.Div(
                        [dcc.Input(id="data-question-input", type="text", placeholder="e.g. Which contracts need health checks?", className="question-input"), html.Button("Ask data", id="ask-data-button", n_clicks=0, className="primary-button")],
                        className="question-row",
                    ),
                    dcc.Dropdown(
                        id="suggested-question",
                        options=[
                            {"label": text, "value": text} for text in SUGGESTED_QUESTIONS
                        ],
                        placeholder="Suggested questions", clearable=True, className="suggested-questions",
                    ),
                    dcc.Loading(
                        html.Div([html.Div("Ready for a local workbook question.", className="answer-summary"), html.Div("Workbook mode uses deterministic pandas calculations.", className="answer-source")], id="data-answer", className="answer-panel"),
                        type="circle",
                        color=COLORS[0],
                    ),
                ], className="ask-section",
            ),
            html.Div(id="salesperson-detail-section"),
            html.Section(
                [
                    html.Div([html.Div("OPPORTUNITY", className="eyebrow"), html.H3("Customer whitespace and collaboration", className="section-title compact-title")]),
                    html.Div(
                        [
                            _chart_panel(
                                whitespace_fig,
                                "Ranks customers by estimated product whitespace. It matters for prioritising evidence-based cross-sell conversations rather than applying one recommendation to every account.",
                            ),
                            _chart_panel(
                                synergy_fig,
                                "Shows referral volume and conversion between salespeople. It matters for identifying productive specialist partnerships and collaboration gaps.",
                            ),
                        ],
                        className="dashboard-grid two-column",
                    ),
                    html.H4("Priority cross-sell conversations", className="subsection-title"),
                    _data_table(whitespace.head(30), ["Customer", "Customer Segment", "Account Owner", "Current Products", "Current Product MRR", "Missing Products", "Whitespace Score", "Estimated Annual Potential", "Recommended Product", "Recommended Specialist", "Estimate Basis", "Next Action"], 7, filter_action="native"),
                    html.H4("Referral partnerships", className="subsection-title"),
                    _data_table(synergy.head(30), ["From Salesperson", "To Salesperson", "Referrals", "Accepted", "Converted", "Conversion Rate", "SynergyType", "SynergyStrength"], 7),
                ], className="content-section",
            ),
            html.Section(
                [
                    html.Div([html.Div("RENEWAL OPERATIONS", className="eyebrow"), html.H3("Contract renewals and health checks", className="section-title compact-title")]),
                    html.Div(
                        [
                            html.Div([html.Label("Salesperson", className="control-label"), dcc.Dropdown(id="renewal-owner-filter", options=[{"label": "All salespeople", "value": "all"}] + salesperson_options, value="all", clearable=False)]),
                            html.Div([html.Label("Priority", className="control-label"), dcc.Dropdown(id="renewal-priority-filter", options=priority_options, multi=True, placeholder="All priorities")]),
                            html.Div([html.Label("Service", className="control-label"), dcc.Dropdown(id="renewal-service-filter", options=service_options, placeholder="All services")]),
                            html.Div([html.Label("Customer segment", className="control-label"), dcc.Dropdown(id="renewal-segment-filter", options=segment_options, placeholder="All segments")]),
                        ], className="filter-grid",
                    ),
                    html.Div(id="renewal-count", className="table-context"),
                    _table_panel(
                        dash_table.DataTable(
                            id="renewal-table", data=table_records(renewals, renewal_columns), columns=table_columns(_safe_columns(renewals, renewal_columns)),
                            page_size=10, sort_action="native", filter_action="native", **TABLE_STYLE,
                        ),
                        "Shows contract renewal timing, value, risk, audit changes, and recommended action. It matters for prioritising health checks before revenue is at risk.",
                    ),
                ], className="content-section",
            ),
            dcc.Store(id="salesperson-feature-store", data=profiles.to_dict("records")),
        ], className="portal manager-portal",
    )


def register_manager_callbacks(
    app, data: dict[str, pd.DataFrame], boundary_message: str = "No data was sent outside this machine."
) -> None:
    """Register local manager interactions."""

    profiles = create_performance_profiles(data)
    contracts, services = data["Contracts"].copy(), data["ContractServices"].copy()
    billing, monthly = data["ExistingCustomerBilling"].copy(), data["MonthlyPerformance"].copy()
    monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
    current_year = int(monthly["Month"].dt.year.max())
    monthly = monthly[monthly["Month"].dt.year.eq(current_year)]
    renewals = _renewal_view(data)
    meetings = data.get("Meetings", pd.DataFrame()).copy()
    notes = data.get("OpportunityNotes", pd.DataFrame()).copy()
    projects = data.get("Projects", pd.DataFrame()).copy()
    tickets = data.get("OpportunityTickets", pd.DataFrame()).copy()
    tasks = data.get("TicketTasks", pd.DataFrame()).copy()
    pipeline_forecast = build_pipeline_forecast(data, save_model=False)
    opportunity_view = create_opportunity_features(data)
    opportunity_view["CreatedDate"] = pd.to_datetime(opportunity_view["CreatedDate"], errors="coerce")
    opportunity_view = opportunity_view[opportunity_view["CreatedDate"].dt.year.eq(current_year)]
    forecast_columns = pipeline_forecast.opportunity_forecast[
        ["OpportunityID", "AdjustedWinProbability", "ForecastRevenue", "ProbabilityAdjustment"]
    ]
    opportunity_view = opportunity_view.merge(forecast_columns, on="OpportunityID", how="left")
    customer_names = data.get("Customers", pd.DataFrame())[["CustomerID", "CustomerName"]]
    if not meetings.empty:
        meetings = meetings.merge(customer_names, on="CustomerID", how="left")

    @app.callback(Output("salesperson-detail-section", "children"), Input("salesperson-dropdown", "value"))
    def render_salesperson_detail(salesperson_id):
        selected = profiles[profiles["SalespersonID"] == salesperson_id]
        if selected.empty:
            return html.Div()
        row = selected.iloc[0]
        drivers = performance_drivers(profiles, salesperson_id)
        recommendations = recommendations_for_salesperson(data, profiles, salesperson_id)
        pipeline_actions = pipeline_forecast.suggestions[
            pipeline_forecast.suggestions["SalespersonID"] == salesperson_id
        ].sort_values("Priority")
        pipeline_recommendations = [
            {"Action": action.Action, "Evidence": action.Evidence, "Type": action.ActionType}
            for action in pipeline_actions.head(4).itertuples(index=False)
        ]
        recommendations = (pipeline_recommendations + recommendations)[:6]
        owned_contracts = contracts[contracts["AccountOwnerID"] == salesperson_id].copy().sort_values("DaysToRenewal")
        owned_ids = set(owned_contracts["ContractID"])
        owned_services = services[services["ContractID"].isin(owned_ids)].copy()
        billing_months = pd.to_datetime(billing["BillingMonth"], errors="coerce")
        current_billing = billing[
            (billing["AccountOwnerID"] == salesperson_id)
            & billing_months.eq(billing_months.max())
        ].copy()
        health_checks = int(pd.to_numeric(owned_contracts["HealthCheckRequired"], errors="coerce").fillna(0).sum())
        person_meetings = meetings[meetings["SalespersonID"] == salesperson_id].copy().sort_values("MeetingDate", ascending=False)
        person_notes = notes[notes["SalespersonID"] == salesperson_id].copy()
        waiting_notes = person_notes[person_notes["ResponseStatus"].astype(str).str.lower().eq("waiting response")]
        person_opportunities = opportunity_view[opportunity_view["SalespersonID"] == salesperson_id].copy()
        person_forecast = pipeline_forecast.salesperson_summary[
            pipeline_forecast.salesperson_summary["SalespersonID"] == salesperson_id
        ]
        forecast_row = person_forecast.iloc[0] if not person_forecast.empty else pd.Series(dtype=float)
        critical_findings = int(person_meetings.get("CriticalFindingFlag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        blocked_projects = int(row.get("blocked_opportunity_project_count", 0))
        average_wait = float(row.get("average_waiting_response_age_days", 0))

        driver_fig = px.bar(
            drivers, x="Index", y="Driver", orientation="h", color="Direction",
            title="Performance Drivers vs Peer Benchmark", color_discrete_map={"+": COLORS[0], "-": COLORS[3]},
            custom_data=["Actual", "Peer Benchmark"],
        )
        driver_fig.add_vline(x=100, line_dash="dash", line_color="#8a9892")
        driver_fig.update_traces(hovertemplate="%{y}<br>Index: %{x:.0f}%<br>Actual: %{customdata[0]:.2f}<br>Peer benchmark: %{customdata[1]:.2f}<extra></extra>")
        driver_fig.update_xaxes(ticksuffix="%", range=[0, max(165, drivers["Index"].max() + 10)])
        _chart_style(driver_fig, 390, False)

        trend = monthly[monthly["SalespersonID"] == salesperson_id].copy()
        trend["Month"] = pd.to_datetime(trend["Month"], errors="coerce")
        trend_fig = px.area(trend, x="Month", y="Revenue", title="Monthly Revenue Trend", color_discrete_sequence=[COLORS[1]])
        trend_fig.update_yaxes(tickprefix="£", tickformat="~s")
        _chart_style(trend_fig, 300, False)

        gap = float(row["performance_gap"])
        score_color = COLORS[0] if row["performance_score"] >= 70 else COLORS[2] if row["performance_score"] >= 55 else COLORS[3]
        max_meetings = max(int(row["total_meetings"] * 1.5), int(row["total_meetings"]) + 10, 1)
        max_cross_sell = max(int(row["cross_sell_opportunities"] * 2), int(row["cross_sell_opportunities"]) + 5, 1)
        return html.Section(
            [
                html.Div(
                    [
                        html.Div([html.Div("SALESPERSON PERFORMANCE", className="eyebrow"), html.H3(row["Salesperson"], className="person-name"), html.Div(f"{row.get('Segment', '')} | {row.get('PrimarySpecialism', '')} | {row.get('Region', '')}", className="person-meta")]),
                        html.Div([html.Div(str(row["performance_score"]), className="score-number"), html.Div("/ 100", className="score-denominator")], className="score-ring", style={"background": f"conic-gradient({score_color} {row['performance_score']}%, #dfe7e3 0)"}),
                    ], className="person-hero",
                ),
                html.Div(
                    [
                        _metric(f"{current_year} YTD Revenue", _money(row["total_revenue"], True), f"Peer median {_money(row['benchmark_revenue'], True)}", "positive"),
                        _metric("YTD Target", _money(row["expected_revenue"], True), "Target prorated to snapshot date"),
                        _metric("YTD Pace Gap", _money(gap, True), f"{row['target_attainment']:.0%} YTD target attainment", "positive" if gap >= 0 else "danger"),
                        _metric("Gross Profit", _money(row["total_gross_profit"], True), "Measured workbook total"),
                        _metric("Win Rate", f"{row['win_rate']:.1%}", f"{row['opportunities_won']:,.0f} opportunities won"),
                        _metric("Renewal Health Checks", f"{health_checks:,}", f"{row['contracts_owned']:,.0f} contracts owned", "warning" if health_checks else "positive"),
                    ], className="metric-grid six-up salesperson-metrics",
                ),
                html.Div(
                    [
                        _metric("Annual Target", _money(forecast_row.get("AnnualTarget", 0), True), "Full-year target"),
                        _metric("Open Pipeline", _money(forecast_row.get("OpenPipeline", 0), True), f"{forecast_row.get('OpenOpportunities', 0):,.0f} opportunities", "info"),
                        _metric("Weighted Pipeline", _money(forecast_row.get("WeightedPipelineForecast", 0), True), f"{forecast_row.get('WeightedCoverage', 0):.0%} of remaining target", "info"),
                        _metric("Year-End Forecast", _money(forecast_row.get("ForecastYearEndRevenue", 0), True), "YTD plus probability-adjusted pipeline", "positive"),
                        _metric("Forecast Gap", _money(forecast_row.get("ForecastGap", 0), True), "Against annual target", "danger" if forecast_row.get("ForecastGap", 0) < 0 else "positive"),
                        _metric("Achievability", str(forecast_row.get("Achievability", "Not available")), f"{forecast_row.get('AchievabilityScore', 0):,.0f} / 100", "warning" if forecast_row.get("Achievability") == "Possible" else "danger" if forecast_row.get("Achievability") == "At risk" else "positive"),
                    ],
                    className="metric-grid six-up salesperson-metrics salesperson-pipeline-metrics",
                ),
                html.Div(
                    [
                        _metric("Customer Meetings", f"{len(person_meetings):,}", f"{row.get('account_health_check_meeting_count', 0):,.0f} account health checks", "info"),
                        _metric("Critical Findings", f"{critical_findings:,}", "Meeting observations requiring attention", "danger" if critical_findings else "positive"),
                        _metric("Waiting Responses", f"{len(waiting_notes):,}", f"Average age {average_wait:.1f} days", "warning" if len(waiting_notes) else "positive"),
                        _metric("Support Escalations", f"{row.get('support_escalation_meeting_count', 0):,.0f}", "Meeting-level escalations", "warning"),
                        _metric("Blocked Projects", f"{blocked_projects:,}", f"{row.get('active_opportunity_project_count', 0):,.0f} active projects", "danger" if blocked_projects else "positive"),
                        _metric("Open Tickets", f"{row.get('open_opportunity_ticket_count', 0):,.0f}", f"{row.get('overdue_delivery_task_count', 0):,.0f} overdue tasks", "warning"),
                    ],
                    className="metric-grid six-up salesperson-metrics operational-person-metrics",
                ),
                html.Div(
                    [
                        _chart_panel(
                            driver_fig,
                            "Indexes this salesperson's activity and outcomes against a relevant peer benchmark. It matters for identifying the specific drivers behind a performance gap.",
                        ),
                        html.Div(
                            [html.Div("RECOMMENDATIONS", className="eyebrow"), html.Ol([html.Li([html.Div(item["Action"], className="recommendation-action"), html.Div(item["Evidence"], className="recommendation-evidence"), html.Span(item["Type"], className="recommendation-type")]) for item in recommendations], className="recommendation-list")],
                            className="recommendations-panel",
                        ),
                    ], className="dashboard-grid two-column drivers-grid",
                ),
                html.Div(
                    [
                        _chart_panel(
                            trend_fig,
                            "Shows the selected salesperson's monthly revenue movement. It matters for distinguishing a sustained trend from a single strong or weak month.",
                        ),
                        html.Div(
                            [
                                html.Div("ACTIVITY SENSITIVITY", className="eyebrow"), html.H4("Meetings and cross-sell scenario", className="subsection-title scenario-title"),
                                html.Label("Meetings", className="control-label"),
                                dcc.Slider(id="scenario-meetings", min=0, max=max_meetings, step=1, value=int(row["total_meetings"]), marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                                html.Label("Cross-sell opportunities", className="control-label scenario-label"),
                                dcc.Slider(id="scenario-cross-sell", min=0, max=max_cross_sell, step=1, value=int(row["cross_sell_opportunities"]), marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                                html.Div(id="scenario-estimate", className="scenario-output"),
                            ], className="scenario-panel",
                        ),
                    ], className="dashboard-grid two-column",
                ),
                html.Div(
                    [
                        html.Div([html.Div("CUSTOMER MEETINGS", className="eyebrow"), html.H4("Meeting notes and critical findings", className="subsection-title operational-title")]),
                        _table_panel(
                            dash_table.DataTable(
                                id="salesperson-meetings-table",
                                data=table_records(person_meetings, ["MeetingID", "MeetingDate", "CustomerName", "CustomerRelationship", "MeetingType", "OpportunityID", "CriticalSeverity", "FollowUpStatus"]),
                                columns=table_columns(["MeetingID", "MeetingDate", "CustomerName", "CustomerRelationship", "MeetingType", "OpportunityID", "CriticalSeverity", "FollowUpStatus"]),
                                page_size=8,
                                sort_action="native",
                                filter_action="native",
                                cell_selectable=True,
                                **{
                                    **TABLE_STYLE,
                                    "style_data_conditional": TABLE_STYLE["style_data_conditional"] + [
                                        {"if": {"filter_query": '{CriticalSeverity} = "Critical"'}, "backgroundColor": "#f8e4e6", "color": "#7e2630", "fontWeight": "700"},
                                        {"if": {"filter_query": '{CriticalSeverity} = "High"'}, "backgroundColor": "#fff2df", "color": "#744817"},
                                    ],
                                },
                            ),
                            "Shows the selected salesperson's customer meetings, opportunity links, severity, and follow-up status. It matters because clicking a row exposes the notes and critical findings behind the activity count.",
                        ),
                        html.Div(id="meeting-detail-panel"),
                    ],
                    className="operational-block",
                ),
                html.Div(
                    [
                        html.Div([html.Div("OPPORTUNITY DELIVERY", className="eyebrow"), html.H4("Projects, tickets, and task status", className="subsection-title operational-title")]),
                        _table_panel(
                            dash_table.DataTable(
                                id="salesperson-opportunities-table",
                                data=table_records(person_opportunities, ["OpportunityID", "CustomerName", "Product", "Stage", "PipelineStage", "PipelineValue", "ExpectedCloseDate", "ForecastCategory", "PipelineRisk", "AdjustedWinProbability", "ForecastRevenue", "MeetingCount", "WaitingResponseCount", "ProjectStage", "DeliveryHealth", "OpenTicketCount", "BlockedTaskCount"]),
                                columns=table_columns(["OpportunityID", "CustomerName", "Product", "Stage", "PipelineStage", "PipelineValue", "ExpectedCloseDate", "ForecastCategory", "PipelineRisk", "AdjustedWinProbability", "ForecastRevenue", "MeetingCount", "WaitingResponseCount", "ProjectStage", "DeliveryHealth", "OpenTicketCount", "BlockedTaskCount"]),
                                page_size=8,
                                sort_action="native",
                                filter_action="native",
                                cell_selectable=True,
                                **{
                                    **TABLE_STYLE,
                                    "style_data_conditional": TABLE_STYLE["style_data_conditional"] + [
                                        {"if": {"filter_query": '{DeliveryHealth} = "Red"'}, "backgroundColor": "#f8e4e6", "color": "#7e2630", "fontWeight": "700"},
                                        {"if": {"filter_query": '{DeliveryHealth} = "Amber"'}, "backgroundColor": "#fff2df", "color": "#744817"},
                                    ],
                                },
                            ),
                            "Connects opportunities with forecast value, customer activity, project health, tickets, and blocked tasks. It matters because clicking a row reveals the operational evidence behind pipeline risk.",
                        ),
                        html.Div(id="opportunity-detail-panel"),
                    ],
                    className="operational-block",
                ),
                html.Details([html.Summary("Customers currently billing"), _data_table(current_billing.head(100), ["CustomerID", "ServiceCategory", "Service", "MRR", "TotalBilled", "RenewalDate", "PaymentStatus", "ContractLinkStatus"], 8)], className="data-details"),
                html.Details([html.Summary("Contracts owned"), _data_table(owned_contracts, ["CustomerID", "ContractName", "ContractStatus", "CurrentEndDate", "DaysToRenewal", "ContractMRR", "ContractARR", "RenewalRisk", "EndDateChangeCount", "RollbackCount", "SuggestedAction"], 8)], className="data-details"),
                html.Details([html.Summary("Services linked to contracts"), _data_table(owned_services, ["CustomerID", "ContractID", "ServiceCategory", "Service", "ServiceMRR", "ServiceARR", "ServiceStatus"], 8)], className="data-details"),
            ], className="content-section salesperson-section",
        )

    @app.callback(
        Output("meeting-detail-panel", "children"),
        Input("salesperson-meetings-table", "active_cell"),
        State("salesperson-meetings-table", "derived_virtual_data"),
        State("salesperson-meetings-table", "data"),
        prevent_initial_call=True,
    )
    def render_meeting_detail(active_cell, visible_rows, all_rows):
        rows = visible_rows or all_rows
        if not active_cell or not rows:
            return html.Div()
        row_index = active_cell.get("row", -1)
        if row_index < 0 or row_index >= len(rows):
            return html.Div()
        meeting_id = rows[row_index].get("MeetingID")
        selected = meetings[meetings["MeetingID"] == meeting_id]
        if selected.empty:
            return html.Div()
        meeting = selected.iloc[0]
        critical = bool(meeting.get("CriticalFindingFlag", False))
        opportunity = meeting.get("OpportunityID")
        opportunity_text = "No opportunity linked" if pd.isna(opportunity) or not opportunity else str(opportunity)
        return html.Div(
            [
                html.Div(
                    [
                        html.Div([html.Div(str(meeting.get("MeetingType", "Meeting")), className="detail-kicker"), html.H4(str(meeting.get("Subject", meeting_id)), className="detail-title")]),
                        html.Div(str(meeting.get("CriticalSeverity", "None")), className=f"severity-badge severity-{str(meeting.get('CriticalSeverity', 'none')).lower()}"),
                    ],
                    className="detail-heading",
                ),
                html.Div(
                    [
                        html.Div([html.Span("Meeting", className="detail-label"), html.Strong(str(meeting_id))]),
                        html.Div([html.Span("Customer", className="detail-label"), html.Strong(str(meeting.get("CustomerName", meeting.get("CustomerID", ""))))]),
                        html.Div([html.Span("Opportunity", className="detail-label"), html.Strong(opportunity_text)]),
                        html.Div([html.Span("Follow-up", className="detail-label"), html.Strong(str(meeting.get("FollowUpStatus", "")))]),
                    ],
                    className="detail-meta-grid",
                ),
                html.Div([html.Div("MEETING SUMMARY", className="detail-label"), html.P(str(meeting.get("MeetingSummary", "")))], className="detail-copy"),
                html.Div([html.Div("SALESPERSON NOTES", className="detail-label"), html.P(str(meeting.get("SalespersonNotes", "")))], className="detail-copy notes-copy"),
                html.Div(
                    [html.Div("CRITICAL FINDING", className="detail-label"), html.Strong(str(meeting.get("CriticalFinding", "")))],
                    className="critical-finding",
                ) if critical else html.Div(),
                html.Div([html.Div("NEXT ACTION", className="detail-label"), html.Strong(str(meeting.get("NextAction", ""))), html.Span(f"Due {pd.to_datetime(meeting.get('ActionDueDate')):%d %b %Y}", className="detail-date")], className="next-action"),
            ],
            className=f"record-detail {'record-critical' if critical else ''}",
        )

    @app.callback(
        Output("opportunity-detail-panel", "children"),
        Input("salesperson-opportunities-table", "active_cell"),
        State("salesperson-opportunities-table", "derived_virtual_data"),
        State("salesperson-opportunities-table", "data"),
        prevent_initial_call=True,
    )
    def render_opportunity_detail(active_cell, visible_rows, all_rows):
        rows = visible_rows or all_rows
        if not active_cell or not rows:
            return html.Div()
        row_index = active_cell.get("row", -1)
        if row_index < 0 or row_index >= len(rows):
            return html.Div()
        opportunity_id = rows[row_index].get("OpportunityID")
        selected = opportunity_view[opportunity_view["OpportunityID"] == opportunity_id]
        if selected.empty:
            return html.Div()
        opportunity = selected.iloc[0]
        linked_notes = notes[notes["OpportunityID"] == opportunity_id].copy().sort_values("NoteDate", ascending=False)
        linked_projects = projects[projects["OpportunityID"] == opportunity_id].copy()
        linked_tickets = tickets[tickets["OpportunityID"] == opportunity_id].copy()
        linked_tasks = tasks[tasks["OpportunityID"] == opportunity_id].copy()
        waiting = int(linked_notes["ResponseStatus"].astype(str).str.lower().eq("waiting response").sum()) if not linked_notes.empty else 0
        return html.Div(
            [
                html.Div(
                    [
                        html.Div([html.Div(str(opportunity_id), className="detail-kicker"), html.H4(f"{opportunity.get('CustomerName', opportunity.get('CustomerID', ''))} - {opportunity.get('Product', '')}", className="detail-title")]),
                        html.Div(str(opportunity.get("Stage", "")), className="stage-badge"),
                    ],
                    className="detail-heading",
                ),
                html.Div(
                    [
                        html.Div([html.Span("Pipeline", className="detail-label"), html.Strong(_money(opportunity.get("PipelineValue", 0)))]),
                        html.Div([html.Span("Weighted forecast", className="detail-label"), html.Strong(_money(opportunity.get("ForecastRevenue", 0)))]),
                        html.Div([html.Span("Win probability", className="detail-label"), html.Strong(f"{opportunity.get('AdjustedWinProbability', 0):.0%}")]),
                        html.Div([html.Span("Expected close", className="detail-label"), html.Strong(f"{pd.to_datetime(opportunity.get('ExpectedCloseDate')):%d %b %Y}" if pd.notna(opportunity.get("ExpectedCloseDate")) else "Not set")]),
                        html.Div([html.Span("Meetings", className="detail-label"), html.Strong(f"{opportunity.get('MeetingCount', 0):,.0f}")]),
                        html.Div([html.Span("Waiting responses", className="detail-label"), html.Strong(f"{waiting:,}")]),
                        html.Div([html.Span("Project stage", className="detail-label"), html.Strong(str(opportunity.get("ProjectStage", "Not started")) if pd.notna(opportunity.get("ProjectStage")) else "Not started")]),
                        html.Div([html.Span("Pipeline risk", className="detail-label"), html.Strong(str(opportunity.get("PipelineRisk", "")))]),
                    ],
                    className="detail-meta-grid",
                ),
                html.Div([html.Div("NEXT PIPELINE STEP", className="detail-label"), html.Strong(str(opportunity.get("NextStep", ""))), html.Span(f"Due {pd.to_datetime(opportunity.get('NextStepDueDate')):%d %b %Y}" if pd.notna(opportunity.get("NextStepDueDate")) else "", className="detail-date")], className="next-action"),
                html.H5("Opportunity notes", className="detail-section-title"),
                _data_table(linked_notes, ["NoteDate", "NoteSource", "NoteType", "ResponseStatus", "ResponseAgeDays", "EscalationSeverity", "NoteText", "NextAction"], 5),
                html.H5("Linked projects", className="detail-section-title"),
                _data_table(linked_projects, ["ProjectID", "ProjectName", "ProjectStage", "ProjectStatus", "DeliveryHealth", "PercentComplete", "TargetCompletionDate", "Blocker"], 5),
                html.H5("Opportunity tickets", className="detail-section-title"),
                _data_table(linked_tickets, ["TicketID", "ProjectID", "TicketType", "TicketStatus", "Priority", "DueDate", "EscalationFlag", "LastUpdate"], 6),
                html.H5("Ticket tasks", className="detail-section-title"),
                _data_table(linked_tasks, ["TaskID", "TicketID", "TaskName", "TaskStatus", "TaskOwner", "DueDate", "BlockedReason"], 6),
            ],
            className="record-detail opportunity-detail",
        )

    @app.callback(
        Output("scenario-estimate", "children"), Input("scenario-meetings", "value"),
        Input("scenario-cross-sell", "value"), State("salesperson-dropdown", "value"), prevent_initial_call=True,
    )
    def update_scenario(meetings, cross_sell, salesperson_id):
        selected = profiles[profiles["SalespersonID"] == salesperson_id]
        if selected.empty or meetings is None or cross_sell is None:
            return ""
        row = selected.iloc[0]
        meeting_value = row["total_revenue"] / max(row["total_meetings"], 1) * 0.20
        cross_sell_value = row["cross_sell_revenue"] / max(row["cross_sell_opportunities"], 1)
        impact = (meetings - row["total_meetings"]) * meeting_value + (cross_sell - row["cross_sell_opportunities"]) * cross_sell_value
        return [html.Div(f"Estimated change: {_money(impact, True)}", className=f"scenario-value {'positive-text' if impact >= 0 else 'negative-text'}"), html.Div("Sensitivity basis: 20% of observed revenue per meeting plus average cross-sell revenue per opportunity. This is not the pipeline forecast or a guarantee.", className="scenario-note")]

    @app.callback(
        Output("data-answer", "children"), Input("ask-data-button", "n_clicks"),
        Input("data-question-input", "n_submit"), Input("suggested-question", "value"),
        State("data-question-input", "value"), State("question-mode", "value"), prevent_initial_call=True,
    )
    def ask_local_data(_clicks, _submit, suggested, typed, mode):
        question = suggested if ctx.triggered_id == "suggested-question" and suggested else typed
        answer = answer_with_local_llm(question or "", data) if mode == "llm" else answer_data_question(question or "", data)
        table = answer.table.copy()
        result = html.Div()
        if not table.empty:
            preferred = [
                "Rank", "Salesperson", "Meetings", "Period Start", "Period End",
                "performance_score", "total_revenue", "expected_revenue", "performance_gap", "win_rate", "support_status",
                "Priority", "CustomerName", "ContractName", "DaysToRenewal", "ContractARR", "RenewalRisk", "SuggestedAction",
                "Customer", "Account Owner", "Recommended Product", "Estimated Annual Potential",
                "Next Action", "Whitespace Score", "Estimate Basis",
                "From Salesperson", "To Salesperson", "Referrals", "Conversion Rate",
                "MeetingID", "MeetingDate", "MeetingType", "CustomerRelationship", "Subject",
                "DurationMinutes", "MeetingStatus", "MeetingSummary", "SalespersonNotes",
                "CriticalSeverity", "CriticalFinding", "NextAction", "ActionDueDate", "FollowUpStatus",
                "NoteID", "NoteDate", "OpportunityID", "Product", "OpportunityType",
                "PipelineStage", "PipelineValue", "ExpectedGrossProfit", "WinProbability", "ForecastCategory",
                "ExpectedCloseDate", "NextStep", "NextStepDueDate", "DaysInStage", "PipelineRisk",
                "WaitingResponseCount", "NoteType", "ResponseStatus", "ResponseAgeDays", "EscalationSeverity",
                "ProjectID", "ProjectStage", "ProjectStatus", "DeliveryHealth", "PercentComplete",
                "OpenTicketCount", "BlockedTicketCount", "OpenTaskCount", "BlockedTaskCount",
                "YTDRevenue", "AnnualTarget", "TargetGap", "OpenPipeline", "WeightedPipelineForecast",
                "ForecastYearEndRevenue", "ForecastGap", "PipelineCoverage", "AchievabilityScore", "Achievability",
                "Action", "Evidence", "ActionType", "ForecastRevenue", "AdjustedWinProbability",
                "Information Needed", "Why It Matters", "Example Request",
            ]
            columns = [column for column in preferred if column in table.columns] or list(table.columns[:10])
            result = _data_table(table, columns, 7)
        visual = _answer_visual_panel(answer.visual)
        meeting_detail = html.Div()
        if "Meetings last week" in answer.title and not table.empty and {"Period Start", "Period End"}.issubset(table.columns):
            period_start = pd.to_datetime(table["Period Start"], errors="coerce").min()
            period_end = pd.to_datetime(table["Period End"], errors="coerce").max()
            if pd.notna(period_start) and pd.notna(period_end):
                detail = meeting_records_between(data, period_start, period_end)
                if not detail.empty:
                    meeting_detail = html.Div(
                        [
                            html.A(
                                "View last week's meeting records",
                                href="#last-week-meeting-records",
                                className="answer-action-link",
                            ),
                            html.Div(
                                [
                                    html.H4("Last week's meeting records", className="subsection-title"),
                                    _data_table(
                                        detail,
                                        [
                                            "MeetingDate", "Salesperson", "CustomerName", "OpportunityID",
                                            "MeetingType", "Subject", "MeetingStatus", "CriticalSeverity",
                                            "FollowUpStatus", "NextAction",
                                        ],
                                        10,
                                        filter_action="native",
                                    ),
                                ],
                                id="last-week-meeting-records",
                                className="answer-detail-section",
                            ),
                        ]
                    )
        interpretation = html.Div()
        if answer.interpretation:
            interpretation = html.Div(
                [
                    html.Span("Interpreted as", className="interpretation-label"),
                    *[
                        html.Span(
                            [html.Strong(f"{label}: "), str(value)],
                            className="interpretation-chip",
                        )
                        for label, value in answer.interpretation.items()
                    ],
                ],
                className="answer-interpretation",
            )
        return [
            html.Div(answer.title, className="answer-title"),
            interpretation,
            html.Div(answer.summary, className="answer-summary"),
            visual,
            result,
            meeting_detail,
            html.Div(
                f"Source: {answer.source}. {boundary_message}",
                className="answer-source",
            ),
        ]

    @app.callback(
        Output("renewal-table", "data"), Output("renewal-count", "children"),
        Input("renewal-owner-filter", "value"), Input("renewal-priority-filter", "value"),
        Input("renewal-service-filter", "value"), Input("renewal-segment-filter", "value"),
    )
    def filter_renewals(owner, priorities, service, segment):
        filtered = renewals.copy()
        if owner and owner != "all":
            filtered = filtered[filtered["AccountOwnerID"] == owner]
        if priorities:
            filtered = filtered[filtered["Priority"].isin(priorities)]
        if service:
            filtered = filtered[filtered["Services"].fillna("").str.contains(re.escape(service), case=False)]
        if segment:
            filtered = filtered[filtered["Segment"] == segment]
        columns = _safe_columns(filtered, [
            "Priority", "CustomerName", "Salesperson", "ContractName", "ContractStatus", "Services",
            "CurrentEndDate", "DaysToRenewal", "ContractARR", "RenewalRisk", "EndDateChangeCount",
            "RollbackCount", "HealthCheckReason", "SuggestedAction",
        ])
        return table_records(filtered, columns), f"{len(filtered):,} contracts match the current filters"
