"""Local visual recommendations for deterministic workbook answers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AnswerVisual:
    chart_type: str
    title: str
    data: pd.DataFrame
    x: str
    y: str
    color: str | None = None
    orientation: str = "v"
    barmode: str = "group"
    value_prefix: str = ""
    value_suffix: str = ""
    explanation: str = ""


def _has_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in frame.columns for column in columns)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _date_activity_visual(
    answer_title: str,
    table: pd.DataFrame,
) -> AnswerVisual | None:
    if not _has_columns(table, ["MeetingDate", "Salesperson"]):
        return None
    rows = table.copy()
    rows["MeetingDate"] = pd.to_datetime(rows["MeetingDate"], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["MeetingDate"])
    if rows.empty:
        return None
    grouped = (
        rows.groupby(["MeetingDate", "Salesperson"], as_index=False)
        .size()
        .rename(columns={"size": "Meetings"})
        .sort_values(["MeetingDate", "Salesperson"])
    )
    return AnswerVisual(
        chart_type="bar",
        title=f"{answer_title} by date",
        data=grouped,
        x="MeetingDate",
        y="Meetings",
        color="Salesperson",
        barmode="stack",
        value_suffix=" meetings",
        explanation=(
            "Shows meeting volume over the selected period, split by salesperson. "
            "It helps distinguish one strong day from consistent activity across the range."
        ),
    )


def recommend_answer_visual(
    answer_title: str,
    table: pd.DataFrame,
    question: str = "",
    interpretation: dict[str, str] | None = None,
) -> AnswerVisual | None:
    """Choose a useful chart from the verified result table, without model inference."""

    if table.empty:
        return None
    lowered = question.casefold()

    if _has_columns(table, ["Salesperson", "Meetings"]):
        rows = table.copy()
        rows["Meetings"] = _numeric(rows, "Meetings")
        rows = rows.sort_values("Meetings", ascending=False)
        period = (interpretation or {}).get("Period")
        title = "Meetings by salesperson"
        if period:
            title = f"{title}: {period}"
        return AnswerVisual(
            chart_type="bar",
            title=title,
            data=rows,
            x="Salesperson",
            y="Meetings",
            value_suffix=" meetings",
            explanation=(
                "Shows the full salesperson meeting distribution behind the answer, "
                "so the leader and the rest of the team can be compared at a glance."
            ),
        )

    dated_activity = _date_activity_visual(answer_title, table)
    if dated_activity is not None:
        return dated_activity

    if _has_columns(table, ["Salesperson", "performance_score"]):
        rows = table.copy()
        rows["performance_score"] = _numeric(rows, "performance_score")
        rows = rows.sort_values("performance_score")
        return AnswerVisual(
            chart_type="bar",
            title="Performance score by salesperson",
            data=rows,
            x="performance_score",
            y="Salesperson",
            color="support_status" if "support_status" in rows else None,
            orientation="h",
            value_suffix="/100",
            explanation=(
                "Shows the ranked performance-score spread, including support status when available. "
                "It helps compare the magnitude of gaps rather than only the ordering."
            ),
        )

    if _has_columns(table, ["Customer", "Estimated Annual Potential"]):
        rows = table.copy()
        rows["Estimated Annual Potential"] = _numeric(rows, "Estimated Annual Potential")
        rows = rows.sort_values("Estimated Annual Potential", ascending=True).tail(12)
        return AnswerVisual(
            chart_type="bar",
            title="Estimated annual whitespace potential",
            data=rows,
            x="Estimated Annual Potential",
            y="Customer",
            color="Recommended Product" if "Recommended Product" in rows else None,
            orientation="h",
            value_prefix="£",
            explanation=(
                "Shows which accounts have the largest directional whitespace estimates. "
                "It helps prioritise follow-up while keeping the detailed estimate basis in the table."
            ),
        )

    if _has_columns(table, ["CustomerName", "ContractARR", "DaysToRenewal"]):
        rows = table.copy()
        rows["ContractARR"] = _numeric(rows, "ContractARR")
        rows = rows.sort_values("DaysToRenewal", ascending=True).head(12)
        return AnswerVisual(
            chart_type="bar",
            title="Renewal value by urgency",
            data=rows,
            x="ContractARR",
            y="CustomerName",
            color="RenewalRisk" if "RenewalRisk" in rows else None,
            orientation="h",
            value_prefix="£",
            explanation=(
                "Shows renewal value for the most urgent contracts. "
                "It helps balance days-to-renewal urgency with commercial exposure."
            ),
        )

    if _has_columns(table, ["Salesperson", "YTDRevenue", "WeightedPipelineForecast"]):
        rows = table.copy()
        value_columns = ["YTDRevenue", "WeightedPipelineForecast"]
        melted = rows.melt(
            id_vars=["Salesperson"],
            value_vars=value_columns,
            var_name="Measure",
            value_name="Value",
        )
        return AnswerVisual(
            chart_type="bar",
            title="Revenue and weighted pipeline by salesperson",
            data=melted,
            x="Salesperson",
            y="Value",
            color="Measure",
            barmode="group",
            value_prefix="£",
            explanation=(
                "Compares recognised revenue with probability-adjusted open pipeline. "
                "It helps show whether the forecast depends on already-booked revenue or future conversion."
            ),
        )

    if _has_columns(table, ["Salesperson", "ForecastGap"]):
        rows = table.copy()
        rows["ForecastGap"] = _numeric(rows, "ForecastGap")
        rows = rows.sort_values("ForecastGap", ascending=True)
        return AnswerVisual(
            chart_type="bar",
            title="Forecast gap by salesperson",
            data=rows,
            x="ForecastGap",
            y="Salesperson",
            color="Achievability" if "Achievability" in rows else None,
            orientation="h",
            value_prefix="£",
            explanation=(
                "Shows where probability-adjusted forecast gaps are concentrated. "
                "It helps prioritise pipeline coverage actions."
            ),
        )

    if _has_columns(table, ["ResponseAgeDays", "Salesperson"]):
        rows = table.copy()
        rows["ResponseAgeDays"] = _numeric(rows, "ResponseAgeDays")
        rows = (
            rows.groupby("Salesperson", as_index=False)["ResponseAgeDays"]
            .max()
            .sort_values("ResponseAgeDays", ascending=True)
        )
        return AnswerVisual(
            chart_type="bar",
            title="Oldest waiting response by salesperson",
            data=rows,
            x="ResponseAgeDays",
            y="Salesperson",
            orientation="h",
            value_suffix=" days",
            explanation=(
                "Shows the oldest unanswered opportunity note by salesperson. "
                "It helps identify where customer follow-up age is becoming a risk."
            ),
        )

    if _has_columns(table, ["From Salesperson", "To Salesperson", "Conversion Rate"]):
        rows = table.copy()
        rows["Conversion Rate"] = _numeric(rows, "Conversion Rate")
        rows["Partnership"] = rows["From Salesperson"].astype(str) + " -> " + rows["To Salesperson"].astype(str)
        rows = rows.sort_values("Conversion Rate", ascending=True).tail(12)
        return AnswerVisual(
            chart_type="bar",
            title="Referral conversion by partnership",
            data=rows,
            x="Conversion Rate",
            y="Partnership",
            orientation="h",
            value_suffix="%",
            explanation=(
                "Shows referral conversion rate for each salesperson partnership. "
                "It helps separate high-volume collaboration from high-conversion collaboration."
            ),
        )

    if "chart" in lowered or "visual" in lowered or "graph" in lowered:
        numeric_columns = [
            column for column in table.columns if pd.api.types.is_numeric_dtype(table[column])
        ]
        text_columns = [
            column for column in table.columns if not pd.api.types.is_numeric_dtype(table[column])
        ]
        if numeric_columns and text_columns:
            rows = table.copy()
            value_column = numeric_columns[0]
            label_column = text_columns[0]
            rows[value_column] = _numeric(rows, value_column)
            rows = rows.sort_values(value_column, ascending=True).tail(12)
            return AnswerVisual(
                chart_type="bar",
                title=f"{value_column} by {label_column}",
                data=rows,
                x=value_column,
                y=label_column,
                orientation="h",
                explanation=(
                    "Shows the strongest numeric comparison available in the verified result table."
                ),
            )

    return None
