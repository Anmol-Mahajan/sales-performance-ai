"""Execute validated query plans against local pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import create_opportunity_features
from .meeting_queries import (
    last_complete_month_bounds,
    last_complete_week_bounds,
    meeting_records_between,
)
from .query_planning import QueryPlan


@dataclass
class PlannedQueryResult:
    title: str
    summary: str
    table: pd.DataFrame
    source: str
    interpretation: dict[str, str]


def workbook_snapshot_date(data: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Return the latest recorded operational snapshot without using wall-clock time."""

    contracts = data.get("Contracts", pd.DataFrame())
    if "SnapshotDate" in contracts:
        snapshots = pd.to_datetime(contracts["SnapshotDate"], errors="coerce").dropna()
        if not snapshots.empty:
            return snapshots.max().normalize()

    candidates: list[pd.Timestamp] = []
    for sheet, column in [
        ("Contracts", "SnapshotDate"),
        ("Meetings", "MeetingDate"),
        ("OpportunityNotes", "NoteDate"),
        ("Projects", "LastUpdatedDate"),
    ]:
        frame = data.get(sheet, pd.DataFrame())
        if column in frame:
            if sheet == "Meetings" and "MeetingStatus" in frame:
                frame = frame[
                    frame["MeetingStatus"].astype(str).str.casefold().eq("held")
                ]
            values = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not values.empty:
                candidates.append(values.max().normalize())
    return max(candidates) if candidates else pd.Timestamp.today().normalize()


def _money(value: float) -> str:
    return f"£{float(value):,.0f}"


def _execute_opportunity_list(
    plan: QueryPlan, data: dict[str, pd.DataFrame]
) -> PlannedQueryResult | None:
    actionable = plan.time_scope == "upcoming" or "Stage" in plan.filters
    if plan.operation not in {"list", "status"} or not actionable:
        return None

    rows = create_opportunity_features(data)
    snapshot = workbook_snapshot_date(data)
    for column in ["ExpectedCloseDate", "CloseDate", "NextStepDueDate"]:
        if column in rows:
            rows[column] = pd.to_datetime(rows[column], errors="coerce")

    if "SalespersonID" in plan.filters:
        rows = rows[
            rows["SalespersonID"].astype(str).eq(str(plan.filters["SalespersonID"]))
        ]
    if "CustomerID" in plan.filters:
        rows = rows[rows["CustomerID"].astype(str).eq(str(plan.filters["CustomerID"]))]
    if "OpportunityType" in plan.filters:
        rows = rows[
            rows["OpportunityType"].astype(str).str.casefold().eq(
                str(plan.filters["OpportunityType"]).casefold()
            )
        ]
    if "Stage" in plan.filters:
        rows = rows[
            rows["Stage"].astype(str).str.casefold().eq(str(plan.filters["Stage"]).casefold())
        ]
        if str(plan.filters["Stage"]).casefold() == "open" and "CloseDate" in rows:
            rows = rows[rows["CloseDate"].isna()]
    if plan.filters.get("CloseDate") == "Missing":
        rows = rows[rows["CloseDate"].isna()]
    if plan.filters.get("ExpectedCloseDate") == "OnOrAfterSnapshot":
        rows = rows[rows["ExpectedCloseDate"].ge(snapshot)]

    if plan.sort_field and plan.sort_field in rows:
        rows = rows.sort_values(
            [plan.sort_field, "OpportunityID"],
            ascending=[plan.sort_direction == "ascending", True],
        )
    rows = rows.copy()
    columns = [
        column for column in [
            "OpportunityID", "CustomerName", "Salesperson", "Product", "OpportunityType",
            "PipelineStage", "PipelineValue", "ExpectedGrossProfit", "WinProbability",
            "ForecastCategory", "ExpectedCloseDate", "NextStep", "NextStepDueDate",
            "DaysInStage", "PipelineRisk", "WaitingResponseCount", "ProjectID",
            "ProjectStatus", "DeliveryHealth",
        ] if column in rows
    ]

    salesperson = plan.filters.get("SalespersonName", "The team")
    opportunity_type = plan.filters.get("OpportunityType")
    type_text = f" {str(opportunity_type).lower()}" if opportunity_type else ""
    if rows.empty:
        summary = (
            f"No open{type_text} opportunities for {salesperson} have an expected close date on or "
            f"after the workbook snapshot of {snapshot:%d %b %Y}."
        )
    else:
        total = pd.to_numeric(rows["PipelineValue"], errors="coerce").fillna(0).sum()
        table_note = (
            f" The table shows the first {plan.limit:,} by expected close date."
            if len(rows) > plan.limit else ""
        )
        summary = (
            f"{salesperson} has {len(rows):,} upcoming open{type_text} opportunities worth "
            f"{_money(total)}, expected to close from {rows['ExpectedCloseDate'].min():%d %b %Y} "
            f"to {rows['ExpectedCloseDate'].max():%d %b %Y}. Upcoming means open, not already "
            f"closed, and due on or after the workbook snapshot of {snapshot:%d %b %Y}.{table_note}"
        )
    title = f"Upcoming {str(opportunity_type).lower()} opportunities" if opportunity_type else "Upcoming opportunities"
    return PlannedQueryResult(
        title=title,
        summary=summary,
        table=rows[columns].head(plan.limit),
        source="Validated query plan over Opportunities, Customers, Salespeople, meetings, notes, and linked delivery records",
        interpretation=plan.interpretation(snapshot),
    )


def _meeting_period(
    plan: QueryPlan, data: dict[str, pd.DataFrame]
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None, str]:
    if plan.time_scope == "last_complete_week":
        period_start, period_end, latest_date = last_complete_week_bounds(data)
        return period_start, period_end, latest_date, "the latest complete workbook week"
    if plan.time_scope == "last_complete_month":
        period_start, period_end, latest_date = last_complete_month_bounds(data)
        period_label = (
            period_start.strftime("%B %Y")
            if period_start is not None
            else "the previous complete workbook month"
        )
        return period_start, period_end, latest_date, period_label
    if plan.time_scope in {"explicit_day", "explicit_range", "explicit_month"}:
        period_start = pd.to_datetime(plan.filters.get("PeriodStart"), errors="coerce")
        period_end = pd.to_datetime(plan.filters.get("PeriodEnd"), errors="coerce")
        if pd.isna(period_start) or pd.isna(period_end):
            return None, None, None, "selected period"
        return (
            period_start,
            period_end,
            None,
            str(plan.filters.get("PeriodLabel", period_start.strftime("%d %b %Y"))),
        )
    if plan.time_scope == "last_n_days":
        reference_date = workbook_snapshot_date(data)
        days = max(1, int(plan.filters.get("DaysBack", 1)))
        return (
            reference_date - pd.Timedelta(days=days - 1),
            reference_date,
            reference_date,
            f"the last {days:,} days",
        )
    if plan.time_scope == "next_n_days":
        reference_date = workbook_snapshot_date(data)
        days = max(1, int(plan.filters.get("DaysForward", 1)))
        return (
            reference_date + pd.Timedelta(days=1),
            reference_date + pd.Timedelta(days=days),
            reference_date,
            f"the next {days:,} days",
        )
    if plan.time_scope == "upcoming":
        reference_date = workbook_snapshot_date(data)
        start = reference_date + pd.Timedelta(days=1)
        meetings = data.get("Meetings", pd.DataFrame()).copy()
        end = start
        if "MeetingDate" in meetings:
            meetings["MeetingDate"] = pd.to_datetime(
                meetings["MeetingDate"], errors="coerce"
            )
            meetings = meetings[meetings["MeetingDate"].gt(reference_date)]
            if "MeetingStatus" in meetings:
                meetings = meetings[
                    meetings["MeetingStatus"].astype(str).str.casefold().isin(
                        {"scheduled", "planned", "confirmed", "tentative", "booked"}
                    )
                ]
            if not meetings.empty:
                end = meetings["MeetingDate"].max().normalize()
        return start, end, reference_date, f"after {reference_date:%d %b %Y}"
    return None, None, None, "selected period"


def _meeting_ranking(
    rows: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    if rows.empty or "SalespersonID" not in rows:
        return pd.DataFrame(
            columns=["Rank", "Salesperson", "Meetings", "Period Start", "Period End"]
        )
    counts = rows.groupby("SalespersonID").size().rename("Meetings").reset_index()
    names = rows[["SalespersonID", "Salesperson"]].drop_duplicates("SalespersonID")
    counts = counts.merge(names, on="SalespersonID", how="left")
    counts["Salesperson"] = counts["Salesperson"].fillna(counts["SalespersonID"])
    counts["Rank"] = counts["Meetings"].rank(method="dense", ascending=False).astype(int)
    counts["Period Start"] = period_start
    counts["Period End"] = period_end
    counts = counts.sort_values(["Meetings", "Salesperson"], ascending=[False, True])
    return counts[["Rank", "Salesperson", "Meetings", "Period Start", "Period End"]]


def _meeting_titles(
    plan: QueryPlan, period_label: str, salesperson: str | None
) -> tuple[str, str]:
    upcoming = plan.filters.get("MeetingTiming") == "Upcoming" or plan.time_scope == "upcoming"
    if upcoming:
        if plan.time_scope == "upcoming":
            rank_title, list_title = "Upcoming meeting ranking", "Upcoming meetings"
        elif plan.time_scope == "explicit_day":
            rank_title = f"Upcoming meeting ranking for {period_label}"
            list_title = f"Upcoming meetings on {period_label}"
        else:
            rank_title = f"Upcoming meeting ranking for {period_label}"
            list_title = f"Upcoming meetings in {period_label}"
    elif plan.time_scope == "last_complete_week":
        rank_title, list_title = "Meetings last week", "Meetings last week"
    elif plan.time_scope == "explicit_day":
        rank_title = f"Meeting ranking for {period_label}"
        list_title = f"Meetings on {period_label}"
    elif plan.time_scope == "last_n_days":
        rank_title = f"Meeting ranking for {period_label}"
        list_title = f"Meetings in {period_label}"
    else:
        rank_title = f"Meeting ranking for {period_label}"
        list_title = f"Meetings in {period_label}"
    if salesperson:
        list_title += f" for {salesperson}"
    return rank_title, list_title


def _execute_meeting_query(
    plan: QueryPlan, data: dict[str, pd.DataFrame]
) -> PlannedQueryResult | None:
    """Rank or list held or upcoming meetings for a validated workbook period."""

    if plan.time_scope not in {
        "last_complete_week",
        "last_complete_month",
        "last_n_days",
        "next_n_days",
        "explicit_day",
        "explicit_range",
        "explicit_month",
        "upcoming",
    }:
        return None

    period_start, period_end, reference_date, period_label = _meeting_period(plan, data)
    salesperson = plan.filters.get("SalespersonName")
    upcoming = (
        plan.filters.get("MeetingTiming") == "Upcoming"
        or plan.time_scope == "upcoming"
    )
    interpretation = plan.interpretation(
        workbook_snapshot_date(data) if upcoming else None
    )
    rank_title, list_title = _meeting_titles(plan, period_label, salesperson)
    if period_start is None or period_end is None:
        return PlannedQueryResult(
            title=rank_title if plan.operation == "rank" else list_title,
            summary=(
                "A period-based meeting answer needs dated meeting activity in the local "
                "Meetings or Activities sheet."
            ),
            table=pd.DataFrame(),
            source="Local workbook availability check",
            interpretation=interpretation,
        )

    period = f"{period_start:%d %b %Y} to {period_end:%d %b %Y}"
    if plan.time_scope == "upcoming":
        interpretation["Period"] = f"After {reference_date:%d %b %Y}"
    else:
        interpretation["Period"] = period

    rows = meeting_records_between(
        data, period_start, period_end, held_only=not upcoming
    )
    if upcoming:
        snapshot = workbook_snapshot_date(data)
        rows = rows[rows["MeetingDate"].gt(snapshot)] if "MeetingDate" in rows else rows.iloc[0:0]
        if "MeetingStatus" in rows:
            rows = rows[
                rows["MeetingStatus"].astype(str).str.casefold().isin(
                    {"scheduled", "planned", "confirmed", "tentative", "booked"}
                )
            ]
        else:
            rows = rows.iloc[0:0]
    if "SalespersonID" in plan.filters:
        if "SalespersonID" in rows:
            rows = rows[
                rows["SalespersonID"].astype(str).eq(str(plan.filters["SalespersonID"]))
            ]
        else:
            rows = rows.iloc[0:0]
    if "CustomerID" in plan.filters:
        if "CustomerID" in rows:
            rows = rows[
                rows["CustomerID"].astype(str).eq(str(plan.filters["CustomerID"]))
            ]
        else:
            rows = rows.iloc[0:0]

    if plan.sort_field and plan.sort_field in rows:
        rows = rows.sort_values(
            plan.sort_field,
            ascending=plan.sort_direction == "ascending",
        )
    rows = rows.copy()

    if plan.operation == "rank":
        ranked = _meeting_ranking(rows, period_start, period_end)
        if ranked.empty:
            if upcoming:
                if plan.time_scope == "upcoming":
                    summary = (
                        "No upcoming meetings are recorded after the workbook snapshot "
                        f"of {workbook_snapshot_date(data):%d %b %Y}. "
                    )
                else:
                    summary = (
                        f"No upcoming meetings are recorded during {period_label} after "
                        f"the workbook snapshot of {workbook_snapshot_date(data):%d %b %Y}. "
                    )
                summary += (
                    "Future meetings need a date after the snapshot and a status such as "
                    "Scheduled, Planned, Confirmed, Tentative, or Booked."
                )
            else:
                summary = f"No held meetings were recorded during {period_label} ({period})."
        else:
            highest = int(ranked["Meetings"].max())
            leaders = ranked.loc[ranked["Meetings"].eq(highest), "Salesperson"].tolist()
            leader_text = leaders[0] if len(leaders) == 1 else f"{', '.join(leaders[:-1])} and {leaders[-1]}"
            meeting_text = f"{highest} meetings" if len(leaders) == 1 else f"{highest} meetings each"
            if upcoming:
                verb = "has" if len(leaders) == 1 else "have"
                summary = (
                    f"{leader_text} {verb} the most upcoming meetings during "
                    f"{period_label}: {meeting_text} from {period}."
                )
            else:
                if plan.time_scope == "explicit_day":
                    summary = (
                        f"{leader_text} held the most meetings on {period_label}: "
                        f"{meeting_text}."
                    )
                elif period_label == period:
                    summary = (
                        f"{leader_text} held the most meetings during {period}: "
                        f"{meeting_text}."
                    )
                else:
                    summary = (
                        f"{leader_text} held the most meetings during {period_label}: "
                        f"{meeting_text} from {period}."
                    )
            if reference_date is not None and plan.time_scope in {
                "last_complete_week",
                "last_complete_month",
            }:
                summary += f" The workbook's latest meeting date is {reference_date:%d %b %Y}."
            elif plan.time_scope == "last_n_days" and reference_date is not None:
                summary += f" The range ends at the workbook snapshot of {reference_date:%d %b %Y}."
        return PlannedQueryResult(
            title=rank_title,
            summary=summary,
            table=ranked.head(plan.limit),
            source="Validated query plan over Meetings and Salespeople",
            interpretation=interpretation,
        )

    if plan.operation not in {"list", "status"}:
        return None
    columns = [
        column
        for column in [
            "MeetingID",
            "MeetingDate",
            "Salesperson",
            "CustomerName",
            "OpportunityID",
            "CustomerRelationship",
            "MeetingType",
            "Subject",
            "DurationMinutes",
            "MeetingStatus",
            "MeetingSummary",
            "SalespersonNotes",
            "CriticalSeverity",
            "CriticalFinding",
            "NextAction",
            "ActionDueDate",
            "FollowUpStatus",
        ]
        if column in rows
    ]

    subject = salesperson if salesperson else "The team"
    if rows.empty:
        if upcoming:
            subject_text = f" for {subject}" if salesperson else ""
            if plan.time_scope == "upcoming":
                summary = (
                    f"No upcoming meetings are recorded{subject_text} after the workbook "
                    f"snapshot of {workbook_snapshot_date(data):%d %b %Y}. "
                )
            else:
                summary = (
                    f"No upcoming meetings are recorded{subject_text} during {period_label} "
                    f"after the workbook snapshot of {workbook_snapshot_date(data):%d %b %Y}. "
                )
            summary += (
                "Future meetings need a date after the snapshot and a status such as "
                "Scheduled, Planned, Confirmed, Tentative, or Booked."
            )
        else:
            summary = (
                f"{subject} had no held meetings recorded during {period_label}, {period}."
            )
    else:
        meeting_word = "meeting" if len(rows) == 1 else "meetings"
        if upcoming:
            verb = "has" if salesperson else "have"
            summary = (
                f"{subject} {verb} {len(rows):,} upcoming {meeting_word} during "
                f"{period_label}, {period}."
            )
        else:
            if plan.time_scope == "explicit_day":
                summary = (
                    f"{subject} held {len(rows):,} {meeting_word} on {period_label}."
                )
            elif period_label == period:
                summary = (
                    f"{subject} held {len(rows):,} {meeting_word} during {period}."
                )
            else:
                summary = (
                    f"{subject} held {len(rows):,} {meeting_word} during {period_label}, {period}."
                )
    if reference_date is not None and plan.time_scope in {
        "last_complete_week",
        "last_complete_month",
    }:
        summary += f" The workbook's latest meeting date is {reference_date:%d %b %Y}."
    elif plan.time_scope == "last_n_days" and reference_date is not None:
        summary += f" The range ends at the workbook snapshot of {reference_date:%d %b %Y}."
    return PlannedQueryResult(
        title=list_title,
        summary=summary,
        table=rows[columns].head(plan.limit),
        source=(
            "Validated query plan over Meetings, Salespeople, Customers, and "
            "Opportunities"
        ),
        interpretation=interpretation,
    )


def execute_query_plan(
    plan: QueryPlan, data: dict[str, pd.DataFrame]
) -> PlannedQueryResult | None:
    """Execute supported plans; return None for legacy handlers not migrated yet."""

    if plan.needs_clarification:
        return None
    if plan.domain == "opportunities":
        return _execute_opportunity_list(plan, data)
    if plan.domain == "meetings":
        return _execute_meeting_query(plan, data)
    return None
