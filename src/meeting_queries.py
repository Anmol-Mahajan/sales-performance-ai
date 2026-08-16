"""Shared deterministic meeting-period and record queries."""

from __future__ import annotations

import pandas as pd


def _meeting_activity(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    meetings = data.get("Meetings", pd.DataFrame()).copy()
    if not meetings.empty and {"SalespersonID", "MeetingDate"}.issubset(meetings.columns):
        activity = meetings.rename(columns={"MeetingDate": "ActivityDate"})
        if "MeetingStatus" in activity:
            activity = activity[
                activity["MeetingStatus"].astype(str).str.casefold().eq("held")
            ]
        return activity

    activity = data.get("Activities", pd.DataFrame()).copy()
    required = {"SalespersonID", "ActivityDate", "ActivityType"}
    if activity.empty or not required.issubset(activity.columns):
        return pd.DataFrame()
    return activity[
        activity["ActivityType"].astype(str).str.strip().str.casefold().eq("meeting")
    ]


def last_complete_week_bounds(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """Return start, end, and latest activity date for the latest complete week."""

    activity = _meeting_activity(data)
    if activity.empty:
        return None, None, None
    activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"], errors="coerce")
    activity = activity.dropna(subset=["ActivityDate"])
    if activity.empty:
        return None, None, None
    latest_date = activity["ActivityDate"].max().normalize()
    current_week_start = latest_date - pd.Timedelta(days=latest_date.weekday())
    return (
        current_week_start - pd.Timedelta(days=7),
        current_week_start - pd.Timedelta(days=1),
        latest_date,
    )


def last_complete_month_bounds(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """Return the previous calendar month relative to the latest meeting date."""

    activity = _meeting_activity(data)
    if activity.empty:
        return None, None, None
    activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"], errors="coerce")
    activity = activity.dropna(subset=["ActivityDate"])
    if activity.empty:
        return None, None, None
    latest_date = activity["ActivityDate"].max().normalize()
    current_month_start = latest_date.replace(day=1)
    period_end = current_month_start - pd.Timedelta(days=1)
    period_start = period_end.replace(day=1)
    return period_start, period_end, latest_date


def last_complete_week_meeting_leaderboard(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """Rank held meetings during the latest complete workbook week."""

    period_start, period_end, latest_date = last_complete_week_bounds(data)
    if period_start is None:
        return pd.DataFrame(), None, None, None
    counts = meeting_leaderboard_between(data, period_start, period_end)
    return counts, period_start, period_end, latest_date


def meeting_leaderboard_between(
    data: dict[str, pd.DataFrame],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """Rank held meetings between two inclusive workbook dates."""

    activity = _meeting_activity(data)
    if activity.empty:
        return pd.DataFrame()
    activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"], errors="coerce")
    activity = activity[activity["ActivityDate"].between(period_start, period_end)]
    counts = activity.groupby("SalespersonID").size().rename("Meetings").reset_index()
    if counts.empty:
        return counts

    people = data.get("Salespeople", pd.DataFrame())
    if {"SalespersonID", "Salesperson"}.issubset(people.columns):
        counts = counts.merge(
            people[["SalespersonID", "Salesperson"]], on="SalespersonID", how="left"
        )
    else:
        counts["Salesperson"] = counts["SalespersonID"]
    counts["Salesperson"] = counts["Salesperson"].fillna(counts["SalespersonID"])
    counts["Rank"] = counts["Meetings"].rank(method="dense", ascending=False).astype(int)
    counts["Period Start"] = period_start
    counts["Period End"] = period_end
    counts = counts.sort_values(["Meetings", "Salesperson"], ascending=[False, True])
    return counts[["Rank", "Salesperson", "Meetings", "Period Start", "Period End"]]


def meeting_records_between(
    data: dict[str, pd.DataFrame],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    held_only: bool = True,
) -> pd.DataFrame:
    """Return detailed meetings between two inclusive workbook dates."""

    meetings = data.get("Meetings", pd.DataFrame()).copy()
    if meetings.empty or "MeetingDate" not in meetings:
        return meetings
    meetings["MeetingDate"] = pd.to_datetime(meetings["MeetingDate"], errors="coerce")
    meetings = meetings[meetings["MeetingDate"].between(period_start, period_end)]
    if held_only and "MeetingStatus" in meetings:
        meetings = meetings[meetings["MeetingStatus"].astype(str).str.casefold().eq("held")]
    people = data.get("Salespeople", pd.DataFrame())
    if {"SalespersonID", "Salesperson"}.issubset(people.columns):
        meetings = meetings.merge(
            people[["SalespersonID", "Salesperson"]].drop_duplicates("SalespersonID"),
            on="SalespersonID",
            how="left",
        )
    customers = data.get("Customers", pd.DataFrame())
    if {"CustomerID", "CustomerName"}.issubset(customers.columns):
        meetings = meetings.merge(
            customers[["CustomerID", "CustomerName"]].drop_duplicates("CustomerID"),
            on="CustomerID",
            how="left",
        )
    return meetings.sort_values(["MeetingDate", "Salesperson"], ascending=[False, True])
