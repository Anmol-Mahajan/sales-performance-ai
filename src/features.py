"""Feature engineering from local Excel data only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _safe_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def _yes_no_to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "yes", "y", "1"])


def create_salesperson_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create salesperson-level features from the local workbook."""

    salespeople = data.get("Salespeople", _empty_frame(["SalespersonID"]))
    monthly = data.get("MonthlyPerformance", pd.DataFrame()).copy()
    opportunities = data.get("Opportunities", pd.DataFrame()).copy()
    contracts = data.get("Contracts", pd.DataFrame())
    renewals = data.get("UpcomingRenewals", pd.DataFrame())
    audit = data.get("ContractAuditLog", pd.DataFrame())
    referrals = data.get("SynergyReferrals", pd.DataFrame())
    meetings = data.get("Meetings", pd.DataFrame()).copy()
    notes = data.get("OpportunityNotes", pd.DataFrame()).copy()
    projects = data.get("Projects", pd.DataFrame()).copy()
    tickets = data.get("OpportunityTickets", pd.DataFrame()).copy()
    tasks = data.get("TicketTasks", pd.DataFrame()).copy()

    base_cols = [
        "SalespersonID",
        "Salesperson",
        "Segment",
        "PrimarySpecialism",
        "Region",
        "Seniority",
    ]
    features = salespeople[[c for c in base_cols if c in salespeople.columns]].copy()
    if "SalespersonID" not in features.columns:
        return pd.DataFrame()

    reference_year = None
    if not monthly.empty and "Month" in monthly:
        monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
        reference_year = int(monthly["Month"].dt.year.max())
        monthly = monthly[monthly["Month"].dt.year.eq(reference_year)]
    if reference_year is not None and not opportunities.empty and "CreatedDate" in opportunities:
        created_dates = pd.to_datetime(opportunities["CreatedDate"], errors="coerce")
        opportunities = opportunities[created_dates.dt.year.eq(reference_year)]
    if reference_year is not None and not meetings.empty and "MeetingDate" in meetings:
        meeting_dates = pd.to_datetime(meetings["MeetingDate"], errors="coerce")
        meetings = meetings[meeting_dates.dt.year.eq(reference_year)]

    if not monthly.empty:
        metric_cols = [
            "Revenue",
            "GrossProfit",
            "Meetings",
            "CustomerReachouts",
            "OpportunitiesCreated",
            "OpportunitiesWon",
            "NewCustomers",
            "CrossSellOpportunities",
            "CrossSellRevenue",
        ]
        available = [c for c in metric_cols if c in monthly.columns]
        monthly_grouped = (
            monthly.groupby("SalespersonID", as_index=False)[available]
            .sum(numeric_only=True)
            .rename(
                columns={
                    "Revenue": "total_revenue",
                    "GrossProfit": "total_gross_profit",
                    "Meetings": "total_meetings",
                    "CustomerReachouts": "total_reachouts",
                    "OpportunitiesCreated": "opportunities_created",
                    "OpportunitiesWon": "opportunities_won",
                    "NewCustomers": "new_customers",
                    "CrossSellOpportunities": "cross_sell_opportunities",
                    "CrossSellRevenue": "cross_sell_revenue",
                }
            )
        )
        features = features.merge(monthly_grouped, on="SalespersonID", how="left")
        mean_columns = [column for column in ["RetentionRate", "TargetAttainment"] if column in monthly.columns]
        if mean_columns:
            monthly_means = monthly.groupby("SalespersonID", as_index=False)[mean_columns].mean().rename(
                columns={
                    "RetentionRate": "average_retention_rate",
                    "TargetAttainment": "average_monthly_target_attainment",
                }
            )
            features = features.merge(monthly_means, on="SalespersonID", how="left")

    if not opportunities.empty and "SalespersonID" in opportunities.columns:
        opp_grouped = opportunities.groupby("SalespersonID").agg(
            local_opportunity_count=("OpportunityID", "nunique"),
            local_pipeline_value=("PipelineValue", _safe_sum),
        )
        if "Stage" in opportunities.columns:
            won = opportunities.assign(
                _won=opportunities["Stage"].astype(str).str.lower().eq("won")
            )
            opp_grouped["local_opportunities_won"] = won.groupby("SalespersonID")["_won"].sum()
        features = features.merge(opp_grouped.reset_index(), on="SalespersonID", how="left")

    if not contracts.empty and "AccountOwnerID" in contracts.columns:
        contract_grouped = contracts.groupby("AccountOwnerID").agg(
            contracts_owned=("ContractID", "nunique"),
            contract_arr_owned=("ContractARR", _safe_sum),
            average_days_to_renewal=("DaysToRenewal", "mean"),
            contract_end_date_change_count=("EndDateChangeCount", _safe_sum),
            rollback_count=("RollbackCount", _safe_sum),
        )
        features = features.merge(
            contract_grouped.reset_index().rename(columns={"AccountOwnerID": "SalespersonID"}),
            on="SalespersonID",
            how="left",
        )

    if not renewals.empty and "AccountOwnerID" in renewals.columns:
        renewal_grouped = renewals.groupby("AccountOwnerID").agg(
            upcoming_renewal_count=("ContractID", "nunique")
        )
        features = features.merge(
            renewal_grouped.reset_index().rename(columns={"AccountOwnerID": "SalespersonID"}),
            on="SalespersonID",
            how="left",
        )

    if not audit.empty and "SalespersonLinked" in audit.columns:
        audit_grouped = audit.groupby("SalespersonLinked").agg(
            audit_log_events=("AuditID", "count")
        )
        features = features.merge(
            audit_grouped.reset_index().rename(columns={"SalespersonLinked": "SalespersonID"}),
            on="SalespersonID",
            how="left",
        )

    if not referrals.empty:
        sent = referrals.groupby("FromSalespersonID").size().rename("synergy_referrals_sent")
        received = referrals.groupby("ToSalespersonID").size().rename("synergy_referrals_received")
        features = features.merge(
            sent.reset_index().rename(columns={"FromSalespersonID": "SalespersonID"}),
            on="SalespersonID",
            how="left",
        )
        features = features.merge(
            received.reset_index().rename(columns={"ToSalespersonID": "SalespersonID"}),
            on="SalespersonID",
            how="left",
        )

    snapshot_dates = pd.to_datetime(contracts.get("SnapshotDate"), errors="coerce")
    snapshot_date = snapshot_dates.max() if snapshot_dates.notna().any() else pd.Timestamp.today().normalize()
    if not meetings.empty and "SalespersonID" in meetings:
        if "MeetingStatus" in meetings:
            meetings = meetings[meetings["MeetingStatus"].astype(str).str.lower().eq("held")]
        meetings["_critical"] = _yes_no_to_bool(meetings.get("CriticalFindingFlag", pd.Series(False, index=meetings.index)))
        due = pd.to_datetime(meetings.get("ActionDueDate"), errors="coerce")
        complete = meetings.get("FollowUpStatus", pd.Series("", index=meetings.index)).astype(str).str.lower().eq("complete")
        meetings["_overdue"] = due.lt(snapshot_date) & ~complete
        meeting_type = meetings.get("MeetingType", pd.Series("", index=meetings.index)).astype(str).str.lower()
        meetings["_new_business"] = meeting_type.str.contains("new business")
        meetings["_health_check"] = meeting_type.str.contains("health check")
        meetings["_support_escalation"] = meeting_type.str.contains("support escalation")
        meeting_grouped = meetings.groupby("SalespersonID").agg(
            operational_meeting_count=("MeetingID", "nunique"),
            new_business_meeting_count=("_new_business", "sum"),
            account_health_check_meeting_count=("_health_check", "sum"),
            support_escalation_meeting_count=("_support_escalation", "sum"),
            critical_meeting_finding_count=("_critical", "sum"),
            overdue_meeting_action_count=("_overdue", "sum"),
        )
        features = features.merge(meeting_grouped.reset_index(), on="SalespersonID", how="left")

    if not notes.empty and "SalespersonID" in notes:
        notes["_waiting"] = notes.get("ResponseStatus", pd.Series("", index=notes.index)).astype(str).str.lower().eq("waiting response")
        notes["_critical"] = _yes_no_to_bool(notes.get("CriticalFindingFlag", pd.Series(False, index=notes.index)))
        notes["_waiting_age"] = pd.to_numeric(notes.get("ResponseAgeDays"), errors="coerce").where(notes["_waiting"], 0)
        note_grouped = notes.groupby("SalespersonID").agg(
            unanswered_opportunity_note_count=("_waiting", "sum"),
            average_waiting_response_age_days=("_waiting_age", lambda values: values[values > 0].mean()),
            critical_opportunity_escalation_count=("_critical", "sum"),
        )
        features = features.merge(note_grouped.reset_index(), on="SalespersonID", how="left")

    if not projects.empty and "SalespersonID" in projects:
        projects["_active"] = ~projects.get("ProjectStatus", pd.Series("", index=projects.index)).astype(str).str.lower().eq("complete")
        projects["_blocked"] = (
            projects.get("ProjectStatus", pd.Series("", index=projects.index)).astype(str).str.lower().eq("on hold")
            | projects.get("DeliveryHealth", pd.Series("", index=projects.index)).astype(str).str.lower().eq("red")
        )
        project_grouped = projects.groupby("SalespersonID").agg(
            active_opportunity_project_count=("_active", "sum"),
            blocked_opportunity_project_count=("_blocked", "sum"),
        )
        features = features.merge(project_grouped.reset_index(), on="SalespersonID", how="left")

    if not tickets.empty and "SalespersonID" in tickets:
        tickets["_open"] = ~tickets.get("TicketStatus", pd.Series("", index=tickets.index)).astype(str).str.lower().eq("resolved")
        tickets["_escalated"] = _yes_no_to_bool(tickets.get("EscalationFlag", pd.Series(False, index=tickets.index)))
        ticket_grouped = tickets.groupby("SalespersonID").agg(
            open_opportunity_ticket_count=("_open", "sum"),
            escalated_opportunity_ticket_count=("_escalated", "sum"),
        )
        features = features.merge(ticket_grouped.reset_index(), on="SalespersonID", how="left")

    if not tasks.empty and "OpportunityID" in tasks and not projects.empty:
        task_owners = projects[["OpportunityID", "SalespersonID"]].drop_duplicates("OpportunityID")
        tasks = tasks.merge(task_owners, on="OpportunityID", how="left")
        task_due = pd.to_datetime(tasks.get("DueDate"), errors="coerce")
        task_complete = tasks.get("TaskStatus", pd.Series("", index=tasks.index)).astype(str).str.lower().eq("complete")
        tasks["_overdue"] = task_due.lt(snapshot_date) & ~task_complete
        task_grouped = tasks.groupby("SalespersonID").agg(
            overdue_delivery_task_count=("_overdue", "sum")
        )
        features = features.merge(task_grouped.reset_index(), on="SalespersonID", how="left")

    numeric_columns = [
        "total_revenue",
        "total_gross_profit",
        "total_meetings",
        "total_reachouts",
        "opportunities_created",
        "opportunities_won",
        "new_customers",
        "cross_sell_opportunities",
        "cross_sell_revenue",
        "average_retention_rate",
        "average_monthly_target_attainment",
        "local_opportunity_count",
        "local_pipeline_value",
        "local_opportunities_won",
        "contracts_owned",
        "contract_arr_owned",
        "average_days_to_renewal",
        "contract_end_date_change_count",
        "rollback_count",
        "upcoming_renewal_count",
        "audit_log_events",
        "synergy_referrals_sent",
        "synergy_referrals_received",
        "operational_meeting_count",
        "new_business_meeting_count",
        "account_health_check_meeting_count",
        "support_escalation_meeting_count",
        "critical_meeting_finding_count",
        "overdue_meeting_action_count",
        "unanswered_opportunity_note_count",
        "average_waiting_response_age_days",
        "critical_opportunity_escalation_count",
        "active_opportunity_project_count",
        "blocked_opportunity_project_count",
        "open_opportunity_ticket_count",
        "escalated_opportunity_ticket_count",
        "overdue_delivery_task_count",
    ]
    for column in numeric_columns:
        if column not in features.columns:
            features[column] = 0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)

    created = features["opportunities_created"].replace(0, np.nan)
    features["win_rate"] = (features["opportunities_won"] / created).fillna(0)
    revenue = features["total_revenue"].replace(0, np.nan)
    meetings = features["total_meetings"].replace(0, np.nan)
    reachouts = features["total_reachouts"].replace(0, np.nan)
    wins = features["opportunities_won"].replace(0, np.nan)
    features["gross_margin_pct"] = (features["total_gross_profit"] / revenue).fillna(0)
    features["reachouts_per_meeting"] = (features["total_reachouts"] / meetings).fillna(0)
    features["meetings_per_opportunity"] = (features["total_meetings"] / created).fillna(0)
    features["meetings_per_win"] = (features["total_meetings"] / wins).fillna(0)
    features["revenue_per_meeting"] = (features["total_revenue"] / meetings).fillna(0)
    features["gross_profit_per_reachout"] = (features["total_gross_profit"] / reachouts).fillna(0)
    features["cross_sell_revenue_pct"] = (features["cross_sell_revenue"] / revenue).fillna(0)
    features["cross_sell_opportunity_rate"] = (features["cross_sell_opportunities"] / created).fillna(0)
    return features


def create_opportunity_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create an opportunity-level operational view from linked local sheets."""

    opportunities = data.get("Opportunities", pd.DataFrame()).copy()
    if opportunities.empty or "OpportunityID" not in opportunities:
        return pd.DataFrame()
    customers = data.get("Customers", pd.DataFrame())
    salespeople = data.get("Salespeople", pd.DataFrame())
    meetings = data.get("Meetings", pd.DataFrame()).copy()
    notes = data.get("OpportunityNotes", pd.DataFrame()).copy()
    projects = data.get("Projects", pd.DataFrame()).copy()
    tickets = data.get("OpportunityTickets", pd.DataFrame()).copy()
    tasks = data.get("TicketTasks", pd.DataFrame()).copy()

    result = opportunities.copy()
    if {"CustomerID", "CustomerName"}.issubset(customers.columns):
        result = result.merge(customers[["CustomerID", "CustomerName"]], on="CustomerID", how="left")
    if {"SalespersonID", "Salesperson"}.issubset(salespeople.columns):
        result = result.merge(salespeople[["SalespersonID", "Salesperson"]], on="SalespersonID", how="left")

    if not meetings.empty and "OpportunityID" in meetings:
        linked = meetings.dropna(subset=["OpportunityID"]).copy()
        linked["_critical"] = _yes_no_to_bool(linked.get("CriticalFindingFlag", pd.Series(False, index=linked.index)))
        meeting_summary = linked.groupby("OpportunityID").agg(
            MeetingCount=("MeetingID", "nunique"),
            LatestMeetingDate=("MeetingDate", "max"),
            CriticalMeetingFindings=("_critical", "sum"),
        )
        result = result.merge(meeting_summary.reset_index(), on="OpportunityID", how="left")

    if not notes.empty and "OpportunityID" in notes:
        notes["_waiting"] = notes.get("ResponseStatus", pd.Series("", index=notes.index)).astype(str).str.lower().eq("waiting response")
        notes["_critical"] = _yes_no_to_bool(notes.get("CriticalFindingFlag", pd.Series(False, index=notes.index)))
        note_summary = notes.groupby("OpportunityID").agg(
            NoteCount=("NoteID", "nunique"),
            WaitingResponseCount=("_waiting", "sum"),
            OldestWaitingDays=("ResponseAgeDays", "max"),
            CriticalNoteCount=("_critical", "sum"),
            LatestNoteDate=("NoteDate", "max"),
        )
        result = result.merge(note_summary.reset_index(), on="OpportunityID", how="left")

    if not projects.empty and "OpportunityID" in projects:
        project_columns = [
            "OpportunityID", "ProjectID", "ProjectName", "ProjectStage", "ProjectStatus",
            "DeliveryHealth", "PercentComplete", "TargetCompletionDate", "Blocker",
        ]
        result = result.merge(projects[[column for column in project_columns if column in projects]], on="OpportunityID", how="left")

    if not tickets.empty and "OpportunityID" in tickets:
        tickets["_open"] = ~tickets.get("TicketStatus", pd.Series("", index=tickets.index)).astype(str).str.lower().eq("resolved")
        tickets["_blocked"] = tickets.get("TicketStatus", pd.Series("", index=tickets.index)).astype(str).str.lower().eq("blocked")
        ticket_summary = tickets.groupby("OpportunityID").agg(
            TicketCount=("TicketID", "nunique"),
            OpenTicketCount=("_open", "sum"),
            BlockedTicketCount=("_blocked", "sum"),
        )
        result = result.merge(ticket_summary.reset_index(), on="OpportunityID", how="left")

    if not tasks.empty and "OpportunityID" in tasks:
        tasks["_open"] = ~tasks.get("TaskStatus", pd.Series("", index=tasks.index)).astype(str).str.lower().eq("complete")
        tasks["_blocked"] = tasks.get("TaskStatus", pd.Series("", index=tasks.index)).astype(str).str.lower().eq("blocked")
        task_summary = tasks.groupby("OpportunityID").agg(
            OpenTaskCount=("_open", "sum"),
            BlockedTaskCount=("_blocked", "sum"),
        )
        result = result.merge(task_summary.reset_index(), on="OpportunityID", how="left")

    numeric = [
        "MeetingCount", "CriticalMeetingFindings", "NoteCount", "WaitingResponseCount",
        "OldestWaitingDays", "CriticalNoteCount", "TicketCount", "OpenTicketCount",
        "BlockedTicketCount", "OpenTaskCount", "BlockedTaskCount", "PercentComplete",
    ]
    for column in numeric:
        if column not in result:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    return result


def create_customer_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create customer-level features from the local workbook."""

    customers = data.get("Customers", _empty_frame(["CustomerID"]))
    billing = data.get("ExistingCustomerBilling", pd.DataFrame())
    contracts = data.get("Contracts", pd.DataFrame())
    services = data.get("ContractServices", pd.DataFrame())
    summary = data.get("CustomerContractSummary", pd.DataFrame())

    base_cols = [
        "CustomerID",
        "CustomerName",
        "Segment",
        "Region",
        "CustomerSince",
        "ExistingCustomer",
    ]
    features = customers[[c for c in base_cols if c in customers.columns]].copy()
    if "CustomerID" not in features.columns:
        return pd.DataFrame()

    if not summary.empty:
        summary_cols = [
            "CustomerID",
            "AccountOwnerID",
            "TotalMRR",
            "TotalARR",
            "ActiveContractCount",
            "ServiceCount",
            "NearestContractEndDate",
            "DaysToNearestRenewal",
            "HealthCheckRequired",
            "RollbackCount",
            "EndDateChangeCount",
            "SuggestedManagerAction",
        ]
        selected = summary[[c for c in summary_cols if c in summary.columns]].rename(
            columns={
                "TotalMRR": "total_mrr",
                "TotalARR": "total_arr",
                "ActiveContractCount": "number_of_active_contracts",
                "ServiceCount": "number_of_services_billed",
                "NearestContractEndDate": "nearest_renewal_date",
                "DaysToNearestRenewal": "days_to_nearest_renewal",
                "HealthCheckRequired": "health_check_required",
                "RollbackCount": "rollback_count",
                "EndDateChangeCount": "end_date_change_count",
            }
        )
        features = features.merge(selected, on="CustomerID", how="left")

    if not contracts.empty:
        contract_grouped = contracts.groupby("CustomerID").agg(
            contract_total_mrr=("ContractMRR", _safe_sum),
            contract_total_arr=("ContractARR", _safe_sum),
            contract_count=("ContractID", "nunique"),
            contract_rollback_count=("RollbackCount", _safe_sum),
            contract_end_date_change_count=("EndDateChangeCount", _safe_sum),
        )
        features = features.merge(contract_grouped.reset_index(), on="CustomerID", how="left")

    if not services.empty:
        service_grouped = services.groupby("CustomerID").agg(
            service_count_from_contracts=("ContractServiceID", "nunique"),
            services_billed_from_contracts=("Service", "nunique"),
        )
        features = features.merge(service_grouped.reset_index(), on="CustomerID", how="left")

    if not billing.empty:
        billing_grouped = billing.groupby("CustomerID").agg(
            billing_total_mrr=("MRR", _safe_sum),
            billing_total_arr=("MRR", lambda s: _safe_sum(s) * 12),
            billing_services=("Service", "nunique"),
        )
        features = features.merge(billing_grouped.reset_index(), on="CustomerID", how="left")

    default_map = {
        "total_mrr": ["contract_total_mrr", "billing_total_mrr"],
        "total_arr": ["contract_total_arr", "billing_total_arr"],
        "number_of_active_contracts": ["contract_count"],
        "number_of_services_billed": ["service_count_from_contracts", "billing_services"],
        "rollback_count": ["contract_rollback_count"],
        "end_date_change_count": ["contract_end_date_change_count"],
    }
    for target, fallbacks in default_map.items():
        if target not in features.columns:
            features[target] = np.nan
        for fallback in fallbacks:
            if fallback in features.columns:
                features[target] = features[target].fillna(features[fallback])
        features[target] = pd.to_numeric(features[target], errors="coerce").fillna(0)

    if "nearest_renewal_date" not in features.columns:
        nearest = contracts.copy()
        if not nearest.empty and "CurrentEndDate" in nearest.columns:
            nearest["CurrentEndDate"] = pd.to_datetime(nearest["CurrentEndDate"], errors="coerce")
            nearest = nearest.groupby("CustomerID")["CurrentEndDate"].min().reset_index()
            features = features.merge(
                nearest.rename(columns={"CurrentEndDate": "nearest_renewal_date"}),
                on="CustomerID",
                how="left",
            )

    if "days_to_nearest_renewal" not in features.columns:
        today = pd.Timestamp.today().normalize()
        features["days_to_nearest_renewal"] = (
            pd.to_datetime(features["nearest_renewal_date"], errors="coerce") - today
        ).dt.days
    features["days_to_nearest_renewal"] = pd.to_numeric(
        features["days_to_nearest_renewal"], errors="coerce"
    ).fillna(9999)

    if "health_check_required" not in features.columns:
        features["health_check_required"] = (
            (features["days_to_nearest_renewal"] <= 90)
            | (features["rollback_count"] > 0)
            | (features["end_date_change_count"] > 0)
        )
    else:
        features["health_check_required"] = _yes_no_to_bool(features["health_check_required"])

    return features


def feature_summary(
    salesperson_features: pd.DataFrame,
    customer_features: pd.DataFrame,
    opportunity_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise engineered feature tables for the data scientist portal."""

    rows = []
    feature_sets = [
        ("Salesperson features", salesperson_features),
        ("Customer features", customer_features),
    ]
    if opportunity_features is not None:
        feature_sets.append(("Opportunity features", opportunity_features))
    for name, df in feature_sets:
        numeric = df.select_dtypes(include=["number"])
        rows.append(
            {
                "Feature Set": name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Numeric Columns": len(numeric.columns),
                "Missing Values": int(df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)
