"""Explainable, local-only manager insights and question answering.

All answers are calculated from DataFrames loaded from the local workbook. This
module does not use an LLM, the internet, an external API, or hosted inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import create_salesperson_features
from .meeting_queries import (
    last_complete_week_meeting_leaderboard as _last_complete_week_meeting_leaderboard,
    meeting_records_between,
)
from .metric_config import load_metric_config
from .pipeline_forecasting import build_pipeline_forecast
from .query_execution import execute_query_plan
from .query_planning import build_query_plan
from .visuals import AnswerVisual, recommend_answer_visual


DRIVER_COLUMNS = {
    "Opportunities": "opportunities_created",
    "Win Rate": "win_rate",
    "Customer Base": "new_customers",
    "Synergy": "synergy_activity",
    "Meetings": "total_meetings",
    "Cross-sell": "cross_sell_opportunities",
}


@dataclass
class DataAnswer:
    title: str
    summary: str
    table: pd.DataFrame
    source: str
    interpretation: dict[str, str] | None = None
    visual: AnswerVisual | None = None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _peer_percentile(profiles: pd.DataFrame, column: str) -> pd.Series:
    """Rank within the narrowest useful local peer cohort."""

    result = pd.Series(index=profiles.index, dtype=float)
    for index, row in profiles.iterrows():
        peers = profiles[profiles["Segment"] == row["Segment"]]
        seniority_peers = peers[peers["Seniority"] == row["Seniority"]]
        if len(seniority_peers) >= 2:
            peers = seniority_peers
        values = _numeric(peers[column])
        rank = values.rank(pct=True, method="average")
        result.loc[index] = float(rank.loc[index]) if index in rank.index else 0.5
    return result.fillna(0.5)


def _money(value: float) -> str:
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    return f"{sign}£{abs(value):,.0f}"


def performance_comparison_metrics() -> pd.DataFrame:
    """Describe exactly how the manager performance comparison is constructed."""

    config = load_metric_config()
    weights = config.get("recommended_composite_score", {}).get("components", {})
    definitions = {
        "commercial_outcome": ("Commercial outcome", "Target Attainment; Gross Margin Pct"),
        "conversion": ("Conversion", "Win Rate"),
        "new_business": ("New business", "New Customers"),
        "existing_customer_growth": (
            "Existing customer growth",
            "Cross-sell Opportunities; Cross-sell Revenue Pct",
        ),
        "retention": ("Retention", "Average Retention Rate"),
        "sales_efficiency": (
            "Sales efficiency",
            "Revenue Per Meeting; Gross Profit Per Reachout",
        ),
    }
    rows = [
        {
            "Comparison Layer": "Composite score",
            "Component": label,
            "Weight": f"{float(weights.get(key, 0)):.0%}",
            "Metrics Used": metrics,
            "Method": "Peer percentile within Segment and, where viable, Seniority",
        }
        for key, (label, metrics) in definitions.items()
    ]
    rows.extend(
        [
            {
                "Comparison Layer": "Diagnostic drivers",
                "Component": "Performance drivers",
                "Weight": "Not scored",
                "Metrics Used": "Opportunities Created; Win Rate; New Customers; Synergy Referrals; Meetings; Cross-sell Opportunities",
                "Method": "Actual value versus peer median",
            },
            {
                "Comparison Layer": "Displayed outcomes",
                "Component": "Commercial context",
                "Weight": "Not scored separately",
                "Metrics Used": "Actual Revenue; Expected Revenue; Performance Gap; Gross Profit; Contracts; Renewal Health Checks",
                "Method": "Displayed alongside the score for manager judgement",
            },
        ]
    )
    return pd.DataFrame(rows)


def create_performance_profiles(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create transparent salesperson scores, targets, gaps, and peer benchmarks."""

    profiles = create_salesperson_features(data).copy()
    targets = data.get("Targets", pd.DataFrame())
    if not targets.empty and "SalespersonID" in targets:
        profiles = profiles.merge(targets, on="SalespersonID", how="left")

    profiles["synergy_activity"] = (
        _numeric(profiles["synergy_referrals_sent"])
        + _numeric(profiles["synergy_referrals_received"])
    )
    target = _numeric(profiles.get("AnnualRevenueTarget", pd.Series(0, index=profiles.index)))
    fallback_target = float(profiles["total_revenue"].median()) if not profiles.empty else 0
    contracts = data.get("Contracts", pd.DataFrame())
    snapshot_dates = pd.to_datetime(contracts.get("SnapshotDate"), errors="coerce")
    snapshot = snapshot_dates.max() if isinstance(snapshot_dates, pd.Series) and snapshot_dates.notna().any() else pd.Timestamp.today().normalize()
    elapsed_fraction = snapshot.dayofyear / (366 if snapshot.is_leap_year else 365)
    profiles["annual_revenue_target"] = target.where(target > 0, fallback_target / max(elapsed_fraction, 0.01))
    profiles["expected_revenue"] = profiles["annual_revenue_target"] * elapsed_fraction
    profiles["performance_gap"] = profiles["total_revenue"] - profiles["expected_revenue"]
    profiles["target_attainment"] = np.where(
        profiles["expected_revenue"] > 0,
        profiles["total_revenue"] / profiles["expected_revenue"],
        0,
    )

    if "Segment" in profiles and profiles["Segment"].notna().any():
        profiles["benchmark_revenue"] = profiles.groupby("Segment")["total_revenue"].transform("median")
    else:
        profiles["benchmark_revenue"] = profiles["total_revenue"].median()

    weights = load_metric_config().get("recommended_composite_score", {}).get("components", {})
    components = {
        "commercial_outcome": (
            _peer_percentile(profiles, "target_attainment")
            + _peer_percentile(profiles, "gross_margin_pct")
        ) / 2,
        "conversion": _peer_percentile(profiles, "win_rate"),
        "new_business": _peer_percentile(profiles, "new_customers"),
        "existing_customer_growth": (
            _peer_percentile(profiles, "cross_sell_opportunities")
            + _peer_percentile(profiles, "cross_sell_revenue_pct")
        ) / 2,
        "retention": _peer_percentile(profiles, "average_retention_rate"),
        "sales_efficiency": (
            _peer_percentile(profiles, "revenue_per_meeting")
            + _peer_percentile(profiles, "gross_profit_per_reachout")
        ) / 2,
    }
    score = pd.Series(0.0, index=profiles.index)
    for component, values in components.items():
        profiles[f"score_{component}"] = values
        score += values * float(weights.get(component, 0))
    profiles["performance_score"] = (score * 100).round().clip(0, 100).astype(int)
    profiles["support_status"] = pd.cut(
        profiles["performance_score"],
        bins=[-1, 54, 69, 84, 100],
        labels=["Needs support", "Watch", "On track", "Leading"],
    ).astype(str)
    return profiles.sort_values("performance_score", ascending=False).reset_index(drop=True)


def performance_drivers(
    profiles: pd.DataFrame, salesperson_id: str
) -> pd.DataFrame:
    """Compare a salesperson's measurable drivers with their peer median."""

    selected = profiles[profiles["SalespersonID"] == salesperson_id]
    if selected.empty:
        return pd.DataFrame(columns=["Driver", "Actual", "Peer Benchmark", "Index", "Direction"])

    row = selected.iloc[0]
    peers = profiles[profiles.get("Segment", "") == row.get("Segment")]
    if len(peers) < 2:
        peers = profiles

    records = []
    for label, column in DRIVER_COLUMNS.items():
        actual = float(row.get(column, 0))
        benchmark = float(_numeric(peers[column]).median())
        index = actual / benchmark * 100 if benchmark else (100 if actual else 0)
        records.append(
            {
                "Driver": label,
                "Actual": actual,
                "Peer Benchmark": benchmark,
                "Index": min(index, 160),
                "Direction": "+" if actual >= benchmark else "-",
            }
        )
    return pd.DataFrame(records).sort_values("Index", ascending=True)


def create_customer_whitespace(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Estimate product whitespace using active products in the local workbook."""

    products = data.get("CustomerProducts", pd.DataFrame()).copy()
    customers = data.get("Customers", pd.DataFrame()).copy()
    contracts = data.get("Contracts", pd.DataFrame()).copy()
    salespeople = data.get("Salespeople", pd.DataFrame()).copy()
    if products.empty or customers.empty:
        return pd.DataFrame()

    if "Active" in products:
        products = products[_numeric(products["Active"]) > 0]
    product_catalogue = sorted(products["Product"].dropna().astype(str).unique())
    products["MonthlyRevenue"] = _numeric(products["MonthlyRevenue"])
    product_value = products.groupby("Product")["MonthlyRevenue"].median().to_dict()
    customer_segments = customers.set_index("CustomerID")["Segment"].to_dict()
    products["Customer Segment"] = products["CustomerID"].map(customer_segments)
    segment_product_value = products.groupby(["Customer Segment", "Product"])["MonthlyRevenue"].median().to_dict()
    customer_product_mrr = products.groupby("CustomerID")["MonthlyRevenue"].sum()
    customer_value_frame = customer_product_mrr.rename("Current Product MRR").reset_index()
    customer_value_frame["Customer Segment"] = customer_value_frame["CustomerID"].map(customer_segments)
    segment_account_mrr = customer_value_frame.groupby("Customer Segment")["Current Product MRR"].median().to_dict()
    owned = products.groupby("CustomerID")["Product"].agg(lambda values: sorted(set(values.astype(str))))

    owner_map = pd.Series(dtype=object)
    if not contracts.empty:
        owner_map = contracts.groupby("CustomerID")["AccountOwnerID"].first()
    name_map = salespeople.set_index("SalespersonID")["Salesperson"].to_dict() if not salespeople.empty else {}
    specialist_rows = salespeople[["Salesperson", "PrimarySpecialism"]].dropna() if not salespeople.empty else pd.DataFrame()

    rows = []
    for customer in customers.itertuples(index=False):
        current = owned.get(customer.CustomerID, [])
        missing = [product for product in product_catalogue if product not in current]
        if not missing:
            continue
        segment = customer.Segment
        current_mrr = float(customer_product_mrr.get(customer.CustomerID, 0))
        segment_median_mrr = float(segment_account_mrr.get(segment, customer_product_mrr.median()))
        account_size_factor = float(np.clip(current_mrr / max(segment_median_mrr, 1), 0.35, 3.0))
        estimates = {
            product: float(segment_product_value.get((segment, product), product_value.get(product, 0)))
            * account_size_factor
            * 12
            for product in missing
        }
        recommended = max(estimates, key=estimates.get)
        specialist = "Review with product specialist"
        if not specialist_rows.empty:
            tokens = set(re.findall(r"[a-z]+", recommended.lower()))
            scores = specialist_rows["PrimarySpecialism"].astype(str).map(
                lambda value: len(tokens & set(re.findall(r"[a-z]+", value.lower())))
            )
            if scores.max() > 0:
                specialist = specialist_rows.loc[scores.idxmax(), "Salesperson"]
        owner_id = owner_map.get(customer.CustomerID, "")
        article = "an" if recommended[:1].casefold() in {"a", "e", "i", "o", "u"} else "a"
        rows.append(
            {
                "Customer": customer.CustomerName,
                "Customer Segment": segment,
                "Account Owner": name_map.get(owner_id, owner_id),
                "Current Products": ", ".join(current) or "None recorded",
                "Missing Products": ", ".join(missing),
                "Whitespace Breadth": len(missing) / max(len(product_catalogue), 1),
                "Current Product MRR": current_mrr,
                "Account Size Factor": account_size_factor,
                "Estimated Annual Potential": round(estimates[recommended], 2),
                "Recommended Product": recommended,
                "Recommended Specialist": specialist,
                "Estimate Basis": "Segment product median adjusted for current account size",
                "Next Action": f"Open {article} {recommended} discovery conversation",
            }
        )
    whitespace = pd.DataFrame(rows)
    if whitespace.empty:
        return whitespace
    potential_rank = whitespace.groupby("Customer Segment")["Estimated Annual Potential"].rank(
        pct=True, method="average"
    )
    whitespace["Whitespace Score"] = (
        whitespace["Whitespace Breadth"] * 60 + potential_rank * 40
    ).round().clip(0, 100).astype(int)
    return whitespace.drop(columns=["Whitespace Breadth"]).sort_values(
        ["Estimated Annual Potential", "Whitespace Score"], ascending=False
    ).reset_index(drop=True)


def create_synergy_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise local referral collaboration between salesperson pairs."""

    referrals = data.get("SynergyReferrals", pd.DataFrame()).copy()
    salespeople = data.get("Salespeople", pd.DataFrame()).copy()
    synergy_map = data.get("SynergyMap", pd.DataFrame()).copy()
    if referrals.empty:
        return pd.DataFrame()

    referrals["Converted"] = referrals["ReferralStatus"].astype(str).str.lower().eq("converted")
    referrals["Accepted"] = referrals["ReferralStatus"].astype(str).str.lower().isin(["accepted", "converted"])
    summary = referrals.groupby(["FromSalespersonID", "ToSalespersonID"], as_index=False).agg(
        Referrals=("ReferralID", "count"),
        Accepted=("Accepted", "sum"),
        Converted=("Converted", "sum"),
    )
    summary["Conversion Rate"] = summary["Converted"] / summary["Referrals"].clip(lower=1)
    if not synergy_map.empty:
        summary = summary.merge(
            synergy_map,
            on=["FromSalespersonID", "ToSalespersonID"],
            how="left",
        )
    names = salespeople.set_index("SalespersonID")["Salesperson"].to_dict() if not salespeople.empty else {}
    summary["From Salesperson"] = summary["FromSalespersonID"].map(names).fillna(summary["FromSalespersonID"])
    summary["To Salesperson"] = summary["ToSalespersonID"].map(names).fillna(summary["ToSalespersonID"])
    return summary.sort_values(["Converted", "Referrals"], ascending=False).reset_index(drop=True)


def recommendations_for_salesperson(
    data: dict[str, pd.DataFrame], profiles: pd.DataFrame, salesperson_id: str
) -> list[dict[str, str]]:
    """Generate ranked, evidence-based actions from measurable local gaps."""

    selected = profiles[profiles["SalespersonID"] == salesperson_id]
    if selected.empty:
        return []
    row = selected.iloc[0]
    drivers = performance_drivers(profiles, salesperson_id).set_index("Driver")
    recommendations: list[dict[str, str]] = []

    if drivers.loc["Cross-sell", "Direction"] == "-":
        whitespace = create_customer_whitespace(data)
        owned = whitespace[whitespace["Account Owner"] == row["Salesperson"]]
        top = owned.iloc[0] if not owned.empty else (whitespace.iloc[0] if not whitespace.empty else None)
        evidence = "Cross-sell activity is below the peer benchmark."
        action = "Increase structured cross-sell discovery with existing customers."
        if top is not None:
            action = f"Start with {top['Customer']} for a {top['Recommended Product']} conversation."
            evidence += f" The workbook shows {len(owned):,} owned customers with product whitespace."
        recommendations.append({"Action": action, "Evidence": evidence, "Type": "Growth"})

    if drivers.loc["Meetings", "Direction"] == "-":
        actual = drivers.loc["Meetings", "Actual"]
        benchmark = drivers.loc["Meetings", "Peer Benchmark"]
        uplift = max(int(round(benchmark - actual)), 1)
        recommendations.append(
            {
                "Action": f"Plan approximately {uplift} additional qualified meetings over the measured period.",
                "Evidence": f"Meetings are {max(0, 100 - drivers.loc['Meetings', 'Index']):.0f}% below the peer benchmark.",
                "Type": "Coaching",
            }
        )

    if drivers.loc["Win Rate", "Direction"] == "-":
        recommendations.append(
            {
                "Action": "Review qualification and next-step discipline on open opportunities.",
                "Evidence": f"Win rate is {row['win_rate']:.1%} versus a {drivers.loc['Win Rate', 'Peer Benchmark']:.1%} peer benchmark.",
                "Type": "Conversion",
            }
        )

    if drivers.loc["Synergy", "Direction"] == "-":
        links = data.get("SynergyMap", pd.DataFrame())
        names = data.get("Salespeople", pd.DataFrame()).set_index("SalespersonID")["Salesperson"].to_dict()
        available = links[links["FromSalespersonID"] == salesperson_id].sort_values("SynergyStrength", ascending=False)
        partner = names.get(available.iloc[0]["ToSalespersonID"], "a relevant specialist") if not available.empty else "a relevant specialist"
        recommendations.append(
            {
                "Action": f"Partner with {partner} on the next relevant specialist opportunity.",
                "Evidence": "Referral activity is below the peer benchmark and a mapped local synergy relationship is available.",
                "Type": "Collaboration",
            }
        )

    contracts = data.get("Contracts", pd.DataFrame())
    owned = contracts[contracts["AccountOwnerID"] == salesperson_id] if not contracts.empty else pd.DataFrame()
    health_count = int(_numeric(owned.get("HealthCheckRequired", pd.Series(dtype=float))).sum())
    if health_count:
        recommendations.append(
            {
                "Action": f"Prioritise health checks for {health_count} owned contracts before renewal conversations.",
                "Evidence": "These contracts are flagged by renewal timing, rolling status, date changes, or rollback history.",
                "Type": "Renewal",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "Action": "Maintain the current activity mix and review customer whitespace for the next growth conversation.",
                "Evidence": "No major driver is below the current peer median.",
                "Type": "Maintain",
            }
        )
    return recommendations[:4]


def _find_salespeople(question: str, profiles: pd.DataFrame) -> pd.DataFrame:
    """Find exact or first-name matches, including possessive name forms."""

    lowered = question.casefold()
    exact_indexes = []
    for index, row in profiles.iterrows():
        salesperson_id = str(row["SalespersonID"]).casefold()
        full_name = str(row["Salesperson"]).casefold()
        if salesperson_id in lowered or full_name in lowered:
            exact_indexes.append(index)
    if exact_indexes:
        return profiles.loc[exact_indexes]

    first_name_indexes = []
    for index, row in profiles.iterrows():
        first_name = str(row["Salesperson"]).split()[0].casefold()
        pattern = rf"\b{re.escape(first_name)}(?:['’]s|s)?\b"
        if re.search(pattern, lowered):
            first_name_indexes.append(index)
    return profiles.loc[first_name_indexes]


def _find_salesperson(question: str, profiles: pd.DataFrame) -> pd.Series | None:
    matches = _find_salespeople(question, profiles)
    return matches.iloc[0] if len(matches) == 1 else None


def _ambiguous_salesperson_answer(matches: pd.DataFrame) -> DataAnswer:
    options = matches[
        [
            column for column in [
                "SalespersonID", "Salesperson", "Segment", "PrimarySpecialism", "Region"
            ] if column in matches
        ]
    ].copy()
    options["Example Request"] = options.apply(
        lambda row: f"What is the status of {row['Salesperson']}'s most recent project?",
        axis=1,
    )
    names = ", ".join(options["Salesperson"].astype(str))
    return DataAnswer(
        "Please clarify the salesperson",
        f"That first name matches more than one salesperson: {names}. Use a full name or salesperson ID so I do not select the wrong person.",
        options,
        "Manager clarification required; matching Salespeople records",
    )


def _requested_ranking_count(question: str, default: int = 5) -> int:
    """Read a small requested result count from phrases such as 'bottom two'."""

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    ranking_terms = r"top|bottom|best|worst|highest|lowest"
    number_terms = r"\d+|" + "|".join(number_words)
    patterns = [
        rf"\b(?:{ranking_terms})\s+({number_terms})\b",
        rf"\b({number_terms})\s+(?:{ranking_terms})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if not match:
            continue
        token = match.group(1)
        count = int(token) if token.isdigit() else number_words[token]
        return max(1, min(count, 25))
    return default


def _find_customer(question: str, data: dict[str, pd.DataFrame]) -> pd.Series | None:
    lowered = question.lower()
    for _, row in data.get("Customers", pd.DataFrame()).iterrows():
        if str(row["CustomerID"]).lower() in lowered or str(row["CustomerName"]).lower() in lowered:
            return row
    return None


def _find_opportunity_id(question: str, data: dict[str, pd.DataFrame]) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", question.lower())
    for value in data.get("Opportunities", pd.DataFrame()).get("OpportunityID", pd.Series(dtype=str)):
        opportunity_id = str(value)
        if re.sub(r"[^a-z0-9]", "", opportunity_id.lower()) in normalized:
            return opportunity_id
    return None


def _find_record_id(
    question: str,
    data: dict[str, pd.DataFrame],
    sheet: str,
    column: str,
) -> str | None:
    """Find a workbook identifier despite spaces or punctuation in the question."""

    normalized = re.sub(r"[^a-z0-9]", "", question.casefold())
    frame = data.get(sheet, pd.DataFrame())
    if column not in frame:
        return None
    for value in frame[column].dropna():
        record_id = str(value)
        compact = re.sub(r"[^a-z0-9]", "", record_id.casefold())
        if compact and compact in normalized:
            return record_id
    return None


def is_delivery_question(question: str, data: dict[str, pd.DataFrame]) -> bool:
    """Recognise delivery intent from wording or a project, ticket, or task ID."""

    lowered = question.casefold()
    if any(term in lowered for term in ["project", "ticket", "task", "delivery", "implementation"]):
        return True
    return bool(re.search(r"\b(?:prj|tkt|task)[-_ ]?\d+\b", lowered))


def _latest_data_date(data: dict[str, pd.DataFrame]) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    for sheet, column in [
        ("Contracts", "SnapshotDate"),
        ("Meetings", "MeetingDate"),
        ("OpportunityNotes", "NoteDate"),
        ("Projects", "LastUpdatedDate"),
    ]:
        frame = data.get(sheet, pd.DataFrame())
        if column in frame:
            values = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not values.empty:
                candidates.append(values.max())
    return max(candidates) if candidates else pd.Timestamp.today().normalize()


def create_project_delivery_view(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create one project-first view with linked opportunity, ticket, and task facts."""

    projects = data.get("Projects", pd.DataFrame()).copy()
    if projects.empty:
        return projects

    opportunities = data.get("Opportunities", pd.DataFrame())
    opportunity_columns = [
        column for column in [
            "OpportunityID", "Product", "Stage", "PipelineValue", "ExpectedCloseDate",
            "PipelineRisk",
        ] if column in opportunities
    ]
    if "OpportunityID" in opportunity_columns:
        opportunity_details = opportunities[opportunity_columns].drop_duplicates("OpportunityID")
        opportunity_details = opportunity_details.rename(columns={"Stage": "OpportunityStage"})
        projects = projects.merge(opportunity_details, on="OpportunityID", how="left")

    customers = data.get("Customers", pd.DataFrame())
    if {"CustomerID", "CustomerName"}.issubset(customers.columns):
        projects = projects.merge(
            customers[["CustomerID", "CustomerName"]].drop_duplicates("CustomerID"),
            on="CustomerID", how="left",
        )
    people = data.get("Salespeople", pd.DataFrame())
    if {"SalespersonID", "Salesperson"}.issubset(people.columns):
        projects = projects.merge(
            people[["SalespersonID", "Salesperson"]].drop_duplicates("SalespersonID"),
            on="SalespersonID", how="left",
        )

    tickets = data.get("OpportunityTickets", pd.DataFrame()).copy()
    if not tickets.empty and "ProjectID" in tickets:
        ticket_status = tickets.get("TicketStatus", pd.Series("", index=tickets.index)).astype(str).str.casefold()
        tickets["_open"] = ~ticket_status.eq("resolved")
        tickets["_blocked"] = ticket_status.eq("blocked")
        tickets["_escalated"] = tickets.get(
            "EscalationFlag", pd.Series(False, index=tickets.index)
        ).fillna(False).astype(bool)
        ticket_summary = tickets.groupby("ProjectID").agg(
            TicketCount=("TicketID", "size"),
            OpenTicketCount=("_open", "sum"),
            BlockedTicketCount=("_blocked", "sum"),
            EscalatedTicketCount=("_escalated", "sum"),
        )
        projects = projects.merge(ticket_summary.reset_index(), on="ProjectID", how="left")

    tasks = data.get("TicketTasks", pd.DataFrame()).copy()
    if not tasks.empty and "ProjectID" in tasks:
        task_status = tasks.get("TaskStatus", pd.Series("", index=tasks.index)).astype(str).str.casefold()
        tasks["_open"] = ~task_status.eq("complete")
        tasks["_blocked"] = task_status.eq("blocked")
        due_dates = pd.to_datetime(tasks.get("DueDate"), errors="coerce")
        tasks["_overdue"] = due_dates.lt(_latest_data_date(data)) & tasks["_open"]
        task_summary = tasks.groupby("ProjectID").agg(
            TaskCount=("TaskID", "size"),
            OpenTaskCount=("_open", "sum"),
            BlockedTaskCount=("_blocked", "sum"),
            OverdueTaskCount=("_overdue", "sum"),
        )
        projects = projects.merge(task_summary.reset_index(), on="ProjectID", how="left")

    count_columns = [
        "TicketCount", "OpenTicketCount", "BlockedTicketCount", "EscalatedTicketCount",
        "TaskCount", "OpenTaskCount", "BlockedTaskCount", "OverdueTaskCount",
    ]
    for column in count_columns:
        if column not in projects:
            projects[column] = 0
        projects[column] = pd.to_numeric(projects[column], errors="coerce").fillna(0).astype(int)

    target_date = pd.to_datetime(projects.get("TargetCompletionDate"), errors="coerce")
    complete = projects.get("ProjectStatus", pd.Series("", index=projects.index)).astype(str).str.casefold().eq("complete")
    projects["ProjectOverdue"] = target_date.lt(_latest_data_date(data)) & ~complete
    return projects


def _clarification_answer(question: str, topic: str = "general") -> DataAnswer:
    """Ask for the minimum information needed instead of guessing intent."""

    guidance = {
        "project": [
            ("Project or opportunity", "Identifies the delivery record", "Show project PRJ0001 or the project linked to OPP0327"),
            ("Required view", "Separates project, ticket, and task status", "Show blocked tasks for PRJ0001"),
            ("Decision needed", "Makes recommendations relevant", "What should we do about delays on PRJ0001?"),
        ],
        "performance": [
            ("Salesperson or team", "Defines whose performance to assess", "Compare Alice Brown with the team"),
            ("Metric", "Defines what good performance means", "Compare revenue, win rate, and meetings"),
            ("Period or benchmark", "Provides a fair comparison", "Use 2026 year to date versus target"),
        ],
        "customer": [
            ("Customer name or ID", "Identifies the account", "Show renewal risk for CUST0066"),
            ("Commercial topic", "Selects the relevant local records", "Show contracts, billing, meetings, or whitespace"),
            ("Period or objective", "Makes the answer actionable", "What should we discuss before the next renewal?"),
        ],
        "forecast": [
            ("Team or salesperson", "Defines forecast ownership", "Show the team forecast or Alice Brown's forecast"),
            ("Target or horizon", "Defines the outcome being tested", "Can the 2026 revenue target be achieved?"),
            ("Scenario", "Defines what can be changed", "Suggest actions to cover the forecast gap"),
        ],
        "general": [
            ("Subject", "Identifies the workbook area to search", "Performance, customer, opportunity, project, or renewal"),
            ("Scope", "Identifies the relevant person or record", "Give a salesperson, customer, contract, opportunity, or project ID"),
            ("Outcome", "Explains what decision the answer should support", "Ask for status, comparison, risk, cause, or suggested action"),
        ],
    }
    rows = guidance.get(topic, guidance["general"])
    table = pd.DataFrame(rows, columns=["Information Needed", "Why It Matters", "Example Request"])
    subject = {
        "project": "project request",
        "performance": "performance request",
        "customer": "customer request",
        "forecast": "forecast request",
        "general": "request",
    }.get(topic, "request")
    return DataAnswer(
        "Please clarify the request",
        f"I cannot answer this {subject} reliably from the wording provided. Add one or more of the details below; I will not assume missing context or invent an answer.",
        table,
        "Manager clarification required; no model inference used",
    )


def _answer_delivery_question(
    question: str,
    data: dict[str, pd.DataFrame],
    person: pd.Series | None,
    customer: pd.Series | None,
    opportunity_id: str | None,
) -> DataAnswer:
    """Answer project, ticket, and task questions from their primary local sheets."""

    lowered = question.casefold()
    project_id = _find_record_id(question, data, "Projects", "ProjectID")
    ticket_id = _find_record_id(question, data, "OpportunityTickets", "TicketID")
    task_id = _find_record_id(question, data, "TicketTasks", "TaskID")
    supplied_identifier = re.search(r"\b(?:prj|tkt|task)[-_ ]?\d+\b", question, flags=re.IGNORECASE)
    if supplied_identifier and not any([project_id, ticket_id, task_id]):
        return DataAnswer(
            "Delivery identifier not found",
            f"{supplied_identifier.group(0)} is not present in the local Projects, Opportunity Tickets, or Ticket Tasks sheets. Check the identifier or provide its linked opportunity ID.",
            pd.DataFrame(),
            "Local project, ticket, and task identifier check",
        )
    has_identifier = any([project_id, ticket_id, task_id, opportunity_id])
    singular_unscoped = any(
        phrase in lowered
        for phrase in ["this project", "that project", "the project", "this ticket", "the ticket", "this task", "the task"]
    )
    aggregate_scope = any(
        phrase in lowered
        for phrase in ["which", "all ", "projects", "tickets", "tasks", "active", "in progress", "blocked", "on hold", "complete", "overdue"]
    )
    if singular_unscoped and not has_identifier and person is None and customer is None and not aggregate_scope:
        return _clarification_answer(question, "project")

    asks_project = "project" in lowered or "delivery" in lowered or "implementation" in lowered
    asks_ticket = "ticket" in lowered
    asks_task = "task" in lowered

    if (asks_task and "projects" not in lowered) or task_id:
        rows = data.get("TicketTasks", pd.DataFrame()).copy()
        if task_id:
            rows = rows[rows["TaskID"].astype(str).eq(task_id)]
        if ticket_id and "TicketID" in rows:
            rows = rows[rows["TicketID"].astype(str).eq(ticket_id)]
        if project_id and "ProjectID" in rows:
            rows = rows[rows["ProjectID"].astype(str).eq(project_id)]
        if opportunity_id and "OpportunityID" in rows:
            rows = rows[rows["OpportunityID"].astype(str).eq(opportunity_id)]
        if person is not None or customer is not None:
            projects = data.get("Projects", pd.DataFrame())
            links = projects[[column for column in ["ProjectID", "SalespersonID", "CustomerID"] if column in projects]].drop_duplicates("ProjectID")
            rows = rows.merge(links, on="ProjectID", how="left")
            if person is not None:
                rows = rows[rows["SalespersonID"].astype(str).eq(str(person["SalespersonID"]))]
            if customer is not None:
                rows = rows[rows["CustomerID"].astype(str).eq(str(customer["CustomerID"]))]
        status = rows.get("TaskStatus", pd.Series("", index=rows.index)).astype(str).str.casefold()
        if "blocked" in lowered:
            rows = rows[status.eq("blocked")]
        elif "in progress" in lowered:
            rows = rows[status.eq("in progress")]
        elif "not started" in lowered:
            rows = rows[status.eq("not started")]
        elif re.search(r"\bcomplete(?:d)?\b", lowered):
            rows = rows[status.eq("complete")]
        if "overdue" in lowered and "DueDate" in rows:
            current_status = rows["TaskStatus"].astype(str).str.casefold()
            rows = rows[pd.to_datetime(rows["DueDate"], errors="coerce").lt(_latest_data_date(data)) & ~current_status.eq("complete")]
        blocked_count = int(rows.get("TaskStatus", pd.Series(dtype=str)).astype(str).str.casefold().eq("blocked").sum())
        task_label = "task" if len(rows) == 1 else "tasks"
        return DataAnswer(
            "Ticket task status",
            f"Found {len(rows):,} matching {task_label}; {blocked_count:,} are blocked.",
            rows.sort_values("DueDate") if not rows.empty and "DueDate" in rows else rows,
            "Ticket Tasks, Projects, and Opportunities sheets",
        )

    if (asks_ticket and "projects" not in lowered) or ticket_id:
        rows = data.get("OpportunityTickets", pd.DataFrame()).copy()
        if ticket_id:
            rows = rows[rows["TicketID"].astype(str).eq(ticket_id)]
        if project_id and "ProjectID" in rows:
            rows = rows[rows["ProjectID"].astype(str).eq(project_id)]
        if opportunity_id and "OpportunityID" in rows:
            rows = rows[rows["OpportunityID"].astype(str).eq(opportunity_id)]
        if person is not None and "SalespersonID" in rows:
            rows = rows[rows["SalespersonID"].astype(str).eq(str(person["SalespersonID"]))]
        if customer is not None and "CustomerID" in rows:
            rows = rows[rows["CustomerID"].astype(str).eq(str(customer["CustomerID"]))]
        status = rows.get("TicketStatus", pd.Series("", index=rows.index)).astype(str).str.casefold()
        if "blocked" in lowered:
            rows = rows[status.eq("blocked")]
        elif "in progress" in lowered:
            rows = rows[status.eq("in progress")]
        elif "waiting" in lowered:
            rows = rows[status.eq("waiting on customer")]
        elif re.search(r"\b(?:resolved|closed)\b", lowered):
            rows = rows[status.eq("resolved")]
        if "overdue" in lowered and "DueDate" in rows:
            current_status = rows["TicketStatus"].astype(str).str.casefold()
            rows = rows[pd.to_datetime(rows["DueDate"], errors="coerce").lt(_latest_data_date(data)) & ~current_status.eq("resolved")]
        blocked_count = int(rows.get("TicketStatus", pd.Series(dtype=str)).astype(str).str.casefold().eq("blocked").sum())
        ticket_label = "ticket" if len(rows) == 1 else "tickets"
        return DataAnswer(
            "Opportunity ticket status",
            f"Found {len(rows):,} matching {ticket_label}; {blocked_count:,} are blocked.",
            rows.sort_values("DueDate") if not rows.empty and "DueDate" in rows else rows,
            "Opportunity Tickets, Projects, and Opportunities sheets",
        )

    rows = create_project_delivery_view(data)
    if project_id and "ProjectID" in rows:
        rows = rows[rows["ProjectID"].astype(str).eq(project_id)]
    if opportunity_id and "OpportunityID" in rows:
        rows = rows[rows["OpportunityID"].astype(str).eq(opportunity_id)]
    if person is not None and "SalespersonID" in rows:
        rows = rows[rows["SalespersonID"].astype(str).eq(str(person["SalespersonID"]))]
    if customer is not None and "CustomerID" in rows:
        rows = rows[rows["CustomerID"].astype(str).eq(str(customer["CustomerID"]))]

    status = rows.get("ProjectStatus", pd.Series("", index=rows.index)).astype(str).str.casefold()
    health = rows.get("DeliveryHealth", pd.Series("", index=rows.index)).astype(str).str.casefold()
    blocker = rows.get("Blocker", pd.Series("", index=rows.index)).fillna("").astype(str).str.strip()
    critical_state = "critical" in lowered or re.search(r"\bred\b", lowered)
    if critical_state:
        critical = health.eq("red") | status.eq("on hold")
        rows = rows[critical]
    elif "blocked" in lowered:
        blocked = status.eq("on hold") | health.eq("red") | blocker.ne("")
        if asks_task:
            blocked = blocked | rows.get("BlockedTaskCount", pd.Series(0, index=rows.index)).gt(0)
        if asks_ticket:
            blocked = blocked | rows.get("BlockedTicketCount", pd.Series(0, index=rows.index)).gt(0)
        rows = rows[blocked]
    elif "on hold" in lowered or "paused" in lowered:
        rows = rows[status.eq("on hold")]
    elif re.search(r"\b(?:active|in progress|current)\b", lowered):
        rows = rows[status.eq("active")]
    elif re.search(r"\bcomplete(?:d)?\b", lowered):
        rows = rows[status.eq("complete")]
    elif "overdue" in lowered:
        rows = rows[rows.get("ProjectOverdue", pd.Series(False, index=rows.index)).fillna(False)]
    elif "at risk" in lowered:
        rows = rows[health.isin(["amber", "red"])]
    else:
        for stage in data.get("Projects", pd.DataFrame()).get("ProjectStage", pd.Series(dtype=str)).dropna().astype(str).unique():
            if stage.casefold() in lowered:
                rows = rows[rows["ProjectStage"].astype(str).str.casefold().eq(stage.casefold())]
                break

    recent_question = any(
        phrase in lowered for phrase in ["most recent", "latest", "newest", "last project"]
    )
    if recent_question and not rows.empty:
        rows = rows.assign(
            _recent_update=pd.to_datetime(rows.get("LastUpdatedDate"), errors="coerce"),
            _recent_start=pd.to_datetime(rows.get("StartDate"), errors="coerce"),
        ).sort_values(
            ["_recent_update", "_recent_start", "ProjectID"],
            ascending=[False, False, False],
            na_position="last",
        ).head(1).drop(columns=["_recent_update", "_recent_start"])

    if rows.empty and opportunity_id:
        opportunities = data.get("Opportunities", pd.DataFrame()).copy()
        opportunity = opportunities[opportunities["OpportunityID"].astype(str).eq(opportunity_id)]
        return DataAnswer(
            "No linked project",
            f"No project is linked to {opportunity_id}. The opportunity exists locally, but project stage, delivery health, tickets, and tasks cannot be reported until a Projects record is linked.",
            opportunity[[column for column in ["OpportunityID", "CustomerID", "SalespersonID", "Product", "Stage", "PipelineValue", "ExpectedCloseDate"] if column in opportunity]].head(1),
            "Opportunities and Projects sheets",
        )
    if rows.empty:
        identifier = project_id or ticket_id or task_id
        detail = f" for {identifier}" if identifier else ""
        return DataAnswer(
            "No matching delivery records",
            f"No local project records match the requested scope{detail}. Check the identifier or clarify the required project status, owner, customer, or opportunity.",
            rows,
            "Projects, Opportunity Tickets, and Ticket Tasks sheets",
        )

    active = int(rows["ProjectStatus"].astype(str).str.casefold().eq("active").sum())
    on_hold = int(rows["ProjectStatus"].astype(str).str.casefold().eq("on hold").sum())
    red = int(rows["DeliveryHealth"].astype(str).str.casefold().eq("red").sum())
    open_tickets = int(pd.to_numeric(rows.get("OpenTicketCount"), errors="coerce").fillna(0).sum())
    blocked_tasks = int(pd.to_numeric(rows.get("BlockedTaskCount"), errors="coerce").fillna(0).sum())
    title = "Project delivery status"
    if critical_state:
        title = "Critical projects"
    elif "blocked" in lowered:
        title = "Blocked projects"
    elif "on hold" in lowered:
        title = "Projects on hold"
    elif "in progress" in lowered or "active" in lowered:
        title = "Active projects"
    project_label = "project" if len(rows) == 1 else "projects"
    ticket_label = "ticket" if open_tickets == 1 else "tickets"
    task_label = "task" if blocked_tasks == 1 else "tasks"
    if recent_question:
        latest = rows.iloc[0]
        owner = str(latest.get("Salesperson", person["Salesperson"] if person is not None else "The salesperson"))
        updated = pd.to_datetime(latest.get("LastUpdatedDate"), errors="coerce")
        updated_text = f"{updated:%d %b %Y}" if pd.notna(updated) else "not recorded"
        summary = (
            f"{owner}'s most recently updated project is {latest.get('ProjectID', '')}, "
            f"{latest.get('ProjectName', '')}. Its status is {latest.get('ProjectStatus', 'not recorded')}, "
            f"the stage is {latest.get('ProjectStage', 'not recorded')}, and delivery health is "
            f"{latest.get('DeliveryHealth', 'not recorded')}. It was last updated on {updated_text} "
            f"and has {open_tickets:,} open {ticket_label} and {blocked_tasks:,} blocked {task_label}."
        )
        title = f"{owner}'s most recent project"
    else:
        if critical_state:
            summary = (
                f"Found {len(rows):,} critical {project_label} with red delivery health: "
                f"{active:,} active and {on_hold:,} on hold. They have {open_tickets:,} "
                f"open {ticket_label} and {blocked_tasks:,} blocked {task_label}."
            )
        else:
            summary = (
                f"Found {len(rows):,} matching {project_label}: {active:,} active and {red:,} red. "
                f"They have {open_tickets:,} open {ticket_label} and {blocked_tasks:,} blocked {task_label}."
            )
    exploratory = any(
        phrase in lowered
        for phrase in ["why", "recommend", "suggest", "what should", "how can", "help", "next action", "delay"]
    )
    if exploratory:
        overdue_projects = int(
            rows.get("ProjectOverdue", pd.Series(False, index=rows.index)).fillna(False).sum()
        )
        overdue_tasks = int(
            pd.to_numeric(rows.get("OverdueTaskCount"), errors="coerce").fillna(0).sum()
        )
        blockers = int(
            rows.get("Blocker", pd.Series("", index=rows.index))
            .fillna("").astype(str).str.strip().ne("").sum()
        )
        actions: list[str] = []
        if overdue_projects:
            overdue_project_label = "project" if overdue_projects == 1 else "projects"
            actions.append(
                f"Reconfirm the target completion date and next milestone for {overdue_projects:,} overdue {overdue_project_label}"
            )
        if overdue_tasks:
            overdue_task_label = "task" if overdue_tasks == 1 else "tasks"
            actions.append(
                f"Review ownership and revised dates for {overdue_tasks:,} overdue {overdue_task_label}"
            )
        if blocked_tasks:
            actions.append(f"Resolve the recorded reasons for {blocked_tasks:,} blocked {task_label}")
        if blockers:
            blocker_project_label = "project" if blockers == 1 else "projects"
            actions.append(
                f"Assign an owner and resolution date to the blocker on {blockers:,} {blocker_project_label}"
            )
        if red and len(actions) < 3:
            actions.append(f"Agree a recovery plan for {red:,} red {project_label}")
        if open_tickets and len(actions) < 3:
            actions.append(f"Triage the {open_tickets:,} open {ticket_label} by priority and due date")
        if not actions:
            actions.append(
                "Confirm whether the concern is the target date, milestone, ticket queue, or task progress"
            )
        action_text = " ".join(
            f"{index}. {action}." for index, action in enumerate(actions[:3], start=1)
        )
        summary += (
            f" Suggested actions: {action_text} To refine the answer, specify whether delay means "
            "the project target date, a milestone, a ticket, or a task."
        )
        title = "Project delivery actions"
    return DataAnswer(
        title,
        summary,
        rows.sort_values(
            ["DeliveryHealth", "BlockedTaskCount", "OpenTicketCount"],
            ascending=[False, False, False],
        ),
        "Projects, Opportunities, Opportunity Tickets, and Ticket Tasks sheets",
    )


def answer_data_question(question: str, data: dict[str, pd.DataFrame]) -> DataAnswer:
    """Answer supported natural-language questions using deterministic local logic."""

    question = (question or "").strip()
    profiles = create_performance_profiles(data)
    people_matches = _find_salespeople(question, profiles)
    if len(people_matches) > 1:
        return _ambiguous_salesperson_answer(people_matches)
    person = people_matches.iloc[0] if len(people_matches) == 1 else None
    customer = _find_customer(question, data)
    lowered = question.lower()
    profile_columns = [
        "Salesperson", "performance_score", "total_revenue", "expected_revenue",
        "performance_gap", "win_rate", "total_meetings", "opportunities_created",
        "cross_sell_opportunities", "support_status",
    ]

    def with_visual(answer: DataAnswer) -> DataAnswer:
        if answer.visual is None:
            answer.visual = recommend_answer_visual(
                answer.title, answer.table, question, answer.interpretation
            )
        return answer

    if not question:
        return with_visual(DataAnswer(
            "Ask a question",
            "Enter a question about performance, renewals, customers, services, whitespace, or collaboration.",
            pd.DataFrame(),
            "Local workbook only",
        ))

    query_plan = build_query_plan(question, data)
    if query_plan.needs_clarification:
        clarification_rows = pd.DataFrame(
            [
                {
                    "Information Needed": "Query meaning",
                    "Why It Matters": message,
                    "Example Request": "Show Chloe Singh's upcoming cross-sell opportunities",
                }
                for message in query_plan.ambiguities
            ]
        )
        return with_visual(DataAnswer(
            "Please clarify the request",
            "The query planner found more than one valid interpretation. Choose the intended workbook view so I do not mix different records.",
            clarification_rows,
            "Manager clarification required; local structured query plan",
            query_plan.interpretation(),
        ))
    planned_result = execute_query_plan(query_plan, data)
    if planned_result is not None:
        return with_visual(DataAnswer(
            planned_result.title,
            planned_result.summary,
            planned_result.table,
            planned_result.source,
            planned_result.interpretation,
        ))

    metric_definitions = data.get("MetricDefinitions", pd.DataFrame())
    metric_language = any(term in lowered for term in ["metric", "define", "definition", "formula", "calculate", "what is"])
    performance_metric_question = (
        any(term in lowered for term in ["metric", "measure", "factor", "component"])
        and any(term in lowered for term in ["performance", "compare", "comparison", "score", "rank"])
    )
    if performance_metric_question:
        comparison = performance_comparison_metrics()
        weights = comparison.loc[comparison["Comparison Layer"] == "Composite score", ["Component", "Weight"]]
        weight_text = ", ".join(
            f"{row.Component} {row.Weight}" for row in weights.itertuples(index=False)
        )
        return with_visual(DataAnswer(
            "Performance comparison metrics",
            f"The composite performance score uses six peer-adjusted components: {weight_text}. Diagnostic drivers and displayed outcomes are shown separately and are not additional score weights. The score supports coaching and benchmarking; it should not be used alone for employment decisions.",
            comparison,
            "Local sales_metrics.yaml and implemented performance profile logic",
        ))
    if metric_language and not metric_definitions.empty:
        normalized_question = re.sub(r"[^a-z0-9]", "", lowered)
        matches = metric_definitions[
            metric_definitions["MetricName"].astype(str).map(
                lambda value: re.sub(r"[^a-z0-9]", "", value.lower()) in normalized_question
            )
        ]
        if matches.empty and any(term in lowered for term in ["available metrics", "metric catalogue", "metric definitions"]):
            matches = metric_definitions
        if not matches.empty:
            first = matches.iloc[0]
            summary = (
                f"{first['MetricName']}: {first['Definition']} "
                f"Formula or logic: {first['FormulaOrLogic']}. Availability: {first['Availability']}."
            )
            return with_visual(DataAnswer(
                "Canonical metric definition",
                summary,
                matches.head(25),
                "Metric Definitions sheet and local sales_metrics.yaml",
            ))

    has_explicit_record = any(
        [
            person is not None,
            customer is not None,
            _find_opportunity_id(question, data) is not None,
            _find_record_id(question, data, "Projects", "ProjectID") is not None,
            _find_record_id(question, data, "OpportunityTickets", "TicketID") is not None,
            _find_record_id(question, data, "TicketTasks", "TaskID") is not None,
        ]
    )
    ambiguous_reference = any(
        phrase in lowered
        for phrase in ["this opportunity", "that opportunity", "can we win this", "is this achievable", "what about this", "tell me more"]
    )
    if ambiguous_reference and not has_explicit_record:
        topic = "forecast" if any(term in lowered for term in ["win", "achiev", "opportunity"]) else "general"
        return _clarification_answer(question, topic)

    pipeline_question = any(
        phrase in lowered
        for phrase in [
            "pipeline forecast", "forecast pipeline", "pipeline revenue", "pipeline coverage",
            "pipeline gap", "cover the pipeline", "cover that pipeline", "cover target",
            "achievable", "achievability", "year-end forecast", "year end forecast",
        ]
    )
    if pipeline_question:
        forecast = build_pipeline_forecast(data, save_model=False)
        summary = forecast.salesperson_summary.copy()
        suggestions = forecast.suggestions.copy()
        if person is not None:
            summary = summary[summary["SalespersonID"] == person["SalespersonID"]]
            suggestions = suggestions[suggestions["SalespersonID"] == person["SalespersonID"]]
        asks_for_actions = any(
            term in lowered for term in ["suggest", "action", "help", "cover", "improve", "close the gap"]
        )
        if person is not None and not summary.empty:
            row = summary.iloc[0]
            gap_text = f"a shortfall of {_money(abs(row['ForecastGap']))}" if row["ForecastGap"] < 0 else f"a surplus of {_money(row['ForecastGap'])}"
            summary_text = (
                f"{row['Salesperson']} has {_money(row['YTDRevenue'])} revenue year to date against an annual target of {_money(row['AnnualTarget'])}. "
                f"The probability-adjusted pipeline forecast is {_money(row['WeightedPipelineForecast'])}, giving a year-end forecast of {_money(row['ForecastYearEndRevenue'])} and {gap_text}. "
                f"Achievability is {str(row['Achievability']).lower()} at {row['AchievabilityScore']:.0f}/100. This is a local scenario forecast, not a guarantee."
            )
        else:
            team = forecast.team_summary
            gap_text = f"a shortfall of {_money(abs(team['ForecastGap']))}" if team["ForecastGap"] < 0 else f"a surplus of {_money(team['ForecastGap'])}"
            summary_text = (
                f"For {team['ForecastYear']:.0f}, team revenue is {_money(team['YTDRevenue'])} year to date against an annual target of {_money(team['AnnualTarget'])}. "
                f"Open pipeline due this year is {_money(team['OpenPipeline'])}; the probability-adjusted forecast contribution is {_money(team['WeightedPipelineForecast'])}. "
                f"Forecast year-end revenue is {_money(team['ForecastYearEndRevenue'])}, leaving {gap_text}. Achievability is {str(team['Achievability']).lower()} at {team['AchievabilityScore']:.0f}/100."
            )
        if asks_for_actions:
            if person is None and not suggestions.empty:
                names = forecast.salesperson_summary[["SalespersonID", "Salesperson", "AchievabilityScore"]]
                suggestions = suggestions.merge(names, on="SalespersonID", how="left").sort_values(
                    ["AchievabilityScore", "Priority"]
                )
            result_table = suggestions.head(25)
            title = "Pipeline coverage actions"
        else:
            result_table = summary.head(15)
            title = "Pipeline revenue forecast"
        return with_visual(DataAnswer(
            title,
            summary_text,
            result_table,
            "Current-year Monthly Performance, Opportunities, local win classifier, meetings, and opportunity notes",
        ))

    last_week_meeting_question = (
        "meeting" in lowered
        and any(phrase in lowered for phrase in ["last week", "previous week", "prior week"])
    )
    if last_week_meeting_question:
        ranked, period_start, period_end, latest_date = _last_complete_week_meeting_leaderboard(data)
        if period_start is None:
            return with_visual(DataAnswer(
                "Meetings last week",
                "A weekly meeting answer needs SalespersonID, ActivityDate, and ActivityType in the local Activities sheet.",
                pd.DataFrame(),
                "Local workbook availability check",
            ))
        period = f"{period_start:%d %b %Y} to {period_end:%d %b %Y}"
        if ranked.empty:
            return with_visual(DataAnswer(
                "Meetings last week",
                f"No meeting activities were recorded during {period}. The latest activity in the workbook is {latest_date:%d %b %Y}.",
                ranked,
                "Activities sheet in the local workbook",
            ))
        highest = int(ranked["Meetings"].max())
        leaders = ranked.loc[ranked["Meetings"] == highest, "Salesperson"].tolist()
        leader_text = leaders[0] if len(leaders) == 1 else f"{', '.join(leaders[:-1])} and {leaders[-1]}"
        meeting_text = f"{highest} meetings" if len(leaders) == 1 else f"{highest} meetings each"
        return with_visual(DataAnswer(
            "Meetings last week",
            f"{leader_text} held the most meetings in the latest complete week available: {meeting_text} during {period}. The workbook's latest activity date is {latest_date:%d %b %Y}.",
            ranked,
            "Meetings and Salespeople sheets in the local workbook",
        ))

    opportunity_id = _find_opportunity_id(question, data)
    response_queue_question = any(
        phrase in lowered
        for phrase in [
            "waiting response", "waiting for a response", "awaiting response",
            "unanswered", "not responded", "no response", "response overdue",
        ]
    )
    if response_queue_question:
        notes = data.get("OpportunityNotes", pd.DataFrame()).copy()
        if notes.empty:
            return with_visual(DataAnswer(
                "Opportunity response queue",
                "The local workbook does not contain opportunity notes yet.",
                notes,
                "Local workbook availability check",
            ))
        notes = notes[notes["ResponseStatus"].astype(str).str.lower().eq("waiting response")]
        if person is not None:
            notes = notes[notes["SalespersonID"] == person["SalespersonID"]]
        if customer is not None:
            notes = notes[notes["CustomerID"] == customer["CustomerID"]]
        if opportunity_id:
            notes = notes[notes["OpportunityID"] == opportunity_id]
        notes = notes.merge(
            data.get("Customers", pd.DataFrame())[["CustomerID", "CustomerName"]],
            on="CustomerID", how="left",
        ).merge(
            data.get("Salespeople", pd.DataFrame())[["SalespersonID", "Salesperson"]],
            on="SalespersonID", how="left",
        )
        notes = notes.sort_values(
            ["CriticalFindingFlag", "ResponseAgeDays"], ascending=[False, False]
        )
        oldest = int(pd.to_numeric(notes.get("ResponseAgeDays"), errors="coerce").max()) if not notes.empty else 0
        critical = int(notes.get("CriticalFindingFlag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        return with_visual(DataAnswer(
            "Opportunity responses waiting",
            f"{len(notes):,} opportunity notes are waiting for a response; {critical:,} are marked critical and the oldest has waited {oldest:,} days.",
            notes,
            "Opportunity Notes, Customers, and Salespeople sheets",
        ))

    if "meeting" in lowered:
        meetings = data.get("Meetings", pd.DataFrame()).copy()
        if meetings.empty:
            return with_visual(DataAnswer(
                "Customer meetings",
                "Detailed meeting records are not available in the local workbook.",
                meetings,
                "Local workbook availability check",
            ))
        if person is not None:
            meetings = meetings[meetings["SalespersonID"] == person["SalespersonID"]]
        if customer is not None:
            meetings = meetings[meetings["CustomerID"] == customer["CustomerID"]]
        if opportunity_id:
            meetings = meetings[meetings["OpportunityID"] == opportunity_id]
        if "critical" in lowered or "risk" in lowered:
            meetings = meetings[
                meetings["CriticalFindingFlag"].fillna(False).astype(bool)
            ]
        if "support" in lowered or "escalation" in lowered:
            meetings = meetings[
                meetings["MeetingType"].astype(str).str.lower().eq("support escalation")
            ]
        meetings = meetings.merge(
            data.get("Customers", pd.DataFrame())[["CustomerID", "CustomerName"]],
            on="CustomerID", how="left",
        ).merge(
            data.get("Salespeople", pd.DataFrame())[["SalespersonID", "Salesperson"]],
            on="SalespersonID", how="left",
        ).sort_values("MeetingDate", ascending=False)
        critical = int(meetings["CriticalFindingFlag"].fillna(False).astype(bool).sum()) if not meetings.empty else 0
        return with_visual(DataAnswer(
            "Customer meeting intelligence",
            f"Found {len(meetings):,} matching meetings, including {critical:,} with critical findings.",
            meetings,
            "Meetings, Customers, Salespeople, and Opportunities sheets",
        ))

    delivery_question = is_delivery_question(question, data)
    if delivery_question:
        return with_visual(_answer_delivery_question(question, data, person, customer, opportunity_id))

    if any(term in lowered for term in ["why", "cause", "reason", "understand", "explain"]):
        if any(term in lowered for term in ["performance", "perform", "revenue", "win", "meeting", "score", "gap"]):
            return _clarification_answer(question, "performance")

    if "escalation" in lowered:
        notes = data.get("OpportunityNotes", pd.DataFrame()).copy()
        if not notes.empty:
            notes = notes[
                notes["NoteType"].astype(str).str.lower().eq("support escalation")
                | notes["CriticalFindingFlag"].fillna(False).astype(bool)
            ]
            if person is not None:
                notes = notes[notes["SalespersonID"] == person["SalespersonID"]]
            if customer is not None:
                notes = notes[notes["CustomerID"] == customer["CustomerID"]]
            if opportunity_id:
                notes = notes[notes["OpportunityID"] == opportunity_id]
            notes = notes.sort_values(["CriticalFindingFlag", "ResponseAgeDays"], ascending=False)
        return DataAnswer(
            "Opportunity escalations",
            f"Found {len(notes):,} matching opportunity escalations and critical notes.",
            notes,
            "Opportunity Notes sheet",
        )

    if customer is not None:
        customer_id = customer["CustomerID"]
        contracts = data.get("Contracts", pd.DataFrame())
        services = data.get("ContractServices", pd.DataFrame())
        billing = data.get("ExistingCustomerBilling", pd.DataFrame())
        rows = contracts[contracts["CustomerID"] == customer_id].copy()
        if "service" in lowered:
            rows = services[services["CustomerID"] == customer_id].copy()
        elif "bill" in lowered or "mrr" in lowered:
            rows = billing[billing["CustomerID"] == customer_id].copy()
        summary = f"Found {len(rows):,} matching local records for {customer['CustomerName']}."
        return with_visual(DataAnswer(f"{customer['CustomerName']} account view", summary, rows.head(25), "Customers, Contracts, Contract Services, and Billing"))

    if person is not None and any(word in lowered for word in ["perform", "revenue", "score", "gap", "how is", "win rate"]):
        selected = profiles[profiles["SalespersonID"] == person["SalespersonID"]][profile_columns]
        summary = (
            f"{person['Salesperson']} has a performance score of {person['performance_score']}/100, "
            f"actual revenue of {_money(person['total_revenue'])}, and a gap of {_money(person['performance_gap'])} "
            "against the local annual target."
        )
        return with_visual(DataAnswer(f"{person['Salesperson']} performance", summary, selected, "Monthly Performance, Targets, and local engineered features"))

    if "renew" in lowered or "health check" in lowered or "rolling contract" in lowered:
        renewals = data.get("UpcomingRenewals", pd.DataFrame()).copy()
        if person is not None:
            renewals = renewals[renewals["AccountOwnerID"] == person["SalespersonID"]]
        days_match = re.search(r"(?:next|within)\s+(\d+)\s+days", lowered)
        if days_match and "DaysToRenewal" in renewals:
            days = int(days_match.group(1))
            renewals = renewals[renewals["DaysToRenewal"].between(0, days)]
        if "health check" in lowered:
            contracts = data.get("Contracts", pd.DataFrame()).copy()
            renewals = contracts[_numeric(contracts["HealthCheckRequired"]) > 0]
            if person is not None:
                renewals = renewals[renewals["AccountOwnerID"] == person["SalespersonID"]]
        renewals = renewals.sort_values("DaysToRenewal") if "DaysToRenewal" in renewals else renewals
        return with_visual(DataAnswer(
            "Renewal and health-check priorities",
            f"Found {len(renewals):,} contracts matching the question. Negative renewal days indicate overdue or rolling contracts.",
            renewals.head(25),
            "Contracts and Upcoming Renewals",
        ))

    if "whitespace" in lowered or "cross-sell" in lowered or "cross sell" in lowered:
        whitespace = create_customer_whitespace(data)
        if person is not None:
            whitespace = whitespace[whitespace["Account Owner"] == person["Salesperson"]]
        ranking_requested = any(
            term in lowered
            for term in ["top", "highest", "largest", "best", "lowest", "smallest"]
        )
        result_limit = _requested_ranking_count(lowered, default=25) if ranking_requested else 25
        ascending = any(term in lowered for term in ["lowest", "smallest"])
        if not whitespace.empty:
            whitespace = whitespace.sort_values(
                ["Estimated Annual Potential", "Whitespace Score"],
                ascending=[ascending, ascending],
            )
        result = whitespace.head(result_limit)
        potential = result["Estimated Annual Potential"].sum() if not result.empty else 0
        return with_visual(DataAnswer(
            "Customer whitespace opportunities",
            f"The local product history identifies {len(result):,} matching customer opportunities with combined estimated annual potential of {_money(potential)}. Estimates are directional, not guarantees.",
            result,
            "Customer Products, Customers, Contracts, and Salespeople",
        ))

    if any(word in lowered for word in ["synergy", "collaborat", "partner", "referral"]):
        synergy = create_synergy_summary(data)
        if person is not None and not synergy.empty:
            synergy = synergy[(synergy["FromSalespersonID"] == person["SalespersonID"]) | (synergy["ToSalespersonID"] == person["SalespersonID"])]
        return with_visual(DataAnswer(
            "Sales collaboration",
            f"Found {len(synergy):,} salesperson referral relationships in the local workbook.",
            synergy.head(20),
            "Synergy Referrals and Synergy Map",
        ))

    high_ranking = any(term in lowered for term in ["top", "best", "highest"])
    low_ranking = any(
        term in lowered
        for term in ["bottom", "worst", "lowest", "underperform", "needs support", "coaching"]
    )
    if high_ranking or low_ranking:
        if "meeting" in lowered:
            metric, label = "total_meetings", "meetings"
        elif "win" in lowered:
            metric, label = "win_rate", "win rate"
        elif "opportunit" in lowered:
            metric, label = "opportunities_created", "opportunities created"
        elif "revenue" in lowered:
            metric, label = "total_revenue", "revenue"
        else:
            metric, label = "performance_score", "composite performance score"

        count = _requested_ranking_count(lowered)
        ranked = profiles.sort_values(metric, ascending=low_ranking)[profile_columns].head(count).copy()
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))

        def display_value(value: float) -> str:
            if metric == "win_rate":
                return f"{value:.1%}"
            if metric == "total_revenue":
                return _money(value)
            if metric == "performance_score":
                return f"{int(value)}/100"
            return f"{int(value):,}"

        result_text = ", ".join(
            f"{row['Salesperson']} ({display_value(row[metric])})"
            for _, row in ranked.iterrows()
        )
        direction = "lowest" if low_ranking else "highest"
        title = "Bottom performers" if low_ranking else "Top performers"
        return with_visual(DataAnswer(
            title,
            f"The {count} {direction} results by {label} are {result_text}. "
            "Use the underlying measures and context alongside the composite score.",
            ranked,
            "Monthly Performance, Targets, and local engineered features",
        ))

    if "missing" in lowered or "data quality" in lowered:
        rows = []
        for sheet, frame in data.items():
            count = int(frame.isna().sum().sum())
            if count:
                rows.append({"Sheet": sheet, "Rows": len(frame), "Missing Values": count})
        table = pd.DataFrame(rows).sort_values("Missing Values", ascending=False) if rows else pd.DataFrame()
        return with_visual(DataAnswer("Local data quality", f"{len(table):,} sheets contain one or more missing values.", table, "All loaded workbook sheets"))

    if any(word in lowered for word in ["revenue", "meeting", "opportunit", "win rate", "score"]):
        metric = "total_meetings" if "meeting" in lowered else "opportunities_created" if "opportunit" in lowered else "win_rate" if "win" in lowered else "total_revenue"
        ranked = profiles.sort_values(metric, ascending=False)[profile_columns]
        return with_visual(DataAnswer("Team performance", "Team results ranked from the local workbook.", ranked, "Monthly Performance and local performance profiles"))

    if any(term in lowered for term in ["why", "cause", "reason", "understand", "explain"]):
        topic = "performance" if any(term in lowered for term in ["performance", "revenue", "win", "meeting"]) else "general"
        return _clarification_answer(question, topic)
    if any(term in lowered for term in ["customer", "account", "client"]):
        return _clarification_answer(question, "customer")
    if any(term in lowered for term in ["forecast", "target", "future", "achieve"]):
        return _clarification_answer(question, "forecast")
    return _clarification_answer(question, "general")
