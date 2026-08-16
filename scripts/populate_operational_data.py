"""Populate reproducible synthetic meetings, notes, projects, tickets, and tasks.

LOCAL-ONLY MODEL BOUNDARY:
This script reads and updates only the local project workbook. It does not use
the internet, external APIs, hosted models, or real customer information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_PATH = PROJECT_ROOT / "data" / "MSP_Sales_Performance_Raw_Data_With_Common_Metrics.xlsx"
RANDOM_SEED = 20260815


def _choice(rng: np.random.Generator, values: list, probabilities: list[float] | None = None):
    return values[int(rng.choice(len(values), p=probabilities))]


def _business_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))


def _owner_customers(
    salesperson_id: str,
    contracts: pd.DataFrame,
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
) -> list[str]:
    owned = contracts.loc[contracts["AccountOwnerID"] == salesperson_id, "CustomerID"].dropna().tolist()
    owned.extend(
        opportunities.loc[opportunities["SalespersonID"] == salesperson_id, "CustomerID"].dropna().tolist()
    )
    return sorted(set(owned)) or customers["CustomerID"].dropna().tolist()


def build_current_year_performance(
    rng: np.random.Generator,
    monthly: pd.DataFrame,
    salespeople: pd.DataFrame,
    targets: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add reproducible current-year YTD performance while preserving history."""

    monthly = monthly.copy()
    if "DataOrigin" not in monthly:
        monthly["DataOrigin"] = "Original Workbook"
    historical = monthly[monthly["DataOrigin"] != "Synthetic Current Year"].copy()
    historical["Month"] = pd.to_datetime(historical["Month"], errors="coerce")
    baseline = historical[historical["Month"].dt.year == snapshot_date.year - 1].copy()
    target_map = targets.set_index("SalespersonID")["AnnualRevenueTarget"].to_dict()
    source_metrics = [
        "CustomerReachouts", "Meetings", "OpportunitiesCreated", "OpportunitiesWon",
        "NewCustomers", "CrossSellOpportunities", "Revenue", "GrossProfit",
        "CrossSellRevenue", "RetentionRate",
    ]
    rows = []
    for person_index, salesperson_id in enumerate(salespeople["SalespersonID"], start=1):
        person_history = baseline[baseline["SalespersonID"] == salesperson_id].copy()
        if person_history.empty:
            person_history = baseline.copy()
        person_growth = 0.97 + person_index * 0.012
        for month_number in range(1, snapshot_date.month + 1):
            month = pd.Timestamp(snapshot_date.year, month_number, 1)
            prior = person_history[person_history["Month"].dt.month == month_number]
            source = prior.iloc[0] if not prior.empty else person_history.iloc[0]
            is_current_month = month_number == snapshot_date.month
            completion = snapshot_date.day / snapshot_date.days_in_month if is_current_month else 1.0
            row = source.copy()
            row["Month"] = month
            row["SalespersonID"] = salesperson_id
            for metric in source_metrics:
                value = float(pd.to_numeric(pd.Series([source.get(metric, 0)]), errors="coerce").fillna(0).iloc[0])
                noise = float(rng.normal(1.0, 0.07))
                updated = max(0, value * person_growth * noise * completion)
                row[metric] = int(round(updated)) if metric in {
                    "CustomerReachouts", "Meetings", "OpportunitiesCreated", "OpportunitiesWon",
                    "NewCustomers", "CrossSellOpportunities",
                } else round(updated, 2)
            row["OpportunitiesWon"] = min(row["OpportunitiesWon"], row["OpportunitiesCreated"])
            row["GrossProfit"] = min(row["GrossProfit"], row["Revenue"] * 0.55)
            row["CrossSellRevenue"] = min(row["CrossSellRevenue"], row["Revenue"] * 0.35)
            annual_target = float(target_map.get(salesperson_id, 0))
            monthly_target = annual_target / 12 * completion
            row["TargetAttainment"] = row["Revenue"] / monthly_target if monthly_target else 0
            row["DataOrigin"] = "Synthetic Current Year"
            rows.append(row)
    current = pd.DataFrame(rows)
    return pd.concat([historical, current], ignore_index=True).sort_values(
        ["Month", "SalespersonID"]
    ).reset_index(drop=True)


def build_current_year_opportunities(
    rng: np.random.Generator,
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
    salespeople: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add current pipeline records with fields needed for local forecasting."""

    opportunities = opportunities.copy()
    if "DataOrigin" not in opportunities:
        opportunities["DataOrigin"] = "Original Workbook"
    base = opportunities[opportunities["DataOrigin"] != "Synthetic Current Year"].copy()
    for column, default in {
        "ExpectedCloseDate": pd.NaT,
        "PipelineStage": "Historical",
        "WinProbability": 0.0,
        "ForecastCategory": "Historical",
        "LastActivityDate": pd.NaT,
        "NextStep": "Historical record",
        "NextStepDueDate": pd.NaT,
        "DaysInStage": 0,
        "PipelineRisk": "Historical",
    }.items():
        if column not in base:
            base[column] = default
    base["ExpectedCloseDate"] = pd.to_datetime(base["ExpectedCloseDate"], errors="coerce").fillna(
        pd.to_datetime(base.get("CloseDate"), errors="coerce")
    )
    historical_stage = base["Stage"].astype(str).str.lower()
    base["WinProbability"] = np.where(historical_stage.eq("won"), 1.0, 0.0)

    products = sorted(base["Product"].dropna().astype(str).unique())
    customer_ids = customers["CustomerID"].dropna().tolist()
    people_ids = salespeople["SalespersonID"].dropna().tolist()
    stage_probability = {
        "Discovery": 0.12,
        "Qualification": 0.28,
        "Solution Design": 0.45,
        "Proposal": 0.62,
        "Negotiation": 0.76,
        "Verbal Commit": 0.90,
    }
    next_steps = {
        "Discovery": "Confirm business outcomes and executive sponsor",
        "Qualification": "Validate budget, authority, need, and timeline",
        "Solution Design": "Complete technical scope and solution workshop",
        "Proposal": "Review proposal and commercial assumptions",
        "Negotiation": "Resolve commercial and procurement actions",
        "Verbal Commit": "Confirm signature and implementation handover date",
    }
    rows = []
    start_index = max(
        [int(value.replace("OPP", "")) for value in base["OpportunityID"].astype(str) if value.startswith("OPP")]
        or [0]
    )
    current_dates = _business_dates(pd.Timestamp(snapshot_date.year, 1, 2), snapshot_date)
    for index in range(1, 181):
        salesperson_id = people_ids[(index - 1) % len(people_ids)]
        created_date = pd.Timestamp(rng.choice(current_dates))
        customer_id = str(rng.choice(customer_ids))
        status = _choice(rng, ["Open", "Won", "Lost"], [0.64, 0.22, 0.14])
        pipeline_stage = _choice(
            rng,
            list(stage_probability),
            [0.13, 0.18, 0.20, 0.23, 0.17, 0.09],
        )
        value = round(float(np.clip(rng.lognormal(mean=10.55, sigma=0.62), 12000, 145000)), 2)
        gross_margin = float(rng.uniform(0.27, 0.49))
        if status == "Open":
            close_date = pd.NaT
            expected_close = snapshot_date + pd.Timedelta(days=int(rng.integers(7, 155)))
            probability = float(np.clip(stage_probability[pipeline_stage] + rng.normal(0, 0.04), 0.05, 0.95))
            forecast_category = "Commit" if pipeline_stage == "Verbal Commit" else "Best Case" if pipeline_stage in {"Proposal", "Negotiation"} else "Pipeline"
        else:
            close_date = min(created_date + pd.Timedelta(days=int(rng.integers(18, 120))), snapshot_date)
            expected_close = close_date
            probability = 1.0 if status == "Won" else 0.0
            forecast_category = "Closed"
        last_activity = min(snapshot_date, created_date + pd.Timedelta(days=int(rng.integers(1, max(2, (snapshot_date - created_date).days + 1)))))
        days_in_stage = int(rng.integers(3, 85)) if status == "Open" else 0
        risk = "High" if status == "Open" and (days_in_stage > 55 or (snapshot_date - last_activity).days > 20) else "Medium" if status == "Open" and days_in_stage > 35 else "Low"
        rows.append(
            {
                "OpportunityID": f"OPP{start_index + index:04d}",
                "CustomerID": customer_id,
                "SalespersonID": salesperson_id,
                "CreatedDate": created_date,
                "CloseDate": close_date,
                "ExpectedCloseDate": expected_close,
                "Product": str(rng.choice(products)),
                "OpportunityType": _choice(rng, ["New Customer", "Cross-sell", "Upsell"], [0.38, 0.39, 0.23]),
                "Stage": status,
                "PipelineStage": pipeline_stage if status == "Open" else status,
                "PipelineValue": value,
                "ExpectedGrossProfit": round(value * gross_margin, 2),
                "SalesCycleDays": int((snapshot_date - created_date).days if status == "Open" else (close_date - created_date).days),
                "WinProbability": round(probability, 3),
                "ForecastCategory": forecast_category,
                "LastActivityDate": last_activity,
                "NextStep": next_steps[pipeline_stage] if status == "Open" else "Closed opportunity review",
                "NextStepDueDate": min(snapshot_date + pd.Timedelta(days=int(rng.integers(2, 18))), expected_close) if status == "Open" else pd.NaT,
                "DaysInStage": days_in_stage,
                "PipelineRisk": risk,
                "DataOrigin": "Synthetic Current Year",
            }
        )
    current = pd.DataFrame(rows)
    return pd.concat([base, current], ignore_index=True).sort_values(
        ["CreatedDate", "OpportunityID"]
    ).reset_index(drop=True)


def build_meetings(
    rng: np.random.Generator,
    salespeople: pd.DataFrame,
    customers: pd.DataFrame,
    opportunities: pd.DataFrame,
    contracts: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    historical_dates = _business_dates("2025-01-01", "2025-12-31")
    recent_dates = _business_dates("2026-01-01", snapshot_date)
    last_week_start = snapshot_date.normalize() - pd.Timedelta(days=snapshot_date.weekday() + 7)
    last_week_dates = _business_dates(last_week_start, last_week_start + pd.Timedelta(days=6))
    meeting_dates = list(rng.choice(historical_dates, 400, replace=True))
    meeting_dates.extend(rng.choice(recent_dates, 200, replace=True))
    meeting_dates.extend(rng.choice(last_week_dates, 40, replace=True))
    meeting_dates = sorted(pd.Timestamp(value).normalize() for value in meeting_dates)

    opportunity_groups = {
        salesperson_id: frame.reset_index(drop=True)
        for salesperson_id, frame in opportunities.groupby("SalespersonID")
    }
    existing_map = customers.set_index("CustomerID")["ExistingCustomer"].to_dict()
    customer_name_map = customers.set_index("CustomerID")["CustomerName"].to_dict()
    meeting_types = [
        "New Business Discovery",
        "Opportunity Review",
        "Account Health Check",
        "Regular Account Review",
        "Support Escalation",
        "Renewal Planning",
    ]
    type_probabilities = [0.17, 0.27, 0.18, 0.17, 0.11, 0.10]
    critical_findings = {
        "New Business Discovery": [
            "Executive sponsor has not yet been confirmed",
            "Budget approval depends on a board review",
            "Customer has a fixed migration deadline",
        ],
        "Opportunity Review": [
            "Technical scope has expanded beyond the original estimate",
            "Procurement requested revised commercial terms",
            "Competitor is active in the final evaluation",
        ],
        "Account Health Check": [
            "Service adoption is below the expected level",
            "Customer raised recurring service-quality concerns",
            "Key stakeholder satisfaction has declined",
        ],
        "Regular Account Review": [
            "Customer requested an executive service review",
            "Unused licences are creating value concerns",
            "A new cyber requirement is not covered by the current service",
        ],
        "Support Escalation": [
            "Critical incident remains open and is affecting operations",
            "Customer requested senior ownership of the escalation",
            "Repeated incidents are creating renewal risk",
        ],
        "Renewal Planning": [
            "Renewal decision is blocked by unresolved service concerns",
            "Customer requested a price and value review before renewal",
            "Contract scope no longer matches the operating requirement",
        ],
    }
    normal_outcomes = {
        "New Business Discovery": "Customer confirmed the initial business outcomes and agreed to a qualification workshop.",
        "Opportunity Review": "The opportunity remains active with scope, stakeholders, and next commercial step confirmed.",
        "Account Health Check": "Account health was reviewed across adoption, value, support, and renewal readiness.",
        "Regular Account Review": "Service performance and planned business changes were reviewed with the customer.",
        "Support Escalation": "The escalation was reviewed with clear technical ownership and a customer update cadence.",
        "Renewal Planning": "Renewal scope, decision process, and value evidence were reviewed with the customer.",
    }

    rows = []
    people_ids = salespeople["SalespersonID"].tolist()
    for index, meeting_date in enumerate(meeting_dates, start=1):
        salesperson_id = str(rng.choice(people_ids))
        meeting_type = _choice(rng, meeting_types, type_probabilities)
        person_opportunities = opportunity_groups.get(salesperson_id, pd.DataFrame())
        link_opportunity = meeting_type in {"New Business Discovery", "Opportunity Review"} or rng.random() < 0.30
        opportunity_id = None
        if link_opportunity and not person_opportunities.empty:
            candidate = person_opportunities.iloc[int(rng.integers(0, len(person_opportunities)))]
            opportunity_id = candidate["OpportunityID"]
            customer_id = candidate["CustomerID"]
        else:
            customer_id = str(rng.choice(_owner_customers(
                salesperson_id, contracts, opportunities, customers
            )))

        is_existing = bool(existing_map.get(customer_id, 0))
        if meeting_type == "New Business Discovery" and is_existing:
            meeting_type = "Opportunity Review"
        critical_probability = 0.36 if meeting_type == "Support Escalation" else 0.18
        critical = rng.random() < critical_probability
        severity = _choice(
            rng,
            ["Medium", "High", "Critical"],
            [0.48, 0.38, 0.14],
        ) if critical else "None"
        finding = _choice(rng, critical_findings[meeting_type]) if critical else "None identified"
        follow_up = _choice(rng, ["Open", "In Progress", "Complete"], [0.33, 0.31, 0.36])
        due_date = meeting_date + pd.Timedelta(days=int(rng.integers(2, 22)))
        customer_name = customer_name_map.get(customer_id, customer_id)
        summary = normal_outcomes[meeting_type]
        notes = (
            f"Met with {customer_name} regarding {meeting_type.lower()}. {summary} "
            f"Customer context: {'existing managed-service account' if is_existing else 'new customer prospect'}. "
            f"Critical finding: {finding}. Next step is owned by {salesperson_id}."
        )
        rows.append(
            {
                "MeetingID": f"MTG{index:05d}",
                "MeetingDate": meeting_date,
                "SalespersonID": salesperson_id,
                "CustomerID": customer_id,
                "OpportunityID": opportunity_id,
                "CustomerRelationship": "Existing Customer" if is_existing else "New Customer",
                "MeetingType": meeting_type,
                "Subject": f"{customer_name} - {meeting_type}",
                "DurationMinutes": int(_choice(rng, [30, 45, 60, 90], [0.18, 0.34, 0.39, 0.09])),
                "Attendees": _choice(rng, ["Customer lead and salesperson", "Customer lead, technical specialist, and salesperson", "Customer sponsor and salesperson"]),
                "MeetingStatus": "Held",
                "MeetingSummary": summary,
                "SalespersonNotes": notes,
                "CriticalFindingFlag": bool(critical),
                "CriticalSeverity": severity,
                "CriticalFinding": finding,
                "NextAction": _choice(rng, ["Send written follow-up", "Schedule technical workshop", "Confirm commercial response", "Update account action plan", "Escalate to service leadership"]),
                "ActionOwner": salesperson_id,
                "ActionDueDate": due_date,
                "FollowUpStatus": follow_up,
                "LastUpdatedDate": min(due_date, snapshot_date),
            }
        )
    return pd.DataFrame(rows)


def build_opportunity_notes(
    rng: np.random.Generator,
    opportunities: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    note_types = ["Customer Feedback", "Commercial Question", "Technical Query", "Procurement Update", "Support Escalation"]
    note_text = {
        "Customer Feedback": "Customer asked for confirmation that the proposed service outcomes match the operating priorities discussed.",
        "Commercial Question": "Customer requested a written response on pricing, contract term, and implementation assumptions.",
        "Technical Query": "Customer needs clarification on integration, security controls, dependencies, and migration responsibilities.",
        "Procurement Update": "Customer procurement requested updated documentation before the opportunity can progress.",
        "Support Escalation": "Customer reported an unresolved service issue that may affect confidence in the opportunity.",
    }
    rows = []
    weighted = opportunities.copy()
    weighted["_weight"] = np.where(weighted["Stage"].astype(str).str.lower().eq("open"), 3.0, 1.0)
    sampled = weighted.sample(520, replace=True, weights="_weight", random_state=RANDOM_SEED)
    for index, opportunity in enumerate(sampled.itertuples(index=False), start=1):
        note_type = _choice(rng, note_types, [0.22, 0.25, 0.24, 0.17, 0.12])
        waiting = index <= 85 or rng.random() < 0.10
        no_response = not waiting and rng.random() < 0.16
        if waiting:
            age = int(rng.integers(1, 46))
            note_date = snapshot_date - pd.Timedelta(days=age)
            status = "Waiting Response"
            response_required = True
            response_date = pd.NaT
            waiting_since = note_date
            response_age = age
        else:
            start = max(pd.Timestamp("2025-01-01"), pd.Timestamp(opportunity.CreatedDate))
            possible_dates = pd.date_range(start, snapshot_date)
            note_date = pd.Timestamp(rng.choice(possible_dates))
            response_required = not no_response
            status = "No Response Required" if no_response else "Responded"
            response_date = pd.NaT if no_response else min(
                note_date + pd.Timedelta(days=int(rng.integers(1, 8))), snapshot_date
            )
            waiting_since = pd.NaT
            response_age = 0 if no_response else int((response_date - note_date).days)
        severity = _choice(
            rng,
            ["Low", "Medium", "High", "Critical"],
            [0.18, 0.43, 0.31, 0.08],
        ) if note_type == "Support Escalation" or waiting else "Low"
        critical = severity in {"High", "Critical"} and (waiting or note_type == "Support Escalation")
        rows.append(
            {
                "NoteID": f"NOTE{index:05d}",
                "NoteDate": note_date,
                "CustomerID": opportunity.CustomerID,
                "OpportunityID": opportunity.OpportunityID,
                "SalespersonID": opportunity.SalespersonID,
                "NoteSource": _choice(rng, ["Customer", "Customer Procurement", "Customer Technical Lead", "Internal Service Team"], [0.48, 0.18, 0.20, 0.14]),
                "NoteType": note_type,
                "NoteText": note_text[note_type],
                "ResponseRequired": response_required,
                "ResponseStatus": status,
                "WaitingSince": waiting_since,
                "ResponseDate": response_date,
                "ResponseAgeDays": response_age,
                "EscalationSeverity": severity,
                "CriticalFindingFlag": bool(critical),
                "NextAction": "Respond to customer and update opportunity" if waiting else "Monitor through the next opportunity review",
                "ActionDueDate": note_date + pd.Timedelta(days=3) if waiting else pd.NaT,
            }
        )
    return pd.DataFrame(rows).sort_values(["NoteDate", "NoteID"]).reset_index(drop=True)


def build_projects_and_delivery(
    rng: np.random.Generator,
    opportunities: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = opportunities[opportunities["Stage"].astype(str).str.lower().isin(["open", "won"])].copy()
    sampled = eligible.sample(min(150, len(eligible)), random_state=RANDOM_SEED)
    project_rows = []
    ticket_rows = []
    task_rows = []
    ticket_index = 1
    task_index = 1
    project_managers = ["Maya Lewis", "Oliver Grant", "Priya Shah", "Thomas Reed"]
    for project_index, opportunity in enumerate(sampled.itertuples(index=False), start=1):
        is_open = str(opportunity.Stage).lower() == "open"
        stages = ["Discovery", "Solution Design", "Planning"] if is_open else ["Planning", "Implementation", "User Acceptance Testing", "Go Live", "Complete", "On Hold"]
        stage = _choice(rng, stages)
        percent_by_stage = {
            "Discovery": (5, 20), "Solution Design": (15, 35), "Planning": (20, 45),
            "Implementation": (35, 75), "User Acceptance Testing": (70, 90),
            "Go Live": (88, 99), "Complete": (100, 100), "On Hold": (20, 80),
        }
        low, high = percent_by_stage[stage]
        percent_complete = low if low == high else int(rng.integers(low, high + 1))
        health = "Red" if stage == "On Hold" else _choice(rng, ["Green", "Amber", "Red"], [0.68, 0.25, 0.07])
        project_id = f"PRJ{project_index:04d}"
        start_date = max(pd.Timestamp("2025-01-01"), pd.Timestamp(opportunity.CreatedDate)) + pd.Timedelta(days=int(rng.integers(5, 45)))
        target_date = max(start_date + pd.Timedelta(days=int(rng.integers(45, 180))), snapshot_date - pd.Timedelta(days=40))
        blocker = "None"
        if health != "Green":
            blocker = _choice(rng, ["Waiting for customer technical information", "Resource availability", "Unresolved support dependency", "Commercial scope confirmation"])
        project_rows.append(
            {
                "ProjectID": project_id,
                "OpportunityID": opportunity.OpportunityID,
                "CustomerID": opportunity.CustomerID,
                "SalespersonID": opportunity.SalespersonID,
                "ProjectName": f"{opportunity.Product} delivery",
                "ProjectStage": stage,
                "ProjectStatus": "Complete" if stage == "Complete" else "On Hold" if stage == "On Hold" else "Active",
                "DeliveryHealth": health,
                "PercentComplete": percent_complete,
                "ProjectManager": str(rng.choice(project_managers)),
                "StartDate": start_date,
                "TargetCompletionDate": target_date,
                "NextMilestone": "Service handover" if percent_complete > 80 else "Customer stage approval",
                "Blocker": blocker,
                "LastUpdatedDate": snapshot_date - pd.Timedelta(days=int(rng.integers(0, 12))),
            }
        )

        for _ in range(int(rng.integers(1, 4))):
            status = _choice(rng, ["Open", "In Progress", "Waiting on Customer", "Blocked", "Resolved"], [0.15, 0.31, 0.17, 0.09, 0.28])
            priority = _choice(rng, ["Low", "Medium", "High", "Critical"], [0.14, 0.45, 0.33, 0.08])
            ticket_id = f"TKT{ticket_index:05d}"
            created_date = max(start_date, snapshot_date - pd.Timedelta(days=int(rng.integers(4, 150))))
            due_date = created_date + pd.Timedelta(days=int(rng.integers(5, 35)))
            ticket_rows.append(
                {
                    "TicketID": ticket_id,
                    "ProjectID": project_id,
                    "OpportunityID": opportunity.OpportunityID,
                    "CustomerID": opportunity.CustomerID,
                    "SalespersonID": opportunity.SalespersonID,
                    "TicketType": _choice(rng, ["Opportunity Handover", "Technical Design", "Implementation", "Customer Action", "Support Escalation"]),
                    "TicketTitle": f"{opportunity.Product} - delivery action",
                    "TicketStatus": status,
                    "Priority": priority,
                    "AssignedTeam": _choice(rng, ["Professional Services", "Service Desk", "Solutions Architecture", "Customer Success"]),
                    "CreatedDate": created_date,
                    "DueDate": due_date,
                    "ResolvedDate": min(due_date, snapshot_date) if status == "Resolved" else pd.NaT,
                    "EscalationFlag": bool(priority in {"High", "Critical"} and status in {"Blocked", "Waiting on Customer"}),
                    "LastUpdate": "Customer response required" if status == "Waiting on Customer" else "Work is progressing against the current task plan",
                }
            )
            for task_number in range(1, int(rng.integers(2, 6))):
                task_status = "Complete" if status == "Resolved" else _choice(rng, ["Not Started", "In Progress", "Waiting", "Blocked", "Complete"], [0.18, 0.30, 0.15, 0.09, 0.28])
                task_due = due_date - pd.Timedelta(days=int(rng.integers(0, 5)))
                task_rows.append(
                    {
                        "TaskID": f"TASK{task_index:06d}",
                        "TicketID": ticket_id,
                        "ProjectID": project_id,
                        "OpportunityID": opportunity.OpportunityID,
                        "TaskName": f"Task {task_number}: {_choice(rng, ['Confirm requirements', 'Complete design', 'Customer approval', 'Configure service', 'Validate handover'])}",
                        "TaskStatus": task_status,
                        "TaskOwner": _choice(rng, ["Sales", "Solutions", "Delivery", "Customer", "Service Desk"]),
                        "DueDate": task_due,
                        "CompletedDate": min(task_due, snapshot_date) if task_status == "Complete" else pd.NaT,
                        "BlockedReason": _choice(rng, ["Customer input outstanding", "Technical dependency", "Resource constraint"]) if task_status == "Blocked" else "",
                    }
                )
                task_index += 1
            ticket_index += 1
    return pd.DataFrame(project_rows), pd.DataFrame(ticket_rows), pd.DataFrame(task_rows)


def update_reference_sheets(
    relationships: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relationship_rows = pd.DataFrame(
        [
            {"Entity": "Meetings", "Primary Key": "MeetingID", "Foreign Key / Link": "SalespersonID, CustomerID, OpportunityID", "Relationship": "Meeting belongs to a salesperson and customer and may support an opportunity"},
            {"Entity": "OpportunityNotes", "Primary Key": "NoteID", "Foreign Key / Link": "OpportunityID, CustomerID, SalespersonID", "Relationship": "Customer or internal note requiring an opportunity response"},
            {"Entity": "Projects", "Primary Key": "ProjectID", "Foreign Key / Link": "OpportunityID", "Relationship": "Delivery project created from an opportunity"},
            {"Entity": "OpportunityTickets", "Primary Key": "TicketID", "Foreign Key / Link": "ProjectID, OpportunityID", "Relationship": "Operational ticket linked to project and opportunity"},
            {"Entity": "TicketTasks", "Primary Key": "TaskID", "Foreign Key / Link": "TicketID, ProjectID, OpportunityID", "Relationship": "Task status contributing to ticket and project progress"},
        ]
    )
    relationships = pd.concat([relationships, relationship_rows], ignore_index=True)
    relationships = relationships.drop_duplicates(subset=["Entity"], keep="last")
    feature_rows = pd.DataFrame(
        [
            {"Feature": "OperationalMeetingCount", "Category": "Activity", "Definition": "Held meetings recorded at event level", "WhyItMatters": "Supports customer and opportunity context"},
            {"Feature": "CriticalMeetingFindings", "Category": "Risk", "Definition": "Meetings with a critical finding flag", "WhyItMatters": "Surfaces account and opportunity risks"},
            {"Feature": "WaitingOpportunityResponses", "Category": "Responsiveness", "Definition": "Opportunity notes currently waiting for salesperson response", "WhyItMatters": "Identifies customer follow-up exposure"},
            {"Feature": "AverageResponseAgeDays", "Category": "Responsiveness", "Definition": "Average waiting age for unanswered opportunity notes", "WhyItMatters": "Measures response backlog"},
            {"Feature": "BlockedProjects", "Category": "Delivery", "Definition": "Opportunity-linked projects with red health or on-hold status", "WhyItMatters": "Connects delivery blockers to commercial risk"},
            {"Feature": "OpenOpportunityTickets", "Category": "Delivery", "Definition": "Unresolved tickets linked to opportunities", "WhyItMatters": "Shows implementation and escalation workload"},
            {"Feature": "AdjustedWinProbability", "Category": "Pipeline", "Definition": "Blended local probability adjusted for risk and customer-response signals", "WhyItMatters": "Supports probability-adjusted pipeline revenue"},
            {"Feature": "WeightedPipelineForecast", "Category": "Pipeline", "Definition": "Open pipeline value multiplied by adjusted win probability", "WhyItMatters": "Estimates achievable pipeline contribution"},
            {"Feature": "PipelineCoverage", "Category": "Pipeline", "Definition": "Open pipeline divided by remaining annual target", "WhyItMatters": "Shows whether enough raw pipeline exists"},
            {"Feature": "AchievabilityScore", "Category": "Pipeline", "Definition": "Coverage, weighted coverage, and YTD pace combined into a transparent 0-100 score", "WhyItMatters": "Supports manager judgement about annual target risk"},
        ]
    )
    dictionary = pd.concat([dictionary, feature_rows], ignore_index=True)
    dictionary = dictionary.drop_duplicates(subset=["Feature"], keep="last")
    return relationships, dictionary


def main() -> None:
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")
    data = pd.read_excel(WORKBOOK_PATH, sheet_name=None)
    rng = np.random.default_rng(RANDOM_SEED)
    contracts = data["Contracts"].copy()
    snapshot_values = pd.to_datetime(contracts.get("SnapshotDate"), errors="coerce")
    snapshot_date = snapshot_values.max()
    if pd.isna(snapshot_date):
        snapshot_date = pd.Timestamp("2026-08-15")

    monthly = build_current_year_performance(
        rng,
        data["MonthlyPerformance"],
        data["Salespeople"],
        data["Targets"],
        snapshot_date,
    )
    opportunities = build_current_year_opportunities(
        rng,
        data["Opportunities"],
        data["Customers"],
        data["Salespeople"],
        snapshot_date,
    )

    meetings = build_meetings(
        rng, data["Salespeople"], data["Customers"], opportunities, contracts, snapshot_date
    )
    notes = build_opportunity_notes(rng, opportunities, snapshot_date)
    projects, tickets, tasks = build_projects_and_delivery(rng, opportunities, snapshot_date)
    relationships, dictionary = update_reference_sheets(
        data.get("DataRelationships", pd.DataFrame(columns=["Entity", "Primary Key", "Foreign Key / Link", "Relationship"])),
        data.get("FeatureDictionary", pd.DataFrame(columns=["Feature", "Category", "Definition", "WhyItMatters"])),
    )

    generated = {
        "MonthlyPerformance": monthly,
        "Opportunities": opportunities,
        "Meetings": meetings,
        "OpportunityNotes": notes,
        "Projects": projects,
        "OpportunityTickets": tickets,
        "TicketTasks": tasks,
        "DataRelationships": relationships,
        "FeatureDictionary": dictionary,
    }
    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for sheet, frame in generated.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)

    for sheet, frame in generated.items():
        print(f"{sheet}: {len(frame):,} rows")
    print(f"Updated local workbook: {WORKBOOK_PATH}")


if __name__ == "__main__":
    main()
