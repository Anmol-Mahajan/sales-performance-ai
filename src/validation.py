"""Validation helpers for the local Excel workbook."""

from __future__ import annotations

import re

import pandas as pd

from .data_loader import REQUIRED_SHEETS


REQUIRED_COLUMNS = {
    "Salespeople": [
        "SalespersonID",
        "Salesperson",
        "Segment",
        "PrimarySpecialism",
        "Region",
    ],
    "Customers": [
        "CustomerID",
        "CustomerName",
        "Segment",
        "Region",
        "CustomerSince",
        "ExistingCustomer",
    ],
    "ExistingCustomerBilling": [
        "BillingID",
        "CustomerID",
        "BillingMonth",
        "ServiceCategory",
        "Service",
        "ContractID",
        "AccountOwnerID",
        "MRR",
        "TotalBilled",
        "GrossProfit",
        "BillingStatus",
        "RenewalDate",
    ],
    "Contracts": [
        "ContractID",
        "CustomerID",
        "AccountOwnerID",
        "ContractStatus",
        "CurrentEndDate",
        "DaysToRenewal",
        "EndDateChangeCount",
        "RollbackCount",
        "HealthCheckRequired",
        "SuggestedAction",
        "ContractMRR",
        "ContractARR",
        "RenewalRisk",
    ],
    "ContractServices": [
        "ContractServiceID",
        "ContractID",
        "CustomerID",
        "ServiceCategory",
        "Service",
        "ServiceMRR",
        "ServiceARR",
        "ServiceStatus",
    ],
    "ContractAuditLog": [
        "AuditID",
        "ContractID",
        "CustomerID",
        "ChangeDate",
        "ChangeType",
        "RollbackFlag",
        "ApprovalStatus",
        "SalespersonLinked",
    ],
    "UpcomingRenewals": [
        "Priority",
        "ContractID",
        "CustomerID",
        "AccountOwnerID",
        "CurrentEndDate",
        "DaysToRenewal",
        "ContractMRR",
        "ContractARR",
        "RenewalRisk",
        "HealthCheckReason",
        "SuggestedAction",
    ],
    "CustomerContractSummary": [
        "CustomerID",
        "AccountOwnerID",
        "ContractCount",
        "ActiveContractCount",
        "UpcomingRenewalCount",
        "ServiceCount",
        "TotalMRR",
        "TotalARR",
        "NearestContractEndDate",
        "DaysToNearestRenewal",
        "HealthCheckRequired",
        "SuggestedManagerAction",
    ],
    "MonthlyPerformance": [
        "Month",
        "SalespersonID",
        "CustomerReachouts",
        "Meetings",
        "OpportunitiesCreated",
        "OpportunitiesWon",
        "NewCustomers",
        "CrossSellOpportunities",
        "Revenue",
        "GrossProfit",
        "CrossSellRevenue",
        "WinRate",
    ],
    "Opportunities": [
        "OpportunityID",
        "CustomerID",
        "SalespersonID",
        "CreatedDate",
        "Product",
        "OpportunityType",
        "Stage",
        "PipelineValue",
        "ExpectedCloseDate",
        "PipelineStage",
        "WinProbability",
        "ForecastCategory",
        "LastActivityDate",
        "NextStep",
        "NextStepDueDate",
        "DaysInStage",
        "PipelineRisk",
    ],
    "SynergyReferrals": [
        "ReferralID",
        "ReferralDate",
        "FromSalespersonID",
        "ToSalespersonID",
        "ProductArea",
        "ReferralStatus",
    ],
    "SynergyMap": [
        "FromSalespersonID",
        "ToSalespersonID",
        "SynergyType",
        "SynergyStrength",
    ],
    "Meetings": [
        "MeetingID", "MeetingDate", "SalespersonID", "CustomerID", "MeetingType",
        "MeetingSummary", "SalespersonNotes", "CriticalFindingFlag", "NextAction",
    ],
    "OpportunityNotes": [
        "NoteID", "NoteDate", "CustomerID", "OpportunityID", "SalespersonID",
        "NoteText", "ResponseRequired", "ResponseStatus", "ResponseAgeDays",
    ],
    "Projects": [
        "ProjectID", "OpportunityID", "CustomerID", "SalespersonID", "ProjectStage",
        "ProjectStatus", "DeliveryHealth", "PercentComplete",
    ],
    "OpportunityTickets": [
        "TicketID", "ProjectID", "OpportunityID", "TicketStatus", "Priority", "DueDate",
    ],
    "TicketTasks": [
        "TaskID", "TicketID", "ProjectID", "OpportunityID", "TaskStatus", "DueDate",
    ],
}

DATE_COLUMNS = {
    "Customers": ["CustomerSince"],
    "ExistingCustomerBilling": ["BillingMonth", "RenewalDate"],
    "Contracts": ["OriginalStartDate", "CurrentStartDate", "OriginalEndDate", "CurrentEndDate", "SnapshotDate"],
    "ContractServices": ["ServiceStartDate", "ServiceEndDate"],
    "ContractAuditLog": ["ChangeDate", "PreviousEndDate", "NewEndDate"],
    "UpcomingRenewals": ["CurrentEndDate"],
    "CustomerContractSummary": ["NearestContractEndDate"],
    "MonthlyPerformance": ["Month"],
    "Opportunities": ["CreatedDate", "CloseDate", "ExpectedCloseDate", "LastActivityDate", "NextStepDueDate"],
    "SynergyReferrals": ["ReferralDate"],
    "Meetings": ["MeetingDate", "ActionDueDate", "LastUpdatedDate"],
    "OpportunityNotes": ["NoteDate", "WaitingSince", "ResponseDate", "ActionDueDate"],
    "Projects": ["StartDate", "TargetCompletionDate", "LastUpdatedDate"],
    "OpportunityTickets": ["CreatedDate", "DueDate", "ResolvedDate"],
    "TicketTasks": ["DueDate", "CompletedDate"],
}

ID_COLUMNS = {
    "Salespeople": ["SalespersonID"],
    "Customers": ["CustomerID"],
    "ExistingCustomerBilling": ["BillingID"],
    "Contracts": ["ContractID"],
    "ContractServices": ["ContractServiceID"],
    "ContractAuditLog": ["AuditID"],
    "Opportunities": ["OpportunityID"],
    "SynergyReferrals": ["ReferralID"],
    "Meetings": ["MeetingID"],
    "OpportunityNotes": ["NoteID"],
    "Projects": ["ProjectID"],
    "OpportunityTickets": ["TicketID"],
    "TicketTasks": ["TaskID"],
}

RELATIONSHIP_CHECKS = [
    ("Meetings", "SalespersonID", "Salespeople", "SalespersonID"),
    ("Meetings", "CustomerID", "Customers", "CustomerID"),
    ("Meetings", "OpportunityID", "Opportunities", "OpportunityID"),
    ("OpportunityNotes", "OpportunityID", "Opportunities", "OpportunityID"),
    ("Projects", "OpportunityID", "Opportunities", "OpportunityID"),
    ("OpportunityTickets", "ProjectID", "Projects", "ProjectID"),
    ("OpportunityTickets", "OpportunityID", "Opportunities", "OpportunityID"),
    ("TicketTasks", "TicketID", "OpportunityTickets", "TicketID"),
    ("TicketTasks", "OpportunityID", "Opportunities", "OpportunityID"),
]

CONSISTENCY_FLAG_COLUMNS = [
    "Severity",
    "Sheet",
    "RecordID",
    "Field",
    "Issue",
    "ExpectedValue",
    "ActualValue",
    "SuggestedAction",
]


def _normalise_text(value: object) -> str:
    """Normalise labels for consistency checks without changing source data."""

    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _record_id(sheet: str, row: pd.Series) -> str:
    candidates = ID_COLUMNS.get(sheet, []) + [
        "OpportunityID", "ProjectID", "TicketID", "CustomerID", "SalespersonID"
    ]
    for column in candidates:
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return str(row.name)


def data_consistency_flags(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return actionable record-level flags for conflicting cross-sheet data."""

    rows: list[dict[str, object]] = []

    def add_flag(
        severity: str,
        sheet: str,
        record_id: object,
        field: str,
        issue: str,
        expected: object,
        actual: object,
        action: str,
    ) -> None:
        rows.append(
            {
                "Severity": severity,
                "Sheet": sheet,
                "RecordID": str(record_id),
                "Field": field,
                "Issue": issue,
                "ExpectedValue": "" if pd.isna(expected) else str(expected),
                "ActualValue": "" if pd.isna(actual) else str(actual),
                "SuggestedAction": action,
            }
        )

    customers = data.get("Customers", pd.DataFrame()).copy()
    customer_name_by_id: dict[str, object] = {}
    existing_customer_by_id: dict[str, object] = {}
    if {"CustomerID", "CustomerName"}.issubset(customers.columns):
        usable = customers.dropna(subset=["CustomerID"]).copy()
        customer_name_by_id = dict(
            zip(usable["CustomerID"].astype(str), usable["CustomerName"])
        )
        if "ExistingCustomer" in usable:
            existing_customer_by_id = dict(
                zip(usable["CustomerID"].astype(str), usable["ExistingCustomer"])
            )

        for index, row in customers.iterrows():
            if pd.isna(row.get("CustomerID")) or not str(row.get("CustomerID", "")).strip():
                add_flag(
                    "Error", "Customers", index, "CustomerID", "Missing customer ID",
                    "A unique customer ID", row.get("CustomerID"), "Assign a customer ID before using this record",
                )
            if pd.isna(row.get("CustomerName")) or not str(row.get("CustomerName", "")).strip():
                add_flag(
                    "Error", "Customers", row.get("CustomerID", index), "CustomerName",
                    "Missing customer name", "A canonical customer name", row.get("CustomerName"),
                    "Add the approved customer name to the customer master",
                )

        named = usable.dropna(subset=["CustomerName"]).copy()
        named["_normalised_name"] = named["CustomerName"].map(_normalise_text)
        for normalised_name, group in named.groupby("_normalised_name"):
            customer_ids = sorted(group["CustomerID"].astype(str).unique())
            if normalised_name and len(customer_ids) > 1:
                add_flag(
                    "Warning", "Customers", ", ".join(customer_ids), "CustomerName",
                    "Customer name is assigned to multiple IDs", "One canonical customer ID",
                    "; ".join(f"{row.CustomerID}: {row.CustomerName}" for row in group.itertuples()),
                    "Confirm whether these records should be merged or renamed",
                )

        for customer_id, group in usable.groupby(usable["CustomerID"].astype(str)):
            names = group["CustomerName"].dropna().astype(str)
            if names.map(_normalise_text).nunique() > 1:
                add_flag(
                    "Error", "Customers", customer_id, "CustomerName",
                    "Customer ID has conflicting names", names.iloc[0], "; ".join(sorted(names.unique())),
                    "Choose one canonical name for this customer ID",
                )

    valid_customer_ids = set(customer_name_by_id)
    for sheet, frame in data.items():
        if sheet == "Customers" or "CustomerID" not in frame:
            continue
        for _, row in frame.loc[frame["CustomerID"].notna()].iterrows():
            customer_id = str(row["CustomerID"])
            if valid_customer_ids and customer_id not in valid_customer_ids:
                add_flag(
                    "Error", sheet, _record_id(sheet, row), "CustomerID",
                    "Customer ID is missing from the customer master", "An ID in Customers",
                    customer_id, "Correct the ID or add the customer to the customer master",
                )
                continue
            if "CustomerName" in frame and customer_id in customer_name_by_id:
                expected_name = customer_name_by_id[customer_id]
                actual_name = row["CustomerName"]
                if _normalise_text(expected_name) != _normalise_text(actual_name):
                    add_flag(
                        "Error", sheet, _record_id(sheet, row), "CustomerName",
                        "Customer name does not match the customer master", expected_name,
                        actual_name, "Replace the name with the canonical customer-master value",
                    )

    opportunities = data.get("Opportunities", pd.DataFrame())
    opportunity_lookup: dict[str, pd.Series] = {}
    if "OpportunityID" in opportunities:
        opportunity_lookup = {
            str(row["OpportunityID"]): row
            for _, row in opportunities.dropna(subset=["OpportunityID"]).drop_duplicates("OpportunityID").iterrows()
        }

    for sheet in ["Meetings", "OpportunityNotes", "Projects"]:
        frame = data.get(sheet, pd.DataFrame())
        if "OpportunityID" not in frame:
            continue
        for _, row in frame.loc[frame["OpportunityID"].notna()].iterrows():
            opportunity_id = str(row["OpportunityID"])
            parent = opportunity_lookup.get(opportunity_id)
            if parent is None:
                continue
            for field, issue in [
                ("CustomerID", "Customer differs from the linked opportunity"),
                ("SalespersonID", "Salesperson differs from the linked opportunity"),
            ]:
                if field not in row or field not in parent or pd.isna(row[field]) or pd.isna(parent[field]):
                    continue
                if str(row[field]) != str(parent[field]):
                    add_flag(
                        "Error", sheet, _record_id(sheet, row), field, issue, parent[field], row[field],
                        f"Align {field} with opportunity {opportunity_id}",
                    )

    meetings = data.get("Meetings", pd.DataFrame())
    if {"CustomerID", "CustomerRelationship"}.issubset(meetings.columns):
        for _, row in meetings.loc[meetings["CustomerID"].notna()].iterrows():
            customer_id = str(row["CustomerID"])
            if customer_id not in existing_customer_by_id:
                continue
            value = _normalise_text(existing_customer_by_id[customer_id])
            expected = "Existing Customer" if value in {"true", "yes", "y", "1", "existing"} else "New Customer"
            if _normalise_text(row["CustomerRelationship"]) != _normalise_text(expected):
                add_flag(
                    "Warning", "Meetings", _record_id("Meetings", row), "CustomerRelationship",
                    "Customer relationship does not match the customer master", expected,
                    row["CustomerRelationship"], "Refresh the meeting relationship from Customers",
                )

    projects = data.get("Projects", pd.DataFrame())
    project_lookup: dict[str, pd.Series] = {}
    if "ProjectID" in projects:
        project_lookup = {
            str(row["ProjectID"]): row
            for _, row in projects.dropna(subset=["ProjectID"]).drop_duplicates("ProjectID").iterrows()
        }
    tickets = data.get("OpportunityTickets", pd.DataFrame())
    for _, row in tickets.iterrows():
        if pd.isna(row.get("ProjectID")) or str(row.get("ProjectID", "")).strip() == "":
            continue
        project_id = str(row["ProjectID"])
        project = project_lookup.get(project_id)
        if project is None:
            continue
        if "OpportunityID" in row and "OpportunityID" in project and str(row["OpportunityID"]) != str(project["OpportunityID"]):
            add_flag(
                "Error", "OpportunityTickets", _record_id("OpportunityTickets", row), "OpportunityID",
                "Ticket opportunity differs from its linked project", project["OpportunityID"],
                row["OpportunityID"], f"Align the ticket with project {project_id}",
            )

    ticket_lookup: dict[str, pd.Series] = {}
    if "TicketID" in tickets:
        ticket_lookup = {
            str(row["TicketID"]): row
            for _, row in tickets.dropna(subset=["TicketID"]).drop_duplicates("TicketID").iterrows()
        }
    tasks = data.get("TicketTasks", pd.DataFrame())
    for _, row in tasks.iterrows():
        if pd.isna(row.get("TicketID")):
            continue
        ticket_id = str(row["TicketID"])
        ticket = ticket_lookup.get(ticket_id)
        if ticket is None:
            continue
        for field in ["ProjectID", "OpportunityID"]:
            if field not in row or field not in ticket or pd.isna(row[field]) or pd.isna(ticket[field]):
                continue
            if str(row[field]) != str(ticket[field]):
                add_flag(
                    "Error", "TicketTasks", _record_id("TicketTasks", row), field,
                    f"Task {field} differs from its parent ticket", ticket[field], row[field],
                    f"Align the task with ticket {ticket_id}",
                )

    if not opportunities.empty and {"Stage", "ExpectedCloseDate"}.issubset(opportunities.columns):
        snapshot_dates = []
        for sheet, column in [
            ("Contracts", "SnapshotDate"), ("Meetings", "MeetingDate"),
            ("OpportunityNotes", "NoteDate"), ("Opportunities", "LastActivityDate"),
        ]:
            frame = data.get(sheet, pd.DataFrame())
            if column in frame:
                parsed = pd.to_datetime(frame[column], errors="coerce").dropna()
                if not parsed.empty:
                    snapshot_dates.append(parsed.max())
        snapshot_date = max(snapshot_dates) if snapshot_dates else pd.Timestamp.today().normalize()
        expected_close = pd.to_datetime(opportunities["ExpectedCloseDate"], errors="coerce")
        open_mask = ~opportunities["Stage"].astype(str).str.lower().isin(["won", "lost", "closed"])
        for _, row in opportunities.loc[open_mask & expected_close.lt(snapshot_date)].iterrows():
            add_flag(
                "Warning", "Opportunities", _record_id("Opportunities", row), "ExpectedCloseDate",
                "Open opportunity has a stale expected close date", snapshot_date.date(),
                pd.to_datetime(row["ExpectedCloseDate"]).date(), "Update the close date or close the opportunity",
            )

    flags = pd.DataFrame(rows, columns=CONSISTENCY_FLAG_COLUMNS)
    if flags.empty:
        return flags
    severity_order = pd.Categorical(flags["Severity"], ["Error", "Warning", "Information"], ordered=True)
    return flags.assign(_severity_order=severity_order).sort_values(
        ["_severity_order", "Sheet", "RecordID"]
    ).drop(columns="_severity_order").reset_index(drop=True)


def check_required_sheets_exist(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per required sheet with presence and row count."""

    rows = []
    for sheet in REQUIRED_SHEETS:
        exists = sheet in data
        rows.append(
            {
                "Check": "Required sheet",
                "Sheet": sheet,
                "Field": "",
                "Status": "Pass" if exists else "Fail",
                "Value": len(data[sheet]) if exists else 0,
                "Message": "Sheet found" if exists else "Required sheet missing",
            }
        )
    return pd.DataFrame(rows)


def check_required_columns_exist(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check configured required columns for every available required sheet."""

    rows = []
    for sheet, columns in REQUIRED_COLUMNS.items():
        if sheet not in data:
            continue
        actual = set(data[sheet].columns)
        for column in columns:
            exists = column in actual
            rows.append(
                {
                    "Check": "Required column",
                    "Sheet": sheet,
                    "Field": column,
                    "Status": "Pass" if exists else "Fail",
                    "Value": int(exists),
                    "Message": "Column found" if exists else "Column missing",
                }
            )
    return pd.DataFrame(rows)


def count_missing_values(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count missing values by sheet and column."""

    rows = []
    for sheet, df in data.items():
        missing = df.isna().sum()
        for column, count in missing.items():
            if count:
                rows.append(
                    {
                        "Check": "Missing values",
                        "Sheet": sheet,
                        "Field": column,
                        "Status": "Warn",
                        "Value": int(count),
                        "Message": f"{count} missing value(s)",
                    }
                )
    if not rows:
        rows.append(
            {
                "Check": "Missing values",
                "Sheet": "All",
                "Field": "",
                "Status": "Pass",
                "Value": 0,
                "Message": "No missing values detected",
            }
        )
    return pd.DataFrame(rows)


def count_duplicate_ids(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count duplicate IDs for configured primary key columns."""

    rows = []
    for sheet, columns in ID_COLUMNS.items():
        if sheet not in data:
            continue
        for column in columns:
            if column not in data[sheet].columns:
                continue
            count = int(data[sheet][column].duplicated().sum())
            rows.append(
                {
                    "Check": "Duplicate IDs",
                    "Sheet": sheet,
                    "Field": column,
                    "Status": "Pass" if count == 0 else "Warn",
                    "Value": count,
                    "Message": "No duplicate IDs" if count == 0 else f"{count} duplicate ID row(s)",
                }
            )
    return pd.DataFrame(rows)


def detect_invalid_dates(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Detect values that cannot be parsed as dates in known date columns."""

    rows = []
    for sheet, columns in DATE_COLUMNS.items():
        if sheet not in data:
            continue
        df = data[sheet]
        for column in columns:
            if column not in df.columns:
                continue
            raw = df[column]
            non_empty = raw.notna()
            parsed = pd.to_datetime(raw, errors="coerce")
            invalid = int((non_empty & parsed.isna()).sum())
            rows.append(
                {
                    "Check": "Invalid dates",
                    "Sheet": sheet,
                    "Field": column,
                    "Status": "Pass" if invalid == 0 else "Warn",
                    "Value": invalid,
                    "Message": "Dates parse cleanly" if invalid == 0 else f"{invalid} invalid date value(s)",
                }
            )
    return pd.DataFrame(rows)


def validate_relationships(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check that populated foreign keys resolve to local parent records."""

    rows = []
    for child_sheet, child_column, parent_sheet, parent_column in RELATIONSHIP_CHECKS:
        if child_sheet not in data or parent_sheet not in data:
            continue
        child = data[child_sheet]
        parent = data[parent_sheet]
        if child_column not in child or parent_column not in parent:
            continue
        values = child[child_column].dropna().astype(str)
        parent_values = set(parent[parent_column].dropna().astype(str))
        orphan_count = int((~values.isin(parent_values)).sum())
        rows.append(
            {
                "Check": "Relationship integrity",
                "Sheet": child_sheet,
                "Field": child_column,
                "Status": "Pass" if orphan_count == 0 else "Fail",
                "Value": orphan_count,
                "Message": f"All links resolve to {parent_sheet}" if orphan_count == 0 else f"{orphan_count} orphan link(s) to {parent_sheet}",
            }
        )
    return pd.DataFrame(rows)


def validate_metric_calculations(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Expose the local YAML formula execution report as validation checks."""

    report = data.get("MetricCalculationReport", pd.DataFrame())
    if report.empty:
        return pd.DataFrame(
            [{
                "Check": "Metric calculation",
                "Sheet": "MonthlyPerformance",
                "Field": "",
                "Status": "Warn",
                "Value": 0,
                "Message": "No metric calculation report available",
            }]
        )
    return pd.DataFrame(
        [
            {
                "Check": "Metric calculation",
                "Sheet": "MonthlyPerformance",
                "Field": row.Metric,
                "Status": "Pass" if row.Status == "Recomputed" else "Warn",
                "Value": int(row.Status == "Recomputed"),
                "Message": f"{row.Status}: {row.Message}",
            }
            for row in report.itertuples(index=False)
        ]
    )


def validation_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a single validation summary DataFrame."""

    parts = [
        check_required_sheets_exist(data),
        check_required_columns_exist(data),
        count_missing_values(data),
        count_duplicate_ids(data),
        detect_invalid_dates(data),
        validate_relationships(data),
        validate_metric_calculations(data),
    ]
    return pd.concat(parts, ignore_index=True)


def row_counts_by_sheet(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return row and column counts for each loaded sheet."""

    return pd.DataFrame(
        [
            {"Sheet": sheet, "Rows": len(df), "Columns": len(df.columns)}
            for sheet, df in sorted(data.items())
        ]
    )
