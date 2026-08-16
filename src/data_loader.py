"""Local Excel loading utilities.

LOCAL-ONLY MODEL BOUNDARY:
This module reads only local Excel workbooks. It does not use the internet,
does not call external APIs, and does not send organisation data off-machine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metric_config import DEFAULT_METRICS_PATH, load_metric_config, recompute_derived_metrics
from .settings import get_settings


DEFAULT_EXCEL_PATH = get_settings().workbook_path

REQUIRED_SHEETS = [
    "Salespeople",
    "Customers",
    "ExistingCustomerBilling",
    "Contracts",
    "ContractServices",
    "ContractAuditLog",
    "UpcomingRenewals",
    "CustomerContractSummary",
    "MonthlyPerformance",
    "Opportunities",
    "SynergyReferrals",
    "SynergyMap",
]

OPTIONAL_SHEETS = [
    "CustomerProducts",
    "Activities",
    "Meetings",
    "OpportunityNotes",
    "Projects",
    "OpportunityTickets",
    "TicketTasks",
    "Targets",
    "BillingSummaryByService",
    "FeatureDictionary",
    "ContractFeatureNotes",
    "DataRelationships",
    "MetricDefinitions",
]


def load_sales_data(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    required_sheets: list[str] | None = None,
    include_optional: bool = True,
    metrics_path: str | Path | None = DEFAULT_METRICS_PATH,
) -> dict[str, pd.DataFrame]:
    """Load local Excel sheets into DataFrames.

    The workbook stays on the user's machine and is processed with local Python
    libraries only. No external service or hosted model is used.
    """

    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Local Excel workbook not found: {path}")

    required = required_sheets or REQUIRED_SHEETS
    requested = list(required)
    if include_optional:
        requested.extend(sheet for sheet in OPTIONAL_SHEETS if sheet not in requested)

    with pd.ExcelFile(path) as excel:
        missing = sorted(set(required) - set(excel.sheet_names))
        if missing:
            raise ValueError(f"Workbook is missing required sheets: {', '.join(missing)}")

        available = [sheet for sheet in requested if sheet in excel.sheet_names]
        data = {sheet: pd.read_excel(excel, sheet_name=sheet) for sheet in available}
    metric_file = Path(metrics_path) if metrics_path is not None else None
    if "MonthlyPerformance" in data and metric_file is not None and metric_file.exists():
        data["MonthlyPerformance"], report = recompute_derived_metrics(
            data["MonthlyPerformance"], load_metric_config(metric_file)
        )
        data["MetricCalculationReport"] = report
    return data


def get_sheet_names(excel_path: str | Path = DEFAULT_EXCEL_PATH) -> list[str]:
    """Return the sheet names from the local workbook."""

    return pd.ExcelFile(Path(excel_path)).sheet_names
