"""Build validated, compositional plans for local workbook questions."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .settings import get_settings
from .time_periods import parse_meeting_period


@dataclass
class QueryPlan:
    question: str
    domain: str = "unknown"
    operation: str = "list"
    filters: dict[str, Any] = field(default_factory=dict)
    time_scope: str | None = None
    sort_field: str | None = None
    sort_direction: str = "ascending"
    limit: int = 25
    confidence: float = 0.0
    ambiguities: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.ambiguities)

    def interpretation(self, snapshot: pd.Timestamp | None = None) -> dict[str, str]:
        """Return manager-readable labels for the plan executed by pandas."""

        labels: dict[str, str] = {
            "Domain": self.domain.replace("_", " ").title(),
            "Action": self.operation.replace("_", " ").title(),
        }
        display_names = {
            "SalespersonName": "Salesperson",
            "CustomerName": "Customer",
            "OpportunityType": "Opportunity type",
            "Stage": "Status",
            "MeetingTiming": "Meeting timing",
        }
        for key, value in self.filters.items():
            if key.endswith("ID") or key in {
                "CloseDate",
                "ExpectedCloseDate",
                "PeriodStart",
                "PeriodEnd",
                "PeriodLabel",
                "DaysBack",
                "DaysForward",
            }:
                continue
            labels[display_names.get(key, key.replace("_", " ").title())] = str(value)
        if self.time_scope == "upcoming":
            if self.domain == "meetings":
                labels["Period"] = (
                    f"After {snapshot:%d %b %Y}"
                    if snapshot is not None
                    else "After workbook snapshot"
                )
            else:
                labels["Expected close"] = (
                    f"{snapshot:%d %b %Y} onwards"
                    if snapshot is not None
                    else "Workbook snapshot onwards"
                )
        elif self.time_scope == "last_complete_week":
            labels["Period"] = "Latest complete workbook week"
        elif self.time_scope == "last_complete_month":
            labels["Period"] = "Previous complete workbook month"
        elif self.time_scope == "explicit_month":
            labels["Period"] = str(self.filters.get("PeriodLabel", "Selected month"))
        elif self.time_scope in {"explicit_day", "explicit_range"}:
            labels["Period"] = str(self.filters.get("PeriodLabel", "Selected period"))
        elif self.time_scope == "last_n_days":
            labels["Period"] = f"Last {int(self.filters.get('DaysBack', 1)):,} workbook days"
        elif self.time_scope == "next_n_days":
            labels["Period"] = f"Next {int(self.filters.get('DaysForward', 1)):,} workbook days"
        if self.sort_field:
            direction = "earliest first" if self.sort_direction == "ascending" else "latest first"
            sort_names = {
                "ExpectedCloseDate": "Expected close date",
                "MeetingDate": "Meeting date",
                "PipelineValue": "Pipeline value",
            }
            labels["Sort"] = f"{sort_names.get(self.sort_field, self.sort_field.replace('_', ' '))} ({direction})"
        return labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "domain": self.domain,
            "operation": self.operation,
            "filters": self.filters,
            "time_scope": self.time_scope,
            "sort_field": self.sort_field,
            "sort_direction": self.sort_direction,
            "limit": self.limit,
            "confidence": self.confidence,
            "ambiguities": self.ambiguities,
        }


def _normalise(text: str) -> str:
    value = text.casefold().replace("’", "'")
    value = re.sub(r"[-_/]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains(text: str, phrase: str) -> bool:
    normalised = _normalise(phrase)
    return bool(re.search(rf"\b{re.escape(normalised)}\b", text))


@lru_cache(maxsize=4)
def load_query_vocabulary(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or get_settings().query_intents_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Local query intent configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if "domains" not in config:
        raise ValueError("Query intent configuration is missing domains")
    return config


def _matched_label(text: str, choices: dict[str, list[str]]) -> str | None:
    matches = [
        (label, alias)
        for label, aliases in choices.items()
        for alias in aliases
        if _contains(text, alias)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(_normalise(item[1])))[0]


def _domain_scores(text: str, vocabulary: dict[str, Any]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for domain, definition in vocabulary.get("domains", {}).items():
        aliases = definition.get("aliases", [])
        score = sum(max(1, len(_normalise(alias).split())) for alias in aliases if _contains(text, alias))
        if score:
            scores[domain] = score
    return scores


def _extract_salesperson(text: str, data: dict[str, pd.DataFrame]) -> tuple[dict[str, str], list[str]]:
    people = data.get("Salespeople", pd.DataFrame())
    if people.empty or not {"SalespersonID", "Salesperson"}.issubset(people.columns):
        return {}, []
    exact = people[
        people.apply(
            lambda row: _contains(text, str(row["SalespersonID"]))
            or _contains(text, str(row["Salesperson"])),
            axis=1,
        )
    ]
    matches = exact
    if exact.empty:
        matches = people[
            people["Salesperson"].astype(str).map(
                lambda name: bool(
                    re.search(
                        rf"\b{re.escape(_normalise(name).split()[0])}(?:'s|s)?\b", text
                    )
                )
            )
        ]
    if len(matches) == 1:
        row = matches.iloc[0]
        return {
            "SalespersonID": str(row["SalespersonID"]),
            "SalespersonName": str(row["Salesperson"]),
        }, []
    if len(matches) > 1:
        names = ", ".join(matches["Salesperson"].astype(str))
        return {}, [f"The salesperson name matches more than one record: {names}."]
    return {}, []


def _extract_customer(text: str, data: dict[str, pd.DataFrame]) -> dict[str, str]:
    customers = data.get("Customers", pd.DataFrame())
    if customers.empty or not {"CustomerID", "CustomerName"}.issubset(customers.columns):
        return {}
    matches = customers[
        customers.apply(
            lambda row: _contains(text, str(row["CustomerID"]))
            or _contains(text, str(row["CustomerName"])),
            axis=1,
        )
    ]
    if len(matches) != 1:
        return {}
    row = matches.iloc[0]
    return {"CustomerID": str(row["CustomerID"]), "CustomerName": str(row["CustomerName"])}


def _extract_limit(text: str, default: int = 25) -> int:
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    token_pattern = r"\d+|" + "|".join(words)
    match = re.search(
        rf"\b(?:top|bottom|first|next|show)\s+({token_pattern})\b", text
    )
    if not match:
        return default
    token = match.group(1)
    value = int(token) if token.isdigit() else words[token]
    return max(1, min(value, 100))


def build_query_plan(
    question: str,
    data: dict[str, pd.DataFrame],
    vocabulary: dict[str, Any] | None = None,
) -> QueryPlan:
    """Parse independent query slots without accessing any external service."""

    config = vocabulary or load_query_vocabulary()
    text = _normalise(question)
    scores = _domain_scores(text, config)
    domain = max(scores, key=scores.get) if scores else "unknown"
    operation = _matched_label(text, config.get("operations", {})) or "list"
    time_scope = _matched_label(text, config.get("time_scopes", {}))
    filters, ambiguities = _extract_salesperson(text, data)
    filters.update(_extract_customer(text, data))

    if domain == "meetings":
        if time_scope == "upcoming":
            filters["MeetingTiming"] = "Upcoming"
        parsed_period = parse_meeting_period(question)
        if parsed_period is not None:
            time_scope = parsed_period.scope
            filters.update(parsed_period.filters)
            if parsed_period.scope == "next_n_days":
                filters["MeetingTiming"] = "Upcoming"
            if parsed_period.ambiguity:
                ambiguities.append(parsed_period.ambiguity)

    opportunity_type = _matched_label(text, config.get("opportunity_types", {}))
    stage = _matched_label(text, config.get("opportunity_stages", {}))
    if opportunity_type:
        filters["OpportunityType"] = opportunity_type
        if domain == "unknown":
            ambiguities.append(
                f"{opportunity_type} can mean recorded pipeline opportunities or customer product whitespace. "
                "Specify 'open opportunities' or 'customer whitespace'."
            )
    if stage and domain == "opportunities":
        filters["Stage"] = stage
    if domain == "opportunities" and time_scope == "upcoming":
        filters["Stage"] = "Open"
        filters["CloseDate"] = "Missing"
        filters["ExpectedCloseDate"] = "OnOrAfterSnapshot"

    requested_sort = _matched_label(text, config.get("sort_terms", {}))
    sort_direction = requested_sort or "ascending"
    sort_field = "ExpectedCloseDate" if domain == "opportunities" and time_scope == "upcoming" else None
    if domain == "meetings" and time_scope in {
        "last_complete_week",
        "last_complete_month",
        "last_n_days",
        "next_n_days",
        "explicit_day",
        "explicit_range",
        "explicit_month",
        "upcoming",
    }:
        sort_field = "MeetingDate"
        if requested_sort is None:
            sort_direction = (
                "ascending"
                if filters.get("MeetingTiming") == "Upcoming" or time_scope == "upcoming"
                else "descending"
            )
    if domain == "opportunities" and operation == "rank" and "top" in text:
        sort_field, sort_direction = "PipelineValue", "descending"

    confidence = min(
        1.0,
        0.2 + (0.12 * scores.get(domain, 0)) + (0.08 * len(filters)) + (0.1 if time_scope else 0),
    )
    return QueryPlan(
        question=question,
        domain=domain,
        operation=operation,
        filters=filters,
        time_scope=time_scope,
        sort_field=sort_field,
        sort_direction=sort_direction,
        limit=_extract_limit(text),
        confidence=round(confidence, 2),
        ambiguities=ambiguities,
    )
