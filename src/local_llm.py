"""Optional local LLM support through the Ollama command-line application.

The model runs on this machine. Prompts contain only compact slices calculated
from the local workbook, and no customer or salesperson data is sent online.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import pandas as pd

from .insights import (
    DataAnswer,
    answer_data_question,
    create_customer_whitespace,
    create_performance_profiles,
    create_synergy_summary,
    is_delivery_question,
    performance_comparison_metrics,
)
from .pipeline_forecasting import build_pipeline_forecast


DEFAULT_LOCAL_LLM = os.environ.get("SALES_AI_LOCAL_MODEL", "qwen3:4b-instruct")

BRITISH_ENGLISH_REPLACEMENTS = {
    r"\banalyze\b": "analyse",
    r"\banalyzed\b": "analysed",
    r"\banalyzing\b": "analysing",
    r"\bbehavior\b": "behaviour",
    r"\bbehaviors\b": "behaviours",
    r"\borganization\b": "organisation",
    r"\borganizations\b": "organisations",
    r"\borganize\b": "organise",
    r"\borganized\b": "organised",
    r"\bprioritize\b": "prioritise",
    r"\bprioritized\b": "prioritised",
    r"\bsummarize\b": "summarise",
    r"\bsummarized\b": "summarised",
    r"\bmodeling\b": "modelling",
    r"\boptimization\b": "optimisation",
    r"\boptimized\b": "optimised",
    r"\bcenter\b": "centre",
    r"\bcenters\b": "centres",
    r"\bcatalog\b": "catalogue",
    r"\bcolors\b": "colours",
    r"\bcolor\b": "colour",
}


def _use_british_english(response: str) -> str:
    """Apply a small deterministic language guardrail to local LLM prose."""

    response = re.sub(r"\bGBP\s*([0-9])", r"£\1", response, flags=re.IGNORECASE)
    response = re.sub(r"\bGBP\b", "£", response, flags=re.IGNORECASE)
    for pattern, replacement in BRITISH_ENGLISH_REPLACEMENTS.items():
        def preserve_case(match: re.Match) -> str:
            source = match.group(0)
            if source.isupper():
                return replacement.upper()
            if source[:1].isupper():
                return replacement.capitalize()
            return replacement

        response = re.sub(pattern, preserve_case, response, flags=re.IGNORECASE)
    return response


@dataclass
class LocalLLMStatus:
    available: bool
    model_installed: bool
    model: str
    message: str


def _ollama_path() -> str | None:
    candidates = [
        shutil.which("ollama"),
        "/Applications/Ollama.app/Contents/Resources/ollama",
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def local_llm_status(model: str = DEFAULT_LOCAL_LLM) -> LocalLLMStatus:
    """Report whether Ollama and the configured model are available locally."""

    executable = _ollama_path()
    if not executable:
        return LocalLLMStatus(False, False, model, "Ollama is not installed")
    try:
        result = subprocess.run(
            [executable, "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LocalLLMStatus(True, False, model, f"Ollama is installed but unavailable: {exc}")

    installed = result.returncode == 0 and any(
        line.split()[0].split(":latest")[0] == model.split(":latest")[0]
        for line in result.stdout.splitlines()[1:]
        if line.split()
    )
    if installed:
        return LocalLLMStatus(True, True, model, f"{model} is ready locally")
    message = result.stderr.strip() or f"Model {model} has not been downloaded"
    return LocalLLMStatus(True, False, model, message)


def _csv_section(title: str, frame: pd.DataFrame, columns: list[str], limit: int) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return f"\n[{title}]\nNo matching rows."
    display = frame[available].head(limit).copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].round(3)
    return f"\n[{title}]\n{display.to_csv(index=False)}"


def build_question_context(
    question: str,
    data: dict[str, pd.DataFrame],
    base_answer: DataAnswer | None = None,
) -> str:
    """Build a bounded, relevant context without exposing the entire workbook."""

    lowered = question.lower()
    delivery_question = is_delivery_question(question, data)
    if delivery_question:
        answer = base_answer or answer_data_question(question, data)
        return _csv_section(
            "VERIFIED DELIVERY RESULT",
            answer.table,
            [
                "ProjectID", "OpportunityID", "CustomerName", "Salesperson", "ProjectName",
                "ProjectStage", "ProjectStatus", "DeliveryHealth", "PercentComplete",
                "TargetCompletionDate", "ProjectOverdue", "Blocker", "TicketID", "TicketTitle",
                "TicketStatus", "Priority", "EscalationFlag", "TaskID", "TaskName", "TaskStatus",
                "TaskOwner", "DueDate", "BlockedReason", "OpenTicketCount", "BlockedTicketCount",
                "OpenTaskCount", "BlockedTaskCount", "OverdueTaskCount",
            ],
            12,
        )

    profiles = create_performance_profiles(data)
    sections = [
        _csv_section(
            "TEAM PERFORMANCE",
            profiles,
            [
                "Salesperson", "Segment", "PrimarySpecialism", "total_revenue",
                "expected_revenue", "performance_gap", "performance_score", "win_rate",
                "total_meetings", "opportunities_created", "opportunities_won",
                "cross_sell_opportunities", "cross_sell_revenue", "support_status",
            ],
            12,
        )
    ]

    people = data.get("Salespeople", pd.DataFrame())
    selected_ids: set[str] = set()
    for row in people.itertuples(index=False):
        full_name = str(row.Salesperson).lower()
        first_name = full_name.split()[0]
        first_name_pattern = rf"\b{re.escape(first_name)}(?:['’]s|s)?\b"
        if full_name in lowered or re.search(first_name_pattern, lowered):
            selected_ids.add(str(row.SalespersonID))

    normalized_question = re.sub(r"[^a-z0-9]", "", lowered)
    opportunity_ids = {
        str(value)
        for value in data.get("Opportunities", pd.DataFrame()).get("OpportunityID", pd.Series(dtype=str))
        if re.sub(r"[^a-z0-9]", "", str(value).lower()) in normalized_question
    }
    salesperson_opportunity_ids = set()
    if selected_ids:
        opportunities = data.get("Opportunities", pd.DataFrame())
        if {"SalespersonID", "OpportunityID"}.issubset(opportunities.columns):
            salesperson_opportunity_ids = set(
                opportunities.loc[
                    opportunities["SalespersonID"].astype(str).isin(selected_ids),
                    "OpportunityID",
                ].astype(str)
            )

    if any(
        phrase in lowered
        for phrase in [
            "pipeline", "achievable", "achievability", "year-end forecast",
            "year end forecast", "cover target", "forecast revenue",
        ]
    ):
        forecast = build_pipeline_forecast(data, save_model=False)
        salesperson_forecast = forecast.salesperson_summary.copy()
        opportunity_forecast = forecast.opportunity_forecast.copy()
        suggestions = forecast.suggestions.copy()
        if selected_ids:
            salesperson_forecast = salesperson_forecast[
                salesperson_forecast["SalespersonID"].astype(str).isin(selected_ids)
            ]
            opportunity_forecast = opportunity_forecast[
                opportunity_forecast["SalespersonID"].astype(str).isin(selected_ids)
            ]
            suggestions = suggestions[
                suggestions["SalespersonID"].astype(str).isin(selected_ids)
            ]
        sections.extend(
            [
                _csv_section(
                    "TEAM PIPELINE FORECAST",
                    pd.DataFrame([forecast.team_summary]),
                    list(forecast.team_summary),
                    1,
                ),
                _csv_section(
                    "SALESPERSON PIPELINE ACHIEVABILITY",
                    salesperson_forecast,
                    [
                        "Salesperson", "YTDRevenue", "AnnualTarget", "TargetGap",
                        "OpenPipeline", "WeightedPipelineForecast", "ForecastYearEndRevenue",
                        "ForecastGap", "PipelineCoverage", "WeightedCoverage",
                        "AchievabilityScore", "Achievability", "WaitingResponseOpportunities",
                        "StalledOpportunities",
                    ],
                    15,
                ),
                _csv_section(
                    "PIPELINE COVERAGE ACTIONS",
                    suggestions,
                    ["SalespersonID", "Priority", "Action", "Evidence", "ActionType"],
                    30,
                ),
                _csv_section(
                    "CURRENT OPPORTUNITY FORECAST",
                    opportunity_forecast,
                    [
                        "OpportunityID", "CustomerName", "Salesperson", "Product", "PipelineStage",
                        "PipelineValue", "ExpectedCloseDate", "ForecastCategory", "PipelineRisk",
                        "AdjustedWinProbability", "ForecastRevenue", "WaitingResponseCount",
                        "CriticalMeetingFindings", "DaysInStage", "NextStep", "NextStepDueDate",
                    ],
                    35,
                ),
            ]
        )

    if "meeting" in lowered:
        meetings = data.get("Meetings", pd.DataFrame()).copy()
        if selected_ids and "SalespersonID" in meetings:
            meetings = meetings[meetings["SalespersonID"].astype(str).isin(selected_ids)]
        if opportunity_ids and "OpportunityID" in meetings:
            meetings = meetings[meetings["OpportunityID"].astype(str).isin(opportunity_ids)]
        if "critical" in lowered and "CriticalFindingFlag" in meetings:
            meetings = meetings[meetings["CriticalFindingFlag"].fillna(False).astype(bool)]
        if "MeetingDate" in meetings:
            meetings = meetings.sort_values("MeetingDate", ascending=False)
        sections.append(
            _csv_section(
                "CUSTOMER MEETINGS",
                meetings,
                [
                    "MeetingID", "MeetingDate", "SalespersonID", "CustomerID", "OpportunityID",
                    "CustomerRelationship", "MeetingType", "MeetingSummary", "SalespersonNotes",
                    "CriticalSeverity", "CriticalFinding", "NextAction", "ActionDueDate", "FollowUpStatus",
                ],
                35,
            )
        )

    if any(term in lowered for term in ["note", "response", "unanswered", "escalation", "waiting"]):
        notes = data.get("OpportunityNotes", pd.DataFrame()).copy()
        if selected_ids and "SalespersonID" in notes:
            notes = notes[notes["SalespersonID"].astype(str).isin(selected_ids)]
        if opportunity_ids and "OpportunityID" in notes:
            notes = notes[notes["OpportunityID"].astype(str).isin(opportunity_ids)]
        if any(term in lowered for term in ["waiting", "unanswered", "not responded", "no response"]):
            notes = notes[notes.get("ResponseStatus", "").astype(str).str.lower().eq("waiting response")]
        if "NoteDate" in notes:
            notes = notes.sort_values(["CriticalFindingFlag", "ResponseAgeDays"], ascending=False)
        sections.append(
            _csv_section(
                "OPPORTUNITY NOTES AND RESPONSES",
                notes,
                [
                    "NoteID", "NoteDate", "CustomerID", "OpportunityID", "SalespersonID",
                    "NoteSource", "NoteType", "NoteText", "ResponseRequired", "ResponseStatus",
                    "ResponseAgeDays", "EscalationSeverity", "CriticalFindingFlag", "NextAction",
                ],
                35,
            )
        )

    if any(term in lowered for term in ["project", "ticket", "task", "delivery", "implementation"]):
        for title, sheet, columns in [
            (
                "OPPORTUNITY PROJECTS", "Projects",
                ["ProjectID", "OpportunityID", "CustomerID", "SalespersonID", "ProjectName", "ProjectStage", "ProjectStatus", "DeliveryHealth", "PercentComplete", "TargetCompletionDate", "Blocker"],
            ),
            (
                "OPPORTUNITY TICKETS", "OpportunityTickets",
                ["TicketID", "ProjectID", "OpportunityID", "TicketType", "TicketStatus", "Priority", "DueDate", "EscalationFlag", "LastUpdate"],
            ),
            (
                "TICKET TASKS", "TicketTasks",
                ["TaskID", "TicketID", "ProjectID", "OpportunityID", "TaskName", "TaskStatus", "TaskOwner", "DueDate", "BlockedReason"],
            ),
        ]:
            frame = data.get(sheet, pd.DataFrame()).copy()
            if selected_ids and "SalespersonID" in frame:
                frame = frame[frame["SalespersonID"].astype(str).isin(selected_ids)]
            elif salesperson_opportunity_ids and "OpportunityID" in frame:
                frame = frame[frame["OpportunityID"].astype(str).isin(salesperson_opportunity_ids)]
            if opportunity_ids and "OpportunityID" in frame:
                frame = frame[frame["OpportunityID"].astype(str).isin(opportunity_ids)]
            sections.append(_csv_section(title, frame, columns, 35))

    if any(term in lowered for term in ["renew", "contract", "health check", "rollback", "end date"]):
        renewals = data.get("UpcomingRenewals", pd.DataFrame()).copy()
        if selected_ids:
            renewals = renewals[renewals["AccountOwnerID"].astype(str).isin(selected_ids)]
        sections.append(
            _csv_section(
                "RENEWAL PRIORITIES",
                renewals.sort_values("DaysToRenewal"),
                [
                    "Priority", "ContractID", "CustomerID", "AccountOwnerID", "ContractName",
                    "ContractStatus", "DaysToRenewal", "ContractARR", "RenewalRisk",
                    "EndDateChangeCount", "RollbackCount", "HealthCheckReason", "SuggestedAction",
                ],
                35,
            )
        )

    if any(term in lowered for term in ["customer", "whitespace", "cross-sell", "cross sell", "product", "service"]):
        whitespace = create_customer_whitespace(data)
        if selected_ids:
            selected_names = set(people[people["SalespersonID"].astype(str).isin(selected_ids)]["Salesperson"])
            whitespace = whitespace[whitespace["Account Owner"].isin(selected_names)]
        sections.append(
            _csv_section(
                "CUSTOMER WHITESPACE",
                whitespace,
                [
                    "Customer", "Account Owner", "Current Products", "Missing Products",
                    "Customer Segment", "Current Product MRR", "Whitespace Score",
                    "Estimated Annual Potential", "Recommended Product",
                    "Recommended Specialist", "Estimate Basis", "Next Action",
                ],
                30,
            )
        )

    if any(term in lowered for term in ["synergy", "referral", "partner", "collaborat"]):
        synergy = create_synergy_summary(data)
        sections.append(
            _csv_section(
                "REFERRAL PARTNERSHIPS",
                synergy,
                [
                    "From Salesperson", "To Salesperson", "Referrals", "Accepted",
                    "Converted", "Conversion Rate", "SynergyType", "SynergyStrength",
                ],
                25,
            )
        )

    performance_metric_question = (
        any(term in lowered for term in ["metric", "measure", "factor", "component"])
        and any(term in lowered for term in ["performance", "compare", "comparison", "score", "rank"])
    )
    if performance_metric_question:
        comparison = performance_comparison_metrics()
        sections.append(
            _csv_section(
                "PERFORMANCE COMPARISON LOGIC",
                comparison,
                ["Comparison Layer", "Component", "Weight", "Metrics Used", "Method"],
                10,
            )
        )
    elif any(term in lowered for term in ["metric", "define", "definition", "formula", "calculate", "what is"]):
        definitions = data.get("MetricDefinitions", pd.DataFrame()).copy()
        if not definitions.empty:
            normalized_question = re.sub(r"[^a-z0-9]", "", lowered)
            matched = definitions[
                definitions["MetricName"].astype(str).map(
                    lambda value: re.sub(r"[^a-z0-9]", "", value.lower()) in normalized_question
                )
            ]
            if matched.empty:
                matched = definitions
            sections.append(
                _csv_section(
                    "CANONICAL METRIC DEFINITIONS",
                    matched,
                    [
                        "MetricName", "Category", "Definition", "FormulaOrLogic",
                        "PrimarySource", "Direction", "Availability", "ModelUse", "LeakageRisk",
                    ],
                    30,
                )
            )

    sheet_summary = pd.DataFrame(
        [
            {"Sheet": name, "Rows": len(frame), "Columns": ", ".join(map(str, frame.columns))}
            for name, frame in data.items()
        ]
    )
    sections.append(_csv_section("AVAILABLE LOCAL DATA", sheet_summary, list(sheet_summary.columns), 25))
    return "\n".join(sections)


def answer_with_local_llm(
    question: str,
    data: dict[str, pd.DataFrame],
    model: str = DEFAULT_LOCAL_LLM,
    timeout: int = 120,
) -> DataAnswer:
    """Use a locally installed model to explain local, deterministic data results."""

    base_answer = answer_data_question(question, data)
    lowered = question.lower()
    if base_answer.source.startswith("Manager clarification required"):
        return base_answer

    delivery_question = is_delivery_question(question, data)
    if delivery_question:
        return DataAnswer(
            f"Verified local result: {base_answer.title}",
            base_answer.summary,
            base_answer.table,
            f"{base_answer.source}; deterministic result retained in Local LLM mode",
            base_answer.interpretation,
            base_answer.visual,
        )

    last_week_meeting_question = (
        "meeting" in lowered
        and any(phrase in lowered for phrase in ["last week", "previous week", "prior week"])
    )
    if last_week_meeting_question:
        return DataAnswer(
            f"Verified local result: {base_answer.title}",
            base_answer.summary,
            base_answer.table,
            f"{base_answer.source}; deterministic result retained in Local LLM mode",
            base_answer.interpretation,
            base_answer.visual,
        )

    if base_answer.source.startswith("Validated query plan over Meetings"):
        return DataAnswer(
            f"Verified local result: {base_answer.title}",
            base_answer.summary,
            base_answer.table,
            f"{base_answer.source}; deterministic result retained in Local LLM mode",
            base_answer.interpretation,
            base_answer.visual,
        )

    if base_answer.title in {
        "Top performers", "Bottom performers", "Coaching priorities", "Upcoming opportunities",
        "Upcoming cross-sell opportunities", "Upcoming upsell opportunities",
        "Upcoming renewal opportunities", "Upcoming new customer opportunities",
    }:
        return DataAnswer(
            f"Verified local result: {base_answer.title}",
            base_answer.summary,
            base_answer.table,
            f"{base_answer.source}; deterministic result retained in Local LLM mode",
            base_answer.interpretation,
            base_answer.visual,
        )

    pipeline_question = any(
        phrase in lowered
        for phrase in [
            "pipeline", "achievable", "achievability", "year-end forecast",
            "year end forecast", "cover target", "forecast revenue",
        ]
    )
    if pipeline_question:
        summary = base_answer.summary
        if "Action" in base_answer.table:
            actions = base_answer.table["Action"].dropna().astype(str).head(3).tolist()
            if actions:
                summary += " Priority actions: " + " ".join(
                    f"{index}. {action}." for index, action in enumerate(actions, start=1)
                )
        return DataAnswer(
            f"Verified local forecast: {base_answer.title}",
            summary,
            base_answer.table,
            f"{base_answer.source}; deterministic forecast retained in Local LLM mode",
            base_answer.interpretation,
            base_answer.visual,
        )
    status = local_llm_status(model)
    if not status.available or not status.model_installed:
        return DataAnswer(
            f"Workbook answer: {base_answer.title}",
            f"{base_answer.summary} Local LLM note: {status.message}.",
            base_answer.table,
            f"{base_answer.source}; local workbook fallback",
            base_answer.interpretation,
            base_answer.visual,
        )

    context = build_question_context(question, data, base_answer)
    result_limit = 12 if delivery_question else 25
    deterministic_table = _csv_section(
        "DETERMINISTIC RESULT TABLE",
        base_answer.table,
        list(base_answer.table.columns),
        result_limit,
    )
    prompt = f"""/no_think
You are the local Sales Performance AI for a managed service provider.
Answer only from the LOCAL DATA CONTEXT below. Never invent a value, customer,
contract, cause, or recommendation. If the context is insufficient, state which
local field or sheet is needed. Treat estimates as estimates and correlations as
associations. Give the direct answer first, then at most three concise actions.
If the request is ambiguous or the context lacks the required scope, do not
guess. Ask one concise clarification question and state which local identifier,
field, period, comparison, or objective the manager should provide.
For peer comparisons, use all matching Segment records unless the user names a
specific peer. Treat year-to-date as a complete period. Do not append a
clarification when the salesperson, period, peer group, and objective are
already supplied.
If verified rows directly answer the request, finish after the answer or
actions and do not add a clarification.
Use British English spelling and wording. Show pounds with the £ symbol, never
the GBP abbreviation. Keep the response under 180 words. Do not use markdown tables.

USER QUESTION:
{question}

DETERMINISTIC WORKBOOK RESULT:
{base_answer.summary}
{deterministic_table}

LOCAL DATA CONTEXT:
{context}
"""
    try:
        result = subprocess.run(
            [
                _ollama_path(),
                "run",
                model,
                "--nowordwrap",
                "--hidethinking",
                "--think=false",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DataAnswer(
            f"Workbook answer: {base_answer.title}",
            f"{base_answer.summary} The optional local explanation was unavailable: {exc}.",
            base_answer.table,
            f"{base_answer.source}; local workbook fallback",
            base_answer.interpretation,
            base_answer.visual,
        )

    response = result.stdout.strip()
    if result.returncode or not response:
        return DataAnswer(
            f"Workbook answer: {base_answer.title}",
            base_answer.summary,
            base_answer.table,
            f"{base_answer.source}; local LLM returned no usable response",
            base_answer.interpretation,
            base_answer.visual,
        )
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", response)
    response = _use_british_english(response.replace("\r", "").strip())
    resolved_whitespace_ranking = (
        base_answer.title == "Customer whitespace opportunities"
        and not base_answer.table.empty
        and any(term in lowered for term in ["top", "highest", "largest", "lowest"])
    )
    if resolved_whitespace_ranking:
        response = re.sub(
            r"\n+\s*Clarification:.*$", "", response, flags=re.IGNORECASE | re.DOTALL
        ).strip()
    if not response:
        return DataAnswer(
            f"Workbook answer: {base_answer.title}",
            base_answer.summary,
            base_answer.table,
            f"{base_answer.source}; local LLM response was empty after cleanup",
            base_answer.interpretation,
            base_answer.visual,
        )
    return DataAnswer(
        f"Local LLM: {base_answer.title}",
        response,
        base_answer.table,
        f"Local workbook plus {model} running in this private runtime",
        base_answer.interpretation,
        base_answer.visual,
    )
