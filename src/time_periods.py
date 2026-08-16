"""Parse manager-entered meeting periods without external services."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

import pandas as pd


MONTH_NUMBERS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fourteen": 14,
    "thirty": 30,
    "sixty": 60,
    "ninety": 90,
}

_MONTH_PATTERN = "|".join(
    re.escape(name) for name in sorted(MONTH_NUMBERS, key=len, reverse=True)
)


@dataclass(frozen=True)
class ParsedPeriod:
    scope: str
    filters: dict[str, Any]
    ambiguity: str | None = None


def _timestamp(year: int, month: int, day: int) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def _date_matches(question: str) -> list[tuple[int, int, pd.Timestamp]]:
    """Return non-overlapping explicit dates in their original text order."""

    text = question.casefold().replace("’", "'")
    extractors: list[tuple[str, Callable[[re.Match[str]], pd.Timestamp | None]]] = [
        (
            r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
            lambda match: _timestamp(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            ),
        ),
        (
            r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b",
            lambda match: _timestamp(
                int(match.group(3)), int(match.group(2)), int(match.group(1))
            ),
        ),
        (
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\s+(20\d{{2}})\b",
            lambda match: _timestamp(
                int(match.group(3)), MONTH_NUMBERS[match.group(2)], int(match.group(1))
            ),
        ),
        (
            rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(20\d{{2}})\b",
            lambda match: _timestamp(
                int(match.group(3)), MONTH_NUMBERS[match.group(1)], int(match.group(2))
            ),
        ),
    ]
    candidates: list[tuple[int, int, pd.Timestamp]] = []
    for pattern, parser in extractors:
        for match in re.finditer(pattern, text):
            value = parser(match)
            if value is not None:
                candidates.append((match.start(), match.end(), value))

    selected: list[tuple[int, int, pd.Timestamp]] = []
    for candidate in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(candidate[0] < end and candidate[1] > start for start, end, _ in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item[0])


def _shared_year_range(question: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Parse ranges where the year is written only once at the end."""

    text = question.casefold().replace("’", "'")
    full = re.search(
        rf"\b(?:from|between)\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})"
        rf"(?:\s+(20\d{{2}}))?\s+(?:to|and|until|through)\s+"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\s+(20\d{{2}})\b",
        text,
    )
    if full:
        end_year = int(full.group(6))
        start = _timestamp(
            int(full.group(3) or end_year),
            MONTH_NUMBERS[full.group(2)],
            int(full.group(1)),
        )
        end = _timestamp(end_year, MONTH_NUMBERS[full.group(5)], int(full.group(4)))
        if start is not None and end is not None:
            return start, end

    same_month = re.search(
        rf"\b(?:from|between)\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?:to|and|until|through)\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"({_MONTH_PATTERN})\s+(20\d{{2}})\b",
        text,
    )
    if not same_month:
        return None
    year = int(same_month.group(4))
    month = MONTH_NUMBERS[same_month.group(3)]
    start = _timestamp(year, month, int(same_month.group(1)))
    end = _timestamp(year, month, int(same_month.group(2)))
    return (start, end) if start is not None and end is not None else None


def _period_filters(start: pd.Timestamp, end: pd.Timestamp, label: str) -> dict[str, str]:
    return {
        "PeriodStart": start.strftime("%Y-%m-%d"),
        "PeriodEnd": end.strftime("%Y-%m-%d"),
        "PeriodLabel": label,
    }


def parse_meeting_period(question: str) -> ParsedPeriod | None:
    """Parse explicit, rolling, or calendar periods from a meeting question."""

    shared_range = _shared_year_range(question)
    explicit_dates = _date_matches(question)
    if shared_range is not None:
        start, end = shared_range
    elif len(explicit_dates) >= 2:
        start, end = explicit_dates[0][2], explicit_dates[1][2]
    else:
        start = end = None

    if start is not None and end is not None:
        if end < start:
            return ParsedPeriod(
                "explicit_range",
                {},
                "The requested meeting period ends before it starts. Provide the start date first, for example 'from 1 July 2026 to 15 July 2026'.",
            )
        label = f"{start:%d %b %Y} to {end:%d %b %Y}"
        return ParsedPeriod("explicit_range", _period_filters(start, end, label))

    if len(explicit_dates) == 1:
        day = explicit_dates[0][2]
        label = day.strftime("%d %b %Y")
        return ParsedPeriod("explicit_day", _period_filters(day, day, label))

    normalised = re.sub(r"[-_/]+", " ", question.casefold())
    normalised = re.sub(r"\s+", " ", normalised).strip()
    number_pattern = r"\d{1,4}|" + "|".join(NUMBER_WORDS)
    rolling = re.search(
        rf"\b(?:last|past|previous|prior)\s+({number_pattern})\s+days?\b",
        normalised,
    )
    if rolling:
        token = rolling.group(1)
        days = int(token) if token.isdigit() else NUMBER_WORDS[token]
        if days < 1 or days > 3660:
            return ParsedPeriod(
                "last_n_days",
                {},
                "Choose a rolling meeting period between 1 and 3,660 days.",
            )
        return ParsedPeriod(
            "last_n_days",
            {"DaysBack": days, "PeriodLabel": f"last {days:,} days"},
        )

    future = re.search(
        rf"\b(?:next|coming|following)\s+({number_pattern})\s+days?\b",
        normalised,
    )
    if future:
        token = future.group(1)
        days = int(token) if token.isdigit() else NUMBER_WORDS[token]
        if days < 1 or days > 3660:
            return ParsedPeriod(
                "next_n_days",
                {},
                "Choose an upcoming meeting period between 1 and 3,660 days.",
            )
        return ParsedPeriod(
            "next_n_days",
            {"DaysForward": days, "PeriodLabel": f"the next {days:,} days"},
        )

    month = re.search(
        rf"\b(?:(?:in|during|for)\s+)?(?:the\s+)?(?:month\s+of\s+)?"
        rf"({_MONTH_PATTERN})\s+(20\d{{2}})\b",
        normalised,
    )
    if not month:
        return None
    period_start = pd.Timestamp(
        year=int(month.group(2)), month=MONTH_NUMBERS[month.group(1)], day=1
    )
    period_end = period_start + pd.offsets.MonthEnd(0)
    return ParsedPeriod(
        "explicit_month",
        _period_filters(period_start, period_end, period_start.strftime("%B %Y")),
    )
