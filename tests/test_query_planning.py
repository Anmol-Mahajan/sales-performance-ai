"""Tests for compositional, local-only intent planning and execution."""

from __future__ import annotations

import unittest

from src.data_loader import load_sales_data
from src.insights import answer_data_question
from src.query_execution import execute_query_plan
from src.query_planning import build_query_plan


class QueryPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_sales_data()

    def test_cross_sell_question_builds_compositional_plan(self) -> None:
        plan = build_query_plan(
            "What are the upcoming cross sell opportunities for Chloe Singh?", self.data
        )
        self.assertEqual(plan.domain, "opportunities")
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.filters["SalespersonName"], "Chloe Singh")
        self.assertEqual(plan.filters["OpportunityType"], "Cross-sell")
        self.assertEqual(plan.filters["Stage"], "Open")
        self.assertEqual(plan.filters["CloseDate"], "Missing")
        self.assertEqual(plan.filters["ExpectedCloseDate"], "OnOrAfterSnapshot")
        self.assertEqual(plan.time_scope, "upcoming")
        self.assertEqual(plan.sort_field, "ExpectedCloseDate")

    def test_synonyms_produce_the_same_cross_sell_records(self) -> None:
        plan = build_query_plan(
            "List Chloe Singh's future expansion sale deals", self.data
        )
        result = execute_query_plan(plan, self.data)
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Upcoming cross-sell opportunities")
        self.assertEqual(len(result.table), 3)
        self.assertTrue(result.table["OpportunityType"].eq("Cross-sell").all())

    def test_result_limit_is_part_of_the_plan(self) -> None:
        plan = build_query_plan(
            "Show the next two upcoming opportunities for Chloe Singh", self.data
        )
        result = execute_query_plan(plan, self.data)
        self.assertEqual(plan.limit, 2)
        self.assertEqual(len(result.table), 2)

    def test_ambiguous_cross_sell_scope_requests_clarification(self) -> None:
        answer = answer_data_question("Show Chloe's cross sell", self.data)
        self.assertEqual(answer.title, "Please clarify the request")
        self.assertIn("pipeline opportunities or customer product whitespace", answer.table.iloc[0]["Why It Matters"])

    def test_answer_exposes_the_executed_filters(self) -> None:
        answer = answer_data_question(
            "What are the upcoming cross-sell opportunities for Chloe Singh?", self.data
        )
        self.assertEqual(answer.interpretation["Domain"], "Opportunities")
        self.assertEqual(answer.interpretation["Salesperson"], "Chloe Singh")
        self.assertEqual(answer.interpretation["Opportunity type"], "Cross-sell")
        self.assertEqual(answer.interpretation["Status"], "Open")
        self.assertIn("onwards", answer.interpretation["Expected close"])

    def test_team_suggestion_uses_full_scope_without_a_person(self) -> None:
        answer = answer_data_question(
            "What are the upcoming cross-sell opportunities?", self.data
        )
        self.assertEqual(answer.title, "Upcoming cross-sell opportunities")
        self.assertIn("The team has", answer.summary)
        self.assertTrue(answer.table["OpportunityType"].eq("Cross-sell").all())

    def test_person_last_week_meetings_builds_scoped_plan(self) -> None:
        plan = build_query_plan("What are the last week meetings by Alice?", self.data)
        self.assertEqual(plan.domain, "meetings")
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.time_scope, "last_complete_week")
        self.assertEqual(plan.filters["SalespersonName"], "Alice Brown")
        self.assertEqual(plan.sort_field, "MeetingDate")
        self.assertEqual(plan.sort_direction, "descending")

    def test_most_meetings_remains_a_ranking_plan(self) -> None:
        plan = build_query_plan("Who held the most meetings last week?", self.data)
        self.assertEqual(plan.domain, "meetings")
        self.assertEqual(plan.operation, "rank")

    def test_named_month_builds_a_calendar_month_meeting_plan(self) -> None:
        plan = build_query_plan(
            "Who held the most meetings in the month of July 2026?", self.data
        )
        self.assertEqual(plan.domain, "meetings")
        self.assertEqual(plan.operation, "rank")
        self.assertEqual(plan.time_scope, "explicit_month")
        self.assertEqual(plan.filters["PeriodStart"], "2026-07-01")
        self.assertEqual(plan.filters["PeriodEnd"], "2026-07-31")
        self.assertEqual(plan.filters["PeriodLabel"], "July 2026")

    def test_last_month_builds_a_relative_month_meeting_plan(self) -> None:
        plan = build_query_plan("Who held the most meetings last month?", self.data)
        self.assertEqual(plan.domain, "meetings")
        self.assertEqual(plan.operation, "rank")
        self.assertEqual(plan.time_scope, "last_complete_month")
        self.assertEqual(plan.sort_field, "MeetingDate")

    def test_specific_day_builds_a_single_day_meeting_plan(self) -> None:
        plan = build_query_plan(
            "Who held the most meetings on 6 August 2026?", self.data
        )
        self.assertEqual(plan.time_scope, "explicit_day")
        self.assertEqual(plan.filters["PeriodStart"], "2026-08-06")
        self.assertEqual(plan.filters["PeriodEnd"], "2026-08-06")

    def test_shared_year_date_range_builds_an_explicit_period(self) -> None:
        plan = build_query_plan(
            "Who held the most meetings between 1 July and 15 July 2026?",
            self.data,
        )
        self.assertEqual(plan.time_scope, "explicit_range")
        self.assertEqual(plan.filters["PeriodStart"], "2026-07-01")
        self.assertEqual(plan.filters["PeriodEnd"], "2026-07-15")

    def test_last_number_of_days_is_stored_as_a_rolling_period(self) -> None:
        plan = build_query_plan(
            "Who held the most meetings in the last 14 days?", self.data
        )
        self.assertEqual(plan.time_scope, "last_n_days")
        self.assertEqual(plan.filters["DaysBack"], 14)

    def test_upcoming_named_month_keeps_timing_and_period_separate(self) -> None:
        plan = build_query_plan(
            "What are Alice's upcoming meetings in September 2026?", self.data
        )
        self.assertEqual(plan.time_scope, "explicit_month")
        self.assertEqual(plan.filters["MeetingTiming"], "Upcoming")
        self.assertEqual(plan.filters["SalespersonName"], "Alice Brown")
        self.assertEqual(plan.filters["PeriodStart"], "2026-09-01")
        self.assertEqual(plan.sort_direction, "ascending")

    def test_next_number_of_days_implies_upcoming_meetings(self) -> None:
        plan = build_query_plan(
            "What meetings are scheduled in the next 30 days?", self.data
        )
        self.assertEqual(plan.time_scope, "next_n_days")
        self.assertEqual(plan.filters["DaysForward"], 30)
        self.assertEqual(plan.filters["MeetingTiming"], "Upcoming")
        self.assertEqual(plan.sort_direction, "ascending")

    def test_reversed_date_range_requests_clarification(self) -> None:
        plan = build_query_plan(
            "Show meetings from 15 July 2026 to 1 July 2026", self.data
        )
        self.assertTrue(plan.needs_clarification)
        self.assertIn("ends before it starts", plan.ambiguities[0])

    def test_suggested_questions_do_not_embed_people_or_record_ids(self) -> None:
        from app.manager_portal import SUGGESTED_QUESTIONS

        suggestions = " ".join(SUGGESTED_QUESTIONS).casefold()
        for name in self.data["Salespeople"]["Salesperson"].astype(str):
            self.assertNotIn(name.casefold(), suggestions)
        self.assertNotRegex(suggestions, r"\b(?:prj|opp|tkt|task)\d+\b")


if __name__ == "__main__":
    unittest.main()
