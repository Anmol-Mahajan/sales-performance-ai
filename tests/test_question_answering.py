"""Focused tests for project retrieval and manager clarification behaviour."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.data_loader import load_sales_data
from src.insights import answer_data_question, meeting_records_between
from src.local_llm import _use_british_english, answer_with_local_llm, build_question_context
from src.model_training import analyse_revenue_models


class QuestionAnsweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_sales_data()
        cls.revenue_analysis = analyse_revenue_models(cls.data, save_best=False)

    def test_revenue_model_selects_best_validated_feature_set(self) -> None:
        comparison = self.revenue_analysis.feature_set_comparison
        selected = comparison[comparison["Selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected.iloc[0]["RMSE"]), float(comparison["RMSE"].min()))
        self.assertEqual(
            selected.iloc[0]["Feature Set"], self.revenue_analysis.selected_feature_set
        )

    def test_revenue_model_uses_final_six_month_holdout(self) -> None:
        comparison = self.revenue_analysis.selected_training.model_comparison
        self.assertTrue(comparison["Notes"].str.contains("final 6 month").all())
        self.assertTrue(comparison["Holdout Months"].eq(6).all())
        self.assertTrue(
            (comparison["Training Rows"] + comparison["Test Rows"]).eq(comparison["Rows"]).all()
        )

    def test_revenue_predictors_exclude_same_month_outcomes(self) -> None:
        frame = self.revenue_analysis.selected_training.training_frame
        unlagged_outcomes = {
            "Meetings", "CustomerReachouts", "OpportunitiesCreated", "OpportunitiesWon",
            "GrossProfit", "CrossSellRevenue", "RetentionRate",
        }
        self.assertFalse(unlagged_outcomes.intersection(frame.columns))
        self.assertIn("Revenue", frame.columns)
        self.assertTrue(any(column.startswith("PreviousRevenue") for column in frame.columns))

    def test_project_lookup_starts_from_projects(self) -> None:
        answer = answer_data_question("Show project PRJ0001", self.data)
        self.assertEqual(answer.title, "Project delivery status")
        self.assertEqual(answer.table["ProjectID"].tolist(), ["PRJ0001"])
        self.assertIn("1 matching project", answer.summary)

    def test_possessive_first_name_returns_most_recent_project(self) -> None:
        answer = answer_data_question(
            "What is the status of Emmas most recent project?", self.data
        )
        emma_id = self.data["Salespeople"].loc[
            self.data["Salespeople"]["Salesperson"].eq("Emma Patel"), "SalespersonID"
        ].iloc[0]
        expected = self.data["Projects"].loc[
            self.data["Projects"]["SalespersonID"].eq(emma_id)
        ].assign(
            _update=lambda frame: pd.to_datetime(frame["LastUpdatedDate"], errors="coerce"),
            _start=lambda frame: pd.to_datetime(frame["StartDate"], errors="coerce"),
        ).sort_values(
            ["_update", "_start", "ProjectID"], ascending=[False, False, False]
        ).iloc[0]
        self.assertEqual(len(answer.table), 1)
        self.assertEqual(answer.table.iloc[0]["ProjectID"], expected["ProjectID"])
        self.assertIn("Emma Patel's most recently updated project", answer.summary)

    def test_duplicate_first_name_requests_full_name(self) -> None:
        data = {name: frame.copy() for name, frame in self.data.items()}
        duplicate = data["Salespeople"].loc[
            data["Salespeople"]["Salesperson"].eq("Emma Patel")
        ].iloc[0].copy()
        duplicate["SalespersonID"] = "SP099"
        duplicate["Salesperson"] = "Emma Wilson"
        data["Salespeople"] = pd.concat(
            [data["Salespeople"], duplicate.to_frame().T], ignore_index=True
        )
        answer = answer_data_question("How is Emma performing?", data)
        self.assertEqual(answer.title, "Please clarify the salesperson")
        self.assertEqual(set(answer.table["Salesperson"]), {"Emma Patel", "Emma Wilson"})
        self.assertIn("Example Request", answer.table)

    def test_unlinked_opportunity_is_explained_cleanly(self) -> None:
        answer = answer_data_question("What is the project information for OPP0001?", self.data)
        self.assertEqual(answer.title, "No linked project")
        self.assertIn("No project is linked to OPP0001", answer.summary)
        self.assertEqual(len(answer.table), 1)

    def test_active_project_question_filters_status(self) -> None:
        answer = answer_data_question("What projects are in progress?", self.data)
        self.assertFalse(answer.table.empty)
        self.assertTrue(answer.table["ProjectStatus"].eq("Active").all())

    def test_blocked_projects_do_not_mean_blocked_tasks_only(self) -> None:
        answer = answer_data_question("Which projects are blocked?", self.data)
        native_block = (
            answer.table["ProjectStatus"].eq("On Hold")
            | answer.table["DeliveryHealth"].eq("Red")
            | answer.table["Blocker"].fillna("").astype(str).str.strip().ne("")
        )
        self.assertTrue(native_block.all())

    def test_critical_project_question_filters_delivery_risk(self) -> None:
        answer = answer_data_question("Which project are in critical state?", self.data)
        self.assertEqual(answer.title, "Critical projects")
        critical = (
            answer.table["ProjectStatus"].eq("On Hold")
            | answer.table["DeliveryHealth"].eq("Red")
        )
        self.assertFalse(answer.table.empty)
        self.assertTrue(critical.all())
        self.assertEqual(len(answer.table), 24)

    def test_task_question_returns_task_records(self) -> None:
        answer = answer_data_question("Show tasks for project PRJ0001", self.data)
        self.assertEqual(answer.title, "Ticket task status")
        self.assertTrue(answer.table["ProjectID"].eq("PRJ0001").all())

    def test_unknown_delivery_identifier_does_not_return_every_project(self) -> None:
        answer = answer_data_question("Show PRJ9999", self.data)
        self.assertEqual(answer.title, "Delivery identifier not found")
        self.assertTrue(answer.table.empty)

    def test_vague_and_causal_questions_request_clarification(self) -> None:
        for question in ["What should we do?", "Can we win this?", "Why is revenue low?"]:
            with self.subTest(question=question):
                answer = answer_data_question(question, self.data)
                self.assertEqual(answer.title, "Please clarify the request")
                self.assertEqual(
                    list(answer.table.columns),
                    ["Information Needed", "Why It Matters", "Example Request"],
                )

    def test_last_week_leaderboard_and_detail_counts_reconcile(self) -> None:
        answer = answer_data_question("Who held the most meetings last week?", self.data)
        self.assertIn("Meetings", answer.table.columns)
        period_start = pd.to_datetime(answer.table["Period Start"]).min()
        period_end = pd.to_datetime(answer.table["Period End"]).max()
        detail = meeting_records_between(self.data, period_start, period_end)
        self.assertEqual(int(answer.table["Meetings"].sum()), len(detail))
        self.assertTrue(detail["MeetingDate"].between(period_start, period_end).all())

    def test_last_week_leaderboard_recommends_meeting_bar_chart(self) -> None:
        answer = answer_data_question("Who held the most meetings last week?", self.data)
        self.assertIsNotNone(answer.visual)
        self.assertEqual(answer.visual.chart_type, "bar")
        self.assertEqual(answer.visual.x, "Salesperson")
        self.assertEqual(answer.visual.y, "Meetings")
        self.assertIn("Meetings by salesperson", answer.visual.title)

    def test_last_week_meetings_for_salesperson_return_records(self) -> None:
        answer = answer_data_question(
            "What are the last week meetings by Alice?", self.data
        )
        self.assertEqual(answer.title, "Meetings last week for Alice Brown")
        self.assertEqual(len(answer.table), 3)
        self.assertTrue(answer.table["Salesperson"].eq("Alice Brown").all())
        self.assertTrue(answer.table["MeetingStatus"].str.casefold().eq("held").all())
        self.assertIn("03 Aug 2026 to 09 Aug 2026", answer.summary)
        self.assertEqual(answer.interpretation["Salesperson"], "Alice Brown")

    def test_meeting_record_answer_recommends_date_activity_chart(self) -> None:
        answer = answer_data_question(
            "What are the last week meetings by Alice?", self.data
        )
        self.assertIsNotNone(answer.visual)
        self.assertEqual(answer.visual.x, "MeetingDate")
        self.assertEqual(answer.visual.y, "Meetings")
        self.assertEqual(answer.visual.color, "Salesperson")

    def test_last_week_salesperson_meetings_are_retained_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            answer = answer_with_local_llm(
                "What are the last week meetings by Alice?", self.data
            )
        run.assert_not_called()
        self.assertEqual(
            answer.title,
            "Verified local result: Meetings last week for Alice Brown",
        )
        self.assertEqual(len(answer.table), 3)
        self.assertIsNotNone(answer.visual)
        self.assertEqual(answer.visual.x, "MeetingDate")

    def test_monthly_meeting_ranking_uses_the_requested_calendar_month(self) -> None:
        answer = answer_data_question(
            "Who held the most meetings in the month of July 2026?", self.data
        )
        self.assertEqual(answer.title, "Meeting ranking for July 2026")
        self.assertEqual(answer.table.iloc[0]["Salesperson"], "Emma Patel")
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 8)
        self.assertIn("01 Jul 2026 to 31 Jul 2026", answer.summary)
        self.assertEqual(answer.interpretation["Period"], "01 Jul 2026 to 31 Jul 2026")

    def test_monthly_meeting_detail_can_be_scoped_to_a_salesperson(self) -> None:
        answer = answer_data_question(
            "What meetings did Alice have in July 2026?", self.data
        )
        self.assertEqual(answer.title, "Meetings in July 2026 for Alice Brown")
        self.assertEqual(len(answer.table), 1)
        self.assertTrue(answer.table["Salesperson"].eq("Alice Brown").all())
        self.assertTrue(
            answer.table["MeetingDate"].between("2026-07-01", "2026-07-31").all()
        )

    def test_monthly_meeting_result_is_retained_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            answer = answer_with_local_llm(
                "Who held the most meetings in July 2026?", self.data
            )
        run.assert_not_called()
        self.assertEqual(
            answer.title,
            "Verified local result: Meeting ranking for July 2026",
        )
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 8)

    def test_last_month_meeting_ranking_uses_the_previous_workbook_month(self) -> None:
        answer = answer_data_question(
            "Who held the most meetings last month?", self.data
        )
        self.assertEqual(answer.title, "Meeting ranking for July 2026")
        self.assertEqual(answer.table.iloc[0]["Salesperson"], "Emma Patel")
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 8)
        self.assertIn("01 Jul 2026 to 31 Jul 2026", answer.summary)
        self.assertIn("13 Aug 2026", answer.summary)

    def test_last_month_meeting_detail_can_be_scoped_to_a_salesperson(self) -> None:
        answer = answer_data_question(
            "What meetings did Alice have last month?", self.data
        )
        self.assertEqual(answer.title, "Meetings in July 2026 for Alice Brown")
        self.assertEqual(len(answer.table), 1)
        self.assertTrue(answer.table["Salesperson"].eq("Alice Brown").all())

    def test_last_month_meeting_result_is_retained_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            answer = answer_with_local_llm(
                "Who held the most meetings last month?", self.data
            )
        run.assert_not_called()
        self.assertEqual(
            answer.title,
            "Verified local result: Meeting ranking for July 2026",
        )
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 8)

    def test_specific_day_meeting_ranking(self) -> None:
        answer = answer_data_question(
            "Who held the most meetings on 6 August 2026?", self.data
        )
        self.assertEqual(answer.title, "Meeting ranking for 06 Aug 2026")
        self.assertEqual(answer.table.iloc[0]["Salesperson"], "Emma Patel")
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 4)
        self.assertEqual(answer.interpretation["Period"], "06 Aug 2026 to 06 Aug 2026")

    def test_explicit_meeting_date_range(self) -> None:
        answer = answer_data_question(
            "Who held the most meetings from 1 to 15 July 2026?", self.data
        )
        self.assertEqual(
            answer.title, "Meeting ranking for 01 Jul 2026 to 15 Jul 2026"
        )
        self.assertEqual(answer.table.iloc[0]["Salesperson"], "Harry Evans")
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 6)

    def test_last_number_of_days_uses_the_workbook_snapshot(self) -> None:
        answer = answer_data_question(
            "Who held the most meetings in the last 14 days?", self.data
        )
        self.assertEqual(answer.title, "Meeting ranking for the last 14 days")
        self.assertEqual(answer.table.iloc[0]["Salesperson"], "Emma Patel")
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 12)
        self.assertEqual(answer.interpretation["Period"], "02 Aug 2026 to 15 Aug 2026")
        self.assertIn("workbook snapshot of 15 Aug 2026", answer.summary)

    def test_current_workbook_explains_missing_upcoming_meetings(self) -> None:
        answer = answer_data_question("What are the upcoming meetings?", self.data)
        self.assertEqual(answer.title, "Upcoming meetings")
        self.assertTrue(answer.table.empty)
        self.assertIn("No upcoming meetings are recorded", answer.summary)
        self.assertIn("Scheduled, Planned, Confirmed", answer.summary)

    def test_upcoming_meeting_detail_uses_scheduled_future_records(self) -> None:
        data = {name: frame.copy() for name, frame in self.data.items()}
        scheduled = data["Meetings"].iloc[0].copy()
        scheduled["MeetingID"] = "MTG-FUTURE-001"
        scheduled["MeetingDate"] = pd.Timestamp("2026-09-05")
        scheduled["MeetingStatus"] = "Scheduled"
        scheduled["SalespersonID"] = "SP001"
        scheduled["Subject"] = "September customer planning meeting"
        data["Meetings"] = pd.concat(
            [data["Meetings"], scheduled.to_frame().T], ignore_index=True
        )

        answer = answer_data_question(
            "What are Alice's upcoming meetings in September 2026?", data
        )
        self.assertEqual(
            answer.title, "Upcoming meetings in September 2026 for Alice Brown"
        )
        self.assertEqual(answer.table["MeetingID"].tolist(), ["MTG-FUTURE-001"])
        self.assertEqual(answer.table.iloc[0]["MeetingStatus"], "Scheduled")
        self.assertEqual(answer.interpretation["Meeting timing"], "Upcoming")

    def test_next_number_of_days_uses_the_workbook_snapshot(self) -> None:
        data = {name: frame.copy() for name, frame in self.data.items()}
        scheduled = data["Meetings"].iloc[0].copy()
        scheduled["MeetingID"] = "MTG-FUTURE-030"
        scheduled["MeetingDate"] = pd.Timestamp("2026-08-25")
        scheduled["MeetingStatus"] = "Confirmed"
        scheduled["SalespersonID"] = "SP001"
        data["Meetings"] = pd.concat(
            [data["Meetings"], scheduled.to_frame().T], ignore_index=True
        )

        answer = answer_data_question(
            "What meetings does Alice have in the next 30 days?", data
        )
        self.assertEqual(
            answer.title,
            "Upcoming meetings in the next 30 days for Alice Brown",
        )
        self.assertEqual(answer.table["MeetingID"].tolist(), ["MTG-FUTURE-030"])
        self.assertEqual(answer.interpretation["Period"], "16 Aug 2026 to 14 Sep 2026")

    def test_rolling_meeting_result_is_retained_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            answer = answer_with_local_llm(
                "Who held the most meetings in the last 14 days?", self.data
            )
        run.assert_not_called()
        self.assertEqual(
            answer.title,
            "Verified local result: Meeting ranking for the last 14 days",
        )
        self.assertEqual(int(answer.table.iloc[0]["Meetings"]), 12)

    def test_top_customer_whitespace_limit_is_applied_before_llm_explanation(self) -> None:
        answer = answer_data_question(
            "For Alice Brown, show the top three customer whitespace opportunities by "
            "estimated annual potential.",
            self.data,
        )
        self.assertEqual(answer.title, "Customer whitespace opportunities")
        self.assertEqual(len(answer.table), 3)
        self.assertTrue(answer.table["Account Owner"].eq("Alice Brown").all())
        self.assertTrue(answer.table["Estimated Annual Potential"].is_monotonic_decreasing)

    def test_bottom_two_salespeople_by_performance(self) -> None:
        answer = answer_data_question(
            "Who are the bottom two salesperson by performance?", self.data
        )
        self.assertEqual(answer.title, "Bottom performers")
        self.assertEqual(len(answer.table), 2)
        self.assertEqual(answer.table["Rank"].tolist(), [1, 2])
        self.assertTrue(answer.table["performance_score"].is_monotonic_increasing)
        self.assertIn("2 lowest results by composite performance score", answer.summary)

    def test_bottom_performance_ranking_is_retained_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            answer = answer_with_local_llm(
                "Who are the bottom 2 salespeople by performance?", self.data
            )
        run.assert_not_called()
        self.assertEqual(answer.title, "Verified local result: Bottom performers")
        self.assertEqual(len(answer.table), 2)

    def test_upcoming_opportunities_for_salesperson(self) -> None:
        question = "The upcoming opportunities for Chloe Singh"
        answer = answer_data_question(question, self.data)
        chloe_id = self.data["Salespeople"].loc[
            self.data["Salespeople"]["Salesperson"].eq("Chloe Singh"), "SalespersonID"
        ].iloc[0]
        snapshot = pd.to_datetime(
            self.data["Contracts"]["SnapshotDate"], errors="coerce"
        ).max().normalize()
        expected = self.data["Opportunities"].copy()
        expected["ExpectedCloseDate"] = pd.to_datetime(
            expected["ExpectedCloseDate"], errors="coerce"
        )
        expected["CloseDate"] = pd.to_datetime(expected["CloseDate"], errors="coerce")
        expected = expected[
            expected["SalespersonID"].eq(chloe_id)
            & expected["Stage"].str.casefold().eq("open")
            & expected["CloseDate"].isna()
            & expected["ExpectedCloseDate"].ge(snapshot)
        ]

        self.assertEqual(answer.title, "Upcoming opportunities")
        self.assertEqual(len(answer.table), len(expected))
        self.assertTrue(answer.table["Salesperson"].eq("Chloe Singh").all())
        self.assertTrue(answer.table["ExpectedCloseDate"].is_monotonic_increasing)
        self.assertIn("upcoming open opportunities", answer.summary)

    def test_upcoming_opportunities_are_retained_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            answer = answer_with_local_llm(
                "The upcoming opportunities for Chloe Singh", self.data
            )
        run.assert_not_called()
        self.assertEqual(answer.title, "Verified local result: Upcoming opportunities")
        self.assertFalse(answer.table.empty)

    def test_upcoming_cross_sell_opportunities_apply_type_filter(self) -> None:
        question = "What are the upcoming cross sell opportunities for Chloe Singh?"
        answer = answer_data_question(question, self.data)
        self.assertEqual(answer.title, "Upcoming cross-sell opportunities")
        self.assertFalse(answer.table.empty)
        self.assertTrue(answer.table["Salesperson"].eq("Chloe Singh").all())
        self.assertTrue(answer.table["OpportunityType"].eq("Cross-sell").all())
        self.assertEqual(len(answer.table), 3)
        self.assertIn("upcoming open cross-sell opportunities", answer.summary)

    def test_upcoming_cross_sell_is_retained_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            answer = answer_with_local_llm(
                "What are the upcoming cross-sell opportunities for Chloe Singh?", self.data
            )
        run.assert_not_called()
        self.assertEqual(
            answer.title, "Verified local result: Upcoming cross-sell opportunities"
        )
        self.assertTrue(answer.table["OpportunityType"].eq("Cross-sell").all())
        self.assertEqual(answer.interpretation["Opportunity type"], "Cross-sell")

    def test_project_questions_retain_verified_results_in_llm_mode(self) -> None:
        with patch("src.local_llm.subprocess.run") as run:
            factual = answer_with_local_llm("Show project PRJ0001", self.data)
            exploratory = answer_with_local_llm(
                "What should we do about delays on PRJ0001?", self.data
            )
        run.assert_not_called()
        self.assertTrue(factual.title.startswith("Verified local result"))
        self.assertTrue(exploratory.title.startswith("Verified local result"))
        self.assertIn("Suggested actions", exploratory.summary)
        self.assertTrue(bool(exploratory.table.iloc[0]["ProjectOverdue"]))

    def test_exploratory_project_context_is_compact(self) -> None:
        context = build_question_context("What should we do about delays on PRJ0001?", self.data)
        self.assertLess(len(context), 1500)
        self.assertIn("PRJ0001", context)
        self.assertNotIn("TEAM PERFORMANCE", context)

    def test_british_english_response_guardrail(self) -> None:
        response = _use_british_english("Analyze behavior and prioritize GBP 12,500.")
        self.assertEqual(response, "Analyse behaviour and prioritise £12,500.")


if __name__ == "__main__":
    unittest.main()
