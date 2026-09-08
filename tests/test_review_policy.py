import unittest

from src.data.review_policy import requires_human_review, resolved_annotation_id


class ReviewPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.member = {"annotation_id": 10, "audit_selected": 0}
        self.strong_proposal = {
            "annotation_id": 20,
            "preference": "proposal",
            "confidence": "high",
            "validation_status": "pass",
            "material_difference": 1,
        }

    def test_material_strong_luna_win_is_automatic(self) -> None:
        self.assertFalse(requires_human_review(self.member, self.strong_proposal))
        self.assertEqual(
            resolved_annotation_id(self.member, self.strong_proposal, None, "2026-01-01"),
            (20, "policy:luna"),
        )

    def test_passing_tie_keeps_existing_approved_label(self) -> None:
        tie = self.strong_proposal | {"preference": "tie"}
        self.assertFalse(requires_human_review(self.member, tie))
        self.assertEqual(
            resolved_annotation_id(self.member, tie, None, "2026-01-01"),
            (10, "cohort:approved"),
        )

    def test_safe_existing_label_choices_are_automatic(self) -> None:
        self.assertFalse(requires_human_review(
            self.member, self.strong_proposal | {"preference": "production"}
        ))
        self.assertFalse(requires_human_review(
            self.member,
            self.strong_proposal | {
                "confidence": "low", "material_difference": 0
            },
        ))

    def test_ambiguous_cases_and_audits_require_human_review(self) -> None:
        self.assertTrue(requires_human_review(
            self.member,
            self.strong_proposal | {
                "confidence": "medium", "material_difference": 1
            },
        ))
        self.assertTrue(requires_human_review(
            self.member, self.strong_proposal | {"validation_status": "fail"}
        ))
        self.assertTrue(requires_human_review(
            self.member | {"audit_selected": 1}, self.strong_proposal
        ))

    def test_high_confidence_production_win_is_safe_on_failed_proposal(self) -> None:
        analysis = self.strong_proposal | {
            "preference": "production", "validation_status": "fail"
        }
        self.assertFalse(requires_human_review(self.member, analysis))
        self.assertEqual(
            resolved_annotation_id(self.member, analysis, None, "2026-01-01"),
            (10, "cohort:approved"),
        )

    def test_fresh_human_choice_always_wins(self) -> None:
        decision = {
            "annotation_id": 30,
            "reviewer": "human",
            "decided_at": "2026-02-01",
        }
        self.assertEqual(
            resolved_annotation_id(
                self.member, self.strong_proposal, decision, "2026-01-01"
            ),
            (30, "human"),
        )


if __name__ == "__main__":
    unittest.main()
