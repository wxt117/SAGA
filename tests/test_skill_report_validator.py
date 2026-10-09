from __future__ import annotations

import unittest

from saga.core.skill_report import attach_skill_run_report, validate_skill_run_report, validate_step_result_skill_report


class SkillRunReportValidatorTest(unittest.TestCase):
    def test_valid_report_from_executor_result(self) -> None:
        result = {
            "status": "dry_run",
            "message": "ok",
            "output_dir": "/tmp/out",
            "image_count": 2,
            "expected_count": 2,
            "metrics": {"planned": 2},
            "artifacts": {"report": "/tmp/out/report.json"},
        }
        attach_skill_run_report(
            result,
            step_id="run_skill",
            skill="TraditionalAugmentationSkill",
            dry_run=True,
            elapsed_seconds=0.01,
            params={"target_count": 2},
        )
        validation = validate_step_result_skill_report({"step_id": "run_skill", "skill": "TraditionalAugmentationSkill", **result})
        self.assertTrue(validation["valid"], validation)

    def test_missing_report_is_invalid(self) -> None:
        validation = validate_skill_run_report(None)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["issues"][0]["name"], "missing_skill_run_report")

    def test_step_mismatch_is_invalid(self) -> None:
        result = {"status": "dry_run", "message": "ok"}
        attach_skill_run_report(
            result,
            step_id="a",
            skill="QualityEvaluationSkill",
            dry_run=True,
            elapsed_seconds=0.01,
            params={},
        )
        validation = validate_step_result_skill_report({"step_id": "b", "skill": "QualityEvaluationSkill", **result})
        self.assertFalse(validation["valid"])
        self.assertTrue(any(issue["name"] == "step_id_mismatch" for issue in validation["issues"]))

    def test_not_triggered_status_is_known(self) -> None:
        result = {"status": "not_triggered", "message": "Observer did not raise quality triggers."}
        attach_skill_run_report(
            result,
            step_id="repair_policy",
            skill="RepairPolicySkill",
            dry_run=False,
            elapsed_seconds=0.01,
            params={},
        )
        validation = validate_step_result_skill_report({"step_id": "repair_policy", "skill": "RepairPolicySkill", **result})
        self.assertTrue(validation["valid"], validation)
        self.assertFalse(any(warning["name"] == "unknown_status" for warning in validation["warnings"]))


if __name__ == "__main__":
    unittest.main()
