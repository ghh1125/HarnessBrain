import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.encoding.task_outcome_memory import (
    TaskOutcomeMemory,
    classify_trial_result,
    collect_task_outcomes,
)


class TaskOutcomeMemoryTests(unittest.TestCase):
    def test_classifies_harbor_trials_into_actionable_failure_categories(self):
        passed = classify_trial_result({
            "verifier_result": {"rewards": {"reward": 1.0}},
        })
        self.assertEqual(passed["outcome"], "pass")
        self.assertIsNone(passed["failure_category"])

        nested_timeout = classify_trial_result({
            "exception_info": {
                "exception_type": "AgentTimeoutError",
                "exception_message": "900 second timeout",
            },
        })
        self.assertEqual(nested_timeout["failure_category"], "agent_timeout")
        self.assertEqual(nested_timeout["failure_component"], "agent_loop")

        parse_error = classify_trial_result({
            "exception": "Failed to parse tool arguments: invalid JSON",
        })
        self.assertEqual(parse_error["failure_category"], "tool_argument_parse")
        self.assertEqual(parse_error["failure_component"], "tool_parsing")

        infra = classify_trial_result({
            "exception_info": {
                "exception_type": "EnvironmentStartTimeoutError",
                "exception_message": "docker image startup timed out",
            },
        })
        self.assertEqual(infra["failure_category"], "infra_failure")
        self.assertIsNone(infra["failure_component"])

    def test_collects_task_outcomes_from_job_directory_and_exception_txt(self):
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            timeout_trial = job_dir / "extract-elf__0"
            timeout_trial.mkdir()
            (timeout_trial / "result.json").write_text("{}", encoding="utf-8")
            (timeout_trial / "exception.txt").write_text(
                "Traceback...\nAgentTimeoutError: max duration exceeded",
                encoding="utf-8",
            )
            missing_trial = job_dir / "fix-git__0"
            missing_trial.mkdir()

            outcome = collect_task_outcomes(str(job_dir))

        self.assertEqual(
            outcome["tasks"]["extract-elf"]["failure_category"], "agent_timeout"
        )
        self.assertEqual(
            outcome["tasks"]["fix-git"]["failure_category"], "missing_result"
        )
        self.assertEqual(outcome["summary"]["failure_categories"]["agent_timeout"], 1)
        self.assertEqual(outcome["summary"]["failure_categories"]["missing_result"], 1)

    def test_updates_memory_with_new_pass_attribution_and_guidance(self):
        with tempfile.TemporaryDirectory() as td:
            mem = TaskOutcomeMemory(Path(td))
            mem.update_episode({
                "episode_id": "agent_a",
                "iteration": 1,
                "score": 20.0,
                "task_outcomes": {
                    "extract-elf": {
                        "pass_rate": 0.0,
                        "failure_category": "agent_timeout",
                        "failure_component": "agent_loop",
                    },
                    "fix-git": {
                        "pass_rate": 1.0,
                        "failure_category": None,
                        "failure_component": None,
                    },
                },
            })
            enriched = mem.update_episode({
                "episode_id": "agent_b",
                "iteration": 2,
                "parent_candidate": "agent_a",
                "score": 30.0,
                "task_outcomes": {
                    "extract-elf": {
                        "pass_rate": 1.0,
                        "failure_category": None,
                        "failure_component": None,
                    },
                    "fix-git": {
                        "pass_rate": 0.0,
                        "failure_category": "verifier_failure",
                        "failure_component": None,
                    },
                },
            })
            guidance = mem.guidance_summary()

        self.assertEqual(enriched["task_new_passes"], ["extract-elf"])
        self.assertEqual(enriched["task_regressions"], ["fix-git"])
        self.assertEqual(
            enriched["task_failure_category_deltas"]["improved_categories"],
            {"agent_timeout": 1},
        )
        self.assertEqual(
            enriched["new_pass_attribution"],
            {
                "agent_loop": {
                    "tasks": ["extract-elf"],
                    "source_failure_category": "agent_timeout",
                    "count": 1,
                }
            },
        )
        self.assertEqual(guidance["solved_task_count"], 2)
        self.assertIn("verifier_failure", guidance["persistent_failure_categories"])


if __name__ == "__main__":
    unittest.main()
