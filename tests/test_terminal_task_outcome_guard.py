import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TerminalTaskOutcomeGuardTests(unittest.TestCase):
    def test_task_outcome_memory_only_runs_for_terminal_bench_2(self):
        from src.evolve import _task_outcome_memory_enabled_for_dataset

        self.assertTrue(
            _task_outcome_memory_enabled_for_dataset("terminal-bench@2.0")
        )
        self.assertFalse(
            _task_outcome_memory_enabled_for_dataset("swebench-verified")
        )
        self.assertFalse(
            _task_outcome_memory_enabled_for_dataset("unknown-dataset")
        )

    def test_terminal_runner_assets_exist(self):
        self.assertTrue((ROOT / "scripts" / "run_eval.sh").exists())
        self.assertTrue(
            (ROOT / ".claude" / "skills" / "agent" / "SKILL.md").exists()
        )
        self.assertTrue(
            (ROOT / ".claude" / "skills" / "classification" / "SKILL.md").exists()
        )

    def test_component_memory_import_paths_share_configured_taxonomy(self):
        from src.memory import configure
        import memory.encoding.component_evidence as bare_ce
        import src.memory.encoding.component_evidence as src_ce
        import src.memory.steering.component_playbook as component_playbook
        import src.memory.steering.reading_strategy as reading_strategy

        configure("terminal")
        self.assertIs(bare_ce, src_ce)
        self.assertIn("agent_loop", reading_strategy._valid_components())
        self.assertIn("tool_parsing", reading_strategy._valid_components())
        self.assertNotIn("retrieval", reading_strategy._valid_components())
        self.assertIn("agent_loop", component_playbook._ALL_COMPONENTS)
        self.assertNotIn("retrieval", component_playbook._ALL_COMPONENTS)


if __name__ == "__main__":
    unittest.main()
