
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import yaml

from src import claude_wrapper
from dotenv import load_dotenv






from src.memory import configure as _configure_component_memory
from src.memory.config_loader import (
    module_enabled,
    memory_enabled,
    evidence_quality_enabled,
    memory_efficiency_enabled,
    search_governance_enabled,
    get_memory_config,
)
from src.memory.encoding.episode_recorder import EpisodeRecorder
from src.memory.encoding.component_evidence import ComponentEvidenceBuilder, identify_components
from src.memory.updating.evolution_operators import EvolutionOperators
from src.memory.encoding.inter_component import InterComponentEvidence
from src.memory.steering.proposal_plan import ProposalPlanManager
from src.memory.reporting.metrics_collector import MetricsCollector

try:
    from src.memory.encoding.task_outcome_memory import (
        TaskOutcomeMemory,
        collect_task_outcomes,
    )
except ImportError:
    TaskOutcomeMemory = None
    collect_task_outcomes = None

EVOLVE_DIR = Path(__file__).parent
REPO_ROOT = EVOLVE_DIR.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
AGENTS_DIR = EVOLVE_DIR / "agents"
EVO_WORKSPACE = REPO_ROOT / "workspace" / "evo"
BASELINE_FILES = {"__init__.py", "no_memory.py", "fewshot_memory.py", "fewshot_all.py"}


LOGS_DIR = REPO_ROOT / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"

_interrupted = False
_BENCHMARK_DATASET: str | None = None



READING_CONTEXT_CACHE: dict = {}


_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _pct(val):
    s = f"{val:.1f}%"
    if val >= 60:
        return _green(s)
    elif val >= 40:
        return _yellow(s)
    return _red(s)


def _print_final_test_frontier(
    frontier_path: Path,
    allowed_systems: set[str] | None = None,
    current_test_frontier: dict | None = None,
) -> None:
    if not frontier_path.exists():
        print(f"\n{_ts()} {_yellow('Warning')}: test frontier not found at {frontier_path}")
        return

    try:
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\n{_ts()} {_yellow('Warning')}: could not read test frontier: {exc}")
        return

    allowed_systems = set(allowed_systems or [])
    pareto = []
    source = "benchmark_frontier"
    if current_test_frontier:
        source = current_test_frontier.get("source", "current_run")
        for row in current_test_frontier.get("pareto", []) or []:
            if not isinstance(row, dict):
                continue
            system = row.get("system")
            score = row.get("score", row.get("test_accuracy", row.get("accuracy")))
            if system is None or score is None:
                continue
            pareto.append(
                {
                    "system": system,
                    "test_accuracy": score,
                    "ctx_len": row.get("ctx_len"),
                    "correct": row.get("correct"),
                    "total": row.get("total"),
                }
            )

    if not pareto:
        pareto = frontier.get("_pareto", [])

    if not pareto:
        rows = []
        for dataset, record in frontier.items():
            if dataset.startswith("_") or not isinstance(record, dict):
                continue
            system = record.get("best_system")
            score = record.get("test_accuracy", record.get("accuracy"))
            ctx_len = record.get("ctx_len")
            if system is not None and score is not None:
                rows.append({"system": system, "test_accuracy": score, "ctx_len": ctx_len})
        pareto = sorted(rows, key=lambda r: float(r.get("test_accuracy", 0)), reverse=True)

    if allowed_systems:
        pareto = [
            row for row in pareto
            if str(row.get("system", "")) in allowed_systems
        ]

    if not pareto:
        print(f"\n{_ts()} {_yellow('Warning')}: test frontier is empty for current run")
        return

    pareto = sorted(
        pareto,
        key=lambda r: float(r.get("test_accuracy", r.get("accuracy", 0)) or 0),
        reverse=True,
    )

    print(f"\n{_ts()} {_bold('Final Test Frontier')}  source={source}")
    print(f"{'system':<34} {'test':>8} {'ctx_len':>10}")
    print("-" * 55)
    for entry in pareto:
        system = str(entry.get("system", ""))
        score = float(entry.get("test_accuracy", entry.get("accuracy", 0)) or 0)
        ctx_len = entry.get("ctx_len")
        ctx_text = f"{int(ctx_len):,}ch" if isinstance(ctx_len, (int, float)) else "-"
        print(f"{system:<34} {score:>7.1f}% {ctx_text:>10}")

    best = max(pareto, key=lambda r: float(r.get("test_accuracy", r.get("accuracy", 0)) or 0))
    print(
        f"  {_bold('Best test')}: {best.get('system')} "
        f"{float(best.get('test_accuracy', best.get('accuracy', 0)) or 0):.1f}% "
        f"({int(best.get('ctx_len', 0)):,}ch)"
    )


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def run_cmd(cmd, timeout=7200, cwd=None):
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, returncode=124, stdout="", stderr=f"Timed out after {timeout}s"
        )


def _python_cmd() -> list[str]:

    return [sys.executable]


def run_benchmark(args, timeout=7200):
    extra = ["--dataset", _BENCHMARK_DATASET] if _BENCHMARK_DATASET else []
    return run_cmd(
        _python_cmd() + [str(REPO_ROOT / "main.py"), "benchmark", "--logs-dir", str(LOGS_DIR)] + extra + args,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )


def render_task_prompt(iteration, num_datasets):
    return (
        f"Run iteration {iteration} of the evolution loop. There are {num_datasets} datasets.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        f"- `{EVOLUTION_SUMMARY}` — past results\n"
        f"- `{FRONTIER_VAL}` — frontier\n"
        f"- `{LOGS_DIR / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`"
    )


def count_iterations_from_summary():
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


def _strip_code_fence(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    for part in parts:
        cleaned = part.strip()
        if cleaned.startswith("python"):
            return cleaned[len("python"):].strip()
    return parts[1].strip()


def extract_python_code(raw_output: str) -> str:
    if "```python" in raw_output:
        code = raw_output.split("```python", 1)[1]
        code = code.split("```")[0]
        return code.strip()

    if "class " in raw_output or "import " in raw_output or "from " in raw_output:
        lines = raw_output.split("\n")
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("class ", "import ", "from ")):
                return "\n".join(lines[i:]).strip()

    return _strip_code_fence(raw_output)


def _slug(text: str, fallback: str) -> str:
    import re
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", text.lower()).strip("_")
    value = re.sub(r"_+", "_", value)
    return value[:48] or fallback


def _proposer_llm():
    from src.llm import LLM
    from src.model_config import proposer_config
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    model_cfg = proposer_config(cfg)
    return LLM(
        model=model_cfg["model"],
        api_key=model_cfg.get("api_key"),
        api_base=model_cfg.get("api_base"),
        temperature=0.2,
        max_tokens=4096,
        max_workers=1,
    )


def _proposer_backend(cfg: dict | None = None) -> str:
    from src.model_config import proposer_backend

    cfg = cfg if cfg is not None else yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return proposer_backend(cfg)


def _proposer_claude_model(cfg: dict | None = None) -> str:
    from src.model_config import claude_code_proposer_model

    cfg = cfg if cfg is not None else yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return claude_code_proposer_model(cfg)


def _proposer_label(cfg: dict | None = None) -> tuple[str, str]:
    from src.model_config import proposer_config

    cfg = cfg if cfg is not None else yaml.safe_load(CONFIG_PATH.read_text()) or {}
    backend = _proposer_backend(cfg)
    if backend == "api":
        model_cfg = proposer_config(cfg)
        return backend, str(model_cfg["model"])
    return backend, _proposer_claude_model(cfg)


def _iter_dir(iteration: int) -> Path:
    path = EVO_WORKSPACE / f"iter_{iteration:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _proposer_log_path(iteration: int) -> Path:
    return _iter_dir(iteration) / "proposer_log.txt"


def _write_proposer_log(iteration: int, text: str, mode: str = "a") -> None:
    path = _proposer_log_path(iteration)
    with path.open(mode, encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _write_proposer_usage(
    output_dir: Path,
    iteration: int,
    candidate_id: str,
    call_index: int,
    attempt: int,
    usage: dict | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "proposer_usage.jsonl"
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    row = {
        "iteration": iteration,
        "candidate_id": candidate_id,
        "call_index": call_index,
        "attempt": attempt,
        "model": usage.get("model"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        "usage_source": usage.get("usage_source", "estimated"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return path


def _extract_proposer_output(iteration: int, name: str) -> str:
    log_path = _proposer_log_path(iteration)
    if not log_path.exists():
        return ""
    parts = name.rsplit("_", 1)
    try:
        index = int(parts[-1])
    except (ValueError, IndexError):
        return ""
    log_text = log_path.read_text(encoding="utf-8")
    header = f"## Candidate {index} Attempt 1 Raw Output"
    start = log_text.find(header)
    if start == -1:
        return ""
    start += len(header)
    next_section = log_text.find("## ", start)
    content = log_text[start:next_section].strip() if next_section != -1 else log_text[start:].strip()
    return content[:2000]


def _render_reading_context(context: dict) -> str:
    parts = []
    if context.get("search_guidance"):
        parts.append("## Search Guidance")
        parts.append(json.dumps(context["search_guidance"], ensure_ascii=False, indent=2))
    if context.get("component_evidence"):
        parts.append("## Component Evidence")
        parts.append(json.dumps(context["component_evidence"], ensure_ascii=False, indent=2))
    if context.get("component_playbook"):
        parts.append("## Component Playbook")
        parts.append(json.dumps(context["component_playbook"], ensure_ascii=False, indent=2))
    if context.get("effective_strategies"):
        parts.append("## Effective Strategies")
        parts.append(json.dumps(context["effective_strategies"], ensure_ascii=False, indent=2))
    if context.get("interaction_strategies"):
        parts.append("## Interaction Strategies")
        parts.append(json.dumps(context["interaction_strategies"], ensure_ascii=False, indent=2))
    if context.get("strategy_reuse_targets"):
        parts.append("## Strategy Reuse Targets")
        parts.append(json.dumps(context["strategy_reuse_targets"], ensure_ascii=False, indent=2))
    if context.get("hypothesis_test_targets"):
        parts.append("## Hypothesis Test Targets")
        parts.append(json.dumps(context["hypothesis_test_targets"], ensure_ascii=False, indent=2))
    if context.get("positive_anchors"):
        parts.append("## Positive Anchors")
        parts.append(json.dumps(context["positive_anchors"], ensure_ascii=False, indent=2))
    if context.get("anchor_refinement_targets"):
        parts.append("## Anchor Refinement Targets")
        parts.append(json.dumps(context["anchor_refinement_targets"], ensure_ascii=False, indent=2))
    if context.get("hypothesis_target"):
        parts.append("## Hypothesis Target")
        parts.append(json.dumps(context["hypothesis_target"], ensure_ascii=False, indent=2))
    if context.get("recent_episodes"):
        parts.append("## Recent Episodes")
        parts.append(json.dumps(context["recent_episodes"], ensure_ascii=False, indent=2))
    if context.get("evolution_summary"):
        parts.append("## Evolution Summary")
        parts.append(context["evolution_summary"])
    return "\n\n".join(parts)


_CANDIDATE_TEMPLATE = '''import json
from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


class MyMemory(MemorySystem):
    def __init__(self, llm: LLMCallable):
        super().__init__(llm)   # REQUIRED: must be first line
        self.examples = []      # your state fields here

    def predict(self, input: str) -> tuple[str, dict]:
        prompt = f"""Classify the input. Examples: {self.examples[-5:]}
Input: {input}
Return JSON: {{"reasoning": "...", "final_answer": "..."}}"""
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"n": len(self.examples)}   # REQUIRED: return (str, dict) tuple

    def learn_from_batch(self, batch_results):
        for row in batch_results:
            label = row["ground_truth"]   # REQUIRED: use "ground_truth", NOT "label"/"target"
            self.examples.append({"input": row["input"], "label": label})

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples})   # REQUIRED: return JSON string

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
'''

REGEX_SAFETY_RULE = (
    "Regex safety: never put literal line breaks inside r'...' or r\"...\". "
    "Use single-line patterns plus re.DOTALL or re.MULTILINE for multi-line matching."
)


def _candidate_prompt(iteration: int, index: int, task_prompt: str) -> str:
    summary = EVOLUTION_SUMMARY.read_text()[-8000:] if EVOLUTION_SUMMARY.exists() else ""
    frontier = FRONTIER_VAL.read_text()[-4000:] if FRONTIER_VAL.exists() else ""
    return f"""You are the HarnessBrain proposer. Generate ONE new Python memory-system module.

Context:
{task_prompt}

Recent evolution summary:
{summary or "(none)"}

Current frontier:
{frontier or "(none)"}

Candidate index: {index}

MANDATORY STRUCTURE — you must follow this exactly:

```python
{_CANDIDATE_TEMPLATE}
```

API contracts (violations cause runtime failure):
1. __init__: first line MUST be `super().__init__(llm)` — without it predict() crashes
2. predict: MUST return a 2-tuple `(str, dict)` — returning just a string crashes the harness
3. learn_from_batch: read ground truth as `row["ground_truth"]` — NOT row["label"] or row["target"]
4. get_state: MUST return `json.dumps(...)` — returning a plain dict crashes the harness
5. set_state: receives the JSON string that get_state returned

Additional rules:
- Output ONLY Python code. No markdown fences, no explanation.
- Use self.call_llm(prompt), never self._llm directly.
- Do not call the LLM inside learn_from_batch (keep learning cheap).
- Cold start (empty memory) must work without errors.
- No dataset-specific labels or hardcoded class names.
- Python stdlib only (no new pip packages).
- {REGEX_SAFETY_RULE}

Design a genuinely different mechanism from simple few-shot replay. Examples:
1. label-prior + compact lessons
2. token-overlap retrieval with diversity
3. error-focused contrastive memory
4. compressed rule memory
5. balanced per-label exemplars

Generate candidate {index} for iteration {iteration}.
"""


def _candidate_prompt_optimized(
    iteration: int, index: int, task_prompt: str,
    candidate_role: str = None,
) -> tuple[str, dict]:
    guidance_path = EVO_WORKSPACE.parent / "component_memory" / "search_guidance.json"
    guidance: dict = {}
    if guidance_path.exists():
        try:
            guidance = json.loads(guidance_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    _warm_up_cfg = yaml.safe_load(CONFIG_PATH.read_text()).get("memory_config", {}).get("warm_up", {})
    _warm_up_iters = _warm_up_cfg.get("warm_up_iterations", 2)


    if memory_efficiency_enabled():
        try:
            from src.memory.steering.reading_strategy import ReadingStrategyManager
            _strategy = ReadingStrategyManager()
            _mode = _strategy.decide_reading_mode(iteration, guidance, _warm_up_iters)
            _ctx = _strategy.build_context(
                mode=_mode,
                guidance=guidance,
                iteration=iteration,
                evolution_summary_path=EVOLUTION_SUMMARY,
            )
            if candidate_role and memory_efficiency_enabled():
                _ctx = _strategy.build_role_specific_context(
                    base_context=_ctx,
                    raw_guidance=guidance,
                    candidate_role=candidate_role,
                )
            _ctx_chars = _strategy.estimate_context_chars(_ctx)

            _candidate_name = f"mem_agent_i{iteration}_{index}"
            READING_CONTEXT_CACHE.setdefault(_candidate_name, {}).update({
                "reading_mode": _mode,
                "context_chars": _ctx_chars,
            })

            rendered_context = _render_reading_context(_ctx)
            prompt_body = f"""You are the HarnessBrain proposer. Generate ONE new Python memory-system module.

Context:
{task_prompt}

{rendered_context}

Candidate index: {index}

MANDATORY STRUCTURE — you must follow this exactly:

```python
{_CANDIDATE_TEMPLATE}
```

API contracts (violations cause runtime failure):
1. __init__: first line MUST be `super().__init__(llm)` — without it predict() crashes
2. predict: MUST return a 2-tuple `(str, dict)` — returning just a string crashes the harness
3. learn_from_batch: read ground truth as `row["ground_truth"]` — NOT row["label"] or row["target"]
4. get_state: MUST return `json.dumps(...)` — returning a plain dict crashes the harness
5. set_state: receives the JSON string that get_state returned

Additional rules:
- If a proposal_plan is requested above, output it first as a ```json block, then output the Python code in a ```python block.
- If no proposal_plan is requested, output ONLY the Python code in a ```python block. No prose explanation.
- Use self.call_llm(prompt), never self._llm directly.
- Do not call the LLM inside learn_from_batch (keep learning cheap).
- Cold start (empty memory) must work without errors.
- No dataset-specific labels or hardcoded class names.
- Python stdlib only (no new pip packages).
- {REGEX_SAFETY_RULE}

Design a genuinely different mechanism from simple few-shot replay. Examples:
1. label-prior + compact lessons
2. token-overlap retrieval with diversity
3. error-focused contrastive memory
4. compressed rule memory
5. balanced per-label exemplars

Generate candidate {index} for iteration {iteration}.
"""
            metadata = {
                "history_context_mode": _mode,
                "evolution_summary_chars": _ctx_chars.get("evolution_summary_chars", 0),
                "guidance_chars": _ctx_chars.get("guidance_chars", 0),
                "frontier_chars": 0,
            }
            return prompt_body, metadata
        except Exception as _hm_err:
            print(f"  [hypermem] warning: {_hm_err}, falling back to history-budget logic")


    has_high_priority = bool(guidance.get("high_priority"))
    has_avoid = bool(guidance.get("avoid"))
    guidance_mature = has_high_priority or has_avoid

    if iteration <= _warm_up_iters or not guidance_mature:
        history_mode = "warm_up"
        summary_limit = 8000
        summary = EVOLUTION_SUMMARY.read_text()[-summary_limit:] if EVOLUTION_SUMMARY.exists() else ""
        frontier = FRONTIER_VAL.read_text()[-4000:] if FRONTIER_VAL.exists() else ""
        guidance_text = ""
    else:
        history_mode = "guidance_primary"
        summary_limit = 2000
        summary = EVOLUTION_SUMMARY.read_text()[-summary_limit:] if EVOLUTION_SUMMARY.exists() else ""
        frontier = FRONTIER_VAL.read_text()[-4000:] if FRONTIER_VAL.exists() else ""

        hp_lines = "\n".join(
            f"  [{hp['component']}] {hp['direction']}"
            for hp in guidance.get("high_priority", [])[:5]
        )
        av_lines = "\n".join(
            f"  [{av['component']}] {av['direction']} ({av['reason']})"
            for av in guidance.get("avoid", [])[:6]
        )
        guidance_text = (
            "## Search Guidance (evidence-based, read first)\n"
            f"High Priority:\n{hp_lines or '  (none)'}\n\n"
            f"Avoid:\n{av_lines or '  (none)'}\n"
        )

    prompt_body = f"""You are the HarnessBrain proposer. Generate ONE new Python memory-system module.

Context:
{task_prompt}
"""
    if guidance_text:
        prompt_body += f"\n{guidance_text}\n"

    prompt_body += f"""
Current frontier:
{frontier or "(none)"}

Recent evolution summary:
{summary or "(none)"}

Candidate index: {index}

MANDATORY STRUCTURE — you must follow this exactly:

```python
{_CANDIDATE_TEMPLATE}
```

API contracts (violations cause runtime failure):
1. __init__: first line MUST be `super().__init__(llm)` — without it predict() crashes
2. predict: MUST return a 2-tuple `(str, dict)` — returning just a string crashes the harness
3. learn_from_batch: read ground truth as `row["ground_truth"]` — NOT row["label"] or row["target"]
4. get_state: MUST return `json.dumps(...)` — returning a plain dict crashes the harness
5. set_state: receives the JSON string that get_state returned

Additional rules:
- If a proposal_plan is requested above, output it first as a ```json block, then output the Python code in a ```python block.
- If no proposal_plan is requested, output ONLY the Python code in a ```python block. No prose explanation.
- Use self.call_llm(prompt), never self._llm directly.
- Do not call the LLM inside learn_from_batch (keep learning cheap).
- Cold start (empty memory) must work without errors.
- No dataset-specific labels or hardcoded class names.
- Python stdlib only (no new pip packages).
- {REGEX_SAFETY_RULE}

Design a genuinely different mechanism from simple few-shot replay. Examples:
1. label-prior + compact lessons
2. token-overlap retrieval with diversity
3. error-focused contrastive memory
4. compressed rule memory
5. balanced per-label exemplars

Generate candidate {index} for iteration {iteration}.
"""

    metadata = {
        "history_context_mode": history_mode,
        "evolution_summary_chars": len(summary),
        "guidance_chars": len(guidance_text),
        "frontier_chars": len(frontier),
    }
    return prompt_body, metadata


def _patch_candidate_code(code: str) -> str:
    import re

    if "super().__init__(llm)" not in code:
        code = re.sub(
            r"(def __init__\(self[^)]*\):\s*\n)(\s+)",
            lambda m: m.group(1) + m.group(2) + "super().__init__(llm)\n" + m.group(2),
            code,
            count=1,
        )

    code = re.sub(
        r"(return extract_json_field\([^)]+\))\s*$",
        r"\1, {}",
        code,
        flags=re.MULTILINE,
    )

    for var in ("row", "res", "result", "item", "batch_result", "example", "record", "entry"):
        code = re.sub(rf'\b{var}\["label"\]', f'{var}["ground_truth"]', code)
        code = re.sub(rf"\b{var}\['label'\]", f'{var}["ground_truth"]', code)
        code = re.sub(rf'\b{var}\.get\("label"', f'{var}.get("ground_truth"', code)
        code = re.sub(rf"\b{var}\.get\('label'", f'{var}.get("ground_truth"', code)
        code = re.sub(rf'\b{var}\["true_label"\]', f'{var}["ground_truth"]', code)
        code = re.sub(rf'\b{var}\.get\("true_label"', f'{var}.get("ground_truth"', code)

    if "def set_state" in code:
        lines = code.split("\n")
        in_set_state = False
        set_state_indent = ""
        has_loads = False
        set_state_start = -1
        fixed_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"def set_state\(", stripped):
                in_set_state = True
                set_state_indent = re.match(r"(\s*)", line).group(1)
                set_state_start = i
                has_loads = False
            elif in_set_state:
                if stripped and not line.startswith(set_state_indent + " ") and not line.startswith(set_state_indent + "\t"):
                    in_set_state = False
                elif "json.loads" in stripped:
                    has_loads = True
            fixed_lines.append(line)

        if not has_loads and set_state_start >= 0:
            result = []
            in_set_state = False
            injected = False
            for line in fixed_lines:
                stripped = line.strip()
                if re.match(r"def set_state\(", stripped):
                    in_set_state = True
                    injected = False
                elif in_set_state and not injected and stripped and not stripped.startswith("\"\"\"") and not stripped.startswith("'\"\"'"):
                    indent_chars = len(line) - len(line.lstrip())
                    result.append(" " * indent_chars + "data = json.loads(state)")
                    injected = True
                    in_set_state = False
                if in_set_state or not injected:
                    pass
                result.append(line)
            fixed_lines = result

        in_set_state = False
        set_state_indent = ""
        result = []
        for line in fixed_lines:
            stripped = line.strip()
            if re.match(r"def set_state\(", stripped):
                in_set_state = True
                set_state_indent = re.match(r"(\s*)", line).group(1)
            elif in_set_state and stripped and not line.startswith(set_state_indent + " ") and not line.startswith(set_state_indent + "\t"):
                in_set_state = False
            if in_set_state and "data = json.loads" not in stripped:
                line = re.sub(r'\bstate\.get\(', 'data.get(', line)
                line = re.sub(r'\bstate\["', 'data["', line)
                line = re.sub(r"\bstate\['", "data['", line)
            result.append(line)
        code = "\n".join(result)

    lines = code.split("\n")
    in_get_state = False
    get_state_indent = ""
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"def get_state\(", stripped):
            in_get_state = True
            get_state_indent = re.match(r"(\s*)", line).group(1)
        elif in_get_state and stripped and not line.startswith(get_state_indent + " ") and not line.startswith(get_state_indent + "\t"):
            in_get_state = False
        if in_get_state and stripped.startswith("return ") and "json.dumps" not in stripped:
            ret_val = stripped[len("return "):]
            if ret_val.startswith("{") or ret_val.startswith("dict("):
                indent_chars = len(line) - len(line.lstrip())
                line = " " * indent_chars + "return json.dumps(" + ret_val + ")"
        fixed_lines.append(line)
    code = "\n".join(fixed_lines)

    return code


def _validate_import(name: str) -> subprocess.CompletedProcess:
    code = f"""
import importlib, json, inspect
from src.memory_system import MemorySystem
m = importlib.import_module('src.agents.{name}')
classes = [
    obj for _, obj in inspect.getmembers(m, inspect.isclass)
    if issubclass(obj, MemorySystem) and obj is not MemorySystem
]
assert classes, 'no MemorySystem subclass'
calls = []
def stub_llm(prompt):
    calls.append(prompt)
    return '{{"reasoning":"stub","final_answer":"allergy"}}'
inst = classes[0](stub_llm)
pred = inst.predict('symptoms text')
assert isinstance(pred, tuple) and len(pred) == 2, 'predict must return tuple(answer, metadata)'
assert isinstance(pred[0], str), 'prediction answer must be str'
assert isinstance(pred[1], dict), 'prediction metadata must be dict'
inst.learn_from_batch([
    {{
        'input': 'symptoms text',
        'prediction': pred[0],
        'ground_truth': 'allergy',
        'was_correct': pred[0] == 'allergy',
        'metadata': pred[1],
    }}
])
state = inst.get_state()
assert isinstance(state, str), 'get_state must return str'
json.loads(state or '{{}}')
inst.set_state(state)
print('OK')
"""
    return run_cmd(
        [
            "env",
            f"PYTHONPATH={REPO_ROOT.resolve()}",
            *_python_cmd(),
            "-c",
            code,
        ],
        cwd=str(REPO_ROOT),
        timeout=30,
    )


def _write_candidate_module(name: str, code: str, iteration: int | None = None) -> str:
    if "class " not in code or "MemorySystem" not in code:
        raise ValueError("candidate code does not define a MemorySystem class")
    code = _patch_candidate_code(code)
    _write_agent(name, code)
    if iteration is not None:
        (_iter_dir(iteration) / f"{name}.py").write_text(code, encoding="utf-8")
    return code


def _proposer_cli_complete(prompt, iteration, name, attempt, timeout=2400, model=None):
    model = model or _proposer_claude_model()
    os.environ.pop("CLAUDECODE", None)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        result = claude_wrapper.run(
            prompt=prompt,
            model=model,
            allowed_tools=[],
            cwd=str(EVOLVE_DIR),
            log_dir=str(LOGS_DIR / "claude_sessions"),
            name=f"iter{iteration}_{name}_a{attempt}",
            timeout_seconds=timeout,
            effort="max",
        )
    finally:
        if saved_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
    usage = result.token_usage or {}
    usage.setdefault("model", model)
    usage.setdefault("usage_source", "claude_code")
    return (result.text or ""), usage


def _proposer_api_complete(llm, prompt):
    response = llm(prompt)
    usage = llm.get_last_usage() or {}
    usage.setdefault("model", getattr(llm, "model", None))
    return response, usage


def propose_candidates(task_prompt, iteration, timeout=2400):
    _cfg_full = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    backend, proposer_model = _proposer_label(_cfg_full)
    proposer_source = "API proposer" if backend == "api" else "Claude Code proposer"
    proposer_llm = _proposer_llm() if backend == "api" else None

    print(f"  {_cyan(proposer_source)} generating candidate modules...", flush=True)
    _write_proposer_log(
        iteration,
        "\n".join(
            [
                f"# Iteration {iteration} Proposer Log",
                f"timestamp: {datetime.now().isoformat()}",
                f"proposer_backend: {backend}",
                f"proposer_model: {proposer_model}",
                "",
            ]
        ),
        mode="w",
    )
    (_iter_dir(iteration) / "proposer_usage.jsonl").write_text("", encoding="utf-8")
    candidates = []
    _plan_manager = ProposalPlanManager() if search_governance_enabled() else None

    _warm_up_cfg = yaml.safe_load(CONFIG_PATH.read_text()).get("memory_config", {}).get("warm_up", {})
    _warm_up_iters = _warm_up_cfg.get("warm_up_iterations", 2)
    _soft_iters = _warm_up_cfg.get("soft_guidance_iterations", 4)
    if iteration <= _warm_up_iters:
        _guidance_phase = "none"
    elif iteration <= _soft_iters:
        _guidance_phase = "soft"
    else:
        _guidance_phase = "strong"

    _governance_mode = "audit_only_exploration"
    _candidate_roles = ["free_explore", "free_explore", "free_explore"]
    _current_evidence_maturity: dict = {}
    if search_governance_enabled():
        try:
            _guidance_path = EVO_WORKSPACE.parent / "component_memory" / "search_guidance.json"
            if _guidance_path.exists():
                _guidance_data = json.loads(_guidance_path.read_text(encoding="utf-8"))
                _current_evidence_maturity = _guidance_data.get("evidence_maturity", {}) or {}
                _governance = _guidance_data.get("governance", {}) or {}
                _governance_mode = _governance.get("mode", _governance_mode)
                _candidate_roles = _governance.get("candidate_roles", _candidate_roles)
        except Exception:
            pass

        if iteration > _warm_up_iters:
            if _governance_mode == "strong_component_boundary":
                _guidance_phase = "strong"
            else:
                _guidance_phase = "soft"

    _anchor: dict = {}
    _guidance_for_role: dict = {}
    try:
        _anchor_path = EVO_WORKSPACE.parent / "component_memory" / "search_guidance.json"
        if _anchor_path.exists():
            _guidance_for_role = json.loads(_anchor_path.read_text(encoding="utf-8"))
            _anchor = _guidance_for_role.get("anchor_code", {}) or {}
    except Exception:
        _anchor = {}

    _call_index = 0

    _tok_opt = _cfg_full.get("memory_config", {}).get("token_optimization", {})
    _tok_opt_enabled = (
        memory_efficiency_enabled() and bool(_tok_opt.get("enabled", False))
    )
    _plan_code_sep = _tok_opt_enabled and bool(_tok_opt.get("plan_code_separation", True))
    _history_budget = _tok_opt_enabled and bool(_tok_opt.get("history_budget", True))
    _log_context_mode = _tok_opt_enabled and bool(_tok_opt.get("log_context_mode", True))

    for index in range(1, 4):
        name = f"mem_agent_i{iteration}_{index}"
        _candidate_role = (
            _candidate_roles[index - 1]
            if index - 1 < len(_candidate_roles)
            else "free_explore"
        )
        _assigned_memory_ids: list[str] = []
        _assigned_avoid_memory_ids: list[str] = []
        if search_governance_enabled():
            def _strategy_mid(_item: dict) -> str | None:
                _mid = _item.get("memory_id")
                if _mid:
                    return _mid
                _eid = _item.get("episode_id") or _item.get("candidate_id")
                return f"strategy:{_eid}" if _eid else None

            if _candidate_role == "strategy_reuse":
                _targets = (
                    _guidance_for_role.get("strategy_reuse_targets", [])
                    or _guidance_for_role.get("effective_strategies", [])
                )
                if _targets:
                    _mid = _strategy_mid(_targets[0])
                    if _mid:
                        _assigned_memory_ids.append(_mid)
            elif _candidate_role == "hypothesis_test":
                _targets = _guidance_for_role.get("hypothesis_test_targets", [])
                if _targets:
                    _mid = _strategy_mid(_targets[0])
                    if _mid:
                        _assigned_memory_ids.append(_mid)
            elif _candidate_role == "component_exploit":
                _targets = _guidance_for_role.get("high_priority", [])
                if _targets:
                    _gid = _targets[0].get("guidance_id")
                    if _gid:
                        _assigned_memory_ids.append(f"guidance:{_gid}")
            elif _candidate_role == "avoid_retest":
                _retest = _guidance_for_role.get("retest_pool", [])
                if _retest:
                    _mid = _retest[0].get("memory_id")
                    if _mid:
                        _assigned_avoid_memory_ids.append(_mid)

        _eff = _guidance_for_role.get("effective_strategies", [])
        _reuse_targets = _guidance_for_role.get("strategy_reuse_targets", [])
        if (_candidate_role == "strategy_reuse"
                and not _eff
                and not _reuse_targets):
            _candidate_role = "free_explore"
            print(f"[role fallback] {name}: no usable strategy, downgrade to free_explore")

        READING_CONTEXT_CACHE.setdefault(name, {}).update({
            "evidence_maturity": _current_evidence_maturity,
            "governance_mode": _governance_mode,
            "candidate_role": _candidate_role,
            "assigned_memory_ids": _assigned_memory_ids,
            "assigned_avoid_memory_ids": _assigned_avoid_memory_ids,
            "role_context_mode": (
                f"{_candidate_role}_context"
                if _candidate_role else "default"
            ),
        })
        _prompt_metadata: dict = {}
        if _history_budget:
            prompt, _prompt_metadata = _candidate_prompt_optimized(
                iteration, index, task_prompt, candidate_role=_candidate_role)
        else:
            prompt = _candidate_prompt(iteration, index, task_prompt)
        if _plan_manager is not None and _guidance_phase != "none":
            if _guidance_phase == "soft":
                plan_prompt = _plan_manager.generate_soft_plan_prompt(iteration)
            else:
                plan_prompt = _plan_manager.generate_plan_prompt(iteration)
            prompt = plan_prompt + "\n\n" + prompt
        if search_governance_enabled():
            role_prompt = (
                f"## Candidate Role\n"
                f"governance_mode: {_governance_mode}\n"
                f"candidate_role: {_candidate_role}\n"
                "Follow this role when choosing the edit scope. "
                "strategy_reuse should reuse an effective strategy at its recorded evidence scope; "
                "hypothesis_test should isolate one mechanism from an effective strategy; "
                "anchor_refine is a legacy strategy_reuse role; free_explore may explore broadly; "
                "component_exploit should stay within the target component boundary.\n"
            )
            if _candidate_role == "strategy_reuse" and _anchor.get("code_snippet"):
                role_prompt += (
                    f"\n\n## Anchor Code Reference"
                    f" ({_anchor.get('episode_id')},"
                    f" val={_anchor.get('score')}%)\n\n"
                    "This code is a reference for the effective strategy,"
                    " not a mandatory parent.\n"
                    "Reuse the useful idea at its recorded evidence scope."
                    " Do not claim component causality unless"
                    " evidence_scope is component.\n\n"
                    "```python\n"
                    f"{_anchor['code_snippet']}\n"
                    "```\n\n"
                    "Strategy reuse constraints:\n"
                    "1. Preserve the useful mechanism if applicable.\n"
                    "2. Prefer a small targeted change over a full rewrite.\n"
                    "3. If changing multiple components,"
                    " explain why in proposal_plan.\n"
                    "4. Keep API contracts intact.\n"
                )
            _hypo_targets = _guidance_for_role.get("hypothesis_test_targets", [])
            _eff_strategies = _guidance_for_role.get("effective_strategies", [])
            if _candidate_role == "hypothesis_test" and _eff_strategies:
                _best_strat = max(
                    _eff_strategies,
                    key=lambda x: float(x.get("score") or 0),
                )
                _scope = _best_strat.get("evidence_scope", "unknown")
                _comps = _best_strat.get("components_changed", [])
                _eid = _best_strat.get("episode_id", "")
                _score = _best_strat.get("score", 0)
                if _scope == "interaction" and _comps:
                    _target_comp = _comps[0]
                    role_prompt += (
                        f"\n\n## Hypothesis Test Task\n\n"
                        f"Reference strategy: {_eid} (val={_score}%,"
                        f" scope={_scope})\n"
                        f"That strategy changed these components: {_comps}\n\n"
                        f"Your test boundary:\n"
                        f"1. Only modify logic related to {_target_comp}\n"
                        f"2. Keep other components as simple as possible\n"
                        f"3. Do not copy the full implementation of the reference strategy\n"
                        f"4. Goal: verify whether the {_target_comp} direction is independently effective\n\n"
                        f"Notes:\n"
                        f"- Results will be recorded as hypothesis evidence\n"
                        f"- Do not claim causal relationship yet\n"
                        f"- Keep API contracts intact\n"
                    )
            prompt = role_prompt + "\n\n" + prompt
        last_error = ""
        _first_response: str | None = None
        for attempt in range(1, 4):
            effective_prompt = (
                prompt
                if not last_error
                else prompt
                + f"\n\nPrevious import error:\n{last_error}\nReturn a corrected full module."
            )
            _write_proposer_log(
                iteration,
                f"\n## Candidate {index} Attempt {attempt} Prompt\n\n{effective_prompt}\n",
            )
            if backend == "api":
                response, _usage = _proposer_api_complete(proposer_llm, effective_prompt)
            else:
                response, _usage = _proposer_cli_complete(
                    effective_prompt,
                    iteration,
                    name,
                    attempt,
                    timeout=timeout,
                    model=proposer_model,
                )
            _call_index += 1
            _write_proposer_usage(
                _iter_dir(iteration),
                iteration,
                name,
                _call_index,
                attempt,
                _usage,
            )
            _write_proposer_log(
                iteration,
                f"\n## Candidate {index} Attempt {attempt} Raw Output\n\n{response}\n",
            )
            if _first_response is None:
                _first_response = response
            code = extract_python_code(response) if _plan_code_sep else _strip_code_fence(response)
            try:
                patched_code = _write_candidate_module(name, code, iteration=iteration)
                _write_proposer_log(
                    iteration,
                    f"\n## Candidate {index} Attempt {attempt} Patched Code Written To agents/{name}.py\n\n{patched_code}\n",
                )
            except Exception as exc:
                last_error = str(exc)
                _write_proposer_log(
                    iteration,
                    f"\n## Candidate {index} Attempt {attempt} Write Error\n\n{last_error}\n",
                )
                continue
            result = _validate_import(name)
            if result.returncode == 0:
                candidates.append(
                    {
                        "name": name,
                        "file": f"agents/{name}.py",
                        "hypothesis": f"LLM-generated candidate {index} for iteration {iteration}.",
                        "axis": "llm-proposer",
                        "base_system": "generated",
                        "components": [_slug(code[:80], f"candidate_{index}")],
                    }
                )
                print(f"    {_green('OK')} {name} (attempt {attempt})")
                _write_proposer_log(
                    iteration,
                    f"\n## Candidate {index} Validation\n\nOK: {name} attempt {attempt}\n",
                )
                break
            last_error = (result.stderr or result.stdout)[-2000:]
            print(f"    {_yellow('repair')} {name} attempt {attempt}: {last_error[:160]}")
            _write_proposer_log(
                iteration,
                f"\n## Candidate {index} Validation Error\n\n{last_error}\n",
            )

        if _plan_manager is not None and _first_response is not None:
            _plan = _plan_manager.extract_plan_from_output(_first_response)
            if _plan:
                _validation = _plan_manager.validate_plan(_plan)
                if search_governance_enabled():
                    try:
                        from src.memory.steering.constraint_layer import ConstraintLayer as _CL
                        _inter_ev_path = EVO_WORKSPACE.parent / "component_memory" / "inter_component_evidence.json"
                        _inter_ev2: dict = {"confirmed_conflicts": [], "co_change_patterns": [], "confirmed_synergies": []}
                        if _inter_ev_path.exists():
                            try:
                                _inter_ev2 = json.loads(_inter_ev_path.read_text(encoding="utf-8"))
                            except Exception:
                                pass
                        _m4_result = _CL().validate_proposal_plan(
                            _plan, _inter_ev2, governance_mode=_governance_mode
                        )
                        _plan["m4_constraint_validation"] = _m4_result
                        if _m4_result["constraint_severity"] in ("medium", "high"):
                            _evo_log_path = EVO_WORKSPACE.parent / "component_memory" / "evolution_log.jsonl"
                            _evo_log_path.parent.mkdir(parents=True, exist_ok=True)
                            from datetime import datetime as _dt
                            _m4_entry = {
                                "operation": "M4ConstraintWarning",
                                "episode_id": name,
                                "severity": _m4_result["constraint_severity"],
                                "warnings": _m4_result["constraint_warnings"],
                                "timestamp": _dt.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            }
                            with _evo_log_path.open("a", encoding="utf-8") as _ef:
                                _ef.write(json.dumps(_m4_entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                _plan_manager.save_plan(iteration, name, _plan, _validation)
                if _log_context_mode and _prompt_metadata:
                    _plan_manager.append_context_metadata(
                        iteration, name, _prompt_metadata
                    )
                _write_proposer_log(
                    iteration,
                    f"\n## Candidate {index} Proposal Plan\n\n"
                    + json.dumps(_plan, indent=2, ensure_ascii=False)
                    + f"\ncompliance_score: {_validation['compliance_score']}\n"
                    + f"is_valid: {_validation['is_valid']}\n",
                )
            else:
                _write_proposer_log(
                    iteration,
                    f"\n## Candidate {index} Proposal Plan\n\nWarning: proposer did not generate a proposal_plan\n",
                )
        if _log_context_mode and _prompt_metadata:
            _meta_path = _iter_dir(iteration) / f"{name}.context_meta.json"
            _meta_path.write_text(
                json.dumps(_prompt_metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _write_proposer_log(
                iteration,
                f"\n## Candidate {index} Context Metadata\n\n"
                + json.dumps(_prompt_metadata, indent=2)
                + "\n",
            )

    if not candidates:
        _write_proposer_log(iteration, "\n## Final Candidates\n\nNone\n")
        return False

    PENDING_EVAL.write_text(
        json.dumps({"iteration": iteration, "candidates": candidates}, indent=2),
        encoding="utf-8",
    )
    _write_proposer_log(
        iteration,
        "\n## Final Candidates\n\n"
        + json.dumps(candidates, indent=2)
        + f"\n\npending_eval: {PENDING_EVAL}\n"
        + "proposer_usage: "
        + json.dumps(
            proposer_llm.get_usage()
            if proposer_llm is not None
            else {"backend": backend, "model": proposer_model},
            indent=2,
        )
        + "\n",
    )
    print("  CANDIDATES: " + ", ".join(c["name"] for c in candidates))
    return True


def _write_agent(name: str, body: str) -> None:
    path = AGENTS_DIR / f"{name}.py"
    content = textwrap.dedent(body).lstrip()
    path.write_text(content, encoding="utf-8")
    EVO_WORKSPACE.mkdir(parents=True, exist_ok=True)
    (EVO_WORKSPACE / f"{name}.py").write_text(content, encoding="utf-8")


def propose_template_fallback(iteration: int) -> bool:
    print(f"  {_yellow('using static emergency proposer')}")
    names = {
        "label_frequency": f"label_frequency_memory_i{iteration}",
        "token_overlap": f"token_overlap_memory_i{iteration}",
        "balanced_label": f"balanced_label_memory_i{iteration}",
    }
    _write_agent(
        names["label_frequency"],
        '''
        """Label-frequency memory with compact label priors and examples."""

        import json
        from collections import Counter
        from typing import Any

        from ..llm import LLMCallable
        from ..memory_system import MemorySystem, extract_json_field


        class LabelFrequencyMemory(MemorySystem):
            def __init__(self, llm: LLMCallable):
                super().__init__(llm)
                self.counts = Counter()
                self.examples: list[dict[str, str]] = []

            def predict(self, input: str) -> tuple[str, dict[str, Any]]:
                labels = ", ".join(label for label, _ in self.counts.most_common(30))
                demos = "\\n\\n".join(
                    f"Example input: {e['input']}\\nExample answer: {e['target']}"
                    for e in self.examples[-8:]
                )
                prompt = f"""Answer the classification problem.

        Frequent labels observed so far: {labels or "none"}

        {demos}

        Problem:
        {input}

        Return JSON: {{"reasoning": "...", "final_answer": "..."}}"""
                response = self.call_llm(prompt)
                return extract_json_field(response, "final_answer"), {"labels": len(self.counts)}

            def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
                for row in batch_results:
                    target = row["ground_truth"]
                    if isinstance(target, list):
                        self.counts.update(str(x) for x in target)
                    else:
                        self.counts[str(target)] += 1
                    self.examples.append({"input": row["input"], "target": str(target)})

            def get_state(self) -> str:
                return json.dumps({"counts": dict(self.counts), "examples": self.examples})

            def set_state(self, state: str) -> None:
                data = json.loads(state)
                self.counts = Counter(data.get("counts", {}))
                self.examples = data.get("examples", [])
        ''',
    )
    _write_agent(
        names["token_overlap"],
        '''
        """Token-overlap retrieval memory for compact nearest examples."""

        import json
        import re
        from typing import Any

        from ..llm import LLMCallable
        from ..memory_system import MemorySystem, extract_json_field


        def _tokens(text: str) -> set[str]:
            return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+|[\\u4e00-\\u9fff]", text)}


        class TokenOverlapMemory(MemorySystem):
            def __init__(self, llm: LLMCallable):
                super().__init__(llm)
                self.examples: list[dict[str, Any]] = []

            def predict(self, input: str) -> tuple[str, dict[str, Any]]:
                query = _tokens(input)
                ranked = sorted(
                    self.examples,
                    key=lambda e: len(query & set(e["tokens"])),
                    reverse=True,
                )[:10]
                demos = "\\n\\n".join(
                    f"Similar input: {e['input']}\\nAnswer: {e['target']}" for e in ranked
                )
                prompt = f"""Use the retrieved examples to solve the classification problem.

        {demos}

        Problem:
        {input}

        Return JSON: {{"reasoning": "...", "final_answer": "..."}}"""
                response = self.call_llm(prompt)
                return extract_json_field(response, "final_answer"), {"retrieved": len(ranked)}

            def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
                for row in batch_results:
                    self.examples.append(
                        {
                            "input": row["input"],
                            "target": str(row["ground_truth"]),
                            "tokens": sorted(_tokens(row["input"])),
                        }
                    )

            def get_state(self) -> str:
                return json.dumps({"examples": self.examples})

            def set_state(self, state: str) -> None:
                self.examples = json.loads(state).get("examples", [])
        ''',
    )
    _write_agent(
        names["balanced_label"],
        '''
        """Balanced label memory that keeps a small bank per observed label."""

        import json
        import random
        from collections import defaultdict
        from typing import Any

        from ..llm import LLMCallable
        from ..memory_system import MemorySystem, extract_json_field


        class BalancedLabelMemory(MemorySystem):
            def __init__(self, llm: LLMCallable):
                super().__init__(llm)
                self.by_label: dict[str, list[dict[str, str]]] = defaultdict(list)

            def predict(self, input: str) -> tuple[str, dict[str, Any]]:
                rng = random.Random(hash(input) & 0xFFFFFFFF)
                examples = []
                for label in sorted(self.by_label):
                    pool = self.by_label[label]
                    if pool:
                        examples.append(rng.choice(pool))
                rng.shuffle(examples)
                demos = "\\n\\n".join(
                    f"Input: {e['input']}\\nAnswer: {e['target']}" for e in examples[:16]
                )
                prompt = f"""Classify the problem using a balanced set of label examples.

        {demos}

        Problem:
        {input}

        Return JSON: {{"reasoning": "...", "final_answer": "..."}}"""
                response = self.call_llm(prompt)
                return extract_json_field(response, "final_answer"), {"labels": len(self.by_label)}

            def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
                for row in batch_results:
                    labels = row["ground_truth"]
                    if not isinstance(labels, list):
                        labels = [str(labels)]
                    for label in labels:
                        bucket = self.by_label[str(label)]
                        bucket.append({"input": row["input"], "target": str(row["ground_truth"])})
                        del bucket[:-6]

            def get_state(self) -> str:
                return json.dumps({"by_label": dict(self.by_label)})

            def set_state(self, state: str) -> None:
                data = json.loads(state)
                self.by_label = defaultdict(list, data.get("by_label", {}))
        ''',
    )
    candidates = [
        {
            "name": names["label_frequency"],
            "file": f"agents/{names['label_frequency']}.py",
            "hypothesis": "Compact label priors plus recent examples improve cold-start classification.",
            "axis": "exploration",
            "base_system": "no_memory",
            "components": ["label_prior", "recent_examples"],
        },
        {
            "name": names["token_overlap"],
            "file": f"agents/{names['token_overlap']}.py",
            "hypothesis": "Token-overlap retrieval selects more relevant demonstrations than chronological replay.",
            "axis": "exploitation",
            "base_system": "fewshot_all",
            "components": ["retrieval", "nearest_examples"],
        },
        {
            "name": names["balanced_label"],
            "file": f"agents/{names['balanced_label']}.py",
            "hypothesis": "Balancing examples by label reduces majority-label bias.",
            "axis": "exploration",
            "base_system": "fewshot_all",
            "components": ["balanced_memory", "label_coverage"],
        },
    ]
    PENDING_EVAL.write_text(
        json.dumps({"iteration": iteration, "candidates": candidates}, indent=2),
        encoding="utf-8",
    )
    print("  CANDIDATES: " + ", ".join(c["name"] for c in candidates))
    return True


def validate_candidates(candidates):
    valid = []
    for c in candidates:
        name = c["name"]
        result = run_cmd(
            [
                "env",
                f"PYTHONPATH={REPO_ROOT.resolve()}",
                *_python_cmd(),
                "-c",
                f"from src.agents.{name} import *; print('OK')",
            ],
            cwd=str(REPO_ROOT),
            timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"    {_green('OK')} {name}")
            valid.append(c)
        else:
            print(f"    {_red('FAIL')} {name}")
            if result.stderr:
                print(f"      {_dim(result.stderr[:200])}")
    return valid


def update_evolution_summary(
    iteration,
    candidates,
    val_scores,
    propose_time=None,
    bench_time=None,
    wall_time=None,
):
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    pareto = frontier.get("_pareto", [])
    best_val = pareto[0].get("val_accuracy", 0) if pareto else 0

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            avg_val = val_scores.get(name, 0)
            row = {
                "iteration": iteration,
                "system": name,
                "avg_val": round(avg_val, 1),
                "axis": c.get("axis", "?"),
                "hypothesis": c.get("hypothesis", ""),
                "delta": round(avg_val - best_val, 1) if best_val else None,
                "outcome": f"{avg_val:.1f}% ({avg_val - best_val:+.1f})"
                if avg_val > 0
                else "failed",
            }
            if "components" in c:
                row["components"] = c["components"]
            if i == 0 and wall_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1),
                    "wall": round(wall_time, 1),
                }
            f.write(json.dumps(row) + "\n")


def fresh_start():
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    if AGENTS_DIR.exists():
        files = [f for f in AGENTS_DIR.glob("*.py") if f.name not in BASELINE_FILES]
        for f in files:
            f.unlink()
        if files:
            print(f"  Cleared {len(files)} candidate file(s) from agents/")

    for f in [
        EVOLUTION_SUMMARY,
        FRONTIER_VAL,
        LOGS_DIR / "frontier.json",
        PENDING_EVAL,
    ]:
        if f.exists():
            f.unlink()

    if LOGS_DIR.exists():
        val_files = list(LOGS_DIR.rglob("val.json"))
        for f in val_files:
            f.unlink()
        if val_files:
            print(f"  Cleared {len(val_files)} val result files")

    print(f"  {_green('Fresh start')}: cleared generated agents and log files")


def run_evolve_text(args):
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY, EVO_WORKSPACE

    from src.benchmark import get_model_short_name, load_results
    from src.llm import LLM
    from src.model_config import classifier_config

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    datasets = args.datasets if args.datasets else cfg["datasets"]
    global _BENCHMARK_DATASET
    _BENCHMARK_DATASET = datasets[0] if len(datasets) == 1 else None
    classifier_cfg = classifier_config(cfg)
    classifier_model = args.model or classifier_cfg["model"]
    model_short = get_model_short_name(classifier_model)
    proposer_backend_name, proposer_model = _proposer_label(cfg)

    if args.run_name:
        run_name = args.run_name
    else:
        session_id = cfg.get("session_id", "run")
        ds_tag = "-".join(d[:3] for d in datasets)
        run_name = f"{session_id}_{ds_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    LOGS_DIR = REPO_ROOT / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"

    EVO_WORKSPACE = REPO_ROOT / "workspace" / run_name / "evo"
    os.environ["COMPONENT_MEMORY_DIR"] = str(EVO_WORKSPACE.parent / "component_memory")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EVO_WORKSPACE.parent / "component_memory").mkdir(parents=True, exist_ok=True)

    if args.fresh:
        fresh_start()

    print(
        f"{_ts()} {_bold('Evolution (memory systems)')}  "
        f"run={_cyan(run_name)}  classifier={_cyan(classifier_model)}  "
        f"proposer={_cyan(f'{proposer_backend_name}:{proposer_model}')}  "
        f"iters={args.iterations}  datasets={datasets}"
    )


    baselines = cfg["memory_systems"]["baselines"]
    _baseline_timeout = cfg.get("memory_config", {}).get("baseline_timeout", 900)
    if not args.skip_baseline:
        print(f"\n{_ts()} {_bold('Phase 0: Baselines')}  systems={baselines}")
        for bl in baselines:
            if _interrupted:
                break
            print(f"  {_ts()} benchmarking {_bold(bl)}...", flush=True)
            t0 = time.time()
            result = run_benchmark(["--memory", bl], timeout=_baseline_timeout)
            elapsed = time.time() - t0
            if result.returncode == 124:
                print(f"    {_yellow('TIMEOUT')} {bl}: exceeded {_elapsed(_baseline_timeout)}, skipping baseline")
            elif result.returncode != 0:
                print(f"    {_red('FAIL')} {bl}: {result.stderr[:200]}")
            else:
                print(f"    {_green('OK')} ({_elapsed(elapsed)})")

        run_benchmark(["--frontier", "--model", model_short])

        results = load_results(LOGS_DIR, "val.json")
        for bl in baselines:
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, bl)]
                if k in results and results[k].get("accuracy") is not None
            ]
            if accs:
                avg = sum(accs) / len(accs)
                print(f"    {_bold(bl)}: avg_val={_pct(avg)}")


    start_iteration = count_iterations_from_summary() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()

        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        pareto = frontier.get("_pareto", [])
        best_val = pareto[0].get("val_accuracy", 0) if pareto else 0
        best_sys = pareto[0].get("system", "none") if pareto else "none"

        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={best_sys} @ {_pct(best_val * 100 if best_val <= 1 else best_val)}"
        )
        print(f"{'─' * 60}")

        task_prompt = render_task_prompt(iteration, len(datasets))

        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        propose_start = time.time()
        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        ok = propose_candidates(task_prompt, iteration, timeout=args.propose_timeout)
        propose_time = time.time() - propose_start

        if not ok:
            print(
                f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
            )
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        valid_candidates = validate_candidates(candidates)

        if not valid_candidates:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            _write_proposer_log(
                iteration,
                "\n## Candidate Scores\n\nNo valid candidates.\n",
            )
            update_evolution_summary(
                iteration, candidates, {}, propose_time=propose_time
            )
            continue
        print(
            f"  {_green(f'{len(valid_candidates)} valid')} out of {len(candidates)} candidates"
        )

        bench_start = time.time()
        _candidate_timeout = cfg.get("memory_config", {}).get("candidate_timeout", 900)
        _bench_failures: dict[str, str] = {}
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} system(s) x {len(datasets)} datasets"
            f"  (timeout={_elapsed(_candidate_timeout)}/candidate)"
        )
        for ci, c in enumerate(valid_candidates):
            if _interrupted:
                break
            name = c["name"]
            print(
                f"    [{ci + 1}/{len(valid_candidates)}] {_bold(name)}...", flush=True
            )
            t0 = time.time()
            result = run_benchmark(["--memory", name], timeout=_candidate_timeout)
            elapsed = time.time() - t0
            if result.returncode == 124:
                print(f"      {_yellow('TIMEOUT')} exceeded {_elapsed(_candidate_timeout)}, scoring as 0")
                _bench_failures[name] = "timeout"
            elif result.returncode != 0:
                print(f"      {_red('FAIL')} benchmark crashed ({_elapsed(elapsed)})")
                _bench_failures[name] = "crash"
            else:
                print(f"      {_green('OK')} ({_elapsed(elapsed)})")
        bench_time = time.time() - bench_start

        run_benchmark(["--frontier", "--model", model_short])

        val_scores = {}
        results = load_results(LOGS_DIR, "val.json")
        _recorder = EpisodeRecorder() if memory_enabled() else None
        _ev_builder = ComponentEvidenceBuilder() if evidence_quality_enabled() else None
        _ev_ops = EvolutionOperators() if (evidence_quality_enabled() and module_enabled("evolution_operators")) else None


        _ev_guidance_only = (
            EvolutionOperators()
            if (evidence_quality_enabled() and search_governance_enabled() and not module_enabled("evolution_operators"))
            else None
        )
        _plan_auditor = ProposalPlanManager() if search_governance_enabled() else None
        _total_ep_count = len(_recorder.get_all_episodes()) if _recorder else 0
        for c in valid_candidates:
            name = c["name"]
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, name)]
                if k in results and results[k].get("accuracy") is not None
            ]
            val_scores[name] = sum(accs) / len(accs) if accs else 0
            delta = val_scores[name] - (best_val * 100 if best_val <= 1 else best_val)
            delta_str = f"{delta:+.1f}"
            delta_colored = (
                _green(delta_str)
                if delta > 0
                else (_red(delta_str) if delta < 0 else _dim(delta_str))
            )
            print(
                f"    {_bold(name)}: avg_val={_pct(val_scores[name])}  delta={delta_colored}"
            )
            if _recorder is not None:
                _token_cost = 0
                try:
                    mem_path = LOGS_DIR / datasets[0] / name / model_short / "memory.json"
                    if mem_path.exists():
                        _token_cost = len(mem_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
                _ctx_meta: dict = {}
                _meta_path = EVO_WORKSPACE / f"iter_{iteration:02d}" / f"{name}.context_meta.json"
                _plan_json_path = EVO_WORKSPACE / f"iter_{iteration:02d}" / name / "proposal_plan.json"
                if _meta_path.exists():
                    try:
                        _pm = json.loads(_meta_path.read_text(encoding="utf-8"))
                        _ctx_meta = {
                            "history_context_mode": _pm.get("history_context_mode"),
                            "prompt_chars": {
                                "evolution_summary": _pm.get("evolution_summary_chars", 0),
                                "guidance": _pm.get("guidance_chars", 0),
                                "frontier": _pm.get("frontier_chars", 0),
                                "total": sum(_pm.get(k, 0) for k in (
                                    "evolution_summary_chars", "guidance_chars", "frontier_chars"
                                )),
                            },
                        }
                    except Exception:
                        pass
                elif _plan_json_path.exists():
                    try:
                        _plan_rec = json.loads(_plan_json_path.read_text(encoding="utf-8"))
                        _ctx_meta = {
                            "history_context_mode": _plan_rec.get("history_context_mode"),
                            "prompt_chars": _plan_rec.get("prompt_chars"),
                        }
                    except Exception:
                        pass
                _reading_meta = READING_CONTEXT_CACHE.get(name, {})
                _ep = _recorder.record({
                    "episode_id": name,
                    "iteration": iteration,
                    "parent_candidate": best_sys,
                    "score": val_scores[name],
                    "score_delta": delta,
                    "token_cost": _token_cost,
                    "harness_code_path": str(Path(__file__).resolve()),
                    "proposer_output": _extract_proposer_output(iteration, name),
                    "failure_type": _bench_failures.get(name),
                    "reading_mode": _reading_meta.get("reading_mode", "unknown"),
                    "context_chars": _reading_meta.get("context_chars", {}),
                    "evidence_maturity": _reading_meta.get("evidence_maturity", {}),
                    "governance_mode": _reading_meta.get("governance_mode"),
                    "candidate_role": _reading_meta.get("candidate_role"),
                    "assigned_memory_ids": _reading_meta.get("assigned_memory_ids", []),
                    "assigned_avoid_memory_ids": _reading_meta.get(
                        "assigned_avoid_memory_ids", []
                    ),
                    **_ctx_meta,
                })
                _audit = None
                if _plan_auditor is not None:
                    _audit = _plan_auditor.audit_saved_plan(
                        iteration, name, _ep.get("diff_from_parent", "")
                    )
                    if _audit is not None:
                        _write_proposer_log(
                            iteration,
                            f"\n## Candidate {name} Proposal Plan Audit\n\n"
                            + json.dumps(_audit, indent=2, ensure_ascii=False)
                            + "\n",
                        )
                if _ev_builder is not None:
                    _ev_builder.update(_ep)
                if evidence_quality_enabled():
                    try:
                        from src.memory.updating.memory_utility import MemoryUtilityTracker
                        _utility_tracker = MemoryUtilityTracker()
                        _utility_tracker.update_after_episode(_ep)
                        if _reading_meta.get("candidate_role") == "avoid_retest":
                            for _avoid_mid in _reading_meta.get("assigned_avoid_memory_ids", []):
                                _utility_tracker.update_avoid_after_retest(
                                    memory_id=_avoid_mid,
                                    score_delta=delta,
                                    iteration=iteration,
                                )
                    except Exception as _mu_err:
                        print(f"    [memory_utility] warning: {_mu_err}")
                if _ev_ops is not None:
                    _total_ep_count += 1
                    _ev_ops.run(_ep, _total_ep_count)
                elif _ev_guidance_only is not None:


                    _ev_guidance_only.log_save_event(_ep)
                    _ev_guidance_only.generate_search_guidance(_ep.get("iteration", 0))

                if evidence_quality_enabled():
                    try:
                        from src.memory.updating.guidance_tracker import GuidanceTracker
                        _ga_tracker = GuidanceTracker()
                        _audit_passed = True
                        if _audit is not None:
                            _audit_passed = bool(
                                _audit.get("actual_single_component_edit", True)
                            )
                        else:
                            _plan_path = EVO_WORKSPACE / f"iter_{iteration:02d}" / name / "proposal_plan.json"
                            if _plan_path.exists():
                                try:
                                    _saved = json.loads(_plan_path.read_text(encoding="utf-8"))
                                    _audit_passed = bool(
                                        _saved.get("compliance", {})
                                        .get("audit", {})
                                        .get("actual_single_component_edit", True)
                                    )
                                except Exception:
                                    pass
                        _guidance_path = (
                            EVO_WORKSPACE.parent / "component_memory" / "search_guidance.json"
                        )
                        _cur_guidance: dict = {}
                        if _guidance_path.exists():
                            try:
                                _cur_guidance = json.loads(
                                    _guidance_path.read_text(encoding="utf-8")
                                )
                            except Exception:
                                pass
                        _plan_for_match: dict = {}
                        _plan_path2 = EVO_WORKSPACE / f"iter_{iteration:02d}" / name / "proposal_plan.json"
                        if _plan_path2.exists():
                            try:
                                _saved2 = json.loads(_plan_path2.read_text(encoding="utf-8"))
                                _plan_for_match = _saved2.get("plan", {})
                            except Exception:
                                pass
                        if _plan_for_match:
                            _hp_list = _cur_guidance.get("high_priority", [])
                            _matched_gid = _ga_tracker.match_guidance(
                                _plan_for_match, _hp_list
                            )
                            if _matched_gid:
                                _ga_tracker.update_advantage(
                                    guidance_id=_matched_gid,
                                    episode=_ep,
                                    audit_passed=_audit_passed,
                                )
                        if _total_ep_count % 5 == 0:
                            _pruned = _ga_tracker.prune_guidance()
                            if _pruned > 0:
                                print(f"    Pruned {_pruned} ineffective guidance entries")
                    except Exception as _ga_err:
                        print(f"    [guidance_tracker] warning: {_ga_err}")

                if evidence_quality_enabled():
                    _comps = identify_components(_ep.get("diff_from_parent", ""))["components_changed"]
                    InterComponentEvidence().update(_ep, _comps)
                    InterComponentEvidence().check_synergies()

        _write_proposer_log(
            iteration,
            "\n## Candidate Scores\n\n"
            + json.dumps(val_scores, indent=2)
            + "\n",
        )

        wall_time = time.time() - iter_start
        update_evolution_summary(
            iteration,
            valid_candidates,
            val_scores,
            propose_time=propose_time,
            bench_time=bench_time,
            wall_time=wall_time,
        )

        improved = any(
            v > (best_val * 100 if best_val <= 1 else best_val)
            for v in val_scores.values()
        )
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )


    if _interrupted:
        return

    print(f"\n{_ts()} {_bold('Phase Final: Test evaluation')}")

    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    pareto = frontier.get("_pareto", [])

    test_systems = set(baselines)
    for entry in pareto:
        test_systems.add(entry["system"])
    for key, val in frontier.items():
        if not key.startswith("_") and isinstance(val, dict) and "best_system" in val:
            test_systems.add(val["best_system"])

    for name in sorted(test_systems):
        if name.startswith("mem_agent_"):
            for dataset in datasets:
                stale_test = (
                    REPO_ROOT
                    / "results"
                    / dataset
                    / name
                    / model_short
                    / "test.json"
                )
                stale_test.unlink(missing_ok=True)
        print(f"  {_ts()} test eval: {_bold(name)}", flush=True)
        result = run_benchmark(["--memory", name, "--test"])
        if result.returncode != 0:
            print(f"    {_red('FAIL')} {name} test eval failed")

    run_benchmark(["--frontier", "--test", "--model", model_short])
    current_test_frontier = MetricsCollector()._load_current_run_test_frontier(LOGS_DIR)
    _print_final_test_frontier(
        LOGS_DIR / "frontier.json",
        allowed_systems=test_systems,
        current_test_frontier=current_test_frontier,
    )

    result = run_benchmark(["--results"])
    if result.stdout:
        print(result.stdout)

    print(f"\n{_ts()} {_bold('Evolution complete.')}")

    session_id = cfg.get("session_id", run_name)
    if session_id:
        try:
            _metrics_collector = MetricsCollector()
            metrics = _metrics_collector.collect(session_id, logs_dir=LOGS_DIR)
            _metrics_collector.save(metrics)
            print(f"{_ts()} Metrics saved → workspace/memory/metrics.jsonl  (session={session_id})")
        except Exception as _me:
            print(f"{_ts()} {_yellow('Warning')}: metrics collection failed: {_me}")











def _rate_str(rate):
    s = f"{rate:.0%}"
    if rate >= 0.75:
        return _green(s)
    elif rate >= 0.25:
        return _yellow(s)
    return _red(s)

def _find_dotenv() -> Path | None:
    here = Path(__file__).parent
    for candidate in [here, here.parent, here.parent.parent]:
        p = candidate / ".env"
        if p.exists():
            return p
    return None

_dotenv_path = _find_dotenv()

if _dotenv_path:
    load_dotenv(_dotenv_path, override=True)

def _render_reading_context_terminal(context: dict) -> str:
    parts = []
    if context.get("search_guidance"):
        parts.append("## Search Guidance")
        parts.append(json.dumps(context["search_guidance"], ensure_ascii=False, indent=2))
    if context.get("component_evidence"):
        parts.append("## Component Evidence")
        parts.append(json.dumps(context["component_evidence"], ensure_ascii=False, indent=2))
    if context.get("component_playbook"):
        parts.append("## Component Playbook")
        parts.append(json.dumps(context["component_playbook"], ensure_ascii=False, indent=2))
    if context.get("effective_strategies"):
        parts.append("## Effective Strategies")
        parts.append(json.dumps(context["effective_strategies"], ensure_ascii=False, indent=2))
    if context.get("interaction_strategies"):
        parts.append("## Interaction Strategies")
        parts.append(json.dumps(context["interaction_strategies"], ensure_ascii=False, indent=2))
    if context.get("strategy_reuse_targets"):
        parts.append("## Strategy Reuse Targets")
        parts.append(json.dumps(context["strategy_reuse_targets"], ensure_ascii=False, indent=2))
    if context.get("hypothesis_test_targets"):
        parts.append("## Hypothesis Test Targets")
        parts.append(json.dumps(context["hypothesis_test_targets"], ensure_ascii=False, indent=2))
    if context.get("positive_anchors"):
        parts.append("## Positive Anchors")
        parts.append(json.dumps(context["positive_anchors"], ensure_ascii=False, indent=2))
    if context.get("anchor_refinement_targets"):
        parts.append("## Anchor Refinement Targets")
        parts.append(json.dumps(context["anchor_refinement_targets"], ensure_ascii=False, indent=2))
    if context.get("recent_episodes"):
        parts.append("## Recent Episodes")
        parts.append(json.dumps(context["recent_episodes"], ensure_ascii=False, indent=2))
    if context.get("evolution_summary"):
        parts.append("## Evolution Summary")
        parts.append(context["evolution_summary"])
    return "\n\n".join(parts)

JOBS_DIR = REPO_ROOT / "jobs"

AGENTS_DIR_TERMINAL = EVOLVE_DIR / "harness_agents"

BASELINES = [
    ("kira-baseline", "src.harness_agents.baseline_kira:AgentHarness"),
    ("terminus2-baseline", "src.harness_agents.baseline_terminus2:AgentHarness"),
]

BASELINE_AGENT_NAME = BASELINES[0][0]

BASELINE_IMPORT_PATH = BASELINES[0][1]

DATASET_CONFIGS = {
    "terminal-bench@2.0": {
        "harbor_dataset": "terminal-bench@2.0",
        "n_tasks": 89,
        "description": "Terminal-Bench 2 (89 tasks)",
    },
    "swebench-verified": {
        "harbor_dataset": "swebench-verified",
        "n_tasks": 500,
        "description": "SWE-bench Verified (500 tasks)",
    },
}

DEFAULT_DATASET = "terminal-bench@2.0"

_cfg_path = REPO_ROOT / "config.yaml"

_cfg = yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}

ACTIVE_DATASET = _cfg.get("terminal_dataset", DEFAULT_DATASET)

DATASET_CFG = DATASET_CONFIGS.get(ACTIVE_DATASET, DATASET_CONFIGS[DEFAULT_DATASET])

N_EVAL_TASKS = DATASET_CFG["n_tasks"]

SMOKE_TASK_BY_DATASET = {
    "terminal-bench@2.0": "extract-elf",
    "swebench-verified": "astropy__astropy-12907",
}

SMOKE_TASK = SMOKE_TASK_BY_DATASET.get(ACTIVE_DATASET, None)

DATASET_TASK_DESCRIPTION = {
    "terminal-bench@2.0": (
        "You are optimizing a terminal agent harness "
        "for Terminal-Bench 2 tasks. "
        "Tasks involve shell commands, file operations, "
        "and terminal interactions in a Docker environment. "
        "The agent class is AgentHarness in src/harness_agents/*.py."
    ),
    "swebench-verified": (
        "You are optimizing a coding agent harness "
        "for SWE-bench Verified tasks. "
        "Tasks involve fixing Python repository issues "
        "by modifying source code and passing unit tests. "
        "The agent class is AgentHarness in src/harness_agents/*.py."
    ),
}

TASK_CONTEXT = DATASET_TASK_DESCRIPTION.get(
    ACTIVE_DATASET, DATASET_TASK_DESCRIPTION["terminal-bench@2.0"]
)

DATASET = ACTIVE_DATASET

MODEL = "anthropic/claude-opus-4-6"

DEFAULT_SEARCH_TRIALS = 2

DEFAULT_CONCURRENCY = 1

def _task_outcome_memory_enabled_for_dataset(dataset: str) -> bool:
    return dataset == "terminal-bench@2.0"

os.environ["HARNESSBRAIN_ENABLE_TASK_OUTCOME_MEMORY"] = (
    "1" if _task_outcome_memory_enabled_for_dataset(ACTIVE_DATASET) else "0"
)

PROPOSER_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Agent",
    "Write",
    "Edit",
    "Bash",
]

def run_cmd_terminal(cmd, timeout=7200, cwd=None):
    env = os.environ.copy()
    env["HARBOR_MODEL"] = MODEL
    for key in ("RUNLOOP_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True, env=env
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"Timed out after {timeout}s")

def harbor_run(import_path, job_name, n_trials=2, n_concurrent=10):
    cmd = [
        str(REPO_ROOT / "scripts" / "run_eval.sh"),
        import_path,
        ACTIVE_DATASET,
        str(n_trials),
        str(n_concurrent),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(JOBS_DIR),
    ]

    env = os.environ.copy()
    env["HARBOR_MODEL"] = MODEL

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            timeout=14400,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(cmd, 124, "", "Timed out after 14400s")

    job_dir = JOBS_DIR / job_name
    if result.returncode not in (0, 124):
        print(f"  {_red('harbor failed')} exit={result.returncode} job={job_name}")
        if result.stderr:
            print(f"  {result.stderr[:500]}")
        return job_dir, None

    if result.returncode == 124:
        print(
            f"  {_yellow('harbor timed out')} job={job_name}, reading partial results"
        )

    return job_dir, True

def parse_job_results(job_dir, expected_trials=None):
    task_rewards = {}

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]

        rf = trial_dir / "result.json"
        if not rf.exists():
            task_rewards.setdefault(task, []).append(0.0)
            continue
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            task_rewards.setdefault(task, []).append(0.0)
            continue

        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        task_rewards.setdefault(task, []).append(
            float(reward) if reward is not None else 0.0
        )

    if expected_trials:
        for task, rewards in task_rewards.items():
            if len(rewards) != expected_trials:
                print(
                    f"  {_yellow('warning')}: {task} has {len(rewards)}/{expected_trials} trials"
                )

    return task_rewards

def compute_pass_rates(task_rewards):
    per_task = {}
    total_passes = 0
    total_trials = 0
    for task, rewards in task_rewards.items():
        per_task[task] = sum(r > 0 for r in rewards) / len(rewards) if rewards else 0.0
        total_passes += sum(r > 0 for r in rewards)
        total_trials += len(rewards)

    avg = total_passes / total_trials if total_trials else 0.0
    return per_task, avg

def parse_trial_metrics(job_dir):
    per_task = {}
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]
        rf = trial_dir / "result.json"
        if not rf.exists():
            continue
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ar = r.get("agent_result") or {}
        md = ar.get("metadata") or {}
        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        metrics = {
            "n_input_tokens": ar.get("n_input_tokens"),
            "n_output_tokens": ar.get("n_output_tokens"),
            "n_cache_tokens": ar.get("n_cache_tokens"),
            "cost_usd": ar.get("cost_usd"),
            "n_turns": md.get("n_episodes"),
            "n_api_calls": len(md.get("api_request_times_msec", [])),
            "reward": reward,
        }
        per_task.setdefault(task, []).append(metrics)
    return per_task

def summarize_trial_metrics(trial_metrics):
    all_costs = []
    all_input = []
    all_output = []
    all_cache = []
    all_turns = []
    per_task_summary = {}

    for task, trials in trial_metrics.items():
        task_costs = [t["cost_usd"] for t in trials if t["cost_usd"] is not None]
        task_turns = [t["n_turns"] for t in trials if t["n_turns"] is not None]
        per_task_summary[task] = {
            "mean_cost": round(sum(task_costs) / len(task_costs), 3)
            if task_costs
            else None,
            "mean_turns": round(sum(task_turns) / len(task_turns), 1)
            if task_turns
            else None,
            "n_trials": len(trials),
        }
        all_costs.extend(task_costs)
        all_turns.extend(task_turns)
        all_input.extend(
            t["n_input_tokens"] for t in trials if t["n_input_tokens"] is not None
        )
        all_output.extend(
            t["n_output_tokens"] for t in trials if t["n_output_tokens"] is not None
        )
        all_cache.extend(
            t["n_cache_tokens"] for t in trials if t["n_cache_tokens"] is not None
        )

    n_trials = sum(len(v) for v in trial_metrics.values())
    return {
        "n_trials": n_trials,
        "total_cost_usd": round(sum(all_costs), 2) if all_costs else None,
        "mean_cost_usd": round(sum(all_costs) / len(all_costs), 3)
        if all_costs
        else None,
        "total_input_tokens": sum(all_input) if all_input else None,
        "total_output_tokens": sum(all_output) if all_output else None,
        "total_cache_tokens": sum(all_cache) if all_cache else None,
        "mean_turns": round(sum(all_turns) / len(all_turns), 1) if all_turns else None,
        "per_task": per_task_summary,
    }

def count_iterations():
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter

def update_frontier(candidates_results, metrics=None):
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    metrics = metrics or {}

    for agent_name, (per_task, avg) in candidates_results.items():
        for task, rate in per_task.items():
            current_best = frontier.get(task, {}).get("pass_rate", -1)
            if rate > current_best:
                frontier[task] = {
                    "best_agent": agent_name,
                    "pass_rate": rate,
                }

        current_best_avg = frontier.get("_best", {}).get("avg_pass_rate", -1)
        if avg > current_best_avg:
            frontier["_best"] = {
                "agent": agent_name,
                "avg_pass_rate": avg,
            }

    FRONTIER_VAL.write_text(json.dumps(frontier, indent=2))

def update_evolution_summary_terminal(
    iteration, candidates, results, propose_time=None, bench_time=None, metrics=None
):
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    best_avg = frontier.get("_best", {}).get("avg_pass_rate", 0)
    metrics = metrics or {}

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            per_task, avg = results.get(name, ({}, 0))
            row = {
                "iteration": iteration,
                "agent": name,
                "import_path": c.get("import_path", ""),
                "avg_pass_rate": round(avg, 3),
                "per_task": {k: round(v, 3) for k, v in per_task.items()},
                "hypothesis": c.get("hypothesis", ""),
                "changes": c.get("changes", ""),
                "delta": round(avg - best_avg, 3) if best_avg else None,
                "outcome": f"{avg:.1%} ({avg - best_avg:+.1%})"
                if avg > 0
                else "failed",
            }
            if i == 0 and propose_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1) if bench_time else None,
                }
            if name in metrics:
                row["rollout_metrics"] = metrics[name]
            f.write(json.dumps(row) + "\n")

def propose_claude(task_prompt, iteration, timeout=2400, model=None):
    model = model or _proposer_claude_model()
    os.environ.pop("CLAUDECODE", None)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        result = claude_wrapper.run(
            prompt=task_prompt,
            model=model,
            allowed_tools=PROPOSER_ALLOWED_TOOLS,
            skills=[str(REPO_ROOT / ".claude" / "skills" / "agent")],
            cwd=str(REPO_ROOT),
            log_dir=str(LOGS_DIR / "claude_sessions"),
            name=f"iter{iteration}",
            timeout_seconds=timeout,
            effort="max",
        )
    finally:
        if saved_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    return PENDING_EVAL.exists()


def _extract_json_payload(text: str) -> dict:
    for match in re.finditer(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        try:
            payload = json.loads(match.group(1).strip())
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _module_name(value: str | None, fallback: str) -> str:
    name = _slug(value or fallback, fallback)
    if name[0].isdigit():
        name = f"agent_{name}"
    return name


def propose_api_agent(task_prompt, iteration, timeout=2400):
    backend, proposer_model = _proposer_label()
    assert backend == "api"
    llm = _proposer_llm()
    api_prompt = (
        task_prompt
        + "\n\n## API proposer output contract\n\n"
        "You are running through an API and cannot write files or execute tools. "
        "Return exactly one candidate. Provide one fenced JSON block followed by "
        "one fenced Python block.\n\n"
        "The JSON block must contain: name, hypothesis, changes, expected_efficiency.\n"
        "The Python block must be the complete contents of "
        "`src/harness_agents/<name>.py`, define class `AgentHarness`, and remain "
        "importable as `src.harness_agents.<name>:AgentHarness`.\n"
    )
    try:
        response, usage = _proposer_api_complete(llm, api_prompt)
    except Exception as exc:
        print(f"  {_red('API proposer failed')}: {exc}")
        return False

    log_path = LOGS_DIR / "proposer_api_log.txt"
    log_path.write_text(
        "\n".join(
            [
                f"# Iteration {iteration} API Proposer Log",
                f"timestamp: {datetime.now().isoformat()}",
                f"proposer_backend: api",
                f"proposer_model: {proposer_model}",
                f"usage: {json.dumps(usage, indent=2)}",
                "",
                response,
            ]
        ),
        encoding="utf-8",
    )

    payload = _extract_json_payload(response)
    name = _module_name(payload.get("name"), f"api_agent_i{iteration}_1")
    code = extract_python_code(response)
    if "class AgentHarness" not in code:
        print(f"  {_red('API proposer failed')}: no AgentHarness class in response")
        return False

    AGENTS_DIR_TERMINAL.mkdir(parents=True, exist_ok=True)
    candidate_path = AGENTS_DIR_TERMINAL / f"{name}.py"
    candidate_path.write_text(textwrap.dedent(code).lstrip(), encoding="utf-8")

    candidate = {
        "name": name,
        "import_path": f"src.harness_agents.{name}:AgentHarness",
        "hypothesis": str(payload.get("hypothesis") or "API-generated agent candidate."),
        "changes": str(payload.get("changes") or ""),
        "expected_efficiency": str(payload.get("expected_efficiency") or ""),
    }
    PENDING_EVAL.write_text(
        json.dumps({"iteration": iteration, "candidates": [candidate]}, indent=2),
        encoding="utf-8",
    )
    print(f"  CANDIDATES: {name}")
    return True

def validate_candidate(name, import_path):
    module_path = import_path.split(":")[0]
    result = run_cmd_terminal(
        [*_python_cmd(), "-c", f"from {module_path} import *; print('OK')"],
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        return True
    print(f"  {_red('import FAIL')}: {name}")
    if result.stderr:
        print(f"    {_dim(result.stderr[:300])}")
    return False

def smoke_test(name, import_path, timeout=1800):
    job_name = f"smoke-{name}"
    job_dir = JOBS_DIR / job_name
    if job_dir.exists():
        run_cmd_terminal(["rm", "-rf", str(job_dir)])

    cmd = [
        str(REPO_ROOT / "scripts" / "run_eval.sh"),
        import_path,
        ACTIVE_DATASET,
        "1",
        "1",
    ]
    if SMOKE_TASK:
        cmd += ["-i", SMOKE_TASK]
    cmd += [
        "--job-name",
        job_name,
        "--jobs-dir",
        str(JOBS_DIR),
    ]
    t0 = time.time()
    result = run_cmd_terminal(cmd, timeout=timeout, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(
            f"  {_red('smoke FAIL')}: {name} exit={result.returncode} ({_elapsed(elapsed)})"
        )
        if result.stderr:
            print(f"    {_dim(result.stderr[:300])}")
        return False

    result_file = job_dir / "result.json"
    if not result_file.exists():
        print(f"  {_red('smoke FAIL')}: {name} (no result.json, {_elapsed(elapsed)})")
        return False

    data = json.loads(result_file.read_text())
    n_errors = data.get("stats", {}).get("n_errors", 0)
    if n_errors > 0:
        print(
            f"  {_red('smoke FAIL')}: {name} ({n_errors} errors, {_elapsed(elapsed)})"
        )
        return False

    print(f"  {_green('smoke OK')}: {name} ({_elapsed(elapsed)})")
    return True

def render_task_prompt_terminal(iteration, n_trials):
    return (
        f"{TASK_CONTEXT}\n\n"
        f"Run iteration {iteration} of the scaffold evolution loop (KIRA track). "
        f"Model: {MODEL} (Opus). "
        f"Start from src/harness_agents/baseline_kira.py as the parent.\n\n"
        f"## Dataset: {ACTIVE_DATASET} ({N_EVAL_TASKS} tasks x {n_trials} trials)\n\n"
        f"Focus on scaffold changes that help the agent solve complex, long-horizon tasks.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        f"- `{LOGS_DIR / 'evolution_summary.jsonl'}` — past results\n"
        f"- `{LOGS_DIR / 'frontier_val.json'}` — frontier\n"
        f"- `{LOGS_DIR / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`"
    )

def fresh_start_terminal():
    if AGENTS_DIR_TERMINAL.exists():
        for f in AGENTS_DIR_TERMINAL.iterdir():
            if f.name in (
                "__pycache__",
                "__init__.py",
                "baseline_kira.py",
                "baseline_terminus2.py",
            ):
                continue
            if f.is_dir():
                run_cmd_terminal(["rm", "-rf", str(f)])
                print(f"  Cleared {f.name}/")
            elif f.suffix == ".py":
                f.unlink()
                print(f"  Cleared {f.name}")

    for f in [EVOLUTION_SUMMARY, FRONTIER_VAL, PENDING_EVAL]:
        if f.exists():
            f.unlink()

    print("  Fresh start: cleared generated agents/ files and log files")

def run_evolve_terminal(args):
    global JOBS_DIR, LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    proposer_backend_name, proposer_model = _proposer_label(cfg)

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    JOBS_DIR = REPO_ROOT / "jobs" / run_name
    LOGS_DIR = REPO_ROOT / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR_TERMINAL.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        fresh_start_terminal()

    n_tasks = N_EVAL_TASKS
    print(
        f"{_ts()} {_bold('Evolution (KIRA track)')}  run={_cyan(run_name)}  "
        f"model={_cyan(MODEL)}  proposer={_cyan(f'{proposer_backend_name}:{proposer_model}')}  "
        f"iters={args.iterations}  trials={args.trials}  tasks={n_tasks}"
    )


    baseline_dirs = {}
    if not args.skip_baseline:
        print(
            f"\n{_ts()} {_bold('Phase 0: Baselines')}  agents={len(BASELINES)}  trials={args.trials}"
        )

    for bl_name, bl_import in BASELINES:
        bl_job = f"{bl_name}-t{args.trials}"
        bl_dir = JOBS_DIR / bl_job
        baseline_dirs[bl_name] = bl_dir

        if not args.skip_baseline:
            cached_ok = False
            if bl_dir.exists() and parse_job_results(bl_dir):
                cfg_file = bl_dir / "config.json"
                if cfg_file.exists():
                    try:
                        cfg = json.loads(cfg_file.read_text())
                        cfg_model = cfg.get("model") or cfg.get("agent", {}).get(
                            "model", ""
                        )
                        cfg_attempts = cfg.get("n_attempts", 0)
                        if MODEL not in str(cfg_model):
                            print(
                                f"  {_yellow('stale')} {bl_name}: model mismatch (cached={cfg_model}, want={MODEL}), re-running"
                            )
                        elif cfg_attempts != args.trials:
                            print(
                                f"  {_yellow('stale')} {bl_name}: trials mismatch (cached={cfg_attempts}, want={args.trials}), re-running"
                            )
                        else:
                            cached_ok = True
                    except (json.JSONDecodeError, OSError):
                        pass
                else:
                    cached_ok = True
            if cached_ok:
                print(f"  {_dim('cached')} {bl_name}: {bl_dir}")
            else:
                print(
                    f"  {_ts()} running {_bold(bl_name)}: {n_tasks} tasks x {args.trials} trials...",
                    flush=True,
                )
                t0 = time.time()
                bl_dir, ok = harbor_run(
                    bl_import,
                    bl_job,
                    n_trials=args.trials,
                    n_concurrent=args.concurrent,
                )
                baseline_dirs[bl_name] = bl_dir
                elapsed = time.time() - t0
                if not ok:
                    print(
                        f"  {_red('FAIL')} {bl_name} crashed after {_elapsed(elapsed)}"
                    )
                else:
                    print(f"  {_ts()} {bl_name} completed in {_elapsed(elapsed)}")

    for bl_name, bl_dir in baseline_dirs.items():
        if bl_dir.exists():
            task_rewards = parse_job_results(bl_dir, expected_trials=args.trials)
            if task_rewards:
                per_task, avg = compute_pass_rates(task_rewards)
                update_frontier({bl_name: (per_task, avg)})
                if (
                    memory_enabled()
                    and _task_outcome_memory_enabled_for_dataset(ACTIVE_DATASET)
                    and TaskOutcomeMemory is not None
                    and collect_task_outcomes is not None
                ):
                    try:
                        _baseline_outcomes = collect_task_outcomes(
                            bl_dir,
                            task_rewards=task_rewards,
                            expected_trials=args.trials,
                        )
                        TaskOutcomeMemory().update_episode({
                            "episode_id": bl_name,
                            "iteration": 0,
                            "parent_candidate": None,
                            "score": round(avg * 100, 2),
                            "compare_to_previous": False,
                            "task_outcomes": _baseline_outcomes.get("tasks", {}),
                            "task_outcome_summary": _baseline_outcomes.get("summary", {}),
                        })
                    except Exception as _task_mem_err:
                        print(
                            f"  {_dim(f'task memory seed skipped: {_task_mem_err}')}"
                        )
                if not args.skip_baseline:
                    print(f"  {_bold(bl_name)}: avg={_rate_str(avg)}")
                    max_name = max(len(t) for t in per_task) if per_task else 0
                    for task, rate in sorted(per_task.items()):
                        print(f"    {task:<{max_name}}  {_rate_str(rate)}")


    start_iteration = count_iterations() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        best_avg = frontier.get("_best", {}).get("avg_pass_rate", 0)
        best_agent = frontier.get("_best", {}).get("agent", "none")
        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  frontier={best_agent} @ {best_avg:.1%}"
        )
        print(f"{'─' * 60}")

        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        propose_start = time.time()
        task_prompt = render_task_prompt_terminal(iteration, args.trials)


        _mode = "full_history"
        _recommended_role = "free_explore"
        _governance_mode = "audit_only_exploration"
        if memory_efficiency_enabled() or search_governance_enabled():
            try:
                _guidance_path = REPO_ROOT / "workspace" / "component_memory" / "search_guidance.json"
                _guidance: dict = {}
                if _guidance_path.exists():
                    _guidance = json.loads(_guidance_path.read_text(encoding="utf-8"))

                _warm_up_iters = get_memory_config().get("warm_up", {}).get("warm_up_iterations", 2)

                if memory_efficiency_enabled():
                    from src.memory.steering.reading_strategy import ReadingStrategyManager
                    _strategy = ReadingStrategyManager()
                    _mode = _strategy.decide_reading_mode(iteration, _guidance, _warm_up_iters)
                    _ctx = _strategy.build_context(
                        mode=_mode,
                        guidance=_guidance,
                        iteration=iteration,
                        evolution_summary_path=EVOLUTION_SUMMARY,
                    )
                    _rendered = _render_reading_context_terminal(_ctx)
                    if _rendered:
                        task_prompt = task_prompt + "\n\n" + _rendered

                if search_governance_enabled() and _guidance:
                    from src.memory.steering.constraint_layer import ConstraintLayer
                    _gov = ConstraintLayer().choose_governance_mode(_guidance)
                    _governance_mode = _gov.get("mode", _governance_mode)
                    _roles = _gov.get("candidate_roles", [])
                    _recommended_role = _roles[0] if _roles else "free_explore"
                    if _recommended_role and _recommended_role != "free_explore":
                        task_prompt = (
                            f"[candidate_role: {_recommended_role}]\n\n" + task_prompt
                        )
            except Exception as _ph2_err:
                print(f"  {_dim(f'Phase 2 context skipped: {_ph2_err}')}")

        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        if proposer_backend_name == "api":
            ok = propose_api_agent(task_prompt, iteration, timeout=args.propose_timeout)
        else:
            ok = propose_claude(
                task_prompt,
                iteration,
                timeout=args.propose_timeout,
                model=proposer_model,
            )
        propose_time = time.time() - propose_start

        if not ok:
            print(
                f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
            )
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        for c in candidates:
            if "import_path" in c and ":" in c["import_path"]:
                module, _ = c["import_path"].rsplit(":", 1)
                c["import_path"] = f"{module}:AgentHarness"
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")


        valid = []
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        for ci, c in enumerate(candidates):
            name = c["name"]
            import_path = c["import_path"]
            prefix = f"    [{ci + 1}/{len(candidates)}] {name}:"
            if validate_candidate(name, import_path):
                if args.skip_smoke:
                    print(f"{prefix} {_green('import OK')} (smoke skipped)")
                    valid.append(c)
                elif smoke_test(name, import_path):
                    print(f"{prefix} {_green('import OK + smoke OK')}")
                    valid.append(c)
                else:
                    print(f"{prefix} {_red('smoke FAIL')}")
            else:
                print(f"{prefix} {_red('import FAIL')}")
            if _interrupted:
                break

        if not valid:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            update_evolution_summary_terminal(
                iteration, candidates, {}, propose_time=propose_time
            )
            continue
        print(f"  {_green(f'{len(valid)} valid')} out of {len(candidates)} candidates")


        bench_start = time.time()
        results = {}
        all_metrics = {}
        n_evals = len(valid) * n_tasks * args.trials
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid)} agent(s) x {n_tasks} tasks x {args.trials} trials = {n_evals} evals"
        )
        for ci, c in enumerate(valid):
            if _interrupted:
                break
            name = c["name"]
            import_path = c["import_path"]
            job_name = f"evolve-{name}-t{args.trials}"

            print(f"    [{ci + 1}/{len(valid)}] {_bold(name)}...", flush=True)
            t0 = time.time()
            job_dir, job_result = harbor_run(
                import_path,
                job_name,
                n_trials=args.trials,
                n_concurrent=args.concurrent,
            )
            elapsed = time.time() - t0
            if job_result:
                task_rewards = parse_job_results(job_dir, expected_trials=args.trials)
                per_task, avg = compute_pass_rates(task_rewards)
                results[name] = (per_task, avg)
                task_outcome_payload = {}
                if (
                    _task_outcome_memory_enabled_for_dataset(ACTIVE_DATASET)
                    and collect_task_outcomes is not None
                ):
                    task_outcome_payload = collect_task_outcomes(
                        job_dir,
                        task_rewards=task_rewards,
                        expected_trials=args.trials,
                    )


                if memory_enabled():
                    try:
                        _recorder = EpisodeRecorder()
                        _score = round(avg * 100, 2)
                        _prev_best = max(
                            (ep.get("score", 0) for ep in _recorder.get_all_episodes()),
                            default=best_avg * 100,
                        )
                        _ep_info = {
                            "episode_id": name,
                            "iteration": iteration,
                            "parent_candidate": best_agent,
                            "score": _score,
                            "score_delta": _score - _prev_best,
                            "candidate_role": _recommended_role,
                            "reading_mode": _mode,
                            "governance_mode": _governance_mode,



                            "job_dir": str(job_dir),
                        }
                        if _task_outcome_memory_enabled_for_dataset(ACTIVE_DATASET):
                            _ep_info.update({
                                "per_task_results": {
                                    task: round(rate, 4)
                                    for task, rate in sorted(per_task.items())
                                },
                                "task_outcomes": task_outcome_payload.get("tasks", {}),
                                "task_outcome_summary": task_outcome_payload.get(
                                    "summary", {}
                                ),
                            })
                            if TaskOutcomeMemory is not None:
                                _task_updates = TaskOutcomeMemory().update_episode(
                                    _ep_info
                                )
                                _ep_info.update(_task_updates)
                        _recorded_episode = _recorder.record(_ep_info)
                        if evidence_quality_enabled():
                            ComponentEvidenceBuilder().update(_recorded_episode)
                            _all_eps = _recorder.get_all_episodes()
                            EvolutionOperators().run(_recorded_episode, len(_all_eps))



                            if _recommended_role == "avoid_retest":
                                _avoid_ids = _ep_info.get("assigned_avoid_memory_ids") or []
                                if _avoid_ids:
                                    from memory.updating.memory_utility import MemoryUtilityTracker
                                    _mut = MemoryUtilityTracker()
                                    _ep_delta = _ep_info.get("score_delta", 0)
                                    _ep_iter = _ep_info.get("iteration", 0)
                                    for _mid in _avoid_ids:
                                        _mut.update_avoid_after_retest(
                                            _mid, _ep_delta, _ep_iter
                                        )
                    except Exception as _mem_err:
                        print(f"  {_dim(f'memory update skipped: {_mem_err}')}")

                delta = avg - best_avg
                delta_str = f"{delta:+.1%}"
                delta_colored = (
                    _green(delta_str)
                    if delta > 0
                    else (_red(delta_str) if delta < 0 else _dim(delta_str))
                )

                trial_metrics = parse_trial_metrics(job_dir)
                metrics_summary = summarize_trial_metrics(trial_metrics)
                all_metrics[name] = metrics_summary
                cost_str = (
                    f"${metrics_summary['total_cost_usd']:.2f}"
                    if metrics_summary["total_cost_usd"]
                    else "?"
                )
                print(
                    f"         avg={_rate_str(avg)}  delta={delta_colored}  cost={cost_str}  ({_elapsed(elapsed)})"
                )
                max_name_len = max(len(t) for t in per_task) if per_task else 0
                baseline_per_task = {}
                if FRONTIER_VAL.exists():
                    fr = json.loads(FRONTIER_VAL.read_text())
                    for tk in per_task:
                        baseline_per_task[tk] = fr.get(tk, {}).get("pass_rate", 0)
                for task, rate in sorted(per_task.items()):
                    bl = baseline_per_task.get(task, 0)
                    td = rate - bl
                    td_s = f"{td:+.0%}" if td != 0 else "  ="
                    td_c = (
                        _green(td_s)
                        if td > 0
                        else (_red(td_s) if td < 0 else _dim(td_s))
                    )
                    tm = metrics_summary.get("per_task", {}).get(task, {})
                    tc = (
                        f"${tm['mean_cost']:.2f}"
                        if tm.get("mean_cost") is not None
                        else ""
                    )
                    tt = (
                        f"{tm['mean_turns']:.0f}t"
                        if tm.get("mean_turns") is not None
                        else ""
                    )
                    suffix = f"  {_dim(tc)} {_dim(tt)}" if tc else ""
                    print(
                        f"         {task:<{max_name_len}}  {_rate_str(rate)}  {td_c}{suffix}"
                    )
            else:
                results[name] = ({}, 0)
                print(
                    f"         {_red('FAIL')} benchmark crashed ({_elapsed(elapsed)})"
                )

        bench_time = time.time() - bench_start

        update_frontier(results, metrics=all_metrics)
        update_evolution_summary_terminal(
            iteration,
            valid,
            results,
            propose_time=propose_time,
            bench_time=bench_time,
            metrics=all_metrics,
        )

        wall_time = time.time() - iter_start
        frontier_now = (
            json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        )
        new_best_avg = frontier_now.get("_best", {}).get("avg_pass_rate", 0)
        new_best_agent = frontier_now.get("_best", {}).get("agent", "none")
        improved = new_best_avg > best_avg
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}  frontier={new_best_agent} @ {new_best_avg:.1%}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )


    if _interrupted or not args.full_eval:
        return

    print(f"\n{_ts()} {_bold('Phase Final: 5-trial eval for frontier agents')}")
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    best_agent = frontier.get("_best", {}).get("agent")
    if best_agent and best_agent != BASELINE_AGENT_NAME:
        import_path = None
        for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("agent") == best_agent and row.get("import_path"):
                import_path = row["import_path"]
                break

        if import_path:
            job_name = f"final-{best_agent}-t5"
            print(f"  {_ts()} running {_bold(best_agent)} x 5 trials...", flush=True)
            t0 = time.time()
            job_dir, _ = harbor_run(
                import_path,
                job_name,
                n_trials=5,
                n_concurrent=args.concurrent,
            )
            if job_dir.exists():
                task_rewards = parse_job_results(job_dir, expected_trials=5)
                per_task, avg = compute_pass_rates(task_rewards)
                print(
                    f"  {_bold(best_agent)} (5-trial): avg={_rate_str(avg)}  ({_elapsed(time.time() - t0)})"
                )
                max_name_len = max(len(t) for t in per_task) if per_task else 0
                for task, rate in sorted(per_task.items()):
                    print(f"    {task:<{max_name_len}}  {_rate_str(rate)}")

    print(f"\n{_ts()} {_bold('Evolution complete.')}")





def run_evolve(args):
    task = getattr(args, "task", "text") or "text"
    if task == "terminal":
        _configure_component_memory("terminal")
        run_evolve_terminal(args)
    else:
        _configure_component_memory("text")
        run_evolve_text(args)
