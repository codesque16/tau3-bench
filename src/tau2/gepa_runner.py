"""Programmatic τ² simulation entrypoint for GEPA (tau2-mermaid monorepo).

Mirrors ``uv run tau2config --config <yaml> --run-ids <run_id>`` for a **single**
task, with an optional **retail policy file** swap so GEPA candidates replace
``domains/retail`` policy on disk.

Callers should ``os.chdir(fork_root)`` before importing ``tau2`` so ``DATA_DIR``
and domain data paths resolve correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from tau2.config_cli import _build_run_config, load_yaml_and_prepare_run
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import RewardType
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.environment.toolkit import get_tool_signatures
from tau2.evaluator.evaluator import EvaluationType, _reward_info_payload
from tau2.runner.batch import run_single_task
from tau2.runner.helpers import get_tasks
from tau2.utils.utils import DATA_DIR


def simulation_messages_to_openai_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Flatten τ² messages into OpenAI-style dicts for GEPA / diagnosis formatting."""
    out: list[dict[str, Any]] = []

    def emit_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for tc in tool_calls:
            formatted.append(
                {
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
            )
        return formatted

    for msg in messages:
        if isinstance(msg, MultiToolMessage):
            for tm in msg.tool_messages:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tm.id,
                        "content": tm.content or "",
                    }
                )
            continue
        if isinstance(msg, SystemMessage):
            if msg.content:
                out.append({"role": "system", "content": msg.content})
            continue
        if isinstance(msg, UserMessage):
            row: dict[str, Any] = {"role": "user", "content": msg.content or ""}
            if msg.tool_calls:
                row["tool_calls"] = emit_tool_calls(msg.tool_calls)
            out.append(row)
            continue
        if isinstance(msg, AssistantMessage):
            row = {"role": "assistant", "content": msg.content or ""}
            if msg.reasoning_content:
                row["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls:
                row["tool_calls"] = emit_tool_calls(msg.tool_calls)
            out.append(row)
            continue
        if isinstance(msg, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.id,
                    "content": msg.content or "",
                }
            )
            continue

    return out


def retail_tools_markdown_for_gepa(*, max_chars: int = 24_000) -> str:
    """Best-effort tool listing for qualitative diagnosis prompts (retail)."""
    db = RetailDB.load(RETAIL_DB_PATH)
    tools = RetailTools(db)
    sigs = get_tool_signatures(tools)
    lines = ["# Retail tools (τ-bench signatures)\n"]
    for name, sig in sorted(sigs.items()):
        lines.append(f"## {name}\n{sig.doc}\n")
        if sig.params:
            lines.append(f"params: {json.dumps(sig.params, ensure_ascii=False)[:2000]}\n")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n…(truncated)…"


def format_tau3_reward_for_gepa_diagnosis(
    simulation: SimulationRun,
    *,
    task_id: str,
) -> str:
    """Human-readable reward / evaluation block for LLM diagnosis (failed runs)."""
    ri = simulation.reward_info
    if ri is None:
        return f"task_id={task_id}\n(no reward_info attached)"
    payload = _reward_info_payload(ri)
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)


def path_mismatch_from_reward(simulation: SimulationRun) -> dict[str, Any] | None:
    """Summarize failed action checks in a tau2-mermaid-compatible shape."""
    ri = simulation.reward_info
    if ri is None or not ri.action_checks:
        return None
    failed = [c for c in ri.action_checks if not c.action_match]
    if not failed:
        return None
    return {
        "failed_action_checks": len(failed),
        "samples": [
            {
                "tool": c.action.name if c.action else None,
                "arguments": c.action.arguments if c.action else None,
                "action_match": c.action_match,
                "action_reward": c.action_reward,
            }
            for c in failed[:12]
        ],
    }


def run_gepa_evaluation_task(
    *,
    yaml_config_path: str | Path,
    run_id: str,
    task_id: str,
    candidate_policy_text: str,
    seed: Optional[int] = None,
    gepa_artifact_dir: str | Path,
    merged_overrides: Optional[dict[str, Any]] = None,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    verbose_logs: bool = False,
) -> SimulationRun:
    """Run one retail (or other) τ² task using YAML run settings + a policy file override.

    Args:
        yaml_config_path: Same file passed to ``tau2config --config``.
        run_id: Run id under ``runs:`` (e.g. ``retail_conv_full_gemma4_26b_fp8_cloud_run``).
        task_id: Single task id string (must exist in the task set for this domain).
        candidate_policy_text: Full policy markdown written to a temp file; retail loads it
            via ``retail_policy_path``.
        seed: Optional per-simulation seed (overrides merged YAML seed when set).
        gepa_artifact_dir: Directory under which ``policy_candidate.md`` and per-run artifacts
            are written (must exist or will be created).
        merged_overrides: Deep-merge keys into the merged YAML dict before building
            :class:`TextRunConfig` (e.g. ``{"max_concurrency": 1}``).
        evaluation_type: Passed to :func:`run_single_task`. If this is ``ALL`` (default) and the
            task's ``reward_basis`` includes ``NL_ASSERTION``, it is upgraded to
            ``ALL_WITH_NL_ASSERTIONS`` automatically.
        verbose_logs: Per-task log files under the simulation save dir.

    Returns:
        Completed :class:`SimulationRun` with ``reward_info`` populated.
    """
    art = Path(gepa_artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    policy_path = art / "policy_candidate.md"
    policy_path.write_text(candidate_policy_text, encoding="utf-8")

    merged = load_yaml_and_prepare_run(yaml_config_path, run_id, load_env=True)
    merged = dict(merged)
    merged["num_trials"] = 1
    merged["task_ids"] = [str(task_id)]
    merged["retail_policy_path"] = str(policy_path.resolve())
    if seed is not None:
        merged["seed"] = int(seed)
    base_save = merged.get("save_to") or run_id
    merged["save_to"] = f"{base_save}_gepa/{art.name}"
    merged["fresh"] = False

    if merged_overrides:
        for k, v in merged_overrides.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v

    run_config = _build_run_config(merged)
    if not isinstance(run_config, TextRunConfig):
        raise TypeError(
            "GEPA runner supports text runs only; use a non-voice YAML run template."
        )

    tasks = get_tasks(
        task_set_name=run_config.task_set_name or run_config.domain,
        task_split_name=run_config.task_split_name,
        task_ids=run_config.task_ids,
        num_tasks=run_config.num_tasks,
    )
    if not tasks:
        raise ValueError(f"No tasks loaded for task_ids={run_config.task_ids!r}")
    task = tasks[0]

    effective_seed = int(seed) if seed is not None else run_config.seed
    save_dir = DATA_DIR / "simulations" / (run_config.save_to or "gepa_run")

    # Tasks whose reward_basis includes NL_ASSERTION require ALL_WITH_NL_ASSERTIONS;
    # ALL skips NLEvaluator and evaluate_simulation raises.
    eff_eval = evaluation_type
    if evaluation_type == EvaluationType.ALL:
        ec = task.evaluation_criteria
        if ec is not None and RewardType.NL_ASSERTION in set(ec.reward_basis):
            eff_eval = EvaluationType.ALL_WITH_NL_ASSERTIONS

    return run_single_task(
        run_config,
        task,
        seed=effective_seed,
        evaluation_type=eff_eval,
        save_dir=save_dir,
        verbose_logs=verbose_logs,
    )


def write_gepa_simulation_artifact(
    *,
    simulation: SimulationRun,
    task_id: str,
    dest_path: str | Path,
) -> None:
    """Persist trace + evaluation payload for offline reflector / debugging."""
    p = Path(dest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ri = simulation.reward_info
    payload = {
        "task_id": task_id,
        "reward": ri.reward if ri is not None else None,
        "termination_reason": getattr(simulation.termination_reason, "value", None),
        "messages_openai_style": simulation_messages_to_openai_dicts(simulation.messages),
        "reward_info": _reward_info_payload(ri) if ri is not None else None,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
