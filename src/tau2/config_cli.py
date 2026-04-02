from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_PROVIDER,
    DEFAULT_INTEGRATION_DURATION_SECONDS,
    DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_USER,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_STEPS_SECONDS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_PCM_SAMPLE_RATE,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_RUN_CONCURRENCY,
    DEFAULT_SEED,
    DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
    DEFAULT_TELEPHONY_RATE,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS,
)
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import AudioNativeConfig, TextRunConfig, VoiceRunConfig
from tau2.run import run_domain
from tau2.utils.llm_utils import set_llm_log_mode


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key).replace("-", "_")
            out[key_str] = _normalize_keys(item)
        return out
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    return value


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/object.")
    return _normalize_keys(raw)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries with run-level override precedence."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def _runs_map_from_entries(
    run_entries: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Build run_id -> raw run mapping. Duplicate keys would silently overwrite."""
    return {run_id: dict(run_cfg) for run_id, run_cfg in run_entries}


def _resolve_run_with_base_inheritance(
    run_id: str,
    run_cfg: dict[str, Any],
    all_runs: dict[str, dict[str, Any]],
    visiting: frozenset[str],
) -> dict[str, Any]:
    """Apply ``base_run`` chain: merge resolved parent, then this run's overrides (deep).

    ``base_run`` must name another key under ``runs``. Cycles are rejected.
    The special key ``base_run`` is never passed to :func:`_build_run_config`.
    """
    if run_id in visiting:
        raise ValueError(
            f"base_run cycle: run '{run_id}' appears twice in the inheritance chain"
        )

    base_ref = run_cfg.get("base_run")
    if base_ref is None:
        out = dict(run_cfg)
        out.pop("base_run", None)
        return out

    base_id = str(base_ref)
    if base_id not in all_runs:
        raise ValueError(
            f"Run '{run_id}' references unknown base_run '{base_id}'. "
            f"Available runs: {sorted(all_runs.keys())}"
        )

    base_resolved = _resolve_run_with_base_inheritance(
        base_id,
        all_runs[base_id],
        all_runs,
        visiting | {run_id},
    )
    override = {k: v for k, v in run_cfg.items() if k != "base_run"}
    return _merge_dicts(base_resolved, override)


def _split_root_and_runs(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    runs_raw = cfg.get("runs")
    root_defaults = {k: v for k, v in cfg.items() if k != "runs"}
    if runs_raw is None:
        return root_defaults, [("default", cfg)]

    if isinstance(runs_raw, dict):
        runs: list[tuple[str, dict[str, Any]]] = []
        for run_id, run_cfg in runs_raw.items():
            if not isinstance(run_cfg, dict):
                raise ValueError(f"Run '{run_id}' must be a mapping/object.")
            runs.append((str(run_id), run_cfg))
        return root_defaults, runs

    if isinstance(runs_raw, list):
        runs = []
        for idx, run_cfg in enumerate(runs_raw):
            if not isinstance(run_cfg, dict):
                raise ValueError(f"Run at index {idx} must be a mapping/object.")
            run_id = str(run_cfg.get("id") or f"run_{idx + 1}")
            runs.append((run_id, run_cfg))
        return root_defaults, runs

    raise ValueError("'runs' must be either a mapping or a list of mappings.")


def _select_runs(
    cfg: dict[str, Any],
    runs: list[tuple[str, dict[str, Any]]],
    cli_run_ids: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    requested_ids: set[str] | None = None
    if cli_run_ids:
        requested_ids = {str(item) for item in cli_run_ids}
    else:
        enabled_ids = cfg.get("enabled_run_ids")
        if enabled_ids is not None:
            if not isinstance(enabled_ids, list):
                raise ValueError("'enabled_run_ids' must be a list of run IDs.")
            requested_ids = {str(item) for item in enabled_ids}

    selected: list[tuple[str, dict[str, Any]]] = []
    for run_id, run_cfg in runs:
        if requested_ids is not None:
            if run_id in requested_ids:
                selected.append((run_id, run_cfg))
            continue
        if bool(run_cfg.get("enabled", True)):
            selected.append((run_id, run_cfg))

    if requested_ids is not None and not selected:
        raise ValueError("No matching runs found for selected run IDs.")
    if not selected:
        raise ValueError("No enabled runs found in config.")
    return selected


def _build_audio_native_config(cfg: dict[str, Any]) -> AudioNativeConfig:
    if "audio_native_config" in cfg and isinstance(cfg["audio_native_config"], dict):
        return AudioNativeConfig(**cfg["audio_native_config"])

    provider = cfg.get("audio_native_provider", DEFAULT_AUDIO_NATIVE_PROVIDER)
    model = cfg.get("audio_native_model") or DEFAULT_AUDIO_NATIVE_MODELS.get(
        provider, DEFAULT_AUDIO_NATIVE_MODELS[DEFAULT_AUDIO_NATIVE_PROVIDER]
    )
    return AudioNativeConfig(
        provider=provider,
        model=model,
        cascaded_config_name=cfg.get("cascaded_config"),
        tick_duration_seconds=cfg.get("tick_duration", 0.2),
        max_steps_seconds=cfg.get("max_steps_seconds", DEFAULT_MAX_STEPS_SECONDS),
        pcm_sample_rate=cfg.get("pcm_sample_rate", DEFAULT_PCM_SAMPLE_RATE),
        telephony_rate=cfg.get("telephony_rate", DEFAULT_TELEPHONY_RATE),
        wait_to_respond_threshold_other_seconds=cfg.get(
            "wait_to_respond_other", DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS
        ),
        wait_to_respond_threshold_self_seconds=cfg.get(
            "wait_to_respond_self", DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS
        ),
        yield_threshold_when_interrupted_seconds=cfg.get(
            "yield_when_interrupted", DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS
        ),
        yield_threshold_when_interrupting_seconds=cfg.get(
            "yield_when_interrupting", DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS
        ),
        interruption_check_interval_seconds=cfg.get(
            "interruption_check_interval", DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS
        ),
        integration_duration_seconds=cfg.get(
            "integration_duration", DEFAULT_INTEGRATION_DURATION_SECONDS
        ),
        silence_annotation_threshold_seconds=cfg.get(
            "silence_annotation_threshold",
            DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
        ),
        use_xml_prompt=bool(cfg.get("xml_prompt", False)),
    )


def _build_run_config(cfg: dict[str, Any]) -> TextRunConfig | VoiceRunConfig:
    persona = cfg.get("user_persona")
    if isinstance(persona, dict):
        # Keep parity with tau2 cli behavior; validates early.
        PersonaConfig.from_dict(persona)

    llm_log_mode = cfg.get("llm_log_mode", "latest")
    set_llm_log_mode(llm_log_mode)

    user_llm_args = dict(cfg.get("user_llm_args", cfg.get("llm_args_user", {})) or {})
    if isinstance(cfg.get("user_pricing"), dict):
        user_llm_args["pricing"] = cfg["user_pricing"]

    shared_kwargs = dict(
        domain=cfg["domain"],
        task_set_name=cfg.get("task_set_name"),
        task_split_name=cfg.get("task_split_name", "base"),
        task_ids=(cfg.get("task_ids") or None),
        num_tasks=cfg.get("num_tasks"),
        llm_user=cfg.get("user_llm", cfg.get("llm_user", DEFAULT_LLM_USER)),
        llm_args_user=user_llm_args,
        num_trials=cfg.get("num_trials", DEFAULT_NUM_TRIALS),
        max_errors=cfg.get("max_errors", DEFAULT_MAX_ERRORS),
        timeout=cfg.get("timeout"),
        save_to=cfg.get("save_to"),
        config_name=cfg.get("config_name"),
        max_concurrency=cfg.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
        trial_concurrency=cfg.get("trial_concurrency", 1),
        seed=cfg.get("seed", DEFAULT_SEED),
        log_level=cfg.get("log_level", "INFO"),
        verbose_logs=bool(cfg.get("verbose_logs", False)),
        max_retries=cfg.get("max_retries", DEFAULT_RETRY_ATTEMPTS),
        retry_delay=cfg.get("retry_delay", DEFAULT_RETRY_MIN_WAIT),
        auto_resume=bool(cfg.get("auto_resume", False)),
        auto_review=bool(cfg.get("auto_review", False)),
        review_mode=cfg.get("review_mode", "full"),
        hallucination_retries=cfg.get("hallucination_retries", 3),
        retrieval_config=cfg.get("retrieval_config"),
        retrieval_config_kwargs=cfg.get("retrieval_config_kwargs"),
        fresh=bool(cfg.get("fresh", False)),
    )

    mode = cfg.get("mode")
    use_voice = bool(
        cfg.get("audio_native", False)
        or isinstance(cfg.get("audio_native_config"), dict)
        or mode == "voice"
    )

    if use_voice:
        return VoiceRunConfig(
            **shared_kwargs,
            audio_native_config=_build_audio_native_config(cfg),
            speech_complexity=cfg.get("speech_complexity", "regular"),
            audio_debug=bool(cfg.get("audio_debug", False)),
            audio_taps=bool(cfg.get("audio_taps", False)),
        )

    agent_llm_args = dict(cfg.get("agent_llm_args", cfg.get("llm_args_agent", {})) or {})
    if isinstance(cfg.get("agent_pricing"), dict):
        agent_llm_args["pricing"] = cfg["agent_pricing"]

    return TextRunConfig(
        **shared_kwargs,
        agent=cfg.get("agent", "llm_agent"),
        llm_agent=cfg.get("agent_llm", cfg.get("llm_agent", DEFAULT_LLM_AGENT)),
        llm_args_agent=agent_llm_args,
        user=cfg.get("user", "user_simulator"),
        max_steps=cfg.get("max_steps", DEFAULT_MAX_STEPS),
        enforce_communication_protocol=bool(
            cfg.get("enforce_communication_protocol", False)
        ),
        assistant_solo_mode=bool(cfg.get("assistant_solo_mode", False)),
    )


def _apply_fresh_save_to(cfg: dict[str, Any]) -> None:
    if not bool(cfg.get("fresh", False)):
        return
    base_name = cfg.get("save_to")
    if not base_name:
        raise ValueError(
            "Run has fresh=true but no save_to value. Please provide save_to."
        )
    stamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    cfg["save_to"] = f"{base_name}_{stamp}"


def _effective_run_concurrency(cfg: dict[str, Any], cli_override: int | None) -> int:
    """Root-level YAML ``run_concurrency`` or CLI ``--run-concurrency``; minimum 1."""
    if cli_override is not None:
        return max(1, int(cli_override))
    raw = cfg.get("run_concurrency", DEFAULT_RUN_CONCURRENCY)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _prepare_merged_cfg_for_run(
    run_id: str,
    run_cfg: dict[str, Any],
    all_runs: dict[str, dict[str, Any]],
    root_defaults: dict[str, Any],
    cfg_path: Path,
) -> dict[str, Any]:
    resolved_run = _resolve_run_with_base_inheritance(
        run_id, run_cfg, all_runs, frozenset()
    )
    merged_cfg = _merge_dicts(root_defaults, resolved_run)
    merged_cfg.pop("enabled_run_ids", None)
    merged_cfg.pop("enabled", None)
    merged_cfg.pop("id", None)
    merged_cfg.pop("base_run", None)
    merged_cfg.pop("run_concurrency", None)
    if run_id != "default":
        merged_cfg["config_name"] = run_id
    else:
        merged_cfg.setdefault("config_name", cfg_path.stem)
    _apply_fresh_save_to(merged_cfg)
    return merged_cfg


def _execute_yaml_run(run_id: str, merged_cfg: dict[str, Any]) -> None:
    run_config = _build_run_config(merged_cfg)
    print(f"[tau2config] Running '{run_id}'...")
    run_domain(run_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tau2 from YAML config")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=None,
        help="Optional list of run IDs to execute from multi-run YAML config",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--run-concurrency",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override YAML root run_concurrency: how many top-level runs from this file "
            f"may execute in parallel (default {DEFAULT_RUN_CONCURRENCY} or run_concurrency in YAML)."
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    # Default cwd search + .env next to the config file (import chain also loads .env via tau2.utils.utils).
    load_dotenv()
    load_dotenv(cfg_path.parent / ".env")

    cfg = _load_yaml_config(cfg_path)
    root_defaults, run_entries = _split_root_and_runs(cfg)
    all_runs = _runs_map_from_entries(run_entries)
    selected_runs = _select_runs(cfg, run_entries, args.run_ids)

    prepared: list[tuple[str, dict[str, Any]]] = [
        (
            run_id,
            _prepare_merged_cfg_for_run(
                run_id, run_cfg, all_runs, root_defaults, cfg_path
            ),
        )
        for run_id, run_cfg in selected_runs
    ]

    run_concurrency = _effective_run_concurrency(cfg, args.run_concurrency)

    if run_concurrency <= 1 or len(prepared) <= 1:
        for run_id, merged_cfg in prepared:
            _execute_yaml_run(run_id, merged_cfg)
    else:
        workers = min(run_concurrency, len(prepared))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_execute_yaml_run, run_id, merged_cfg): run_id
                for run_id, merged_cfg in prepared
            }
            for fut in as_completed(futures):
                fut.result()


if __name__ == "__main__":
    main()
