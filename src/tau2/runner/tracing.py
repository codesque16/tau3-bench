from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

from tau2.data_model.simulation import RunConfig
from tau2.data_model.tasks import Task


class RunTracer:
    @contextmanager
    def run_span(self, *, run_id: str):
        yield None

    @contextmanager
    def trial_span(self, *, trial: int, seed: int):
        yield None

    @contextmanager
    def task_span(self, *, task_id: str):
        yield None

    @contextmanager
    def simulation_span(self, *, simulation_id: str, domain: str, task_id: str):
        yield None

    @contextmanager
    def evaluation_span(self):
        yield None

    @contextmanager
    def evaluation_check_span(self, name: str):
        yield None

    def log_experiment_config(
        self,
        *,
        config: RunConfig,
        run_id: str,
        experiment_index: int = 1,
        experiment_total: int = 1,
        mode: str = "conversation",
    ) -> None:
        return None

    def log_task_details(self, task: Task) -> None:
        return None

    def log_task_result(self, *, task_id: str, passed: bool) -> None:
        return None

    def log_metrics(self, metrics: Any) -> None:
        return None

    def finalize_run_span(
        self,
        span: Any,
        *,
        run_id: str,
        pass_hat_1: float | None,
    ) -> None:
        return None

    def finalize_span(
        self,
        span: Any,
        *,
        base_name: str,
        passed: bool,
        **attrs: Any,
    ) -> None:
        return None


class LogfireRunTracer(RunTracer):
    def _set_span_display_name(self, span: Any, name: str) -> None:
        if span is None:
            return
        if hasattr(span, "message"):
            span.message = name
        elif hasattr(span, "update_name"):
            span.update_name(name)
        elif hasattr(span, "set_attribute"):
            span.set_attribute("name", name)

    def __init__(self):
        try:
            import logfire  # type: ignore

            self._logfire = logfire
        except Exception:
            self._logfire = None

    def _span(self, name: str, **attrs: Any):
        if self._logfire is None:
            return nullcontext()
        clean = {k: v for k, v in attrs.items() if v is not None}
        return self._logfire.span(name, **clean)

    def _log(self, name: str, **attrs: Any) -> None:
        if self._logfire is None:
            return
        clean = {k: v for k, v in attrs.items() if v is not None}
        self._logfire.info(name, **clean)

    @contextmanager
    def run_span(self, *, run_id: str):
        with self._span(run_id, run_id=run_id) as span:
            yield span

    @contextmanager
    def trial_span(self, *, trial: int, seed: int):
        with self._span(f"Trial:{trial} [seed={seed}]", trial=trial, seed=seed) as span:
            yield span

    @contextmanager
    def task_span(self, *, task_id: str):
        with self._span(f"Task:{task_id}", task_id=task_id) as span:
            yield span

    @contextmanager
    def simulation_span(self, *, simulation_id: str, domain: str, task_id: str):
        with self._span(
            "simulation",
            simulation_id=simulation_id,
            domain=domain,
            task_id=task_id,
        ) as span:
            yield span

    @contextmanager
    def evaluation_span(self):
        with self._span("evaluation") as span:
            yield span

    @contextmanager
    def evaluation_check_span(self, name: str):
        with self._span(name) as span:
            yield span

    def log_experiment_config(
        self,
        *,
        config: RunConfig,
        run_id: str,
        experiment_index: int = 1,
        experiment_total: int = 1,
        mode: str = "conversation",
    ) -> None:
        config_json = config.model_dump(mode="json")
        self._log(
            "experiment_config",
            config_json=config_json,
            experiment_index=experiment_index,
            experiment_total=experiment_total,
            mode=mode,
            run_id=run_id,
        )

    def log_task_details(self, task: Task) -> None:
        self._log(
            "task_details",
            task_id=task.id,
            user_scenario=str(task.user_scenario),
            actions=[
                a.model_dump(mode="json")
                for a in ((task.evaluation_criteria.actions or []) if task.evaluation_criteria else [])
            ],
            nl_assertions=(task.evaluation_criteria.nl_assertions if task.evaluation_criteria else None),
        )

    def log_task_result(self, *, task_id: str, passed: bool) -> None:
        # Backwards-compatible no-op; task status is now applied by renaming the same span.
        return None

    def log_metrics(self, metrics: Any) -> None:
        self._log("metrics", metrics=metrics.model_dump(mode="json"))

    def finalize_run_span(
        self,
        span: Any,
        *,
        run_id: str,
        pass_hat_1: float | None,
    ) -> None:
        if pass_hat_1 is None:
            name = run_id
        else:
            name = f"{run_id}[{pass_hat_1:.4f}]"
        self._set_span_display_name(span, name)
        if span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("pass_hat_1", pass_hat_1)

    def finalize_span(
        self,
        span: Any,
        *,
        base_name: str,
        passed: bool,
        **attrs: Any,
    ) -> None:
        status = "pass" if passed else "fail"
        self._set_span_display_name(span, f"{base_name} [{status}]")
        if span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("passed", passed)
            for key, value in attrs.items():
                if value is not None:
                    span.set_attribute(key, value)


def build_run_id(config: RunConfig, *, policy_name: Optional[str] = None) -> str:
    domain = config.domain
    agent_model = config.effective_agent_model
    user_model = config.effective_user_model
    policy = policy_name or "policy"
    base = f"[{domain}][{policy}][{agent_model}][{user_model}]"

    label = getattr(config, "config_name", None)
    label = label.strip() if isinstance(label, str) and label.strip() else None
    save = (config.save_to or "").strip()
    fresh = bool(getattr(config, "fresh", False))

    if label:
        if fresh and save:
            tail = f"{label}|{Path(save).name}"
        else:
            tail = label
    elif save:
        tail = Path(save).name
    else:
        tail = None

    if tail:
        return f"{base}[{tail}]"
    return base


def infer_policy_name(config: RunConfig) -> str:
    # Prefer domain policy file basename if available (e.g., RETAIL_POLICY_PATH -> policy.md).
    try:
        import importlib

        module = importlib.import_module(f"tau2.domains.{config.domain}.utils")
        for name in dir(module):
            if name.endswith("_POLICY_PATH"):
                value = getattr(module, name, None)
                if value is not None:
                    return Path(str(value)).name
    except Exception:
        pass

    raw = Path(config.save_to).name if config.save_to else "policy"
    return raw

