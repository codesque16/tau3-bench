import math
import re
from collections import defaultdict

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from tau2.data_model.simulation import Results, TerminationReason


def is_successful(reward: float) -> bool:
    """
    Check if the reward is successful.
    """
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


class AgentMetrics(BaseModel):
    # Core metrics
    avg_reward: float
    pass_hat_ks: dict[int, float]
    pass_hat_ks_db_only: dict[int, float] = {}
    pass_hat_ks_db_and_communication: dict[int, float] = {}
    pass_hat_ks_db_communication_nl: dict[int, float] = {}
    avg_agent_cost: float
    avg_user_cost: float = 0.0
    avg_total_cost: float = 0.0
    median_agent_cost: float = 0.0
    median_user_cost: float = 0.0
    median_total_cost: float = 0.0
    total_agent_cost: float = 0.0
    total_user_cost: float = 0.0
    total_cost: float = 0.0

    # Detailed token/cost telemetry
    total_agent_input_tokens: int = 0
    total_agent_cached_input_tokens: int = 0
    total_agent_output_tokens: int = 0
    total_agent_reasoning_tokens: int = 0
    total_user_input_tokens: int = 0
    total_user_cached_input_tokens: int = 0
    total_user_output_tokens: int = 0
    total_user_reasoning_tokens: int = 0
    agent_cached_input_percentage: float = 0.0
    agent_reasoning_output_percentage: float = 0.0
    user_cached_input_percentage: float = 0.0
    user_reasoning_output_percentage: float = 0.0
    total_agent_input_cost_with_cache: float = 0.0
    total_agent_input_cost_without_cache: float = 0.0
    total_agent_output_cost: float = 0.0
    total_user_input_cost_with_cache: float = 0.0
    total_user_input_cost_without_cache: float = 0.0
    total_user_output_cost: float = 0.0

    # Simulation counts
    total_simulations: int = 0
    total_tasks: int = 0
    infra_error_count: int = 0

    # Action metrics
    total_read_actions: int = 0
    correct_read_actions: int = 0
    total_write_actions: int = 0
    correct_write_actions: int = 0

    # DB match metrics
    db_match_count: int = 0
    db_mismatch_count: int = 0
    db_not_checked: int = 0

    # Authentication metrics
    auth_succeeded: int = 0
    auth_failed: int = 0
    auth_not_needed: int = 0
    auth_not_checked: int = 0

    # Termination reason counts
    termination_user_stop: int = 0
    termination_agent_stop: int = 0
    termination_max_steps: int = 0
    termination_error: int = 0
    termination_infrastructure_error: int = 0

    # Responsiveness metrics (from full-duplex/streaming mode)
    sims_with_unresponsive_period: int = 0
    sims_with_responsiveness_info: int = 0

    # Review error metrics (from LLM judge)
    agent_errors_by_severity: dict[str, int] = {}  # severity -> count
    user_errors_by_severity: dict[str, int] = {}  # severity -> count
    sims_by_max_agent_severity: dict[str, int] = {}  # max severity -> sim count
    sims_by_max_user_severity: dict[str, int] = {}  # max severity -> sim count
    sims_by_first_critical_source: dict[
        str, int
    ] = {}  # "agent" / "user" / "none" -> count
    agent_error_tags_by_severity: dict[
        str, dict[str, int]
    ] = {}  # tag -> severity -> count
    user_error_tags_by_severity: dict[
        str, dict[str, int]
    ] = {}  # tag -> severity -> count

    # Computed properties for convenience
    @property
    def total_agent_errors(self) -> int:
        return sum(self.agent_errors_by_severity.values())

    @property
    def total_user_errors(self) -> int:
        return sum(self.user_errors_by_severity.values())

    @property
    def sims_with_agent_errors(self) -> int:
        return sum(v for k, v in self.sims_by_max_agent_severity.items() if k != "none")

    @property
    def sims_with_user_errors(self) -> int:
        return sum(v for k, v in self.sims_by_max_user_severity.items() if k != "none")

    @property
    def sims_with_critical_agent_errors(self) -> int:
        return self.sims_by_max_agent_severity.get("critical", 0)

    @property
    def sims_with_critical_user_errors(self) -> int:
        return self.sims_by_max_user_severity.get(
            "critical_helped", 0
        ) + self.sims_by_max_user_severity.get("critical_hindered", 0)

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_agent_cost": self.avg_agent_cost,
            "total_simulations": self.total_simulations,
            "total_tasks": self.total_tasks,
            "infra_error_count": self.infra_error_count,
        }
        for k, v in self.pass_hat_ks.items():
            data[f"pass_hat_{k}"] = v
        return data


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def get_metrics_df(results: Results) -> tuple[pd.DataFrame, int]:
    """
    Convert the results to a dataframe and add a column for success.
    Filters out infrastructure errors (simulations that never ran).
    Checks that all simulations have the same number of trials.
    Returns the maximum number of trials that can be used for pass^k metrics.
    """
    df = results.to_df()

    infra_count = (
        df.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    ).sum()
    if infra_count > 0:
        logger.warning(
            f"Excluding {infra_count} infrastructure error simulation(s) from metrics."
        )
        df = df[df.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR]

    if df.empty:
        df["success"] = pd.Series(dtype=bool)
        return df, 0

    df["success"] = df.reward.apply(is_successful)
    if len(df.info_num_trials.unique()) > 1:
        logger.warning(
            f"All simulations must have the same number of trials. Found {df.info_num_trials.unique()}"
        )
    max_k = df.info_num_trials.max()

    task_ids_counts = [(tid, count) for tid, count in df.task_id.value_counts().items()]
    task_ids_counts.sort(key=lambda x: x[1])
    min_k = task_ids_counts[0][1]
    if min_k < max_k:
        logger.warning(
            f"The minimum number of trials for a task is {min_k}, which is less than the expected number of trials {max_k}. Setting max k to {min_k}."
        )
        max_k = min_k
    return df, max_k


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    if df.empty or max_k == 0:
        return pd.DataFrame()
    dfs = []
    for k in range(1, max_k + 1):
        res = df.groupby("task_id")["success"].apply(
            lambda df: pass_hat_k(len(df), df.sum(), k)
        )
        res.name = f"pass^{k}"
        dfs.append(res)
    df_pass_hat_k = pd.concat(dfs, axis=1)
    task_columns = [
        "task_num_agent_actions",
        "task_num_user_actions",
        "task_num_actions",
    ]
    df_task_infos = df.groupby("task_id").first()[task_columns]
    df_pass_hat_k = df_task_infos.join(df_pass_hat_k)
    return df_pass_hat_k


def prepare_dfs(results: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    df_pass_hat_k["num_actions"] = df.groupby("task_id").first()["task_num_actions"]
    df_pass_hat_k = df_pass_hat_k.sort_values(by="num_actions")
    return df, df_pass_hat_k


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute comprehensive metrics for the agent including:
    - average reward and pass^k
    - action metrics (read/write)
    - DB match, authentication, termination stats
    - LLM judge review error stats
    """
    if not results.simulations:
        return AgentMetrics(
            avg_reward=0.0,
            pass_hat_ks={},
            avg_agent_cost=0.0,
        )

    infra_error_count = sum(
        1
        for sim in results.simulations
        if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    )
    evaluated_sims = [
        sim
        for sim in results.simulations
        if sim.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
    ]

    if not evaluated_sims:
        return AgentMetrics(
            avg_reward=0.0,
            pass_hat_ks={},
            avg_agent_cost=0.0,
            total_simulations=0,
            total_tasks=0,
            infra_error_count=infra_error_count,
        )

    df, df_pass_hat_k = prepare_dfs(results)
    avg_reward = df.reward.mean()
    pass_hat_ks = {}
    for column in df_pass_hat_k.columns:
        if match := re.match(r"pass\^(\d+)", column):
            k = int(match.group(1))
            pass_hat_ks[k] = df_pass_hat_k[column].mean()

    def _all_met(checks: list | None) -> bool:
        if not checks:
            return True
        return all(bool(getattr(c, "met", False)) for c in checks)

    def _compute_pass_hat_from_flags(flags_by_task: dict[str, list[bool]]) -> dict[int, float]:
        if not flags_by_task:
            return {}
        min_trials = min((len(v) for v in flags_by_task.values()), default=0)
        if min_trials <= 0:
            return {}
        out: dict[int, float] = {}
        for k in range(1, min_trials + 1):
            per_task_vals: list[float] = []
            for task_flags in flags_by_task.values():
                success_count = sum(1 for f in task_flags if f)
                per_task_vals.append(pass_hat_k(len(task_flags), success_count, k))
            out[k] = sum(per_task_vals) / len(per_task_vals)
        return out

    db_only_flags: dict[str, list[bool]] = defaultdict(list)
    db_comm_flags: dict[str, list[bool]] = defaultdict(list)
    db_comm_nl_flags: dict[str, list[bool]] = defaultdict(list)
    for sim in evaluated_sims:
        ri = sim.reward_info
        db_ok = bool(ri and ri.db_check and ri.db_check.db_match)
        comm_ok = bool(ri and _all_met(ri.communicate_checks))
        nl_ok = bool(ri and _all_met(ri.nl_assertions))
        db_only_flags[str(sim.task_id)].append(db_ok)
        db_comm_flags[str(sim.task_id)].append(db_ok and comm_ok)
        db_comm_nl_flags[str(sim.task_id)].append(db_ok and comm_ok and nl_ok)

    pass_hat_ks_db_only = _compute_pass_hat_from_flags(db_only_flags)
    pass_hat_ks_db_and_communication = _compute_pass_hat_from_flags(db_comm_flags)
    pass_hat_ks_db_communication_nl = _compute_pass_hat_from_flags(db_comm_nl_flags)
    avg_agent_cost = df.agent_cost.mean()
    avg_user_cost = df.user_cost.mean() if "user_cost" in df.columns else 0.0

    # Cost aggregates (fallback-safe for mixed providers)
    agent_cost_series = pd.to_numeric(df.agent_cost, errors="coerce").fillna(0.0)
    user_cost_series = (
        pd.to_numeric(df.user_cost, errors="coerce").fillna(0.0)
        if "user_cost" in df.columns
        else pd.Series([0.0] * len(df))
    )
    total_cost_series = agent_cost_series + user_cost_series
    total_agent_cost = float(agent_cost_series.sum())
    total_user_cost = float(user_cost_series.sum())
    total_cost = float(total_cost_series.sum())
    avg_total_cost = float(total_cost_series.mean()) if len(total_cost_series) else 0.0
    median_agent_cost = float(agent_cost_series.median()) if len(agent_cost_series) else 0.0
    median_user_cost = float(user_cost_series.median()) if len(user_cost_series) else 0.0
    median_total_cost = float(total_cost_series.median()) if len(total_cost_series) else 0.0

    # Token/cost telemetry from message usage (for Vertex-style explicit accounting).
    total_agent_input_tokens = 0
    total_agent_cached_input_tokens = 0
    total_agent_output_tokens = 0
    total_agent_reasoning_tokens = 0
    total_user_input_tokens = 0
    total_user_cached_input_tokens = 0
    total_user_output_tokens = 0
    total_user_reasoning_tokens = 0
    total_agent_input_cost_with_cache = 0.0
    total_agent_input_cost_without_cache = 0.0
    total_agent_output_cost = 0.0
    total_user_input_cost_with_cache = 0.0
    total_user_input_cost_without_cache = 0.0
    total_user_output_cost = 0.0

    for sim in evaluated_sims:
        for msg in sim.get_messages():
            if not hasattr(msg, "usage") or not isinstance(msg.usage, dict):
                continue
            usage = msg.usage
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
            reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
            # For metrics, output tokens should include reasoning tokens so percentages
            # remain bounded and reflect true generated output volume.
            output_tokens = completion_tokens + reasoning_tokens
            input_cost_with_cache = float(usage.get("input_cost_with_cache_usd", 0.0) or 0.0)
            input_cost_without_cache = float(
                usage.get("input_cost_without_cache_usd", 0.0) or 0.0
            )
            output_cost = float(usage.get("output_cost_usd", 0.0) or 0.0)

            if msg.role == "assistant":
                total_agent_input_tokens += prompt_tokens
                total_agent_cached_input_tokens += cached_input_tokens
                total_agent_output_tokens += output_tokens
                total_agent_reasoning_tokens += reasoning_tokens
                total_agent_input_cost_with_cache += input_cost_with_cache
                total_agent_input_cost_without_cache += input_cost_without_cache
                total_agent_output_cost += output_cost
            elif msg.role == "user":
                total_user_input_tokens += prompt_tokens
                total_user_cached_input_tokens += cached_input_tokens
                total_user_output_tokens += output_tokens
                total_user_reasoning_tokens += reasoning_tokens
                total_user_input_cost_with_cache += input_cost_with_cache
                total_user_input_cost_without_cache += input_cost_without_cache
                total_user_output_cost += output_cost

    agent_cached_input_percentage = (
        (100.0 * total_agent_cached_input_tokens / total_agent_input_tokens)
        if total_agent_input_tokens > 0
        else 0.0
    )
    agent_reasoning_output_percentage = (
        (100.0 * total_agent_reasoning_tokens / total_agent_output_tokens)
        if total_agent_output_tokens > 0
        else 0.0
    )
    user_cached_input_percentage = (
        (100.0 * total_user_cached_input_tokens / total_user_input_tokens)
        if total_user_input_tokens > 0
        else 0.0
    )
    user_reasoning_output_percentage = (
        (100.0 * total_user_reasoning_tokens / total_user_output_tokens)
        if total_user_output_tokens > 0
        else 0.0
    )

    # Counts exclude infrastructure errors
    total_simulations = len(evaluated_sims)
    total_tasks = len(set(sim.task_id for sim in evaluated_sims))

    # Action metrics
    total_read_actions = 0
    correct_read_actions = 0
    total_write_actions = 0
    correct_write_actions = 0

    # DB match
    db_match_count = 0
    db_mismatch_count = 0
    db_not_checked = 0

    # Authentication
    auth_succeeded = 0
    auth_failed = 0
    auth_not_needed = 0
    auth_not_checked = 0

    # Termination
    termination_user_stop = 0
    termination_agent_stop = 0
    termination_max_steps = 0
    termination_error = 0

    # Responsiveness (from full-duplex/streaming mode)
    sims_with_unresponsive_period = 0
    sims_with_responsiveness_info = 0

    # Review errors
    agent_errors_by_severity: dict[str, int] = defaultdict(int)
    user_errors_by_severity: dict[str, int] = defaultdict(int)
    sims_by_max_agent_severity: dict[str, int] = defaultdict(int)
    sims_by_max_user_severity: dict[str, int] = defaultdict(int)
    sims_by_first_critical_source: dict[str, int] = defaultdict(int)
    agent_error_tags_by_severity: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    user_error_tags_by_severity: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for sim in evaluated_sims:
        # Action metrics
        if sim.reward_info and sim.reward_info.action_checks:
            partial = sim.reward_info.partial_action_reward
            if partial:
                if partial.get("read"):
                    total_read_actions += partial["read"]["count"]
                    correct_read_actions += partial["read"]["correct"]
                if partial.get("write"):
                    total_write_actions += partial["write"]["count"]
                    correct_write_actions += partial["write"]["correct"]

        # DB match
        if sim.reward_info and sim.reward_info.db_check:
            if sim.reward_info.db_check.db_match:
                db_match_count += 1
            else:
                db_mismatch_count += 1
        else:
            db_not_checked += 1

        # Authentication
        if sim.auth_classification:
            if sim.auth_classification.status == "succeeded":
                auth_succeeded += 1
            elif sim.auth_classification.status == "failed":
                auth_failed += 1
            else:
                auth_not_needed += 1
        else:
            auth_not_checked += 1

        # Termination reason
        if sim.termination_reason == TerminationReason.USER_STOP:
            termination_user_stop += 1
        elif sim.termination_reason == TerminationReason.AGENT_STOP:
            termination_agent_stop += 1
        elif sim.termination_reason == TerminationReason.MAX_STEPS:
            termination_max_steps += 1
        elif sim.termination_reason in (
            TerminationReason.TOO_MANY_ERRORS,
            TerminationReason.AGENT_ERROR,
            TerminationReason.USER_ERROR,
        ):
            termination_error += 1

        # Responsiveness info (from full-duplex/streaming mode)
        if sim.info and "had_unresponsive_period" in sim.info:
            sims_with_responsiveness_info += 1
            if sim.info["had_unresponsive_period"]:
                sims_with_unresponsive_period += 1

        # Review errors
        if sim.review is not None:
            agent_errors = [e for e in sim.review.errors if e.source == "agent"]
            user_errors = [e for e in sim.review.errors if e.source == "user"]

            if agent_errors:
                max_agent_sev = "none"
                for e in agent_errors:
                    severity = e.severity or "unknown"
                    agent_errors_by_severity[severity] += 1
                    # Track max severity for this sim (critical > minor > unknown > none)
                    if severity == "critical":
                        max_agent_sev = "critical"
                    elif severity == "minor" and max_agent_sev != "critical":
                        max_agent_sev = "minor"
                    elif max_agent_sev == "none":
                        max_agent_sev = severity
                    for tag in e.error_tags:
                        agent_error_tags_by_severity[tag][severity] += 1
                sims_by_max_agent_severity[max_agent_sev] += 1
            else:
                sims_by_max_agent_severity["none"] += 1

            if user_errors:
                max_user_sev = "none"
                for e in user_errors:
                    severity = e.severity or "unknown"
                    user_errors_by_severity[severity] += 1
                    # Track max severity (critical_helped/critical_hindered > minor > unknown > none)
                    if severity in ("critical_helped", "critical_hindered"):
                        max_user_sev = severity
                    elif severity == "minor" and max_user_sev not in (
                        "critical_helped",
                        "critical_hindered",
                    ):
                        max_user_sev = "minor"
                    elif max_user_sev == "none":
                        max_user_sev = severity
                    for tag in e.error_tags:
                        user_error_tags_by_severity[tag][severity] += 1
                sims_by_max_user_severity[max_user_sev] += 1
            else:
                sims_by_max_user_severity["none"] += 1

            # Find first critical error source
            def get_error_position(e):
                """Get position for sorting: tick_start for full-duplex, turn_idx for turn-based."""
                if e.tick_start is not None:
                    return e.tick_start
                if e.turn_idx is not None:
                    return e.turn_idx
                return float("inf")

            critical_errors = [
                e
                for e in sim.review.errors
                if (e.source == "agent" and e.severity == "critical")
                or (
                    e.source == "user"
                    and e.severity in ("critical_helped", "critical_hindered")
                )
            ]
            if critical_errors:
                first_critical = min(critical_errors, key=get_error_position)
                sims_by_first_critical_source[first_critical.source] += 1
            else:
                sims_by_first_critical_source["none"] += 1

        elif sim.user_only_review is not None:
            user_errors = sim.user_only_review.errors
            sims_by_max_agent_severity["none"] += 1  # No agent review for user-only
            if user_errors:
                max_user_sev = "none"
                for e in user_errors:
                    severity = e.severity or "unknown"
                    user_errors_by_severity[severity] += 1
                    if severity in ("critical_helped", "critical_hindered"):
                        max_user_sev = severity
                    elif severity == "minor" and max_user_sev not in (
                        "critical_helped",
                        "critical_hindered",
                    ):
                        max_user_sev = "minor"
                    elif max_user_sev == "none":
                        max_user_sev = severity
                    for tag in e.error_tags:
                        user_error_tags_by_severity[tag][severity] += 1
                sims_by_max_user_severity[max_user_sev] += 1

                # Track first critical for user-only review
                critical_user_errors = [
                    e
                    for e in user_errors
                    if e.severity in ("critical_helped", "critical_hindered")
                ]
                if critical_user_errors:
                    sims_by_first_critical_source["user"] += 1
                else:
                    sims_by_first_critical_source["none"] += 1
            else:
                sims_by_max_user_severity["none"] += 1
                sims_by_first_critical_source["none"] += 1

    return AgentMetrics(
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        pass_hat_ks_db_only=pass_hat_ks_db_only,
        pass_hat_ks_db_and_communication=pass_hat_ks_db_and_communication,
        pass_hat_ks_db_communication_nl=pass_hat_ks_db_communication_nl,
        avg_agent_cost=avg_agent_cost,
        avg_user_cost=avg_user_cost,
        avg_total_cost=avg_total_cost,
        median_agent_cost=median_agent_cost,
        median_user_cost=median_user_cost,
        median_total_cost=median_total_cost,
        total_agent_cost=total_agent_cost,
        total_user_cost=total_user_cost,
        total_cost=total_cost,
        total_agent_input_tokens=total_agent_input_tokens,
        total_agent_cached_input_tokens=total_agent_cached_input_tokens,
        total_agent_output_tokens=total_agent_output_tokens,
        total_agent_reasoning_tokens=total_agent_reasoning_tokens,
        total_user_input_tokens=total_user_input_tokens,
        total_user_cached_input_tokens=total_user_cached_input_tokens,
        total_user_output_tokens=total_user_output_tokens,
        total_user_reasoning_tokens=total_user_reasoning_tokens,
        agent_cached_input_percentage=agent_cached_input_percentage,
        agent_reasoning_output_percentage=agent_reasoning_output_percentage,
        user_cached_input_percentage=user_cached_input_percentage,
        user_reasoning_output_percentage=user_reasoning_output_percentage,
        total_agent_input_cost_with_cache=total_agent_input_cost_with_cache,
        total_agent_input_cost_without_cache=total_agent_input_cost_without_cache,
        total_agent_output_cost=total_agent_output_cost,
        total_user_input_cost_with_cache=total_user_input_cost_with_cache,
        total_user_input_cost_without_cache=total_user_input_cost_without_cache,
        total_user_output_cost=total_user_output_cost,
        total_simulations=total_simulations,
        total_tasks=total_tasks,
        infra_error_count=infra_error_count,
        total_read_actions=total_read_actions,
        correct_read_actions=correct_read_actions,
        total_write_actions=total_write_actions,
        correct_write_actions=correct_write_actions,
        db_match_count=db_match_count,
        db_mismatch_count=db_mismatch_count,
        db_not_checked=db_not_checked,
        auth_succeeded=auth_succeeded,
        auth_failed=auth_failed,
        auth_not_needed=auth_not_needed,
        auth_not_checked=auth_not_checked,
        termination_user_stop=termination_user_stop,
        termination_agent_stop=termination_agent_stop,
        termination_max_steps=termination_max_steps,
        termination_error=termination_error,
        termination_infrastructure_error=infra_error_count,
        sims_with_unresponsive_period=sims_with_unresponsive_period,
        sims_with_responsiveness_info=sims_with_responsiveness_info,
        agent_errors_by_severity=dict(agent_errors_by_severity),
        user_errors_by_severity=dict(user_errors_by_severity),
        sims_by_max_agent_severity=dict(sims_by_max_agent_severity),
        sims_by_max_user_severity=dict(sims_by_max_user_severity),
        sims_by_first_critical_source=dict(sims_by_first_critical_source),
        agent_error_tags_by_severity={
            k: dict(v) for k, v in agent_error_tags_by_severity.items()
        },
        user_error_tags_by_severity={
            k: dict(v) for k, v in user_error_tags_by_severity.items()
        },
    )


def display_metrics(metrics: AgentMetrics) -> None:
    print(f"🏆 Average reward: {metrics.avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in metrics.pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
    print(f"💰 Average agent cost: {metrics.avg_agent_cost}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    results = Results.load(Path(args.results))
    metrics = compute_metrics(results)
    display_metrics(metrics)
