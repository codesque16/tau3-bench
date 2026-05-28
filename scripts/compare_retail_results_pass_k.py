#!/usr/bin/env python3
"""
Load one or more tau2 retail-style results.json files and emit:
  - a wide CSV: task_id × run columns; each cell is the success count (integer), or
    ``s/n`` if ``--cell-format ratio``
  - pass^k (k=1..K) per run using C(s,k)/C(n,k) averaged over tasks (arxiv:2406.12045),
    matching tau2-bench ``pass_hat_k`` / ``AgentMetrics`` aggregation.
  - optionally, a second pass^k table restricted to ``--subset-ids`` task IDs.

Each input file must be a monolithic JSON with top-level ``simulation_index`` (list of
entries with task_id, trial, reward). Optional ``info.num_trials`` is used as the
expected trial count when all tasks have the same n.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _is_success(reward: float) -> bool:
    return (1.0 - 1e-6) <= reward <= (1.0 + 1e-6)


def _pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    if num_trials < k:
        return 0.0
    if success_count < k:
        return 0.0
    return math.comb(success_count, k) / math.comb(num_trials, k)


def _aggregate_pass_hat_ks(
    per_task_n_s: dict[str, tuple[int, int]],
    *,
    max_k: int,
    subset: set[str] | None = None,
) -> dict[int, float]:
    tasks = {k: v for k, v in per_task_n_s.items() if subset is None or k in subset}
    if not tasks:
        return {k: 0.0 for k in range(1, max_k + 1)}
    m = len(tasks)
    out: dict[int, float] = {}
    for k in range(1, max_k + 1):
        total = 0.0
        for n, s in tasks.values():
            total += _pass_hat_k(n, s, k)
        out[k] = total / m
    return out


def _load_simulation_index(path: Path) -> tuple[list[dict[str, Any]], int | None]:
    data = json.loads(path.read_text())
    sims = data.get("simulation_index") or data.get("simulations")
    if not isinstance(sims, list):
        raise ValueError(f"{path}: expected 'simulation_index' or 'simulations' list")
    info = data.get("info") or {}
    num_trials = info.get("num_trials")
    return sims, num_trials if isinstance(num_trials, int) else None


def _counts_from_sims(
    sims: list[dict[str, Any]],
) -> tuple[dict[str, tuple[int, int]], int]:
    """Returns (task_id -> (n_trials, success_count)), and min_n across tasks."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in sims:
        tid = str(row.get("task_id", ""))
        if not tid:
            continue
        r = row.get("reward")
        if not isinstance(r, (int, float)):
            continue
        by_task[tid].append(float(r))

    per_task: dict[str, tuple[int, int]] = {}
    min_n = 10**9
    for tid, rewards in by_task.items():
        n = len(rewards)
        s = sum(1 for x in rewards if _is_success(x))
        per_task[tid] = (n, s)
        min_n = min(min_n, n)
    if not per_task:
        return {}, 0
    return per_task, min_n


def _all_task_ids(per_run: Iterable[dict[str, tuple[int, int]]]) -> list[str]:
    ids: set[str] = set()
    for m in per_run:
        ids.update(m.keys())

    def sort_key(t: str) -> tuple[int, str]:
        try:
            return (int(t), t)
        except ValueError:
            return (10**9, t)

    return sorted(ids, key=sort_key)


def _print_pass_k_table(
    run_rows: list[dict[str, Any]],
    max_k: int,
    *,
    global_min_n: int,
    label: str = "",
    subset_size: int | None = None,
) -> None:
    task_count = subset_size if subset_size is not None else None
    header_parts = [f"pass^k"]
    if label:
        header_parts.append(f"[{label}]")
    if task_count is not None:
        header_parts.append(f"(n={task_count} tasks)")
    print(" ".join(header_parts))
    print(f"min trials per task: {global_min_n}")
    hdr = ["run", "n_tasks", "min_n"] + [f"pass^{k}" for k in range(1, max_k + 1)]
    print("\t".join(hdr))
    for r in run_rows:
        row = [
            r["name"],
            str(r["num_tasks"]),
            str(r["min_trials_per_task"]),
            *[f"{r[f'pass^{k}']:.6f}" for k in range(1, max_k + 1)],
        ]
        print("\t".join(row))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_json",
        nargs="+",
        type=Path,
        help="Paths to results.json files",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        help="Short column names for each file (default: run_1, run_2, ...)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Write task × run success counts CSV here",
    )
    parser.add_argument(
        "--out-metrics-json",
        type=Path,
        help="Write pass^k and per-run summaries as JSON",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=4,
        help="Largest k for pass^k (default 4; also capped by min trial count per task)",
    )
    parser.add_argument(
        "--cell-format",
        choices=("int", "ratio"),
        default="int",
        help="CSV cell: raw success count (int) or s/n (ratio). Default: int",
    )
    parser.add_argument(
        "--subset-ids",
        nargs="+",
        help="Compute an additional pass^k table restricted to these task IDs",
    )
    args = parser.parse_args()

    paths: list[Path] = [p.expanduser().resolve() for p in args.results_json]
    names: list[str]
    if args.names:
        if len(args.names) != len(paths):
            parser.error("--names length must match number of results.json paths")
        names = list(args.names)
    else:
        names = [f"run_{i + 1}" for i in range(len(paths))]

    subset: set[str] | None = None
    if args.subset_ids:
        subset = {str(x) for x in args.subset_ids}

    run_per_task: list[dict[str, tuple[int, int]]] = []
    run_min_n: list[int] = []
    declared_trials: list[int | None] = []

    for p in paths:
        if not p.is_file():
            print(f"error: not a file: {p}", file=sys.stderr)
            sys.exit(1)
        sims, nt = _load_simulation_index(p)
        per_task, min_n = _counts_from_sims(sims)
        run_per_task.append(per_task)
        run_min_n.append(min_n)
        declared_trials.append(nt)

    task_ids = _all_task_ids(run_per_task)

    # Determine max_k: min(declared, min_n, arg.max_k)
    global_min_n = min(run_min_n) if run_min_n else 0
    expected = declared_trials[0]
    for dt in declared_trials[1:]:
        if dt != expected:
            expected = None
    cap_n = global_min_n
    if expected is not None:
        cap_n = min(cap_n, expected) if cap_n else expected
    max_k = max(0, min(args.max_k, cap_n))

    # Full-set pass^k rows
    full_rows: list[dict[str, Any]] = []
    for name, per_task, p, mn in zip(names, run_per_task, paths, run_min_n):
        pk = _aggregate_pass_hat_ks(per_task, max_k=max_k) if max_k > 0 else {}
        full_rows.append(
            {
                "name": name,
                "path": str(p),
                "num_tasks": len(per_task),
                "min_trials_per_task": mn,
                **{f"pass^{k}": pk.get(k, 0.0) for k in range(1, max_k + 1)},
            }
        )

    # Subset pass^k rows
    subset_rows: list[dict[str, Any]] = []
    if subset is not None:
        for name, per_task, p, mn in zip(names, run_per_task, paths, run_min_n):
            subset_tasks = {k: v for k, v in per_task.items() if k in subset}
            subset_min_n = min((v[0] for v in subset_tasks.values()), default=0)
            pk = _aggregate_pass_hat_ks(per_task, max_k=max_k, subset=subset) if max_k > 0 else {}
            subset_rows.append(
                {
                    "name": name,
                    "path": str(p),
                    "num_tasks": len(subset_tasks),
                    "min_trials_per_task": subset_min_n,
                    **{f"pass^{k}": pk.get(k, 0.0) for k in range(1, max_k + 1)},
                }
            )

    # CSV output
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            fieldnames = ["task_id", *names]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for tid in task_ids:
                row: dict[str, Any] = {"task_id": tid}
                for name, per_task in zip(names, run_per_task):
                    if tid not in per_task:
                        row[name] = ""
                    else:
                        n, s = per_task[tid]
                        row[name] = f"{s}/{n}" if args.cell_format == "ratio" else str(s)
                w.writerow(row)

    # JSON metrics output
    metrics: dict[str, Any] = {
        "max_k_used": max_k,
        "all_tasks": full_rows,
    }
    if subset_rows:
        metrics["subset_tasks"] = {
            "ids": sorted(subset, key=lambda x: (int(x) if x.isdigit() else 10**9, x)),
            "runs": subset_rows,
        }
    if args.out_metrics_json:
        args.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics_json.write_text(json.dumps(metrics, indent=2))

    # Stdout
    print(f"Tasks (rows): {len(task_ids)}")
    print(f"pass^k uses k=1..{max_k}\n")

    _print_pass_k_table(full_rows, max_k, global_min_n=global_min_n, label="all tasks")

    if subset_rows:
        # min_n across subset tasks in any run
        subset_global_min_n = min(r["min_trials_per_task"] for r in subset_rows)
        n_present = subset_rows[0]["num_tasks"]
        _print_pass_k_table(
            subset_rows,
            max_k,
            global_min_n=subset_global_min_n,
            label=f"subset {len(subset)} ids",
            subset_size=n_present,
        )

    if args.out_csv:
        print(f"Wrote {args.out_csv}")
    if args.out_metrics_json:
        print(f"Wrote {args.out_metrics_json}")


if __name__ == "__main__":
    main()
