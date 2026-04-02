"""Tests for tau2.config_cli YAML helpers (base_run inheritance)."""

import pytest

from tau2.config_cli import (
    _effective_run_concurrency,
    _merge_dicts,
    _resolve_run_with_base_inheritance,
    _runs_map_from_entries,
)


def test_merge_dicts_deep_override():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 9}, "b": 2}
    assert _merge_dicts(base, override) == {"a": 1, "nested": {"x": 1, "y": 9}, "b": 2}


def test_base_run_simple_chain():
    all_runs = {
        "parent": {"domain": "retail", "seed": 1, "agent_llm_args": {"temperature": 0}},
        "child": {
            "base_run": "parent",
            "agent_llm_args": {"reasoning_level": "high"},
        },
    }
    out = _resolve_run_with_base_inheritance(
        "child", all_runs["child"], all_runs, frozenset()
    )
    assert out["domain"] == "retail"
    assert out["seed"] == 1
    assert out["agent_llm_args"] == {
        "temperature": 0,
        "reasoning_level": "high",
    }
    assert "base_run" not in out


def test_base_run_three_levels():
    all_runs = {
        "a": {"k": 1, "d": {"x": 0}},
        "b": {"base_run": "a", "d": {"y": 1}},
        "c": {"base_run": "b", "d": {"x": 9}},
    }
    out = _resolve_run_with_base_inheritance("c", all_runs["c"], all_runs, frozenset())
    assert out["k"] == 1
    assert out["d"] == {"x": 9, "y": 1}


def test_base_run_cycle_raises():
    all_runs = {
        "a": {"base_run": "b", "x": 1},
        "b": {"base_run": "a", "y": 2},
    }
    with pytest.raises(ValueError, match="base_run cycle"):
        _resolve_run_with_base_inheritance("a", all_runs["a"], all_runs, frozenset())


def test_base_run_missing_parent_raises():
    all_runs = {"child": {"base_run": "nope"}}
    with pytest.raises(ValueError, match="unknown base_run"):
        _resolve_run_with_base_inheritance(
            "child", all_runs["child"], all_runs, frozenset()
        )


def test_runs_map_from_entries():
    m = _runs_map_from_entries(
        [("u", {"a": 1}), ("v", {"base_run": "u", "b": 2})]
    )
    assert m["v"]["base_run"] == "u"


def test_effective_run_concurrency_cli_wins():
    assert _effective_run_concurrency({"run_concurrency": 1}, 4) == 4


def test_effective_run_concurrency_yaml():
    assert _effective_run_concurrency({"run_concurrency": 3}, None) == 3
    assert _effective_run_concurrency({}, None) == 1


def test_effective_run_concurrency_invalid_falls_back():
    assert _effective_run_concurrency({"run_concurrency": "bad"}, None) == 1
