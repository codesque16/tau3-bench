"""Tests for per-simulation LLM I/O file capture."""

import json
import time
from pathlib import Path

from tau2.utils.sim_llm_io import (
    infer_actor_from_call_name,
    set_sim_llm_io_root,
    write_sim_llm_io_json,
)


def test_infer_actor():
    assert infer_actor_from_call_name("vertex_agent_response") == "agent"
    assert infer_actor_from_call_name("vertex_user_simulator_response") == "user"
    assert infer_actor_from_call_name("user_simulator_response") == "user"
    assert infer_actor_from_call_name("llm_judge_review") is None


def test_write_sim_llm_io_json_creates_file(tmp_path: Path):
    root = tmp_path / "sim_x"
    root.mkdir()
    set_sim_llm_io_root(root)
    try:
        write_sim_llm_io_json(
            "agent",
            call_name="test",
            payload={"format": "google_genai", "gemini_io_json": {"a": 1}},
        )
        time.sleep(0.2)
        files = list((root / "agent").glob("llm_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["actor"] == "agent"
        assert data["call_name"] == "test"
        assert data["gemini_io_json"] == {"a": 1}
    finally:
        set_sim_llm_io_root(None)
