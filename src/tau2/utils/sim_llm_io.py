"""
Per-simulation LLM I/O capture under artifacts/task_*/sim_*/{agent,user}/.

Used to mirror Logfire raw_io (Gemini) and LiteLLM request/response dumps on disk
so conversation history can be replayed without Logfire.
"""

from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

Actor = Literal["agent", "user"]

sim_llm_io_root: ContextVar[Optional[Path]] = ContextVar("sim_llm_io_root", default=None)


def set_sim_llm_io_root(path: Optional[Path | str]) -> None:
    """Root directory: .../artifacts/task_<id>/sim_<uuid>/. None disables file capture."""
    if path is None:
        sim_llm_io_root.set(None)
        return
    if isinstance(path, str):
        path = Path(path)
    sim_llm_io_root.set(path)


def infer_actor_from_call_name(call_name: Optional[str]) -> Optional[Actor]:
    """Map LiteLLM call_name heuristics to agent vs user; None = skip sim I/O."""
    if not call_name:
        return None
    cn = call_name.lower()
    if any(
        x in cn
        for x in (
            "user_simulator",
            "vertex_user",
            "user_streaming",
            "user_response",
            "interruption_decision",
            "backchannel_decision",
        )
    ):
        return "user"
    if any(
        x in cn
        for x in (
            "agent_response",
            "vertex_agent",
            "agent_gt",
            "agent_solo",
            "interface_agent",
        )
    ):
        return "agent"
    return None


def _write_sim_llm_io_json_sync(
    root: Path,
    actor: Actor,
    call_name: str,
    payload: dict[str, Any],
) -> None:
    """Synchronous write (used from background thread). ``root`` is the sim_* directory."""
    actor_dir = root / actor
    actor_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = actor_dir / f"llm_{ts}.json"
    envelope: dict[str, Any] = {
        "call_name": call_name,
        "actor": actor,
        "timestamp": datetime.now().isoformat(),
        **payload,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def write_sim_llm_io_json(
    actor: Actor,
    *,
    call_name: str,
    payload: dict[str, Any],
) -> None:
    """Queue ``llm_<timestamp>.json`` under ``sim_llm_io_root/<actor>/`` (non-blocking).

    Reads :func:`sim_llm_io_root` in the caller thread (ContextVar is not inherited by
    new threads). A daemon thread performs JSON serialization and disk I/O so LLM /
    Logfire spans are not extended by large file writes.
    """
    root = sim_llm_io_root.get()
    if root is None:
        return

    def _job() -> None:
        try:
            # Copy in the worker so the hot path does not pay for deepcopy of large I/O blobs.
            _write_sim_llm_io_json_sync(
                root, actor, call_name, deepcopy(payload)
            )
        except Exception:
            pass

    threading.Thread(
        target=_job,
        daemon=True,
        name="sim_llm_io",
    ).start()
