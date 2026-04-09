#!/usr/bin/env python3
"""
Integration test: Jinja ``LLMClient`` + Vertex vLLM OpenAI ``/v1/completions``
(same transport as tau3 ``vertex_openai_chat_url`` — see ``LLMClient`` in ``vllm_jinja_client``).

Requires Google credentials that can mint an identity token for the Cloud Run URL.

Examples (from ``tau3-bench-fork/``):

  uv run python tools/test_vllm_jinja_tau3_vertex.py

  uv run python tools/test_vllm_jinja_tau3_vertex.py --request-yaml src/tau2/utils/request.yaml

  uv run python tools/test_vllm_jinja_tau3_vertex.py --no-tool
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.utils import RETAIL_DB_PATH, RETAIL_TASK_SET_PATH
from tau2.utils.vllm_jinja_client import LLMClient, Tool, ToolParameter

DEFAULT_VERTEX_OPENAI_CHAT_URL = "https://gemma4-26b-fp8-gcreopmlnq-uc.a.run.app"
DEFAULT_VERTEX_OPENAI_CHAT_MODEL = "gemma4-26b-fp8"


def _load_yaml(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def _schema_to_tool_param(schema: dict) -> ToolParameter:
    t = str(schema.get("type") or "string")
    enum = schema.get("enum")
    enum_list = [str(v) for v in enum] if isinstance(enum, list) else None

    properties_raw = schema.get("properties")
    props: dict[str, ToolParameter] | None = None
    if isinstance(properties_raw, dict):
        props = {
            str(k): _schema_to_tool_param(v)
            for k, v in properties_raw.items()
            if isinstance(v, dict)
        }

    items_raw = schema.get("items")
    items = items_raw if isinstance(items_raw, dict) else None
    required_raw = schema.get("required")
    required = (
        [str(x) for x in required_raw] if isinstance(required_raw, list) else None
    )

    return ToolParameter(
        type=t,
        description=str(schema.get("description") or ""),
        enum=enum_list,
        properties=props,
        required=required,
        items=items,
        nullable=bool(schema.get("nullable", False)),
    )


def _load_retail_task_user_message(tasks_path: Path, task_id: str) -> str:
    with tasks_path.open(encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid tasks file: {tasks_path}")
    task = next((t for t in tasks if str(t.get("id")) == str(task_id)), None)
    if not isinstance(task, dict):
        raise ValueError(f"Task id {task_id!r} not found in {tasks_path}")

    inst = task.get("user_scenario", {}).get("instructions", {})
    known_info = str(inst.get("known_info") or "").strip()
    reason = str(inst.get("reason_for_call") or "").strip()
    if not known_info and not reason:
        raise ValueError(f"Task id {task_id!r} has no known_info/reason_for_call")
    return f"{known_info}\n{reason}".strip()


def _register_all_retail_tools(client: LLMClient, db_path: Path) -> None:
    toolkit = RetailTools(RetailDB.load(db_path))
    tool_map = toolkit.get_tools()  # name -> tau2 environment Tool

    for tool_name, tool_obj in tool_map.items():
        schema = tool_obj.openai_schema
        fn = schema.get("function", {})
        params = fn.get("parameters", {})
        param_props = params.get("properties", {}) if isinstance(params, dict) else {}
        param_defs = {
            str(k): _schema_to_tool_param(v)
            for k, v in param_props.items()
            if isinstance(v, dict)
        }
        required = (
            [str(x) for x in params.get("required", [])]
            if isinstance(params, dict)
            else []
        )

        def _handler(_tool_name: str):
            def _invoke(**kwargs):
                out = toolkit.use_tool(_tool_name, **kwargs)
                if hasattr(out, "model_dump"):
                    return out.model_dump()
                return out

            return _invoke

        client.register_tool(
            Tool(
                name=str(fn.get("name") or tool_name),
                description=str(fn.get("description") or tool_name),
                parameters=param_defs,
                required=required,
                handler=_handler(tool_name),
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--request-yaml",
        type=Path,
        default=None,
        help="Optional request YAML (vertex URL/model, parameters, enable_thinking).",
    )
    ap.add_argument(
        "--no-tool",
        action="store_true",
        help="Do not register a tool; single-turn reply only.",
    )
    ap.add_argument(
        "--retail-task-id",
        type=str,
        default=None,
        help="Run a real retail benchmark task id (loads tasks.json + all retail tools).",
    )
    ap.add_argument(
        "--retail-tasks-path",
        type=Path,
        default=Path(RETAIL_TASK_SET_PATH),
        help="Path to retail tasks.json.",
    )
    ap.add_argument(
        "--retail-db-path",
        type=Path,
        default=Path(RETAIL_DB_PATH),
        help="Path to retail db.json.",
    )
    ap.add_argument(
        "--policy-path",
        type=Path,
        default=Path("data/tau2/domains/retail/policy_solo.md"),
        help="Path to system policy prompt file.",
    )
    ap.add_argument(
        "--quiet-prompt",
        action="store_true",
        help="Do not print the full rendered prompt in chat().",
    )
    ap.add_argument(
        "--no-raw-json",
        action="store_true",
        help="Disable pretty raw completion JSON logging.",
    )
    ap.add_argument(
        "--trace-json",
        type=Path,
        default=Path("tools/simulation_trace.json"),
        help="Where to write conversation trace JSON.",
    )
    args = ap.parse_args()

    cfg = _load_yaml(args.request_yaml)
    base = (
        os.environ.get("VERTEX_OPENAI_CHAT_URL", "").strip()
        or str(
            cfg.get("vertex_openai_chat_url") or DEFAULT_VERTEX_OPENAI_CHAT_URL
        ).strip()
    )
    model = (
        os.environ.get("VERTEX_OPENAI_CHAT_MODEL", "").strip()
        or str(
            cfg.get("vertex_openai_chat_model") or DEFAULT_VERTEX_OPENAI_CHAT_MODEL
        ).strip()
    )
    params = cfg.get("parameters") if isinstance(cfg.get("parameters"), dict) else {}
    max_tokens = int(params.get("max_tokens", 512))
    temperature = float(params.get("temperature", 0.0))
    top_p = float(params.get("top_p", 0.95))
    timeout_s = int(params.get("timeout_s", 180))
    enable_thinking = bool(cfg.get("enable_thinking", True))
    # Thinking + tool calls need headroom; tiny max_tokens hits finish_reason=length mid-thought.
    if not args.no_tool:
        max_tokens = max(max_tokens, 1024)
    elif enable_thinking:
        max_tokens = max(max_tokens, 256)

    system_prompt = "You are a helpful assistant."
    if args.policy_path.is_file():
        system_prompt = args.policy_path.read_text(encoding="utf-8").strip()

    client = LLMClient(
        endpoint=base.rstrip("/"),
        vertex_openai_model=model,
        bos_token=str(cfg.get("bos_token") or "<bos>"),
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )

    if args.retail_task_id is not None:
        _register_all_retail_tools(client, args.retail_db_path)
        user_msg = _load_retail_task_user_message(
            args.retail_tasks_path, args.retail_task_id
        )
    elif not args.no_tool:
        weather = Tool(
            name="get_weather",
            description="Get the current weather for a location.",
            parameters={
                "location": ToolParameter(type="string", description="City name"),
                "unit": ToolParameter(
                    type="string",
                    description="Temperature unit",
                    enum=["celsius", "fahrenheit"],
                ),
            },
            required=["location"],
            handler=lambda location, unit="celsius": {
                "temperature": 22,
                "unit": unit,
                "condition": "Sunny",
                "location": location,
            },
        )
        client.register_tool(weather)
        user_msg = "What's the weather in Paris?"
    else:
        user_msg = "Say hi in one word."

    extra_gen: dict = {}
    if "stop" in params:
        extra_gen["stop"] = params["stop"]
    if not args.no_tool:
        stop_for_tools = extra_gen.get("stop")
        if isinstance(stop_for_tools, list):
            if "<|tool_response>" not in stop_for_tools:
                stop_for_tools = [*stop_for_tools, "<|tool_response>"]
        else:
            stop_for_tools = ["<turn|>", "<|turn>", "<|tool_response>"]
        extra_gen["stop"] = stop_for_tools
    if "stream" in params:
        extra_gen["stream"] = bool(params["stream"])
    if "seed" in params and params["seed"] is not None:
        extra_gen["seed"] = int(params["seed"])
    for k in (
        "add_special_tokens",
        "skip_special_tokens",
        "include_stop_str_in_output",
    ):
        if k in params:
            extra_gen[k] = bool(params[k])
    extra_gen["log_raw_json"] = not args.no_raw_json
    extra_gen["top_p"] = top_p

    trace_events: list[dict] = []
    try:
        final = client.chat(
            user_msg,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            verbose=not args.quiet_prompt,
            auto_invoke_tools=not args.no_tool,
            trace_events=trace_events,
            **extra_gen,
        )
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    trace_doc = {
        "request_yaml": str(args.request_yaml) if args.request_yaml else None,
        "retail_task_id": args.retail_task_id,
        "endpoint": base.rstrip("/"),
        "model": model,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "system_prompt": system_prompt,
        "initial_user_message": user_msg,
        "events": trace_events,
        "final_assistant_text": final or "",
    }
    args.trace_json.parent.mkdir(parents=True, exist_ok=True)
    args.trace_json.write_text(
        json.dumps(trace_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[trace] wrote simulation trace to {args.trace_json}")

    print("\n--- final assistant text ---")
    print(final or "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
