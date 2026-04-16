#!/usr/bin/env python3
"""
YAML-driven tester for Vertex/OpenAI-compatible ``/v1/chat/completions``.

Usage:
  uv run python tools/test_vertex_chat_completions_yaml.py --request-yaml tools/request_chat_completions.sample.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests
import yaml

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.utils.vertex_endpoint_chat import (
    build_openai_chat_completions_body,
    fetch_google_identity_token_for_audience,
    normalize_vertex_openai_chat_url,
    vertex_openai_chat_id_token_audience,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def _resolve_text_field(cfg: dict[str, Any], key: str) -> str | None:
    val = cfg.get(key)
    return str(val).strip() if val is not None else None


def _load_text_path(path_s: str | None) -> str | None:
    if not path_s:
        return None
    p = Path(path_s).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.is_file():
        raise ValueError(f"Path does not exist: {p}")
    return p.read_text(encoding="utf-8")


def _build_messages(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = cfg.get("messages")
    if isinstance(msgs, list):
        out: list[dict[str, Any]] = []
        for m in msgs:
            if isinstance(m, dict):
                role = str(m.get("role") or "").strip()
                if role:
                    out.append(
                        {"role": role, "content": str(m.get("content") or "")}
                    )
        if out:
            return out

    system_prompt = _resolve_text_field(cfg, "system_prompt")
    if not system_prompt:
        system_prompt = _load_text_path(_resolve_text_field(cfg, "system_prompt_path")) or ""
    user_prompt = _resolve_text_field(cfg, "user_prompt")
    if not user_prompt:
        user_prompt = _load_text_path(_resolve_text_field(cfg, "user_prompt_path")) or ""
    if not user_prompt:
        raise ValueError("Provide either messages[] or user_prompt/user_prompt_path in YAML")

    out2: list[dict[str, Any]] = []
    if system_prompt.strip():
        out2.append({"role": "system", "content": system_prompt})
    out2.append({"role": "user", "content": user_prompt})
    return out2


def _load_tools(cfg: dict[str, Any]) -> list[dict[str, Any]] | None:
    if bool(cfg.get("include_retail_tools", True)):
        db_path_raw = _resolve_text_field(cfg, "retail_db_path")
        db_path = Path(db_path_raw) if db_path_raw else Path(RETAIL_DB_PATH)
        toolkit = RetailTools(RetailDB.load(db_path))
        return [tool.openai_schema for tool in toolkit.get_tools().values()]

    tools = cfg.get("tools")
    if isinstance(tools, list):
        return [t for t in tools if isinstance(t, dict)]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request-yaml", type=Path, required=True)
    ap.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional file to write request/response JSON.",
    )
    args = ap.parse_args()

    cfg = _load_yaml(args.request_yaml)
    base_url = str(cfg.get("vertex_openai_chat_url") or "").strip()
    model = str(cfg.get("vertex_openai_chat_model") or "").strip()
    if not base_url or not model:
        raise ValueError("YAML must include vertex_openai_chat_url and vertex_openai_chat_model")

    params = cfg.get("vertex_endpoint_parameters")
    if params is not None and not isinstance(params, dict):
        raise ValueError("vertex_endpoint_parameters must be a mapping when provided")
    llm_args: dict[str, Any] = {"vertex_endpoint_parameters": dict(params or {})}

    messages = _build_messages(cfg)
    tools = _load_tools(cfg)
    body = build_openai_chat_completions_body(
        llm_args=llm_args,
        openai_messages=messages,
        tools=tools,
        model=model,
    )

    chat_url = normalize_vertex_openai_chat_url(base_url)
    audience = _resolve_text_field(cfg, "vertex_openai_chat_audience")
    audience = audience or vertex_openai_chat_id_token_audience(chat_url)
    token = fetch_google_identity_token_for_audience(audience)

    timeout_s = int(cfg.get("timeout_s") or 180)
    resp = requests.post(
        chat_url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()

    print("\n--- request url ---")
    print(chat_url)
    print("\n--- request body ---")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    print("\n--- response json ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(
                {
                    "request_url": chat_url,
                    "request_body": body,
                    "response": payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n[saved] {args.save_json}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
