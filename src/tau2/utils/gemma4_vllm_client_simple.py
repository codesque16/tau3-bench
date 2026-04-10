"""Simple Gemma vLLM client: request/response first."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from tau2.utils.vertex_endpoint_chat import (
    fetch_google_identity_token_for_audience,
    gemma4_vllm_generation_params_from_llm_args,
    normalize_vertex_openai_chat_url,
    resolve_runtime_seed,
    vertex_openai_chat_id_token_audience,
)


TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"
TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"
TOOL_RESPONSE_OPEN = "<|tool_response>"
TOOL_RESPONSE_CLOSE = "<tool_response|>"
MODEL_TURN_OPEN = "<|turn>model\n"
USER_TURN_OPEN = "<|turn>user\n"
SYSTEM_TURN_OPEN = "<|turn>system\n"

TOOL_CALL_RE = re.compile(r"<\|tool_call>call:(\w+)\{([\s\S]*?)\}<tool_call\|>")
TOOL_RESPONSE_RE = re.compile(
    r"<\|tool_response>response:(\w+)\{value:<\|\"\|>([\s\S]*?)<\|\"\|>\}<tool_response\|>"
)
THOUGHT_BLOCK_RE = re.compile(r"<\|channel>thought\s*\n?([\s\S]*?)<channel\|>")


def build_v1_completions_request(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 3072,
    temperature: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a minimal /v1/completions JSON body."""
    body: dict[str, Any] = {
        "model": model,
        "prompt": str(prompt).rstrip("\n"),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        "stop": [TURN_CLOSE, TURN_OPEN],
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
    }
    if seed is not None:
        body["seed"] = int(seed)
    return body


def _auth_headers(base_url: str) -> dict[str, str]:
    chat_url = normalize_vertex_openai_chat_url(base_url)
    audience = vertex_openai_chat_id_token_audience(chat_url)
    token = fetch_google_identity_token_for_audience(audience)
    return {"Authorization": f"Bearer {token}"}


def post_v1_completions_simple(
    *,
    base_url: str,
    body: dict[str, Any],
    timeout_s: int = 180,
) -> dict[str, Any]:
    """POST to {base_url}/v1/completions with identity-token auth."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    resp = requests.post(
        url,
        headers={**_auth_headers(base_url), "Content-Type": "application/json"},
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("vLLM response is not a JSON object.")
    return payload


def extract_completion_text(payload: dict[str, Any]) -> str:
    """Extract completion text from OpenAI /v1/completions payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    txt = first.get("text")
    return txt if isinstance(txt, str) else ""


def _extract_thought_blocks(text: str) -> str:
    parts: list[str] = []
    pos = 0
    while True:
        i = text.find(THOUGHT_OPEN, pos)
        if i < 0:
            break
        nl = text.find("\n", i)
        if nl < 0:
            break
        j = text.find(THOUGHT_CLOSE, nl + 1)
        if j < 0:
            break
        block = text[nl + 1 : j].strip()
        if block:
            parts.append(block)
        pos = j + len(THOUGHT_CLOSE)
    return "\n\n".join(parts).strip()


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pos = 0
    idx = 0
    while True:
        i = text.find(TOOL_CALL_OPEN, pos)
        if i < 0:
            break
        j = text.find(TOOL_CALL_CLOSE, i + len(TOOL_CALL_OPEN))
        if j < 0:
            break
        payload = text[i + len(TOOL_CALL_OPEN) : j].strip()
        if payload.startswith("call:"):
            raw = payload[len("call:") :]
            name_end = raw.find("{")
            if name_end > 0 and raw.endswith("}"):
                fn_name = raw[:name_end].strip()
                arg_str = raw[name_end + 1 : -1]
                args: dict[str, Any] = _parse_tool_args(arg_str)
                out.append(
                    {
                        "id": f"call_{idx}_{fn_name}",
                        "type": "function",
                        "function": {"name": fn_name, "arguments": json.dumps(args)},
                    }
                )
                idx += 1
        pos = j + len(TOOL_CALL_CLOSE)
    return out


def _parse_tool_args(arg_str: str) -> dict[str, Any]:
    """Parse tool arguments from JSON or Gemma-style key:value format."""
    s = (arg_str or "").strip()
    if not s:
        return {}

    # Case 1: already full JSON object.
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Case 2: JSON object body without outer braces, e.g. '"city":"NYC"'.
    try:
        obj = json.loads("{" + s + "}")
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Case 3: Gemma wrapped quotes: <|"|>value<|"|>
    normalized = s.replace('<|"|>', '"')
    try:
        obj = json.loads("{" + normalized + "}")
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Case 4: key:<|"|>value<|"|> or key:value fallback.
    out: dict[str, Any] = {}
    for m in __import__("re").finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', s):
        out[m.group(1)] = m.group(2)
    for m in __import__("re").finditer(r'(\w+):([^,}<|\n"]+)', s):
        k, v = m.group(1), m.group(2).strip()
        if k not in out and v:
            out[k] = v
    return out


def _extract_text_part(text: str) -> str:
    """Visible assistant text only; strips thought/tool/transcript markers."""
    s = str(text or "")
    s = THOUGHT_BLOCK_RE.sub("", s)
    s = TOOL_CALL_RE.sub("", s)
    s = TOOL_RESPONSE_RE.sub("", s)
    s = s.replace(MODEL_TURN_OPEN, "")
    s = s.replace(USER_TURN_OPEN, "")
    s = s.replace(SYSTEM_TURN_OPEN, "")
    s = s.replace(TURN_CLOSE, "")
    return s.strip()


def parse_request_response_to_messages(
    request_text: str,
    response_text: str,
) -> list[dict[str, Any]]:
    """
    Parse raw transcript (request + response) into a simple OpenAI-style array.

    Output shape:
    {"role": str, "content": str, "thought": str, "tool_calls": list}
    """
    transcript = f"{request_text or ''}{response_text or ''}"
    out: list[dict[str, Any]] = []
    pos = 0

    while True:
        start = transcript.find(TURN_OPEN, pos)
        if start < 0:
            break
        role_start = start + len(TURN_OPEN)
        role_end = transcript.find("\n", role_start)
        if role_end < 0:
            break
        role = transcript[role_start:role_end].strip()
        body_start = role_end + 1

        next_close = transcript.find(TURN_CLOSE, body_start)
        next_turn = transcript.find(TURN_OPEN, body_start)
        end = len(transcript)
        if next_close >= 0:
            end = min(end, next_close)
        if next_turn >= 0:
            end = min(end, next_turn)

        body = transcript[body_start:end]
        out.append(
            {
                "role": "assistant" if role == "model" else role,
                "content": body,
                "thought": _extract_thought_blocks(body),
                "tool_calls": _extract_tool_calls(body),
            }
        )

        pos = end + (len(TURN_CLOSE) if end == next_close else 0)
        if pos <= start:
            break

    return out


def request_and_parse(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 3072,
    temperature: float = 0.0,
    seed: int | None = None,
    timeout_s: int = 180,
) -> dict[str, Any]:
    """Single helper: send request and parse request+response transcript."""
    body = build_v1_completions_request(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    payload = post_v1_completions_simple(base_url=base_url, body=body, timeout_s=timeout_s)
    response_text = extract_completion_text(payload)
    messages = parse_request_response_to_messages(prompt, response_text)
    return {
        "request_url": f"{base_url.rstrip('/')}/v1/completions",
        "request_body": body,
        "response_payload": payload,
        "response_text": response_text,
        "messages": messages,
    }


def _tool_declaration_from_openai_schema(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = str(fn.get("name") or "").strip()
    if not name:
        return ""
    desc = str(fn.get("description") or "")
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "OBJECT"}
    return (
        "<|tool>declaration:"
        + name
        + "{description:<|\"|>"
        + desc
        + "<|\"|>,parameters:"
        + json.dumps(params)
        + "}<tool|>"
    )


def _assistant_tool_call_block(tc: dict[str, Any]) -> str:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = str(fn.get("name") or "")
    args_raw = fn.get("arguments")
    args_s = json.dumps(args_raw, separators=(",", ":")) if isinstance(args_raw, dict) else str(args_raw or "{}")
    return "<|tool_call>call:" + name + "{" + args_s + "}<tool_call|>"


def _tool_response_block_from_api_message(m: dict[str, Any], call_name: str) -> str:
    content = m.get("content")
    value = json.dumps(content, separators=(",", ":")) if isinstance(content, (dict, list)) else str(content or "")
    return (
        "<|tool_response>response:"
        + call_name
        + "{value:<|\"|>"
        + value
        + "<|\"|>}<tool_response|>"
    )


def build_prompt_from_api_messages_simple(
    api_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    enable_thinking: bool,
    assistant_turn_injection_prefix: str,
) -> str:
    out: list[str] = ["<bos>"]
    sys = ""
    for m in api_messages:
        if str(m.get("role") or "") == "system":
            sys = str(m.get("content") or "")
            break
    if sys or tools:
        out.append("<|turn>system\n")
        if enable_thinking:
            out.append("<|think|>")
        out.append(sys)
        for t in tools or []:
            d = _tool_declaration_from_openai_schema(t)
            if d:
                out.append(d)
        out.append("<turn|>\n")

    i = 0
    while i < len(api_messages):
        m = api_messages[i]
        role = str(m.get("role") or "")
        if role == "system":
            i += 1
            continue
        if role == "user":
            out.append("<|turn>user\n")
            out.append(str(m.get("content") or ""))
            out.append("<turn|>\n")
            i += 1
            continue
        if role == "assistant":
            out.append("<|turn>model\n")
            out.append(str(m.get("content") or ""))
            tcs = m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []
            by_id: dict[str, str] = {}
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    by_id[str(tc.get("id") or "")] = str(fn.get("name") or "unknown_tool")
                    out.append(_assistant_tool_call_block(tc))
            j = i + 1
            tool_blocks: list[str] = []
            while j < len(api_messages) and str((api_messages[j] or {}).get("role") or "") == "tool":
                tm = api_messages[j]
                cname = by_id.get(str(tm.get("tool_call_id") or ""), "unknown_tool")
                tool_blocks.append(_tool_response_block_from_api_message(tm, cname))
                j += 1
            # Keep loop-test ordering in a single model turn:
            # text_part -> tool_call(s) -> tool_response(s) -> <turn|>
            if tool_blocks:
                out.extend(tool_blocks)
            out.append("<turn|>\n")
            i = j
            continue
        i += 1

    out.append("<|turn>model\n")
    atip = str(assistant_turn_injection_prefix or "")
    if atip:
        out.append(atip if atip.endswith("\n") else atip + "\n")
    return "".join(out)


def _build_loop_seed_request_text(
    api_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    enable_thinking: bool,
    assistant_turn_injection_prefix: str,
) -> str:
    """Build initial loop-test style request text once per conversation.

    Seed intentionally excludes injection; injection is appended only at
    send-time after optional thought stripping.
    """
    seed_messages: list[dict[str, Any]] = []
    for m in api_messages:
        role = str((m or {}).get("role") or "")
        if role in ("system", "user"):
            seed_messages.append(m)
    return build_prompt_from_api_messages_simple(
        seed_messages,
        tools,
        enable_thinking=enable_thinking,
        assistant_turn_injection_prefix="",
    )


def _append_new_tool_responses_from_api_messages(
    response_text: str,
    api_messages: list[dict[str, Any]],
    *,
    assistant_turn_injection_prefix: str,
) -> tuple[str, bool, int]:
    """Append new tool responses from trailing tool messages (loop style)."""
    if not api_messages:
        return response_text, False, 0

    tool_name_by_id: dict[str, str] = {}
    for m in api_messages:
        if str((m or {}).get("role") or "") != "assistant":
            continue
        tcs = m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            cid = str(tc.get("id") or "")
            if cid:
                tool_name_by_id[cid] = str(fn.get("name") or "unknown_tool")

    total_tool_messages = 0
    for m in api_messages:
        if str((m or {}).get("role") or "") == "tool":
            total_tool_messages += 1

    appended = False
    new_response = response_text
    for m in api_messages:
        if str((m or {}).get("role") or "") != "tool":
            continue
        cid = str(m.get("tool_call_id") or "")
        name = tool_name_by_id.get(cid, "unknown_tool")
        block = _tool_response_block_from_api_message(m, name)
        if block:
            new_response += block
            appended = True

    if appended:
        atip = str(assistant_turn_injection_prefix or "")
        new_response += TURN_CLOSE + MODEL_TURN_OPEN
        if atip:
            new_response += atip if atip.endswith("\n") else atip + "\n"
    return new_response, appended, total_tool_messages


def _strip_thought_blocks_for_request_text(text: str) -> str:
    """Remove closed thought blocks from transcript before next request send."""
    return THOUGHT_BLOCK_RE.sub("", str(text or ""))


def _sanitize_tool_response_markers_for_request_text(text: str) -> str:
    """Normalize duplicate/dangling tool-response markers in outgoing request."""
    s = str(text or "")
    # Collapse accidental duplicated opener before a canonical response block.
    s = s.replace(
        TOOL_RESPONSE_OPEN + TOOL_RESPONSE_OPEN + "response:",
        TOOL_RESPONSE_OPEN + "response:",
    )
    # If a stray tool-response opener appears before a tool call, drop it.
    s = s.replace(
        TOOL_RESPONSE_OPEN + TOOL_CALL_OPEN,
        TOOL_CALL_OPEN,
    )
    # Trim trailing dangling opener (no payload), if present.
    if s.endswith(TOOL_RESPONSE_OPEN):
        s = s[: -len(TOOL_RESPONSE_OPEN)]
    return s


def _append_injection_suffix(text: str, injection_prefix: str) -> str:
    """Append injection at end (once), preserving caller-provided thought priming."""
    s = str(text or "")
    atip = str(injection_prefix or "")
    if not atip:
        return s
    atip_norm = atip if atip.endswith("\n") else atip + "\n"
    if s.endswith(atip_norm):
        return s
    return s + atip_norm


def call_gemma4_vllm_for_vertex_openai_simple(
    *,
    base_url: str,
    model: str,
    llm_args: dict[str, Any],
    api_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    runtime_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gp = gemma4_vllm_generation_params_from_llm_args(llm_args)
    seed = runtime_seed if runtime_seed is not None else resolve_runtime_seed(llm_args)
    atip = str(gp.get("prompt_injection") or gp.get("assistant_turn_injection_prefix") or "")
    use_loop_state = bool((llm_args or {}).get("vertex_use_gemma4_vllm_loop_state"))
    strip_thought_in_request = bool(
        (llm_args or {}).get("vertex_strip_thought_blocks_from_request", False)
    )

    if use_loop_state:
        state = llm_args.setdefault("_gemma4_vllm_loop_state", {})
        has_history = any(
            str((m or {}).get("role") or "") in {"assistant", "tool"}
            for m in (api_messages or [])
        )
        if not has_history:
            state.clear()
        request_text = str(state.get("request_text") or "")
        response_text = str(state.get("response_text") or "")
        processed_tool_messages = int(state.get("processed_tool_messages") or 0)
        if not request_text:
            request_text = _build_loop_seed_request_text(
                api_messages,
                tools,
                enable_thinking=bool(gp["enable_thinking"]),
                assistant_turn_injection_prefix=atip,
            )
            response_text = ""
            processed_tool_messages = 0
            state["request_text"] = request_text
            state["response_text"] = response_text
        tool_messages = [
            m for m in (api_messages or []) if str((m or {}).get("role") or "") == "tool"
        ]
        new_tool_messages = tool_messages[processed_tool_messages:]
        if new_tool_messages:
            merged_for_tools = [m for m in api_messages if str((m or {}).get("role") or "") != "tool"]
            merged_for_tools.extend(new_tool_messages)
            response_text, _, processed_tool_messages = _append_new_tool_responses_from_api_messages(
                response_text,
                merged_for_tools,
                assistant_turn_injection_prefix=atip,
            )
            state["response_text"] = response_text
            state["processed_tool_messages"] = processed_tool_messages

        prompt_base = request_text + response_text
        if strip_thought_in_request:
            # First strip historical thought blocks from the carried transcript...
            prompt = _strip_thought_blocks_for_request_text(prompt_base)
            # ...then re-append the configured injection so it is preserved.
            prompt = _append_injection_suffix(prompt, atip)
        else:
            prompt = prompt_base
        prompt = _sanitize_tool_response_markers_for_request_text(prompt)
    else:
        prompt = build_prompt_from_api_messages_simple(
            api_messages,
            tools,
            enable_thinking=bool(gp["enable_thinking"]),
            assistant_turn_injection_prefix=atip,
        )
    body = build_v1_completions_request(
        model=model,
        prompt=prompt,
        max_tokens=int(gp["max_tokens"]),
        temperature=float(gp["temperature"]),
        seed=seed,
    )
    payload = post_v1_completions_simple(base_url=base_url, body=body, timeout_s=int(gp["timeout"]))
    raw = extract_completion_text(payload)
    if use_loop_state:
        loop_state = llm_args.setdefault("_gemma4_vllm_loop_state", {})
        loop_state["response_text"] = str(loop_state.get("response_text") or "") + raw
    arr = parse_request_response_to_messages(prompt, raw)
    last = arr[-1] if arr else {"content": "", "thought": "", "tool_calls": []}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    raw_content = str(last.get("content") or "")
    msg: dict[str, Any] = {"role": "assistant", "content": _extract_text_part(raw_content)}
    if str(last.get("thought") or "").strip():
        msg["reasoning_content"] = str(last.get("thought") or "")
    tcs = last.get("tool_calls") if isinstance(last.get("tool_calls"), list) else []
    if tcs:
        msg["tool_calls"] = tcs
    full_payload: dict[str, Any] = {
        "_tau2_gemma4_vllm_client_simple": True,
        "_tau2_gemma4_vllm_request_url": f"{base_url.rstrip('/')}/v1/completions",
        "_tau2_gemma4_vllm_request_payload": body,
        "_tau2_gemma4_vllm_transcript": {"prompt": prompt, "completion": raw, "prompt_plus_completion": prompt + raw},
        "_tau2_gemma4_vllm_response_after_chat_template": {
            "reasoning_content": msg.get("reasoning_content", ""),
            "content": msg.get("content", ""),
            "tool_calls": tcs,
            "conversation_array": arr,
        },
        "object": "text_completion",
        "choices": [{"text": raw, "finish_reason": "stop", "index": 0}],
        "usage": usage,
        "model": model,
    }
    pred: dict[str, Any] = {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage}
    return full_payload, pred
