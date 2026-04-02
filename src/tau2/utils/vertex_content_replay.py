"""
Rebuild google.genai `types.Part` lists for history replay from `response.model_dump(mode="json")`.

When thinking is enabled, a model turn may include `Part(text=..., thought=True)` before
`Part(function_call=..., thought_signature=...)`. Rebuilding only from `content` + `tool_calls`
drops thought parts and breaks the next `generate_content` request.
"""

from __future__ import annotations

import base64
from typing import Any


def _decode_thought_signature(sig: Any) -> bytes | None:
    if sig is None:
        return None
    if isinstance(sig, bytes):
        return sig
    if isinstance(sig, str) and sig.strip():
        s = sig.strip()
        pad = "=" * (-len(s) % 4)
        for candidate in (s + pad, s):
            for decoder in (base64.b64decode, base64.urlsafe_b64decode):
                try:
                    return decoder(candidate)
                except Exception:
                    continue
        return None
    if isinstance(sig, dict) and sig.get("_encoding") == "base64":
        data = sig.get("_data")
        if isinstance(data, str):
            try:
                return base64.b64decode(data)
            except Exception:
                return None
    return None


def enrich_raw_candidates_parts_from_response(
    raw: dict[str, Any] | None,
    response: Any,
) -> dict[str, Any] | None:
    """
    Copy `thought_signature` from live SDK `Part` objects into `raw` JSON.

    `model_dump(mode="json")` can omit or mishandle opaque bytes; replay must match
    the live request. The global `_thought_signatures` map is unsafe across turns
    because `call_0`-style ids are reused each response.
    """
    if not isinstance(raw, dict):
        return raw
    try:
        cands = getattr(response, "candidates", None) or []
        if not cands:
            return raw
        live_parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
        rc = raw.get("candidates")
        if not isinstance(rc, list) or not rc:
            return raw
        parts_json = (rc[0].get("content") or {}).get("parts")
        if not isinstance(parts_json, list):
            return raw
        for i, live_part in enumerate(live_parts):
            if i >= len(parts_json):
                break
            pj = parts_json[i]
            if not isinstance(pj, dict):
                continue
            live_fn = getattr(live_part, "function_call", None)
            if isinstance(pj.get("function_call"), dict) and live_fn is not None:
                pj["function_call"]["id"] = getattr(live_fn, "id", None)
            sig = getattr(live_part, "thought_signature", None)
            if not isinstance(sig, bytes):
                continue
            enc = base64.b64encode(sig).decode("ascii")
            if pj.get("function_call") is not None:
                pj["thought_signature"] = enc
            elif pj.get("text") and pj.get("thought"):
                pj["thought_signature"] = enc
    except Exception:
        return raw
    return raw


def build_model_parts_from_raw_data(
    raw_data: dict[str, Any] | None, types: Any
) -> list[Any] | None:
    """
    Return a list of `types.Part` mirroring the API response order, or None if unavailable.

    Only used when the message was produced by this SDK path (has `candidates[].content.parts`).
    """
    if not isinstance(raw_data, dict):
        return None
    candidates = raw_data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content") or {}
    parts_data = content.get("parts")
    if not isinstance(parts_data, list) or not parts_data:
        return None

    parts: list[Any] = []
    for p in parts_data:
        if not isinstance(p, dict):
            continue
        fn = p.get("function_call")
        if isinstance(fn, dict):
            # Preserve API ids as stored (do not synthesize call_{idx}); None is valid.
            call_id = fn.get("id")
            args = fn.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            part_kwargs: dict[str, Any] = {
                "function_call": types.FunctionCall(
                    id=call_id,
                    name=fn.get("name") or "",
                    args=args,
                )
            }
            sig = _decode_thought_signature(p.get("thought_signature"))
            if sig is not None:
                part_kwargs["thought_signature"] = sig
            parts.append(types.Part(**part_kwargs))
            continue

        text = p.get("text")
        if isinstance(text, str) and text:
            part_kwargs = {"text": text}
            if bool(p.get("thought")):
                part_kwargs["thought"] = True
            sig = _decode_thought_signature(p.get("thought_signature"))
            if sig is not None:
                part_kwargs["thought_signature"] = sig
            parts.append(types.Part(**part_kwargs))

    return parts if parts else None


def should_replay_thought_parts(llm_args: dict[str, Any] | None) -> bool:
    """Default True: replay thought parts from raw_data when present."""
    if not isinstance(llm_args, dict):
        return True
    value = llm_args.get("include_thoughts_in_history", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def register_tool_names_from_parts(
    parts: list[Any], tool_name_by_id: dict[str, str]
) -> None:
    """Fill tool_name_by_id from FunctionCall parts for subsequent ToolMessage routing."""
    for part in parts:
        fn_call = getattr(part, "function_call", None)
        if fn_call is None:
            continue
        call_id = getattr(fn_call, "id", None)
        key = "" if call_id is None else str(call_id)
        name = getattr(fn_call, "name", None) or ""
        if name:
            tool_name_by_id[key] = str(name)
