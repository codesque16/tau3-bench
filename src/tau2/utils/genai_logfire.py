from __future__ import annotations

from contextlib import nullcontext
import base64
import json
import os
import time
from typing import Any

from loguru import logger


def _genai_retryable(exc: BaseException) -> bool:
    """429 / transient overload — safe to retry with backoff."""
    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError):
            return int(getattr(exc, "code", 0) or 0) in (429, 500, 502, 503, 504)
    except Exception:
        pass
    return False


def _generate_content_with_backoff(
    client: Any,
    *,
    model: str,
    contents: list[Any],
    config: Any,
    actor: str,
    call_name: str,
) -> Any:
    """
    Truncated exponential backoff for rate limits / transient errors (default: 429, 5xx).
    Env: TAU2_GENAI_MAX_RETRIES (default 8), TAU2_GENAI_RETRY_BASE_S (default 1.0),
    TAU2_GENAI_RETRY_MAX_S (default 60.0).
    """
    max_retries = max(1, int(os.getenv("TAU2_GENAI_MAX_RETRIES", "8")))
    base_s = float(os.getenv("TAU2_GENAI_RETRY_BASE_S", "1.0"))
    cap_s = float(os.getenv("TAU2_GENAI_RETRY_MAX_S", "60.0"))
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_exc = e
            if attempt >= max_retries - 1 or not _genai_retryable(e):
                raise
            delay = min(cap_s, base_s * (2**attempt))
            logger.warning(
                "[{}] {} attempt {}/{} failed ({}); retrying in {:.1f}s",
                actor,
                call_name,
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _include_thoughts_in_rendering(config: Any) -> bool:
    # Per-call config wins; env var is a global override.
    cfg_value = getattr(config, "include_thoughts_in_history", None)
    if cfg_value is None and hasattr(config, "model_dump"):
        try:
            dumped = config.model_dump(mode="json")
            if isinstance(dumped, dict):
                cfg_value = dumped.get("include_thoughts_in_history")
        except Exception:
            cfg_value = None
    if cfg_value is None:
        cfg_value = os.getenv("TAU2_INCLUDE_THOUGHTS_IN_HISTORY", "1")
    if isinstance(cfg_value, str):
        return cfg_value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(cfg_value)


def _safe_logfire_attrs(config: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if config is None:
        return attrs
    for key in ("temperature", "max_output_tokens", "seed"):
        value = getattr(config, key, None)
        if value is not None:
            attrs[f"gen_ai.request.{key}"] = value
    return attrs


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        # Preserve opaque binary fields (e.g., thought_signature) losslessly.
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def _extract_text_output(response: Any) -> tuple[str, str]:
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "", ""
    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        text = getattr(part, "text", None)
        if not isinstance(text, str) or not text:
            continue
        is_thought = bool(getattr(part, "thought", False))
        if is_thought:
            reasoning_parts.append(text)
        else:
            text_parts.append(text)
    return "\n".join(reasoning_parts).strip(), "\n".join(text_parts).strip()


def _extract_output_text_blocks(response: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return blocks
    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        text = getattr(part, "text", None)
        if not isinstance(text, str) or not text:
            continue
        if bool(getattr(part, "thought", False)):
            continue
        item: dict[str, str] = {"text": text}
        thought_signature = getattr(part, "thought_signature", None)
        if isinstance(thought_signature, bytes):
            item["thought_signature"] = base64.b64encode(thought_signature).decode("ascii")
        elif isinstance(thought_signature, str):
            item["thought_signature"] = thought_signature
        blocks.append(item)
    return blocks


def _extract_reasoning_blocks(response: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return blocks
    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        text = getattr(part, "text", None)
        if not isinstance(text, str) or not text:
            continue
        if not bool(getattr(part, "thought", False)):
            continue
        item: dict[str, str] = {"text": text}
        thought_signature = getattr(part, "thought_signature", None)
        if isinstance(thought_signature, bytes):
            item["thought_signature"] = base64.b64encode(thought_signature).decode(
                "ascii"
            )
        elif isinstance(thought_signature, str):
            item["thought_signature"] = thought_signature
        blocks.append(item)
    return blocks


def _json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _extract_response_tool_calls(response: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return out
    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        fn_call = getattr(part, "function_call", None)
        if fn_call is None:
            continue
        args = getattr(fn_call, "args", None) or {}
        out.append(
            {
                "id": getattr(fn_call, "id", "") or "",
                "type": "function",
                "function": {
                    "name": getattr(fn_call, "name", "") or "",
                    "arguments": _json_dumps_safe(_to_jsonable(args)),
                },
            }
        )
    return out


def _contents_to_all_messages_events(contents: list[Any], config: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    include_thoughts = _include_thoughts_in_rendering(config)
    system_instruction = getattr(config, "system_instruction", None)
    if isinstance(system_instruction, str) and system_instruction.strip():
        events.append({"role": "system", "content": system_instruction})

    for content in contents:
        role = getattr(content, "role", None) or "user"
        parts = getattr(content, "parts", None) or []
        text_blocks: list[str] = []
        thinking_blocks: list[dict[str, str]] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                if include_thoughts and bool(getattr(part, "thought", False)):
                    thought_item: dict[str, str] = {"text": text}
                    thought_signature = getattr(part, "thought_signature", None)
                    if isinstance(thought_signature, (str, bytes)):
                        thought_item["thought_signature"] = (
                            thought_signature.decode("utf-8", errors="replace")
                            if isinstance(thought_signature, bytes)
                            else thought_signature
                        )
                    thinking_blocks.append(thought_item)
                else:
                    text_blocks.append(text)
            fn_call = getattr(part, "function_call", None)
            if fn_call is not None:
                args = getattr(fn_call, "args", {}) or {}
                tc: dict[str, Any] = {
                    "id": getattr(fn_call, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": getattr(fn_call, "name", "") or "",
                        "arguments": _json_dumps_safe(_to_jsonable(args)),
                    },
                }
                thought_signature = getattr(part, "thought_signature", None)
                if isinstance(thought_signature, (str, bytes)):
                    tc["thought_signature"] = (
                        thought_signature.decode("utf-8", errors="replace")
                        if isinstance(thought_signature, bytes)
                        else thought_signature
                    )
                tool_calls.append(tc)
        if tool_calls:
            msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            mixed_content: list[dict[str, str]] = []
            for thought in thinking_blocks:
                thought_event: dict[str, str] = {
                    "type": "thinking",
                    "text": thought["text"],
                }
                if thought.get("thought_signature"):
                    thought_event["thought_signature"] = thought["thought_signature"]
                mixed_content.append(thought_event)
            for text in text_blocks:
                mixed_content.append({"type": "text", "text": text})
            if mixed_content:
                msg["content"] = mixed_content
            events.append(msg)
            continue
        if role == "function":
            # Convert function response content into OpenAI-style tool messages.
            emitted = False
            for part in parts:
                fn_resp = getattr(part, "function_response", None)
                if fn_resp is not None:
                    fn_name = getattr(fn_resp, "name", "") or ""
                    fn_payload = _to_jsonable(getattr(fn_resp, "response", None))
                    tool_call_id = getattr(fn_resp, "id", "") or ""
                    events.append(
                        {
                            "role": "tool",
                            # Keep LangChain-style key for Logfire all_messages_events rendering.
                            "id": tool_call_id,
                            "name": fn_name,
                            "content": _json_dumps_safe(fn_payload),
                        }
                    )
                    emitted = True
            if not emitted:
                events.append({"role": "tool", "name": "", "content": ""})
            continue

        out_role = "assistant" if role == "model" else "user"
        if out_role == "assistant" and include_thoughts and thinking_blocks:
            mixed_content = []
            for thought in thinking_blocks:
                thought_event: dict[str, str] = {
                    "type": "thinking",
                    "text": thought["text"],
                }
                if thought.get("thought_signature"):
                    thought_event["thought_signature"] = thought["thought_signature"]
                mixed_content.append(thought_event)
            for text in text_blocks:
                mixed_content.append({"type": "text", "text": text})
            events.append({"role": out_role, "content": mixed_content})
        else:
            events.append({"role": out_role, "content": "\n".join(text_blocks)})
    return events


def _response_to_all_messages_event(
    *,
    include_thoughts: bool,
    reasoning: str,
    reasoning_blocks: list[dict[str, str]],
    output_text_blocks: list[dict[str, str]],
    output_text: str,
    response_tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    mixed_content: list[dict[str, str]] = []
    if include_thoughts and reasoning_blocks:
        for block in reasoning_blocks:
            thought_event: dict[str, str] = {
                "type": "thinking",
                "text": block.get("text", ""),
            }
            if block.get("thought_signature"):
                thought_event["thought_signature"] = block["thought_signature"]
            mixed_content.append(thought_event)
    elif include_thoughts and reasoning:
        mixed_content.append({"type": "thinking", "text": reasoning})
    if output_text_blocks:
        for block in output_text_blocks:
            text_event: dict[str, str] = {"type": "text", "text": block.get("text", "")}
            if block.get("thought_signature"):
                text_event["thought_signature"] = block["thought_signature"]
            mixed_content.append(text_event)
    elif output_text:
        mixed_content.append({"type": "text", "text": output_text})

    if response_tool_calls:
        msg: dict[str, Any] = {"role": "assistant", "tool_calls": response_tool_calls}
        if mixed_content:
            msg["content"] = mixed_content
        return msg
    if mixed_content:
        return {"role": "assistant", "content": mixed_content}
    return {"role": "assistant", "content": output_text}


def _flatten_llm_messages_attrs(
    *,
    all_messages_events: list[dict[str, Any]],
    response_data: dict[str, Any],
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for i, msg in enumerate(all_messages_events):
        prefix = f"llm.input_messages.{i}.message"
        role = msg.get("role")
        if role is not None:
            attrs[f"{prefix}.role"] = role
        if "name" in msg and msg.get("name") is not None:
            attrs[f"{prefix}.name"] = msg.get("name")
        if "content" in msg:
            attrs[f"{prefix}.content"] = msg.get("content")
            content_value = msg.get("content")
            if isinstance(content_value, list):
                thinking_texts: list[str] = []
                for k, part in enumerate(content_value):
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    part_text = part.get("text")
                    if part_type is not None:
                        attrs[f"{prefix}.contents.{k}.message_content.type"] = part_type
                    if part_text is not None:
                        attrs[f"{prefix}.contents.{k}.message_content.text"] = part_text
                        if part_type == "thinking" and isinstance(part_text, str):
                            thinking_texts.append(part_text)
                    part_sig = part.get("thought_signature")
                    if part_sig is not None:
                        attrs[f"{prefix}.contents.{k}.message_content.thought_signature"] = (
                            part_sig
                        )
                if thinking_texts:
                    attrs[f"{prefix}.reasoning"] = "\n\n".join(
                        t for t in thinking_texts if t
                    )
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for j, tc in enumerate(tool_calls):
                tc_prefix = f"{prefix}.tool_calls.{j}.tool_call"
                tc_id = tc.get("id")
                if tc_id is not None:
                    attrs[f"{tc_prefix}.id"] = tc_id
                fn = tc.get("function") or {}
                if fn.get("name") is not None:
                    attrs[f"{tc_prefix}.function.name"] = fn.get("name")
                if fn.get("arguments") is not None:
                    attrs[f"{tc_prefix}.function.arguments"] = fn.get("arguments")

    out_msg = (response_data.get("message") or {}) if isinstance(response_data, dict) else {}
    out_prefix = "llm.output_messages.0.message"
    if out_msg.get("role") is not None:
        attrs[f"{out_prefix}.role"] = out_msg.get("role")
    if out_msg.get("content") is not None:
        attrs[f"{out_prefix}.content"] = out_msg.get("content")
    if out_msg.get("reasoning_content"):
        attrs[f"{out_prefix}.reasoning"] = out_msg.get("reasoning_content")
    out_tool_calls = out_msg.get("tool_calls")
    if isinstance(out_tool_calls, list):
        for j, tc in enumerate(out_tool_calls):
            tc_prefix = f"{out_prefix}.tool_calls.{j}.tool_call"
            tc_id = tc.get("id")
            if tc_id is not None:
                attrs[f"{tc_prefix}.id"] = tc_id
            fn = tc.get("function") or {}
            if fn.get("name") is not None:
                attrs[f"{tc_prefix}.function.name"] = fn.get("name")
            if fn.get("arguments") is not None:
                attrs[f"{tc_prefix}.function.arguments"] = fn.get("arguments")
    return attrs


def _infer_tool_round(contents: list[Any]) -> int:
    """
    Infer tool round from request history.

    We treat each function-response turn in the outgoing contents as one completed
    tool round. This matches the reference logging shape where tool_round increases
    as the conversation alternates through tool calls.
    """
    rounds = 0
    for content in contents:
        if (getattr(content, "role", None) or "") != "function":
            continue
        parts = getattr(content, "parts", None) or []
        if any(getattr(part, "function_response", None) is not None for part in parts):
            rounds += 1
    return rounds


def genai_generate_with_logfire(
    *,
    client: Any,
    model: str,
    contents: list[Any],
    config: Any,
    actor: str,
    call_name: str,
) -> Any:
    """
    Call google.genai generate_content with optional Logfire tracing.

    If logfire is not installed/configured, this function behaves like a direct SDK call.
    """
    try:
        import logfire  # type: ignore

        actor_for_span = "assistant" if actor == "agent" else actor
        span_cm = logfire.span(
            f"google.genai generate_content [{actor_for_span}]",
            _span_name=f"google.genai.generate_content [{actor_for_span}]",
            _tags=["LLM"],
            actor=actor,
            call_name=call_name,
            model=model,
            llm_system="google",
            llm_model_name=model,
            gen_ai_operation_name="generate_content",
            gen_ai_request_model=model,
            gen_ai_response_model=model,
            gen_ai_system="google",
            **_safe_logfire_attrs(config),
        )
    except Exception:
        logfire = None
        span_cm = nullcontext()

    logger.info(
        f"[{actor}] {call_name}: model={model} contents={len(contents)}"
    )

    request_data = {
        "model": model,
        "contents": _to_jsonable(contents),
        "config": _to_jsonable(config),
    }
    tool_round = _infer_tool_round(contents)

    with span_cm as span:
        try:
            response = _generate_content_with_backoff(
                client,
                model=model,
                contents=contents,
                config=config,
                actor=actor,
                call_name=call_name,
            )
        except Exception as e:
            logger.error(f"[{actor}] {call_name} failed: {e}")
            raise

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if usage is not None:
            prompt_tokens = (
                getattr(usage, "prompt_token_count", None)
                or getattr(usage, "input_token_count", None)
                or 0
            )
            completion_tokens = (
                getattr(usage, "candidates_token_count", None)
                or getattr(usage, "output_token_count", None)
                or 0
            )
            total_tokens = (
                getattr(usage, "total_token_count", None)
                or int(prompt_tokens) + int(completion_tokens)
            )
            logger.debug(
                f"{call_name} usage: prompt_tokens={int(prompt_tokens)} completion_tokens={int(completion_tokens)}"
            )

        reasoning, output_text = _extract_text_output(response)
        include_thoughts = _include_thoughts_in_rendering(config)
        reasoning_blocks = _extract_reasoning_blocks(response)
        output_text_blocks = _extract_output_text_blocks(response)
        response_tool_calls = _extract_response_tool_calls(response)
        response_data = {
            "message": {
                "role": "assistant",
                "content": (None if response_tool_calls and not output_text else output_text),
                "reasoning_content": (reasoning if include_thoughts and reasoning else None),
                "tool_calls": response_tool_calls or None,
            }
        }
        gemini_io_json = {
            "tool_round": tool_round,
            "request": request_data,
            "response": _to_jsonable(response),
        }
        input_messages_events = _contents_to_all_messages_events(contents, config)
        all_messages_events = input_messages_events + [
            _response_to_all_messages_event(
                include_thoughts=include_thoughts,
                reasoning=reasoning,
                reasoning_blocks=reasoning_blocks,
                output_text_blocks=output_text_blocks,
                output_text=output_text,
                response_tool_calls=response_tool_calls,
            )
        ]
        request_data_for_model_run = {
            "model": model,
            "messages": input_messages_events,
        }

        if logfire is not None:
            if span is not None and hasattr(span, "set_attribute"):
                # Keep both underscore and dotted variants for compatibility with UI renderers.
                span.set_attribute("llm.model_name", model)
                span.set_attribute("llm.system", "google")
                span.set_attribute("gen_ai.operation.name", "generate_content")
                span.set_attribute("gen_ai.request.model", model)
                span.set_attribute("gen_ai.response.model", model)
                span.set_attribute("gen_ai.system", "google")
                system_instruction = getattr(config, "system_instruction", None)
                if isinstance(system_instruction, str) and system_instruction.strip():
                    span.set_attribute(
                        "gen_ai.system_instructions",
                        [{"content": system_instruction, "type": "text"}],
                    )
                span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))
                span.set_attribute("gen_ai.usage.output_tokens", int(completion_tokens))
                span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))
                span.set_attribute("llm.token_count.prompt", int(prompt_tokens))
                span.set_attribute("llm.token_count.completion", int(completion_tokens))
                span.set_attribute("llm.token_count.total", int(total_tokens))
                span.set_attribute("all_messages_events", all_messages_events)
                span.set_attribute("request_data", request_data_for_model_run)
                span.set_attribute("response_data", response_data)
                span.set_attribute("input.mime_type", "application/json")
                span.set_attribute("input.value", {"messages": all_messages_events})
                span.set_attribute("output.mime_type", "application/json")
                span.set_attribute("output.value", response_data)
                span.set_attribute("gemini_io_json", gemini_io_json)
                span.set_attribute("tool_round", tool_round)
                flat_attrs = _flatten_llm_messages_attrs(
                    all_messages_events=input_messages_events,
                    response_data=response_data,
                )
                for key, value in flat_attrs.items():
                    span.set_attribute(key, value)
            logfire.info(
                "gemini.generate_content.raw_io",
                model=model,
                tool_round=tool_round,
                gemini_io_json=gemini_io_json,
            )

    # After the span closes: disk I/O must not run inside ``with span_cm`` or Logfire
    # shows the LLM span as "ongoing" for the full JSON-serialize + write duration.
    if actor in ("agent", "user"):
        try:
            from tau2.utils.sim_llm_io import write_sim_llm_io_json

            write_sim_llm_io_json(
                actor,
                call_name=call_name,
                payload={
                    "format": "google_genai",
                    "model": model,
                    "tool_round": tool_round,
                    "gemini_io_json": gemini_io_json,
                },
            )
        except Exception:
            pass

    logger.info(f"[{actor}] {call_name}: success")
    return response
