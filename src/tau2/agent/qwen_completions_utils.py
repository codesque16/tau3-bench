"""
Qwen3 chat-template utilities for ``openai_completions_agent``.

Use with a Qwen jinja template (e.g. ``qwen35-27b-fp8.jinja``) and set
``agent_llm_args.completion_format: qwen`` (or ``chat_template_format: qwen``).

Handles:
  - Prompt rendering (no Gemma token escaping)
  - ``reasoning`` → ``reasoning_content`` for multi-turn history
  - Parsing ``<think>…</think>`` from completions output
  - Parsing Qwen XML tool calls (``<tool_call><function=…>``)
  - Default stop token ``<|im_end|>``
  - Length-truncation continuation wrappers
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

DEFAULT_QWEN_STOP_TOKENS: list[str] = ["<|im_end|>"]

# Opener is in the rendered prompt (``add_generation_prompt``); completions continue
# inside the block — same role as ``<|channel>thought\\n`` in Gemma injection.
_QWEN_THINKING_OPEN = "<think>\n"

_QWEN_THINKING_RE = re.compile(
    r"<think>\s*(.*?)\s*</think>",
    re.DOTALL,
)
_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def completion_format_from_llm_args(llm_args: dict[str, Any] | None) -> str:
    """Return ``gemma`` or ``qwen`` (default ``gemma``)."""
    args = llm_args or {}
    explicit = args.get("completion_format") or args.get("chat_template_format")
    if explicit is not None:
        return str(explicit).strip().lower()
    template = str(args.get("jinja_template_path") or "").lower()
    if "qwen" in template:
        return "qwen"
    return "gemma"


def is_qwen_completion_format(llm_args: dict[str, Any] | None) -> bool:
    return completion_format_from_llm_args(llm_args) == "qwen"


def _coerce_arguments_dict(arguments: Any) -> dict[str, Any]:
    """OpenAI chat export uses JSON strings; Qwen jinja ``|items`` needs a mapping."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _normalize_tool_calls_for_qwen(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_out = dict(tc)
        fn = tc_out.get("function")
        if isinstance(fn, dict):
            fn_out = dict(fn)
            fn_out["arguments"] = _coerce_arguments_dict(fn.get("arguments"))
            tc_out["function"] = fn_out
        else:
            if "name" in tc_out:
                tc_out["arguments"] = _coerce_arguments_dict(tc_out.get("arguments"))
        out.append(tc_out)
    return out


def normalize_messages_for_qwen_template(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Prepare OpenAI-style messages for the Qwen jinja template.

    - ``reasoning`` → ``reasoning_content``
    - ``tool_calls[].function.arguments`` JSON strings → dicts (for ``|items``)
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        reasoning = m.get("reasoning_content")
        if not (isinstance(reasoning, str) and reasoning.strip()):
            alt = m.get("reasoning")
            if isinstance(alt, str) and alt.strip():
                m["reasoning_content"] = alt
        if m.get("tool_calls"):
            m["tool_calls"] = _normalize_tool_calls_for_qwen(m["tool_calls"])
        out.append(m)
    return out


def _load_qwen_template(template_path: str) -> Any:
    try:
        from jinja2 import BaseLoader, Environment, Undefined
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for qwen_completions_utils. Install with: uv add jinja2"
        ) from e

    p = Path(template_path)
    if not p.exists():
        raise FileNotFoundError(
            f"jinja_template_path not found: {template_path!r}. "
            "Set jinja_template_path in agent_llm_args."
        )
    env = Environment(
        loader=BaseLoader(),
        keep_trailing_newline=True,
        undefined=Undefined,
    )
    return env.from_string(p.read_text())


def render_qwen_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    enable_thinking: bool,
    injection_prefix: str | None,
    template_path: str,
) -> str:
    """
    Render a Qwen chat-template prompt for ``/v1/completions``.

    When ``injection_prefix`` is set, it is appended after the rendered prompt.
    The template should end with an open ``<think>`` block so the model
    continues inside that block (same pattern as Gemma channel injection).
    """
    normalized = normalize_messages_for_qwen_template(messages)
    tmpl = _load_qwen_template(template_path)
    rendered = tmpl.render(
        messages=normalized,
        tools=tools or [],
        enable_thinking=enable_thinking,
        add_generation_prompt=True,
    )
    if injection_prefix:
        rendered = rendered + injection_prefix
    return rendered


def _qwen_parameter_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _qwen_parameters_to_dict(params_body: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for m in _QWEN_PARAMETER_RE.finditer(params_body):
        key = m.group(1).strip()
        if key:
            args[key] = _qwen_parameter_value(m.group(2))
    return args


def parse_qwen_completion(
    raw_output: str,
    injection_prefix: str | None = None,
) -> tuple[str | None, list[dict[str, Any]], str]:
    """
    Parse raw model output into (reasoning, tool_calls, content).

    Same flow as ``parse_completion`` (Gemma): merge injection + output, then
    regex-extract thinking and tool calls.

    When ``injection_prefix`` is active the model's output is a *continuation*
    of that prefix — the opening ``<think>\\n`` tag is in the
    rendered prompt, not the output.  We prepend the prefix (and the prompt
    opener when missing) before regex matching so the thinking block is complete.
    """
    prefix = injection_prefix or ""
    full_text = prefix + (raw_output or "")

    # Prompt ends with ``<think>\\n``; completion is body + close tag.
    # Prepend opener so merged text matches a full block (like Gemma's channel tag
    # living in injection_prefix or the template).
    if not full_text.lstrip().startswith("<think>"):
        if "</think>" in full_text or prefix:
            full_text = _QWEN_THINKING_OPEN + full_text

    reasoning: str | None = None
    m = _QWEN_THINKING_RE.search(full_text)
    if m:
        reasoning = m.group(1).strip()
        full_text = full_text[: m.start()] + full_text[m.end() :]

    tool_calls: list[dict[str, Any]] = []
    counter = [0]

    def _replace_tool(mc: re.Match) -> str:
        name = mc.group(1).strip()
        params_body = mc.group(2)
        args = _qwen_parameters_to_dict(params_body)
        counter[0] += 1
        tool_calls.append(
            {
                "id": f"call_{counter[0]:04d}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
        return ""

    full_text = _QWEN_TOOL_CALL_RE.sub(_replace_tool, full_text)
    content = (
        full_text.strip()
        .removesuffix("<|im_end|>")
        .strip()
    )
    return reasoning, tool_calls, content


def clean_qwen_completion_chunk_text(raw: str) -> str:
    """Strip empty thinking markers before concatenating continuation chunks."""
    raw = raw.replace("<think>\n\n</think>", "")
    raw = raw.replace("<think></think>", "")
    return raw


def wrap_truncated_qwen_thought(
    chunk: str,
    *,
    has_open_thinking: bool = False,
) -> str:
    """
    Close a length-truncated turn so the next completion can continue cleanly.

    When ``has_open_thinking`` is True, the prompt already ends inside an open
    ``<think>`` block (first length hit with ``injection_prefix``).
    """
    stripped = chunk.strip()
    if has_open_thinking:
        return (
            f"{stripped}\n"
            "Thought too long ... TRUNCATED\n"
            "</think>\n\n"
        )
    return (
        "<think>\n"
        f"{stripped}\n"
        "Thought too long ... TRUNCATED\n"
        "</think>\n\n"
    )


def resolve_qwen_stop_tokens(llm_args: dict[str, Any] | None) -> list[str]:
    args = llm_args or {}
    custom = args.get("stop_tokens")
    if custom:
        return list(custom)
    return list(DEFAULT_QWEN_STOP_TOKENS)


# Type alias for optional injection into agent without circular imports.
ParseCompletionFn = Callable[
    [str, str | None], tuple[str | None, list[dict[str, Any]], str]
]
