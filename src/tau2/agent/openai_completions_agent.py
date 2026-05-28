"""
OpenAI raw-completions agent for tau3 benchmark.

Uses /v1/completions (not /v1/chat/completions) with client-side Jinja2
rendering of the Gemma4 chat template.  Identical multi-step tool chain to
openai_agent but completely bypasses vLLM's chat-template machinery on the
server — the full prompt string is built here and sent as a raw text prompt.

Advantages
----------
- injection_prefix is appended directly to the rendered prompt string:
  no server-side chat_template_kwargs needed.
- Raw input prompt and raw model output are available immediately in
  logfire — no /render round-trip.
- Easier to debug: what you render is what the model sees.

llm_args keys
-------------
  openai_base_url        : str   — server root, e.g. "http://34.29.173.120:8000"
  openai_api_key         : str   — not used for /v1/completions (auth via URL)
  openai_model           : str   — model name
  temperature            : float — default 0.0
  top_p                  : float — optional, forwarded to /v1/completions
  top_k                  : int   — optional, vLLM-style sampling cap
                          (on vLLM, if temperature≈0, greedy mode resets top_p/top_k
                          in SamplingParams — see vLLM ``sampling_params.py``)
  max_tokens             : int   — default 2048
  logprobs               : bool|int — if true, sends integer logprobs (see top_logprobs)
  top_logprobs           : int   — when logprobs is true, sets logprobs N (default 5)
  seed                   : int   — -1 = fresh random seed
  skip_special_tokens    : bool  — forwarded to /v1/completions (default False);
                                   also read from chat_template_kwargs if unset
  stop_tokens            : list  — default ["<turn|>"]
  enable_thinking        : bool  — default True (adds <|think|> to system block)
  injection_prefix       : str   — appended after <|turn>model\\n on the FIRST
                                   call per user turn (step 0 only); None = off
  jinja_template_path    : str   — path to .jinja file (required)
  completion_format      : str   — ``gemma`` (default) or ``qwen``; use ``qwen`` with
                                   Qwen jinja templates (see qwen_completions_utils)
  chat_template_format   : str   — alias for completion_format
  continue_on_length     : bool  — if true, when finish_reason is ``length`` the
                                   partial output is stripped of Gemma special
                                   tokens, wrapped in a closed
                                   ``<|channel>thought … Thought too long ...
                                   TRUNCATED … <channel|>`` block, appended to
                                   the prompt, and one more completion call is
                                   made before falling back to normal flow
                                   (default False)
  max_length_continuations : int — max extra completion calls when continuing
                                   (default 8). Set to 1 to do exactly one
                                   wrapped retry per length hit.
  openai_connect_timeout_seconds : float — connect timeout for /v1/completions
                                   (default 300)
  openai_read_timeout_seconds    : float — read timeout for /v1/completions
                                   (default 300)
  openai_request_max_retries     : int   — retries for transient request failures
                                   (default 2; total attempts = retries + 1)
  openai_request_retry_backoff_seconds : float — base backoff between retries
                                   (default 2.0; linear backoff by attempt idx)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests
from loguru import logger

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.environment.tool import Tool
from tau2.agent.qwen_completions_utils import (
    clean_qwen_completion_chunk_text,
    is_qwen_completion_format,
    normalize_messages_for_qwen_template,
    parse_qwen_completion,
    render_qwen_prompt,
    resolve_qwen_stop_tokens,
    wrap_truncated_qwen_thought,
)
from tau2.utils.vertex_endpoint_chat import (
    prediction_usage_with_cost,
    resolve_runtime_seed,
    tau_messages_to_openai_chat,
)


# ---------------------------------------------------------------------------
# Gemma special-token ↔ Jinja2-safe placeholder map
# ---------------------------------------------------------------------------

_PLACEHOLDER_MAP: dict[str, str] = {
    "<|turn>":          "GEMMATOK_TURN_OPEN",
    "<turn|>":          "GEMMATOK_TURN_CLOSE",
    "<|channel>":       "GEMMATOK_CHAN_OPEN",
    "<channel|>":       "GEMMATOK_CHAN_CLOSE",
    "<|tool>":          "GEMMATOK_TOOL_OPEN",
    "<tool|>":          "GEMMATOK_TOOL_CLOSE",
    "<|tool_call>":     "GEMMATOK_TCALL_OPEN",
    "<tool_call|>":     "GEMMATOK_TCALL_CLOSE",
    "<|tool_response>": "GEMMATOK_TRESP_OPEN",
    "<tool_response|>": "GEMMATOK_TRESP_CLOSE",
    "<|think|>":        "GEMMATOK_THINK",
    "<|image|>":        "GEMMATOK_IMAGE",
    "<|audio|>":        "GEMMATOK_AUDIO",
    "<|video|>":        "GEMMATOK_VIDEO",
}
_quote_tok = "<|" + '"' + "|>"
_PLACEHOLDER_MAP[_quote_tok] = "GEMMATOK_QUOTE"
_REVERSE_MAP = {v: k for k, v in _PLACEHOLDER_MAP.items()}


def _escape_template(src: str) -> str:
    for tok, ph in _PLACEHOLDER_MAP.items():
        src = src.replace(tok, ph)
    return src


def _unescape_output(text: str) -> str:
    for ph, tok in _REVERSE_MAP.items():
        text = text.replace(ph, tok)
    return text


# ---------------------------------------------------------------------------
# Client-side Jinja2 rendering
# ---------------------------------------------------------------------------

def _load_template(template_path: str) -> Any:
    """Load the Gemma4 Jinja template from the given path."""
    try:
        from jinja2 import BaseLoader, Environment, Undefined
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for openai_completions_agent. "
            "Install with: uv add jinja2"
        ) from e

    p = Path(template_path)
    if not p.exists():
        raise FileNotFoundError(
            f"jinja_template_path not found: {template_path!r}. "
            "Set jinja_template_path in agent_llm_args to a valid .jinja file."
        )
    raw = p.read_text()

    env = Environment(
        loader=BaseLoader(),
        keep_trailing_newline=True,
        undefined=Undefined,
    )
    env.globals["bos_token"] = "<bos>"
    return env.from_string(_escape_template(raw))


def render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    enable_thinking: bool,
    injection_prefix: str | None,
    template_path: str,
) -> str:
    """Render the full prompt string client-side."""
    tmpl = _load_template(template_path)
    rendered = tmpl.render(
        messages=messages,
        tools=tools or [],
        enable_thinking=enable_thinking,
        add_generation_prompt=True,
        injection_prefix=injection_prefix or "",
        inject_thinking=False,
    )
    result = _unescape_output(rendered)
    if injection_prefix:
        result = result + injection_prefix
    return result


# ---------------------------------------------------------------------------
# Completion text parser
# ---------------------------------------------------------------------------

_THINKING_RE  = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL)


def _gemma_args_to_dict(args_str: str) -> dict:
    json_str = args_str.replace('<|"|>', '"').strip()
    if not json_str.startswith("{"):
        json_str = "{" + json_str + "}"
    json_str = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"_raw": args_str}


def parse_completion(
    raw_output: str,
    injection_prefix: str | None = None,
) -> tuple[str | None, list[dict[str, Any]], str]:
    """
    Parse raw model output into (reasoning, tool_calls, content).

    When injection_prefix is active the model's output is a *continuation*
    of that prefix — the opening <|channel>thought\\n tag is in the prefix,
    not the output.  We prepend the prefix before regex matching so the
    thinking block is complete, then strip it back out of the result.
    """
    full_text = (injection_prefix or "") + raw_output

    # <eos> after a tool_call close means the model ended the sequence expecting
    # a tool response. Rewrite to <|tool_response> so the existing
    # removesuffix("<|tool_response>") cleanup leaves clean content.
    full_text = full_text.replace("<tool_call|><eos>", "<tool_call|><|tool_response>")

    reasoning: str | None = None
    m = _THINKING_RE.search(full_text)
    if m:
        reasoning = m.group(1).strip()
        full_text = full_text[:m.start()] + full_text[m.end():]

    tool_calls: list[dict[str, Any]] = []
    counter = [0]

    def _replace(mc: re.Match) -> str:
        name = mc.group(1)
        args = _gemma_args_to_dict(mc.group(2))
        counter[0] += 1
        tool_calls.append({
            "id":       f"call_{counter[0]:04d}",
            "type":     "function",
            "function": {"name": name, "arguments": args},
        })
        return ""

    full_text = _TOOL_CALL_RE.sub(_replace, full_text)
    content = (
        full_text.strip()
        .removesuffix("<turn|>")
        .removesuffix("<|tool_response>")
        .strip()
    )
    return reasoning, tool_calls, content


def _clean_completion_chunk_text(raw: str) -> str:
    """Strip known empty thought markers before concatenating continuation chunks."""
    raw = raw.replace("<|channel>thought<channel|>", "")
    raw = raw.replace("<|channel>thought\n<channel|>", "")
    return raw


# Composite Gemma tokens: a channel/turn opener immediately followed by its
# role/channel label. If we only strip the bare ``<|channel>`` opener the
# ``thought`` label is left behind as plain content; same for ``<|turn>model``
# etc. List them explicitly so the longest-first sort guarantees they are
# stripped before the plain ``<|channel>`` / ``<|turn>`` openers.
_GEMMA_COMPOSITE_TOKENS: tuple[str, ...] = (
    "<|channel>thought",
    "<|turn>model",
    "<|turn>user",
    "<|turn>system",
    "<|turn>assistant",
)


# Every Gemma special token we know about: the keys of the placeholder map
# plus the composite ``opener+label`` forms above. Sorted longest-first so
# that e.g. ``<|channel>thought`` is stripped before ``<|channel>`` and
# ``<|tool_call>`` before ``<|tool>``.
_GEMMA_SPECIAL_TOKENS: tuple[str, ...] = tuple(
    sorted(
        set(_PLACEHOLDER_MAP.keys()) | set(_GEMMA_COMPOSITE_TOKENS),
        key=len,
        reverse=True,
    )
)


def _strip_gemma_special_tokens(text: str) -> str:
    """Remove every known Gemma special token (``<|...>`` / ``<...|>``) and
    its composite ``opener+label`` variants (e.g. ``<|channel>thought``,
    ``<|turn>model``) from ``text``."""
    for tok in _GEMMA_SPECIAL_TOKENS:
        if tok in text:
            text = text.replace(tok, "")
    return text


def _wrap_truncated_thought(chunk: str, has_open_thought: bool = False) -> str:
    """
    Build a closed thought block from a length-truncated completion chunk.

    Gemma cut off mid-stream (``finish_reason == "length"``), so the chunk may
    contain half-emitted special tokens that would confuse the parser or the
    next call. We strip every known special token, then either:

    - ``has_open_thought=True``: the prompt already ends with an open
      ``<|channel>thought`` block (e.g. because ``injection_prefix`` seeded
      one for this turn and this is the first length truncation). Only close
      it — don't open another, or we'd nest openers.
    - ``has_open_thought=False``: prompt has no open thought block, so we
      emit a full ``<|channel>thought … <channel|>`` block.

    Either way the closing block is tagged ``Thought too long ... TRUNCATED``
    so the model can see its previous attempt was cut off and produce a fresh
    final output on the next call.
    """
    stripped = _strip_gemma_special_tokens(chunk).strip()
    if has_open_thought:
        return (
            f"{stripped}\n"
            "Thought too long ... TRUNCATED\n"
            "<channel|>"
        )
    return (
        "<|channel>thought\n"
        f"{stripped}\n"
        "Thought too long ... TRUNCATED\n"
        "<channel|>"
    )


def _merge_openai_completion_predictions(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Combine multiple /v1/completions JSON responses into one OpenAI-shaped dict:
    concatenated choice text, summed usage, last choice metadata otherwise.
    """
    if not preds:
        return {}
    if len(preds) == 1:
        return preds[0]
    texts: list[str] = []
    prompt_sum = completion_sum = total_sum = 0
    last = preds[-1]
    for p in preds:
        ch0 = (p.get("choices") or [{}])[0]
        texts.append(ch0.get("text") or "")
        u = p.get("usage") or {}
        prompt_sum += int(u.get("prompt_tokens") or 0)
        completion_sum += int(u.get("completion_tokens") or 0)
        total_sum += int(u.get("total_tokens") or 0)
    lc = (last.get("choices") or [{}])[0]
    merged_choice = {
        **lc,
        "text": "".join(texts),
        "finish_reason": lc.get("finish_reason"),
    }
    merged_usage = {**(last.get("usage") or {})}
    merged_usage["prompt_tokens"] = prompt_sum
    merged_usage["completion_tokens"] = completion_sum
    merged_usage["total_tokens"] = total_sum or (prompt_sum + completion_sum)
    return {**last, "choices": [merged_choice], "usage": merged_usage}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OpenAICompletionsAgent(LLMAgent):
    """
    Tau3 agent using /v1/completions with client-side Jinja2 chat-template
    rendering.  Supports Gemma4 on vLLM with or without injection prefix.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    def _get_base_url(self) -> str:
        raw = str((self.llm_args or {}).get("openai_base_url") or "").strip()
        if not raw:
            raise ValueError(
                "openai_base_url must be set in llm_args for openai_completions_agent."
            )
        return raw.rstrip("/")

    def _get_model(self) -> str:
        model = str(
            (self.llm_args or {}).get("openai_model")
            or (self.llm or "").strip()
        ).strip()
        if not model:
            raise ValueError(
                "Set openai_model in llm_args (or agent_llm) for openai_completions_agent."
            )
        return model

    def _resolve_pricing(self) -> dict[str, float] | None:
        pricing = (self.llm_args or {}).get("pricing")
        if not isinstance(pricing, dict):
            return None
        try:
            input_cost = float(pricing["input_cost_per_million"])
            output_cost = float(pricing["output_cost_per_million"])
            cached_input_cost = float(
                pricing.get("cached_input_cost_per_million", input_cost)
            )
        except Exception:
            logger.warning("Invalid pricing config for openai_completions_agent: {}", pricing)
            return None
        return {
            "input_cost_per_million": input_cost,
            "cached_input_cost_per_million": cached_input_cost,
            "output_cost_per_million": output_cost,
        }

    def _generate_next_message(self, message, state) -> AssistantMessage:
        if message.role == "user" and getattr(message, "is_audio", False):
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")

        if message.role == "tool" and hasattr(message, "tool_messages"):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        llm_args = self.llm_args or {}
        use_qwen = is_qwen_completion_format(llm_args)
        model = self._get_model()
        base_url = self._get_base_url()
        start_time = time.perf_counter()
        runtime_seed = resolve_runtime_seed(llm_args)

        chat_template_kwargs: dict = llm_args.get("chat_template_kwargs") or {}
        temperature = float(llm_args.get("temperature", 0.0) or 0.0)
        max_tokens = int(llm_args.get("max_tokens") or 2048)
        if llm_args.get("stop_tokens"):
            stop_tokens = list(llm_args["stop_tokens"])
        elif use_qwen:
            stop_tokens = resolve_qwen_stop_tokens(llm_args)
        else:
            stop_tokens = ["<turn|>"]
        enable_thinking = bool(
            llm_args.get("enable_thinking", chat_template_kwargs.get("enable_thinking", True))
        )
        skip_special_tokens = bool(
            llm_args.get(
                "skip_special_tokens",
                chat_template_kwargs.get("skip_special_tokens", False),
            )
        )
        injection_prefix: str | None = (
            llm_args.get("injection_prefix")
            or chat_template_kwargs.get("injection_prefix")
            or None
        )
        template_path: str = llm_args.get("jinja_template_path") or ""
        if not template_path:
            raise ValueError(
                "jinja_template_path must be set in agent_llm_args for openai_completions_agent."
            )

        openai_tools = [tool.openai_schema for tool in self.tools] if self.tools else []

        api_messages = tau_messages_to_openai_chat(
            self.system_prompt,
            state.system_messages + state.messages,
            vertex_include_reasoning_in_request=True,
        )
        if use_qwen:
            api_messages = normalize_messages_for_qwen_template(api_messages)

        # Apply injection prefix only on first call after a user message
        # (not after tool results — template ends with <tool_response|> there)
        last_role = api_messages[-1]["role"] if api_messages else "user"
        active_prefix = injection_prefix if last_role == "user" else None

        if use_qwen:
            prompt_str = render_qwen_prompt(
                messages=api_messages,
                tools=openai_tools or None,
                enable_thinking=enable_thinking,
                injection_prefix=active_prefix,
                template_path=template_path,
            )
        else:
            prompt_str = render_prompt(
                messages=api_messages,
                tools=openai_tools or None,
                enable_thinking=enable_thinking,
                injection_prefix=active_prefix,
                template_path=template_path,
            )

        payload: dict[str, Any] = {
            "model":       model,
            "prompt":      prompt_str,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stop":        stop_tokens,
            "skip_special_tokens": skip_special_tokens,
            "include_stop_str_in_output": True,
        }
        payload["_connect_timeout_seconds"] = float(
            llm_args.get("openai_connect_timeout_seconds", 300.0)
        )
        payload["_read_timeout_seconds"] = float(
            llm_args.get("openai_read_timeout_seconds", 300.0)
        )
        payload["_request_max_retries"] = int(
            llm_args.get("openai_request_max_retries", 2)
        )
        payload["_request_retry_backoff_seconds"] = float(
            llm_args.get("openai_request_retry_backoff_seconds", 2.0)
        )
        if runtime_seed is not None:
            payload["seed"] = int(runtime_seed)

        top_p = llm_args.get("top_p")
        if top_p is not None:
            payload["top_p"] = float(top_p)
        top_k = llm_args.get("top_k")
        if top_k is not None:
            payload["top_k"] = int(top_k)

        # /v1/completions expects ``logprobs`` as a positive int (top-N per position).
        # Mirror chat-style YAML: logprobs: true + top_logprobs: N.
        lp = llm_args.get("logprobs")
        top_lp = llm_args.get("top_logprobs")
        if lp is True or (isinstance(lp, str) and str(lp).lower() in ("true", "1", "yes")):
            n = int(top_lp) if top_lp is not None else 5
            if n > 0:
                payload["logprobs"] = n
        elif isinstance(lp, (int, float)) and int(lp) > 0:
            payload["logprobs"] = int(lp)
        elif top_lp is not None and int(top_lp) > 0:
            payload["logprobs"] = int(top_lp)

        # vLLM: temperature below ~1e-5 triggers greedy sampling and overwrites
        # top_p → 1.0 and top_k → 0 inside SamplingParams (server logs show that).
        _greedy_eps = 1e-5
        if temperature < _greedy_eps and (
            llm_args.get("top_p") is not None or llm_args.get("top_k") is not None
        ):
            logger.debug(
                "openai_completions_agent: vLLM greedy (temperature≈0) resets top_p/top_k "
                "in SamplingParams; request body still has top_p={} top_k={}",
                payload.get("top_p"),
                payload.get("top_k"),
            )

        logger.debug(
            "[openai_completions_agent] calling model={} temperature={} seed={} "
            "prompt_len={} active_prefix={} payload_top_p={} payload_top_k={}",
            model, temperature, runtime_seed, len(prompt_str),
            repr(active_prefix[:40]) if active_prefix else None,
            payload.get("top_p"),
            payload.get("top_k"),
        )

        continue_on_length = bool(llm_args.get("continue_on_length", False))
        _mlc = llm_args.get("max_length_continuations")
        max_length_continuations = max(0, int(8 if _mlc is None else _mlc))

        pred_list: list[dict[str, Any]] = []
        chunk_texts: list[str] = []
        current_prompt = prompt_str
        continuations_done = 0

        while True:
            payload["prompt"] = current_prompt
            pred = _completions_generate_with_logfire(
                base_url=base_url,
                model=model,
                payload=payload,
                api_messages=api_messages,
                openai_tools=openai_tools or None,
                raw_input_prompt=current_prompt,
                active_prefix=active_prefix if continuations_done == 0 else None,
                parse_completion_fn=parse_qwen_completion if use_qwen else parse_completion,
            )
            pred_list.append(pred)
            choices = pred.get("choices") or []
            chunk = (choices[0].get("text") or "") if choices else ""
            finish_reason = (choices[0].get("finish_reason") or "").lower() if choices else ""

            if finish_reason == "length":
                # Visible telemetry: a length truncation happened on this call,
                # regardless of whether we're going to retry or not.
                try:
                    import logfire  # type: ignore

                    logfire.info(
                        "openai.completions.length_truncation",
                        model=model,
                        continue_on_length=continue_on_length,
                        continuations_done=continuations_done,
                        max_length_continuations=max_length_continuations,
                        will_retry=bool(
                            continue_on_length
                            and continuations_done < max_length_continuations
                        ),
                        chunk_len=len(chunk),
                        prompt_len=len(current_prompt),
                    )
                except Exception:
                    pass

            if (
                not continue_on_length
                or finish_reason != "length"
                or continuations_done >= max_length_continuations
            ):
                chunk_texts.append(chunk)
                break

            # Length-truncated turn: strip Gemma special tokens from the partial
            # output and either close the currently-open ``<|channel>thought``
            # block (if injection_prefix opened one this turn and this is the
            # first truncation) or wrap the chunk in a fresh
            # ``<|channel>thought … <channel|>`` block. Tag the close with
            # ``Thought too long ... TRUNCATED`` so the next call sees the
            # previous attempt was cut off and can produce a fresh final output.
            #
            # ``active_prefix`` on the first call ends the rendered prompt with
            # an open ``<|channel>thought\n{injection}…`` block, so on the very
            # first length hit we only emit the closing side. Subsequent
            # retries (``continuations_done > 0``) follow a ``<channel|>`` we
            # already emitted, so a full fresh open+close wrap is correct.
            has_open_thought = bool(active_prefix) and continuations_done == 0
            if use_qwen:
                wrapped = wrap_truncated_qwen_thought(
                    chunk, has_open_thought=has_open_thought
                )
            else:
                wrapped = _wrap_truncated_thought(
                    chunk, has_open_thought=has_open_thought
                )
            chunk_texts.append(wrapped)
            current_prompt = current_prompt + wrapped
            continuations_done += 1
            logger.info(
                "openai_completions_agent: length-truncated turn, wrapped as "
                "TRUNCATED thought and retrying (segment {} / {}, prompt_len={})",
                continuations_done,
                max_length_continuations,
                len(current_prompt),
            )
            try:
                import logfire  # type: ignore

                logfire.info(
                    "openai.completions.length_retry_attempt",
                    model=model,
                    segment=continuations_done,
                    max_length_continuations=max_length_continuations,
                    raw_chunk_len=len(chunk),
                    wrapped_len=len(wrapped),
                    new_prompt_len=len(current_prompt),
                    has_open_thought=has_open_thought,
                    wrapped_preview=wrapped[:400],
                )
            except Exception:
                pass

        pred = _merge_openai_completion_predictions(pred_list)
        elapsed = time.perf_counter() - start_time

        joined = "".join(chunk_texts)
        raw_text = (
            clean_qwen_completion_chunk_text(joined)
            if use_qwen
            else _clean_completion_chunk_text(joined)
        )

        parse_fn = parse_qwen_completion if use_qwen else parse_completion
        reasoning, tool_calls_raw, content = parse_fn(
            raw_output=raw_text,
            injection_prefix=active_prefix,
        )

        # Convert to tau ToolCall objects
        tool_calls: list[ToolCall] = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        usage, cost = prediction_usage_with_cost(pred, pricing=self._resolve_pricing())

        if not content and not tool_calls:
            logger.warning(
                "openai_completions_agent returned empty response; inserting placeholder."
            )
            content = "(No response from model)"

        # Record whether *any* segment in this turn hit ``finish_reason == "length"``
        # so downstream consumers (e.g. _simulation_hit_length in batch.py) can
        # still flag the turn as length-truncated even when the wrapped retry
        # recovered and the final/merged prediction reports ``finish_reason == "stop"``.
        hit_length_any = False
        length_segment_indices: list[int] = []
        for idx, p in enumerate(pred_list):
            _choices = (p.get("choices") or [{}])
            _fr = str((_choices[0] if _choices else {}).get("finish_reason") or "").lower()
            if _fr == "length":
                hit_length_any = True
                length_segment_indices.append(idx)

        raw_data: dict[str, Any] = {
            "completions_response": pred,
            "raw_text": raw_text,
        }
        if hit_length_any:
            # Flag consumed by batch._simulation_hit_length so the task span keeps
            # its ``[length]`` suffix regardless of whether the retry succeeded.
            raw_data["hit_length"] = True
            raw_data["length_segment_indices"] = length_segment_indices
        if len(pred_list) > 1:
            raw_data["completions_responses"] = pred_list
            raw_data["length_continuation_segments"] = len(pred_list)

        return AssistantMessage(
            role="assistant",
            content=content,
            reasoning_content=(reasoning or None),
            tool_calls=tool_calls or None,
            usage=usage,
            cost=cost,
            raw_data=raw_data,
            generation_time_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# HTTP + logfire wrapper for /v1/completions
# ---------------------------------------------------------------------------

def _completions_generate_with_logfire(
    *,
    base_url: str,
    model: str,
    payload: dict[str, Any],
    api_messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]] | None,
    raw_input_prompt: str,
    active_prefix: str | None,
    parse_completion_fn: Any = parse_completion,
) -> dict[str, Any]:
    """
    POST /v1/completions and emit a logfire span + render event.

    The raw input prompt and raw output are logged directly — no /render
    round-trip needed since we built the prompt ourselves.
    """
    from tau2.utils.vertex_endpoint_chat import tool_round_from_openai_messages
    from contextlib import nullcontext

    tool_round = tool_round_from_openai_messages(api_messages)

    try:
        import logfire  # type: ignore
        span_cm = logfire.span(
            f"openai completions [assistant]",
            _span_name="openai.completions [assistant]",
            _tags=["LLM"],
            model=model,
            llm_system="openai",
            llm_model_name=model,
            gen_ai_operation_name="completions",
            gen_ai_request_model=model,
        )
    except Exception:
        logfire = None
        span_cm = nullcontext()

    connect_timeout = float(payload.pop("_connect_timeout_seconds", 300.0))
    read_timeout = float(payload.pop("_read_timeout_seconds", 300.0))
    request_max_retries = max(0, int(payload.pop("_request_max_retries", 2)))
    request_retry_backoff_seconds = max(
        0.0, float(payload.pop("_request_retry_backoff_seconds", 2.0))
    )

    last_error: Exception | None = None
    retries_left = request_max_retries
    while True:
        try:
            r = requests.post(
                base_url.rstrip("/") + "/v1/completions",
                json=payload,
                timeout=(connect_timeout, read_timeout),
            )
            if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"openai_completions_agent transient status {r.status_code}: {r.text[:500]}"
                )
            if r.status_code != 200:
                raise RuntimeError(
                    f"openai_completions_agent: /v1/completions returned {r.status_code}: {r.text[:500]}"
                )
            pred = r.json()
            break
        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
            RuntimeError,
        ) as e:
            last_error = e
            if retries_left <= 0:
                raise
            retry_attempt = (request_max_retries - retries_left) + 1
            sleep_seconds = request_retry_backoff_seconds * retry_attempt
            logger.warning(
                "openai_completions_agent request failed (attempt {} / {}), retrying in {:.1f}s: {}",
                retry_attempt,
                request_max_retries + 1,
                sleep_seconds,
                e,
            )
            time.sleep(sleep_seconds)
            retries_left -= 1

    if "pred" not in locals():
        raise RuntimeError(
            "openai_completions_agent request failed without response"
            + (f": {last_error}" if last_error else "")
        )

    choices = pred.get("choices") or []
    raw_text = (choices[0].get("text") or "") if choices else ""
    finish_reason = (choices[0].get("finish_reason") or "") if choices else ""
    usage = pred.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

    # Parse output for logfire events (same as openai_generate_with_logfire)
    reasoning_only, tool_calls_raw, body_only = parse_completion_fn(
        raw_output=raw_text,
        injection_prefix=active_prefix,
    )
    response_tool_calls = [
        {
            "id": tc.get("id", ""),
            "type": "function",
            "function": {
                "name": (tc.get("function") or {}).get("name", ""),
                "arguments": json.dumps((tc.get("function") or {}).get("arguments") or {}),
            },
        }
        for tc in tool_calls_raw
    ]

    from tau2.utils.genai_logfire import (
        _openai_chat_messages_to_all_messages_events,
        _response_to_all_messages_event,
        _flatten_llm_messages_attrs,
    )

    last_assistant_event = _response_to_all_messages_event(
        include_thoughts=True,
        reasoning=reasoning_only or "",
        reasoning_blocks=[],
        output_text_blocks=[],
        output_text=body_only,
        response_tool_calls=response_tool_calls,
    )
    input_messages_events = _openai_chat_messages_to_all_messages_events(api_messages)
    all_messages_events = input_messages_events + [last_assistant_event]

    response_data = {
        "message": {
            "role": "assistant",
            "content": last_assistant_event.get("content"),
            "reasoning": reasoning_only or None,
            "tool_calls": response_tool_calls or None,
        }
    }
    # request_data = {"model": model, "messages": input_messages_events}

    with span_cm as span:
        if logfire is not None and span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("llm.model_name", model)
            span.set_attribute("llm.system", "openai")
            span.set_attribute("gen_ai.operation.name", "completions")
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("gen_ai.response.model", model)
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
            span.set_attribute("llm.token_count.total", total_tokens)
            span.set_attribute("tool_round", tool_round)
            span.set_attribute("all_messages_events", all_messages_events)
            # span.set_attribute("request_data", request_data)
            span.set_attribute("response_data", response_data)
            # span.set_attribute("input.value", {"messages": all_messages_events})
            span.set_attribute("output.value", response_data)
            # flat_attrs = _flatten_llm_messages_attrs(
            #     all_messages_events=input_messages_events,
            #     response_data=response_data,
            # )
            # for key, value in flat_attrs.items():
            #     span.set_attribute(key, value)

        if logfire is not None:
            logfire.info(
                "openai.completions.render",
                model=model,
                tool_round=tool_round,
                finish_reason=finish_reason,
                injection=repr(active_prefix[:60]) if active_prefix else None,
                input_token_count=prompt_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_input_prompt=raw_input_prompt,
                raw_output=raw_text,
                raw_response=pred,
            )

    return pred


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_openai_completions_agent(tools, domain_policy, **kwargs):
    return OpenAICompletionsAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )

