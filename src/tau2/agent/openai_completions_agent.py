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
  max_tokens             : int   — default 2048
  seed                   : int   — -1 = fresh random seed
  stop_tokens            : list  — default ["<turn|>"]
  enable_thinking        : bool  — default True (adds <|think|> to system block)
  injection_prefix       : str   — appended after <|turn>model\\n on the FIRST
                                   call per user turn (step 0 only); None = off
  jinja_template_path    : str   — path to .jinja file; embedded fallback used
                                   when absent or file not found
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

def _load_template(template_path: str | None = None) -> Any:
    """Load the Gemma4 Jinja template. Falls back to embedded copy."""
    try:
        from jinja2 import BaseLoader, Environment, Undefined
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for openai_completions_agent. "
            "Install with: uv add jinja2"
        ) from e

    raw = None
    pre_escaped = False
    if template_path:
        p = Path(template_path)
        if p.exists():
            raw = p.read_text()

    if raw is None:
        raw = _EMBEDDED_TEMPLATE
        pre_escaped = True  # embedded template already uses placeholder names

    env = Environment(
        loader=BaseLoader(),
        keep_trailing_newline=True,
        undefined=Undefined,
    )
    env.globals["bos_token"] = "<bos>"
    escaped = raw if pre_escaped else _escape_template(raw)
    return env.from_string(escaped)


def render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    enable_thinking: bool,
    injection_prefix: str | None,
    template_path: str | None,
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
        model = self._get_model()
        base_url = self._get_base_url()
        start_time = time.perf_counter()
        runtime_seed = resolve_runtime_seed(llm_args)

        temperature = float(llm_args.get("temperature", 0.0) or 0.0)
        max_tokens = int(llm_args.get("max_tokens") or 2048)
        stop_tokens: list[str] = list(llm_args.get("stop_tokens") or ["<turn|>"])
        enable_thinking = bool(llm_args.get("enable_thinking", True))
        injection_prefix: str | None = llm_args.get("injection_prefix") or None
        template_path: str | None = llm_args.get("jinja_template_path") or None

        openai_tools = [tool.openai_schema for tool in self.tools] if self.tools else []

        api_messages = tau_messages_to_openai_chat(
            self.system_prompt,
            state.system_messages + state.messages,
            vertex_include_reasoning_in_request=True,
        )

        # Apply injection prefix only on first call after a user message
        # (not after tool results — template ends with <tool_response|> there)
        last_role = api_messages[-1]["role"] if api_messages else "user"
        active_prefix = injection_prefix if last_role == "user" else None

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
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
        }
        if runtime_seed is not None:
            payload["seed"] = int(runtime_seed)

        logger.debug(
            "[openai_completions_agent] calling model={} temperature={} seed={} "
            "prompt_len={} active_prefix={}",
            model, temperature, runtime_seed, len(prompt_str),
            repr(active_prefix[:40]) if active_prefix else None,
        )

        pred = _completions_generate_with_logfire(
            base_url=base_url,
            model=model,
            payload=payload,
            api_messages=api_messages,
            openai_tools=openai_tools or None,
            raw_input_prompt=prompt_str,
            active_prefix=active_prefix,
        )
        elapsed = time.perf_counter() - start_time

        choices = pred.get("choices") or []
        raw_text = (choices[0].get("text") or "") if choices else ""

        reasoning, tool_calls_raw, content = parse_completion(
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

        return AssistantMessage(
            role="assistant",
            content=content,
            reasoning_content=(reasoning or None),
            tool_calls=tool_calls or None,
            usage=usage,
            cost=cost,
            raw_data={"completions_response": pred, "raw_text": raw_text},
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

    r = requests.post(
        base_url.rstrip("/") + "/v1/completions",
        json=payload,
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"openai_completions_agent: /v1/completions returned {r.status_code}: {r.text[:500]}"
        )
    pred = r.json()

    choices = pred.get("choices") or []
    raw_text = (choices[0].get("text") or "") if choices else ""
    finish_reason = (choices[0].get("finish_reason") or "") if choices else ""
    usage = pred.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

    with span_cm as span:
        if logfire is not None and span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("llm.model_name", model)
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
            span.set_attribute("tool_round", tool_round)

        if logfire is not None:
            logfire.info(
                "openai.completions.render",
                model=model,
                tool_round=tool_round,
                finish_reason=finish_reason,
                injection=repr(active_prefix[:60]) if active_prefix else None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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


# ---------------------------------------------------------------------------
# Embedded Gemma4 chat template (fallback when jinja_template_path not set)
# ---------------------------------------------------------------------------

_EMBEDDED_TEMPLATE = r"""{%- macro format_parameters(properties, required) -%}
    {%- set standard_keys = ['description', 'type', 'properties', 'required', 'nullable'] -%}
    {%- set ns = namespace(found_first=false) -%}
    {%- for key, value in properties | dictsort -%}
        {%- set add_comma = false -%}
        {%- if key not in standard_keys -%}
            {%- if ns.found_first %},{% endif -%}
            {%- set ns.found_first = true -%}
            {{ key }}:{
            {%- if value['description'] -%}
                description:GEMMATOK_QUOTE{{ value['description'] }}GEMMATOK_QUOTE
                {%- set add_comma = true -%}
            {%- endif -%}
            {%- if value['type'] | upper == 'STRING' -%}
                {%- if value['enum'] -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    enum:{{ format_argument(value['enum']) }}
                {%- endif -%}
            {%- elif value['type'] | upper == 'ARRAY' -%}
                {%- if value['items'] is mapping and value['items'] -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    items:{
                    {%- set ns_items = namespace(found_first=false) -%}
                    {%- for item_key, item_value in value['items'] | dictsort -%}
                        {%- if item_value is not none -%}
                            {%- if ns_items.found_first %},{% endif -%}
                            {%- set ns_items.found_first = true -%}
                            {%- if item_key == 'properties' -%}
                                properties:{
                                {%- if item_value is mapping -%}
                                    {{- format_parameters(item_value, value['items']['required'] | default([])) -}}
                                {%- endif -%}
                                }
                            {%- elif item_key == 'required' -%}
                                required:[
                                {%- for req_item in item_value -%}
                                    GEMMATOK_QUOTE{{- req_item -}}GEMMATOK_QUOTE
                                    {%- if not loop.last %},{% endif -%}
                                {%- endfor -%}
                                ]
                            {%- elif item_key == 'type' -%}
                                {%- if item_value is string -%}
                                    type:{{ format_argument(item_value | upper) }}
                                {%- else -%}
                                    type:{{ format_argument(item_value | map('upper') | list) }}
                                {%- endif -%}
                            {%- else -%}
                                {{ item_key }}:{{ format_argument(item_value) }}
                            {%- endif -%}
                        {%- endif -%}
                    {%- endfor -%}
                    }
                {%- endif -%}
            {%- endif -%}
            {%- if value['nullable'] %}
                {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                nullable:true
            {%- endif -%}
            {%- if value['type'] | upper == 'OBJECT' -%}
                {%- if value['properties'] is defined and value['properties'] is mapping -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    properties:{
                    {{- format_parameters(value['properties'], value['required'] | default([])) -}}
                    }
                {%- elif value is mapping -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    properties:{
                    {{- format_parameters(value, value['required'] | default([])) -}}
                    }
                {%- endif -%}
                {%- if value['required'] -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    required:[
                    {%- for item in value['required'] | default([]) -%}
                        GEMMATOK_QUOTE{{- item -}}GEMMATOK_QUOTE
                        {%- if not loop.last %},{% endif -%}
                    {%- endfor -%}
                    ]
                {%- endif -%}
            {%- endif -%}
            {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
            type:GEMMATOK_QUOTE{{ value['type'] | upper }}GEMMATOK_QUOTE}
        {%- endif -%}
    {%- endfor -%}
{%- endmacro -%}
{%- macro format_function_declaration(tool_data) -%}
    declaration:{{- tool_data['function']['name'] -}}{description:GEMMATOK_QUOTE{{- tool_data['function']['description'] -}}GEMMATOK_QUOTE
    {%- set params = tool_data['function']['parameters'] -%}
    {%- if params -%}
        ,parameters:{
        {%- if params['properties'] -%}
            properties:{ {{- format_parameters(params['properties'], params['required']) -}} },
        {%- endif -%}
        {%- if params['required'] -%}
            required:[
            {%- for item in params['required'] -%}
                GEMMATOK_QUOTE{{- item -}}GEMMATOK_QUOTE
                {{- ',' if not loop.last -}}
            {%- endfor -%}
            ],
        {%- endif -%}
        {%- if params['type'] -%}
            type:GEMMATOK_QUOTE{{- params['type'] | upper -}}GEMMATOK_QUOTE}
        {%- endif -%}
    {%- endif -%}
    }
{%- endmacro -%}
{%- macro format_argument(argument, escape_keys=True) -%}
    {%- if argument is string -%}
        {{- 'GEMMATOK_QUOTE' + argument + 'GEMMATOK_QUOTE' -}}
    {%- elif argument is boolean -%}
        {{- 'true' if argument else 'false' -}}
    {%- elif argument is mapping -%}
        {{- '{' -}}
        {%- set ns = namespace(found_first=false) -%}
        {%- for key, value in argument | dictsort -%}
            {%- if ns.found_first %},{% endif -%}
            {%- set ns.found_first = true -%}
            {%- if escape_keys -%}
                {{- 'GEMMATOK_QUOTE' + key + 'GEMMATOK_QUOTE' -}}
            {%- else -%}
                {{- key -}}
            {%- endif -%}
            :{{- format_argument(value, escape_keys=escape_keys) -}}
        {%- endfor -%}
        {{- '}' -}}
    {%- elif argument is sequence -%}
        {{- '[' -}}
        {%- for item in argument -%}
            {{- format_argument(item, escape_keys=escape_keys) -}}
            {%- if not loop.last %},{% endif -%}
        {%- endfor -%}
        {{- ']' -}}
    {%- else -%}
        {{- argument -}}
    {%- endif -%}
{%- endmacro -%}
{%- macro strip_thinking(text) -%}
    {%- set ns = namespace(result='') -%}
    {%- for part in text.split('GEMMATOK_CHAN_CLOSE') -%}
        {%- if 'GEMMATOK_CHAN_OPEN' in part -%}
            {%- set ns.result = ns.result + part.split('GEMMATOK_CHAN_OPEN')[0] -%}
        {%- else -%}
            {%- set ns.result = ns.result + part -%}
        {%- endif -%}
    {%- endfor -%}
    {{- ns.result | trim -}}
{%- endmacro -%}

{%- macro format_tool_response_block(tool_name, response) -%}
    {{- 'GEMMATOK_TRESP_OPEN' -}}
    {%- if response is mapping -%}
        {{- 'response:' + tool_name + '{' -}}
        {%- for key, value in response | dictsort -%}
            {{- key -}}:{{- format_argument(value, escape_keys=False) -}}
            {%- if not loop.last %},{% endif -%}
        {%- endfor -%}
        {{- '}' -}}
    {%- else -%}
        {{- 'response:' + tool_name + '{value:' + format_argument(response, escape_keys=False) + '}' -}}
    {%- endif -%}
    {{- 'GEMMATOK_TRESP_CLOSE' -}}
{%- endmacro -%}

{%- set ns = namespace(prev_message_type=None) -%}
{%- set loop_messages = messages -%}
{{- bos_token -}}
{%- if (enable_thinking is defined and enable_thinking) or tools or messages[0]['role'] in ['system', 'developer'] -%}
    {{- 'GEMMATOK_TURN_OPENsystem\n' -}}
    {%- if enable_thinking is defined and enable_thinking -%}
        {{- 'GEMMATOK_THINK\n' -}}
        {%- set ns.prev_message_type = 'think' -%}
    {%- endif -%}
    {%- if messages[0]['role'] in ['system', 'developer'] -%}
        {{- messages[0]['content'] | trim -}}
        {%- set loop_messages = messages[1:] -%}
    {%- endif -%}
    {%- if tools -%}
        {%- for tool in tools %}
            {{- 'GEMMATOK_TOOL_OPEN' -}}
            {{- format_function_declaration(tool) | trim -}}
            {{- 'GEMMATOK_TOOL_CLOSE' -}}
        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
    {%- endif -%}
    {{- 'GEMMATOK_TURN_CLOSE\n' -}}
{%- endif %}

{%- set ns_turn = namespace(last_user_idx=-1) -%}
{%- for i in range(loop_messages | length) -%}
    {%- if loop_messages[i]['role'] == 'user' -%}
        {%- set ns_turn.last_user_idx = i -%}
    {%- endif -%}
{%- endfor -%}

{%- for message in loop_messages -%}
    {%- if message['role'] != 'tool' -%}
    {%- set ns.prev_message_type = None -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {%- set prev_nt = namespace(role=None, found=false) -%}
    {%- if loop.index0 > 0 -%}
        {%- for j in range(loop.index0 - 1, -1, -1) -%}
            {%- if not prev_nt.found -%}
                {%- if loop_messages[j]['role'] != 'tool' -%}
                    {%- set prev_nt.role = loop_messages[j]['role'] -%}
                    {%- set prev_nt.found = true -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- set continue_same_model_turn = (role == 'model' and prev_nt.role == 'assistant') -%}
    {%- if not continue_same_model_turn -%}
        {{- 'GEMMATOK_TURN_OPEN' + role + '\n' }}
    {%- endif -%}

    {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
    {%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}
        {{- 'GEMMATOK_CHAN_OPENthought\n' + thinking_text + '\nGEMMATOK_CHAN_CLOSE' -}}
    {%- endif -%}

        {%- if message['tool_calls'] -%}
            {%- for tool_call in message['tool_calls'] -%}
                {%- set function = tool_call['function'] -%}
                {{- 'GEMMATOK_TCALL_OPENcall:' + function['name'] + '{' -}}
                {%- if function['arguments'] is mapping -%}
                    {%- set ns_args = namespace(found_first=false) -%}
                    {%- for key, value in function['arguments'] | dictsort -%}
                        {%- if ns_args.found_first %},{% endif -%}
                        {%- set ns_args.found_first = true -%}
                        {{- key -}}:{{- format_argument(value, escape_keys=False) -}}
                    {%- endfor -%}
                {%- elif function['arguments'] is string -%}
                    {{- function['arguments'] -}}
                {%- endif -%}
                {{- '}GEMMATOK_TCALL_CLOSE' -}}
            {%- endfor -%}
            {%- set ns.prev_message_type = 'tool_call' -%}
        {%- endif -%}

        {%- set ns_tr_out = namespace(flag=false) -%}
        {%- if message.get('tool_responses') -%}
            {%- for tool_response in message['tool_responses'] -%}
                {{- format_tool_response_block(tool_response['name'] | default('unknown'), tool_response['response']) -}}
                {%- set ns_tr_out.flag = true -%}
                {%- set ns.prev_message_type = 'tool_response' -%}
            {%- endfor -%}
        {%- elif message.get('tool_calls') -%}
            {%- set ns_tool_scan = namespace(stopped=false) -%}
            {%- for k in range(loop.index0 + 1, loop_messages | length) -%}
                {%- if ns_tool_scan.stopped -%}
                {%- elif loop_messages[k]['role'] != 'tool' -%}
                    {%- set ns_tool_scan.stopped = true -%}
                {%- else -%}
                    {%- set follow = loop_messages[k] -%}
                    {%- set ns_tname = namespace(name=follow.get('name') | default('unknown')) -%}
                    {%- for tc in message['tool_calls'] -%}
                        {%- if tc.get('id') == follow.get('tool_call_id') -%}
                            {%- set ns_tname.name = tc['function']['name'] -%}
                        {%- endif -%}
                    {%- endfor -%}
                    {%- set tool_body = follow.get('content') -%}
                    {%- if tool_body is string -%}
                        {{- format_tool_response_block(ns_tname.name, tool_body) -}}
                    {%- elif tool_body is sequence and tool_body is not string -%}
                        {%- set ns_txt = namespace(s='') -%}
                        {%- for part in tool_body -%}
                            {%- if part.get('type') == 'text' -%}
                                {%- set ns_txt.s = ns_txt.s + (part.get('text') | default('')) -%}
                            {%- endif -%}
                        {%- endfor -%}
                        {{- format_tool_response_block(ns_tname.name, ns_txt.s) -}}
                    {%- else -%}
                        {{- format_tool_response_block(ns_tname.name, tool_body) -}}
                    {%- endif -%}
                    {%- set ns_tr_out.flag = true -%}
                    {%- set ns.prev_message_type = 'tool_response' -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}

        {%- if message['content'] is string -%}
            {%- if role == 'model' -%}
                {{- strip_thinking(message['content']) -}}
            {%- else -%}
                {{- message['content'] | trim -}}
            {%- endif -%}
        {%- elif message['content'] is sequence -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'text' -%}
                    {%- if role == 'model' -%}
                        {{- strip_thinking(item['text']) -}}
                    {%- else -%}
                        {{- item['text'] | trim -}}
                    {%- endif -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}

    {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}
        {{- 'GEMMATOK_TRESP_OPEN' -}}
    {%- elif not (ns_tr_out.flag and not message.get('content')) -%}
        {{- 'GEMMATOK_TURN_CLOSE\n' -}}
    {%- endif -%}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- 'GEMMATOK_TURN_OPENmodel\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- 'GEMMATOK_CHAN_OPENthought\nGEMMATOK_CHAN_CLOSE' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}
"""
