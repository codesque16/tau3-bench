"""
OpenAI-compatible agent for tau3 benchmark.

Supports any OpenAI-compatible endpoint: vLLM, OpenAI API, Gemma 4 on vLLM, etc.

Key features vs. VertexAgent
------------------------------
- Uses the ``openai`` Python SDK (no google-genai dependency).
- chat_template_kwargs  — vLLM thinking / injection support (enable_thinking,
  inject_thinking, injection_prefix) passed directly in the request body.
- Raw token logging    — when ``log_raw_tokens: true`` in llm_args, calls
  /v1/chat/completions/render (input token IDs) and /tokenize (output token IDs)
  and logs them via loguru at DEBUG level, mirroring the inspector script panels.
- Reasoning extraction — reads message.reasoning / message.reasoning_content from
  the choice, same as vertex_agent's OpenAI path.

llm_args keys
-------------
  openai_base_url        : str   — server root, e.g. "http://34.29.173.120:8000"
  openai_api_key         : str   — API key (default "EMPTY" for vLLM)
  openai_model           : str   — model name sent in the JSON body (falls back to self.llm)
  temperature            : float — default 0.0
  max_tokens             : int   — default 2048
  seed                   : int   — -1 = fresh random seed each call
  tool_choice            : str   — "auto" | "required" | "none" (default "auto")
  chat_template_kwargs   : dict  — forwarded verbatim; supports
                                   enable_thinking, inject_thinking, injection_prefix
  log_raw_tokens         : bool  — if true, log input/output token IDs (DEBUG)
  log_raw_tokens_max_ids : int   — max token IDs to print (default 120)
  skip_special_tokens    : bool  — passed to /render + chat completions (default False)
  pricing                : dict  — input_cost_per_million, output_cost_per_million,
                                   cached_input_cost_per_million (optional)
"""

from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage
from tau2.environment.tool import Tool
from tau2.utils.genai_logfire import openai_generate_with_logfire
from tau2.utils.vertex_endpoint_chat import (
    prediction_assistant_body_text,
    prediction_message_reasoning_text,
    prediction_tool_calls_as_tau,
    prediction_usage_with_cost,
    resolve_runtime_seed,
    tau_messages_to_openai_chat,
)



# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OpenAIAgent(LLMAgent):
    """
    Half-duplex agent using the OpenAI Python SDK (openai-compatible endpoints).

    Designed to work with:
      - vLLM serving Gemma 4 (with chat_template_kwargs for thinking / injection)
      - Standard OpenAI API
      - Any other OpenAI-compatible server
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
        self._openai_client: Any = None

    def _get_base_url(self) -> str:
        raw = str((self.llm_args or {}).get("openai_base_url") or "").strip()
        if not raw:
            raise ValueError(
                "openai_base_url must be set in llm_args for openai_agent. "
                "Example: openai_base_url: 'http://34.29.173.120:8000'"
            )
        return raw.rstrip("/")

    def _get_api_key(self) -> str:
        return str((self.llm_args or {}).get("openai_api_key") or "EMPTY").strip()

    def _get_model(self) -> str:
        model = str(
            (self.llm_args or {}).get("openai_model")
            or (self.llm or "").strip()
        ).strip()
        if not model:
            raise ValueError(
                "Set openai_model in llm_args (or a non-empty agent_llm) for openai_agent."
            )
        return model

    def _get_client(self) -> Any:
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for openai_agent. "
                "Install with: uv add openai"
            ) from e
        self._openai_client = OpenAI(
            base_url=self._get_base_url() + "/v1",
            api_key=self._get_api_key(),
        )
        return self._openai_client

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
            logger.warning("Invalid pricing config for openai_agent: {}", pricing)
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

        openai_tools = [tool.openai_schema for tool in self.tools] if self.tools else []

        api_messages = tau_messages_to_openai_chat(
            self.system_prompt,
            state.system_messages + state.messages,
            vertex_include_reasoning_in_request=bool(
                llm_args.get("vertex_include_reasoning_in_request", True)
            ),
        )

        # Build the request body
        chat_template_kwargs: dict[str, Any] | None = llm_args.get("chat_template_kwargs") or None
        skip_special_tokens = bool(llm_args.get("skip_special_tokens", False))
        tool_choice = str(llm_args.get("tool_choice") or "auto")
        temperature = float(llm_args.get("temperature", 0.0) or 0.0)
        max_tokens = llm_args.get("max_tokens")

        extra_body: dict[str, Any] = {}
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if skip_special_tokens:
            extra_body["skip_special_tokens"] = skip_special_tokens

        # Mirror TAU2_VLLM_MIRROR_CHAT_TEMPLATE_KWARGS_EXTRA_BODY behavior
        import os
        if chat_template_kwargs and os.getenv(
            "TAU2_VLLM_MIRROR_CHAT_TEMPLATE_KWARGS_EXTRA_BODY", ""
        ).lower() in ("1", "true", "yes"):
            extra_body.setdefault("extra_body", {})["chat_template_kwargs"] = chat_template_kwargs

        client = self._get_client()

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,  # type: ignore[arg-type]
            "temperature": temperature,
            "tool_choice": tool_choice if openai_tools else None,
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = int(max_tokens)
        if runtime_seed is not None:
            create_kwargs["seed"] = int(runtime_seed)
        if openai_tools:
            create_kwargs["tools"] = openai_tools  # type: ignore[assignment]
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        logger.debug(
            "[openai_agent] calling model={} temperature={} seed={} tool_choice={}",
            model, temperature, runtime_seed, tool_choice if openai_tools else "n/a",
        )

        response = openai_generate_with_logfire(
            client=client,
            model=model,
            api_messages=api_messages,
            openai_tools=openai_tools or None,
            create_kwargs=create_kwargs,
            actor="agent",
            call_name="openai_agent_response",
            raw_prompt_log_config={
                "base_url": base_url,
                "model": model,
                "chat_template_kwargs": chat_template_kwargs,
                "skip_special_tokens": skip_special_tokens,
            },
        )
        elapsed = time.perf_counter() - start_time

        # Convert to plain dict for reuse of prediction_* helpers
        pred = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}

        content = prediction_assistant_body_text(pred)
        reasoning_only = prediction_message_reasoning_text(pred)
        tool_calls = prediction_tool_calls_as_tau(pred)
        usage, cost = prediction_usage_with_cost(pred, pricing=self._resolve_pricing())

        if not content and not tool_calls:
            logger.warning("openai_agent returned empty response; inserting placeholder.")
            content = "(No response from model)"
        
        return AssistantMessage(
            role="assistant",
            content=content,
            reasoning_content=(reasoning_only or None),
            tool_calls=tool_calls or None,
            usage=usage,
            cost=cost,
            raw_data={"openai_response": pred},
            generation_time_seconds=elapsed,
        )


def create_openai_agent(tools, domain_policy, **kwargs):
    return OpenAIAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
