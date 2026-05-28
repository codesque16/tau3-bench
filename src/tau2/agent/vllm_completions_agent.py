"""
vLLM in-process completions agent for tau3 benchmark.

Same flow as ``openai_completions_agent`` / ``huggingface_completions_agent``:
client-side Jinja2 chat-template rendering, Gemma/Qwen parsing, multi-step tool
flow — but runs ``vllm.LLM.generate()`` in-process (no HTTP server).

llm_args keys (in addition to shared template/sampling keys)
------------------------------------------------------------
  model_id               : str   — HF hub id or local path (default: agent ``llm``)
  adapter_path           : str   — optional LoRA adapter dir (``enable_lora``)
  torch_dtype            : str   — ``bfloat16`` (default), ``float16``, ``float32``
  trust_remote_code      : bool  — passed to vLLM (default True)
  max_model_len          : int   — vLLM context window (default 32768)
  language_model_only    : bool  — skip vision encoder for text-only (default True)
  gpu_memory_utilization : float — vLLM GPU memory fraction (default 0.90)
  enable_prefix_caching  : bool  — reuse KV for shared prompt prefixes (default True)
  enable_chunked_prefill : bool  — chunked prefill (default True)
  tensor_parallel_size   : int   — TP size (default 1)
  max_lora_rank          : int   — when using ``adapter_path`` (default 64)
  vllm_kwargs            : dict  — extra kwargs forwarded to ``vllm.LLM(...)``

For merged LoRA checkpoints, set ``model_id`` to the merged path and leave
``adapter_path`` unset.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent
from tau2.agent.openai_completions_agent import (
    _clean_completion_chunk_text,
    _merge_openai_completion_predictions,
    _wrap_truncated_thought,
    parse_completion,
    render_prompt,
)
from tau2.agent.qwen_completions_utils import (
    clean_qwen_completion_chunk_text,
    is_qwen_completion_format,
    normalize_messages_for_qwen_template,
    parse_qwen_completion,
    render_qwen_prompt,
    resolve_qwen_stop_tokens,
    wrap_truncated_qwen_thought,
)
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.environment.tool import Tool
from tau2.utils.vertex_endpoint_chat import (
    resolve_runtime_seed,
    tau_messages_to_openai_chat,
)


@dataclass(frozen=True)
class _VLLMEngineKey:
    model_id: str
    adapter_path: str
    torch_dtype: str
    trust_remote_code: bool
    max_model_len: int
    language_model_only: bool
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    tensor_parallel_size: int
    max_lora_rank: int
    vllm_kwargs_json: str


class _VLLMEngineHandle:
    """Lazy-loaded vLLM engine, cached per process."""

    _cache: dict[_VLLMEngineKey, "_VLLMEngineHandle"] = {}
    _load_lock = threading.Lock()

    def __init__(
        self,
        *,
        llm: Any,
        model_id: str,
        adapter_path: str | None,
        lora_request: Any | None,
    ):
        self.llm = llm
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.lora_request = lora_request
        self._generate_lock = threading.Lock()

    @classmethod
    def from_llm_args(cls, llm_args: dict[str, Any], llm: str) -> "_VLLMEngineHandle":
        model_id = str(llm_args.get("model_id") or (llm or "").strip()).strip()
        if not model_id:
            raise ValueError(
                "model_id must be set in llm_args (or agent_llm) for "
                "vllm_completions_agent."
            )

        adapter_raw = llm_args.get("adapter_path")
        adapter_path = os.path.expanduser(str(adapter_raw).strip()) if adapter_raw else ""
        torch_dtype = str(llm_args.get("torch_dtype") or "bfloat16").lower()
        trust_remote_code = bool(llm_args.get("trust_remote_code", True))
        max_model_len = int(llm_args.get("max_model_len") or 32768)
        language_model_only = bool(llm_args.get("language_model_only", True))
        gpu_memory_utilization = float(llm_args.get("gpu_memory_utilization", 0.90))
        enable_prefix_caching = bool(llm_args.get("enable_prefix_caching", True))
        enable_chunked_prefill = bool(llm_args.get("enable_chunked_prefill", True))
        tensor_parallel_size = int(llm_args.get("tensor_parallel_size") or 1)
        max_lora_rank = int(llm_args.get("max_lora_rank") or 64)
        extra = llm_args.get("vllm_kwargs") or {}
        vllm_kwargs_json = json.dumps(extra, sort_keys=True)

        key = _VLLMEngineKey(
            model_id=model_id,
            adapter_path=adapter_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            language_model_only=language_model_only,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            tensor_parallel_size=tensor_parallel_size,
            max_lora_rank=max_lora_rank,
            vllm_kwargs_json=vllm_kwargs_json,
        )
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        with cls._load_lock:
            cached = cls._cache.get(key)
            if cached is not None:
                return cached
            handle = cls._load(key, extra)
            cls._cache[key] = handle
            return handle

    @classmethod
    def _load(cls, key: _VLLMEngineKey, extra: dict[str, Any]) -> "_VLLMEngineHandle":
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vllm_completions_agent requires vllm. "
                "Install with: uv sync --extra vllm"
            ) from e

        use_lora = bool(key.adapter_path)
        if use_lora and not Path(key.adapter_path).exists():
            raise FileNotFoundError(f"adapter_path not found: {key.adapter_path!r}")

        llm_kwargs: dict[str, Any] = {
            "model": key.model_id,
            "dtype": key.torch_dtype,
            "trust_remote_code": key.trust_remote_code,
            "max_model_len": key.max_model_len,
            "language_model_only": key.language_model_only,
            "gpu_memory_utilization": key.gpu_memory_utilization,
            "enable_prefix_caching": key.enable_prefix_caching,
            "enable_chunked_prefill": key.enable_chunked_prefill,
            "tensor_parallel_size": key.tensor_parallel_size,
            "generation_config": "vllm",
        }
        if use_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = key.max_lora_rank
        llm_kwargs.update(extra)

        logger.info(
            "vllm_completions_agent: loading engine model={} adapter={} "
            "prefix_caching={} dtype={} max_model_len={} language_model_only={}",
            key.model_id,
            key.adapter_path or None,
            key.enable_prefix_caching,
            key.torch_dtype,
            key.max_model_len,
            key.language_model_only,
        )
        llm = LLM(**llm_kwargs)

        lora_request = None
        if use_lora:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(
                "tau3_adapter",
                1,
                key.adapter_path,
            )

        logger.info("vllm_completions_agent: engine ready (model_id={})", key.model_id)
        return cls(
            llm=llm,
            model_id=key.model_id,
            adapter_path=key.adapter_path or None,
            lora_request=lora_request,
        )


def _build_sampling_params(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    stop_tokens: list[str],
    include_stop_str_in_output: bool,
    seed: int | None,
) -> Any:
    from vllm import SamplingParams

    greedy = temperature < 1e-5
    kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": 0.0 if greedy else float(temperature),
        "stop": stop_tokens or None,
        "include_stop_str_in_output": include_stop_str_in_output,
    }
    if not greedy:
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
    if seed is not None:
        kwargs["seed"] = int(seed)
    return SamplingParams(**kwargs)


def _vllm_local_generate(
    *,
    handle: _VLLMEngineHandle,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    stop_tokens: list[str],
    include_stop_str_in_output: bool,
    seed: int | None,
) -> dict[str, Any]:
    sampling_params = _build_sampling_params(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_tokens=stop_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
        seed=seed,
    )

    gen_kwargs: dict[str, Any] = {
        "prompts": [prompt],
        "sampling_params": sampling_params,
    }
    if handle.lora_request is not None:
        gen_kwargs["lora_request"] = handle.lora_request

    with handle._generate_lock:
        outputs = handle.llm.generate(**gen_kwargs)

    req_out = outputs[0]
    out = req_out.outputs[0]
    raw_text = out.text or ""
    finish_reason = str(out.finish_reason or "stop").lower()
    if finish_reason not in ("stop", "length"):
        finish_reason = "stop"

    prompt_tokens = len(req_out.prompt_token_ids or [])
    completion_tokens = len(out.token_ids or [])

    return {
        "choices": [{"text": raw_text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _vllm_completions_generate_with_logfire(
    *,
    handle: _VLLMEngineHandle,
    model_name: str,
    payload: dict[str, Any],
    api_messages: list[dict[str, Any]],
    raw_input_prompt: str,
    active_prefix: str | None,
    parse_completion_fn: Callable[..., tuple[str | None, list[dict[str, Any]], str]],
) -> dict[str, Any]:
    from contextlib import nullcontext

    from tau2.utils.genai_logfire import (
        _openai_chat_messages_to_all_messages_events,
        _response_to_all_messages_event,
    )
    from tau2.utils.vertex_endpoint_chat import tool_round_from_openai_messages

    tool_round = tool_round_from_openai_messages(api_messages)

    try:
        import logfire  # type: ignore

        span_cm = logfire.span(
            "vllm completions [assistant]",
            _span_name="vllm.completions [assistant]",
            _tags=["LLM"],
            model=model_name,
            llm_system="vllm",
            llm_model_name=model_name,
            gen_ai_operation_name="completions",
            gen_ai_request_model=model_name,
        )
    except Exception:
        logfire = None
        span_cm = nullcontext()

    with span_cm as span:
        generate_start = time.perf_counter()
        pred = _vllm_local_generate(
            handle=handle,
            prompt=str(payload["prompt"]),
            max_tokens=int(payload["max_tokens"]),
            temperature=float(payload.get("temperature", 0.0) or 0.0),
            top_p=payload.get("top_p"),
            top_k=payload.get("top_k"),
            stop_tokens=list(payload.get("stop") or []),
            include_stop_str_in_output=bool(
                payload.get("include_stop_str_in_output", True)
            ),
            seed=payload.get("seed"),
        )
        generate_seconds = time.perf_counter() - generate_start

        choices = pred.get("choices") or []
        raw_text = (choices[0].get("text") or "") if choices else ""
        finish_reason = (choices[0].get("finish_reason") or "") if choices else ""
        usage = pred.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(
            usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

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
                    "arguments": json.dumps(
                        (tc.get("function") or {}).get("arguments") or {}
                    ),
                },
            }
            for tc in tool_calls_raw
        ]

        last_assistant_event = _response_to_all_messages_event(
            include_thoughts=True,
            reasoning=reasoning_only or "",
            reasoning_blocks=[],
            output_text_blocks=[],
            output_text=body_only,
            response_tool_calls=response_tool_calls,
        )
        input_messages_events = _openai_chat_messages_to_all_messages_events(
            api_messages
        )
        all_messages_events = input_messages_events + [last_assistant_event]
        response_data = {
            "message": {
                "role": "assistant",
                "content": last_assistant_event.get("content"),
                "reasoning": reasoning_only or None,
                "tool_calls": response_tool_calls or None,
            }
        }

        if logfire is not None and span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("llm.model_name", model_name)
            span.set_attribute("llm.system", "vllm")
            span.set_attribute("gen_ai.operation.name", "completions")
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.response.model", model_name)
            span.set_attribute("gen_ai.system", "vllm")
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
            span.set_attribute("llm.token_count.total", total_tokens)
            span.set_attribute("generation_time_seconds", generate_seconds)
            span.set_attribute("tool_round", tool_round)
            span.set_attribute("all_messages_events", all_messages_events)
            span.set_attribute("response_data", response_data)
            span.set_attribute("output.value", response_data)

        if logfire is not None:
            logfire.info(
                "vllm.completions.render",
                model=model_name,
                tool_round=tool_round,
                finish_reason=finish_reason,
                injection=repr(active_prefix[:60]) if active_prefix else None,
                input_token_count=prompt_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_seconds=generate_seconds,
                raw_input_prompt=raw_input_prompt,
                raw_output=raw_text,
                raw_response=pred,
            )

    return pred


class VLLMCompletionsAgent(LLMAgent):
    """
    Tau3 agent using in-process ``vllm.LLM.generate()`` with client-side Jinja2
    chat-template rendering. Same flow as ``openai_completions_agent``.
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
        self._vllm_handle: _VLLMEngineHandle | None = None

    def _get_model_id(self) -> str:
        llm_args = self.llm_args or {}
        model_id = str(llm_args.get("model_id") or (self.llm or "").strip()).strip()
        if not model_id:
            raise ValueError(
                "model_id must be set in llm_args (or agent_llm) for "
                "vllm_completions_agent."
            )
        return model_id

    def _get_handle(self) -> _VLLMEngineHandle:
        if self._vllm_handle is None:
            self._vllm_handle = _VLLMEngineHandle.from_llm_args(
                self.llm_args or {},
                self.llm or "",
            )
        return self._vllm_handle

    def _generate_next_message(self, message, state) -> AssistantMessage:
        if message.role == "user" and getattr(message, "is_audio", False):
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")

        if message.role == "tool" and hasattr(message, "tool_messages"):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        llm_args = self.llm_args or {}
        use_qwen = is_qwen_completion_format(llm_args)
        model = self._get_model_id()
        handle = self._get_handle()
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
        injection_prefix: str | None = (
            llm_args.get("injection_prefix")
            or chat_template_kwargs.get("injection_prefix")
            or None
        )
        template_path: str = llm_args.get("jinja_template_path") or ""
        if not template_path:
            raise ValueError(
                "jinja_template_path must be set in agent_llm_args for "
                "vllm_completions_agent."
            )

        openai_tools = [tool.openai_schema for tool in self.tools] if self.tools else []
        api_messages = tau_messages_to_openai_chat(
            self.system_prompt,
            state.system_messages + state.messages,
            vertex_include_reasoning_in_request=True,
        )
        if use_qwen:
            api_messages = normalize_messages_for_qwen_template(api_messages)

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
            "prompt": prompt_str,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop_tokens,
            "include_stop_str_in_output": True,
        }
        if runtime_seed is not None:
            payload["seed"] = int(runtime_seed)
        top_p = llm_args.get("top_p")
        if top_p is not None:
            payload["top_p"] = float(top_p)
        top_k = llm_args.get("top_k")
        if top_k is not None:
            payload["top_k"] = int(top_k)

        logger.debug(
            "[vllm_completions_agent] generating model={} temperature={} seed={} "
            "prompt_len={} active_prefix={} prefix_caching={}",
            model,
            temperature,
            runtime_seed,
            len(prompt_str),
            repr(active_prefix[:40]) if active_prefix else None,
            (self.llm_args or {}).get("enable_prefix_caching", True),
        )

        continue_on_length = bool(llm_args.get("continue_on_length", False))
        _mlc = llm_args.get("max_length_continuations")
        max_length_continuations = max(0, int(8 if _mlc is None else _mlc))

        pred_list: list[dict[str, Any]] = []
        chunk_texts: list[str] = []
        current_prompt = prompt_str
        continuations_done = 0
        parse_fn = parse_qwen_completion if use_qwen else parse_completion

        while True:
            payload["prompt"] = current_prompt
            pred = _vllm_completions_generate_with_logfire(
                handle=handle,
                model_name=model,
                payload=payload,
                api_messages=api_messages,
                raw_input_prompt=current_prompt,
                active_prefix=active_prefix if continuations_done == 0 else None,
                parse_completion_fn=parse_fn,
            )
            pred_list.append(pred)
            choices = pred.get("choices") or []
            chunk = (choices[0].get("text") or "") if choices else ""
            finish_reason = (choices[0].get("finish_reason") or "").lower() if choices else ""

            if finish_reason == "length":
                try:
                    import logfire  # type: ignore

                    logfire.info(
                        "vllm.completions.length_truncation",
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
                "vllm_completions_agent: length-truncated turn, wrapped as "
                "TRUNCATED thought and retrying (segment {} / {}, prompt_len={})",
                continuations_done,
                max_length_continuations,
                len(current_prompt),
            )

        pred = _merge_openai_completion_predictions(pred_list)
        elapsed = time.perf_counter() - start_time

        joined = "".join(chunk_texts)
        raw_text = (
            clean_qwen_completion_chunk_text(joined)
            if use_qwen
            else _clean_completion_chunk_text(joined)
        )
        reasoning, tool_calls_raw, content = parse_fn(
            raw_output=raw_text,
            injection_prefix=active_prefix,
        )

        tool_calls: list[ToolCall] = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        usage = pred.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        usage_out = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        if not content and not tool_calls:
            logger.warning(
                "vllm_completions_agent returned empty response; inserting placeholder."
            )
            content = "(No response from model)"

        hit_length_any = False
        length_segment_indices: list[int] = []
        for idx, p in enumerate(pred_list):
            _choices = p.get("choices") or [{}]
            _fr = str((_choices[0] if _choices else {}).get("finish_reason") or "").lower()
            if _fr == "length":
                hit_length_any = True
                length_segment_indices.append(idx)

        raw_data: dict[str, Any] = {
            "completions_response": pred,
            "raw_text": raw_text,
            "vllm_model_id": model,
            "vllm_adapter_path": (self.llm_args or {}).get("adapter_path"),
        }
        if hit_length_any:
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
            usage=usage_out,
            cost=0.0,
            raw_data=raw_data,
            generation_time_seconds=elapsed,
        )


def create_vllm_completions_agent(tools, domain_policy, **kwargs):
    return VLLMCompletionsAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
