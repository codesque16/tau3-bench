"""
Hugging Face local-completions agent for tau3 benchmark.

Mirrors ``openai_completions_agent``: client-side Jinja2 chat-template rendering,
same Gemma/Qwen parsing and multi-step tool flow — but runs ``model.generate()``
locally via ``transformers`` (+ optional ``peft`` LoRA adapter) instead of
``/v1/completions``.

llm_args keys (in addition to openai_completions_agent template/sampling keys)
-----------------------------------------------------------------------------
  model_id               : str   — HF hub id or local path (default: agent ``llm``)
  adapter_path           : str   — optional LoRA adapter checkpoint directory
  device                 : str   — e.g. ``cuda:0`` (default: cuda:0 if available)
  torch_dtype            : str   — ``bfloat16`` (default), ``float16``, ``float32``
  attn_implementation    : str   — ``flash_attention_2`` (default), ``sdpa``, ``eager``
  trust_remote_code      : bool  — passed to ``from_pretrained`` (default True)

Shared with openai_completions_agent
------------------------------------
  jinja_template_path, completion_format, temperature, top_p, top_k, max_tokens,
  stop_tokens, enable_thinking, injection_prefix, continue_on_length,
  max_length_continuations, seed, skip_special_tokens, chat_template_kwargs
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
class _HFModelKey:
    model_id: str
    adapter_path: str
    device: str
    torch_dtype: str
    attn_implementation: str
    trust_remote_code: bool


class _HFModelHandle:
    """Lazy-loaded model + tokenizer, cached per process."""

    _cache: dict[_HFModelKey, "_HFModelHandle"] = {}
    _load_lock = threading.Lock()

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        adapter_path: str | None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._generate_lock = threading.Lock()

    @classmethod
    def from_llm_args(cls, llm_args: dict[str, Any], llm: str) -> "_HFModelHandle":
        model_id = str(llm_args.get("model_id") or (llm or "").strip()).strip()
        if not model_id:
            raise ValueError(
                "model_id must be set in llm_args (or agent_llm) for "
                "huggingface_completions_agent."
            )

        adapter_raw = llm_args.get("adapter_path")
        adapter_path = os.path.expanduser(str(adapter_raw).strip()) if adapter_raw else ""
        device = str(llm_args.get("device") or _default_device())
        torch_dtype = str(llm_args.get("torch_dtype") or "bfloat16").lower()
        attn_implementation = str(
            llm_args.get("attn_implementation") or "flash_attention_2"
        )
        trust_remote_code = bool(llm_args.get("trust_remote_code", True))

        key = _HFModelKey(
            model_id=model_id,
            adapter_path=adapter_path,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        with cls._load_lock:
            cached = cls._cache.get(key)
            if cached is not None:
                return cached
            handle = cls._load(key)
            cls._cache[key] = handle
            return handle

    @classmethod
    def _load(cls, key: _HFModelKey) -> "_HFModelHandle":
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "huggingface_completions_agent requires torch and transformers. "
                "Install with: uv sync --extra huggingface"
            ) from e

        tok_source = (
            key.adapter_path
            if key.adapter_path and Path(key.adapter_path).exists()
            else key.model_id
        )
        logger.info(
            "huggingface_completions_agent: loading tokenizer from {}",
            tok_source,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tok_source,
            trust_remote_code=key.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = _resolve_torch_dtype(key.torch_dtype)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": key.trust_remote_code,
            # Avoid meta-device init; concurrent .to(cuda) on meta tensors raises.
            "low_cpu_mem_usage": False,
        }
        if key.attn_implementation:
            model_kwargs["attn_implementation"] = key.attn_implementation

        logger.info(
            "huggingface_completions_agent: loading base model {} on {}",
            key.model_id,
            key.device,
        )
        base = cls._load_causal_lm(key.model_id, model_kwargs, key.attn_implementation)

        if key.adapter_path:
            if not Path(key.adapter_path).exists():
                raise FileNotFoundError(
                    f"adapter_path not found: {key.adapter_path!r}"
                )
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "adapter_path was set but peft is not installed. "
                    "Install with: uv sync --extra huggingface"
                ) from e
            logger.info(
                "huggingface_completions_agent: loading LoRA adapter from {}",
                key.adapter_path,
            )
            model = PeftModel.from_pretrained(base, key.adapter_path).eval()
        else:
            model = base.eval()

        model = model.to(key.device)
        logger.info(
            "huggingface_completions_agent: model ready (model_id={} adapter={})",
            key.model_id,
            key.adapter_path or None,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=key.model_id,
            adapter_path=key.adapter_path or None,
        )

    @staticmethod
    def _load_causal_lm(
        model_id: str,
        model_kwargs: dict[str, Any],
        attn_implementation: str,
    ) -> Any:
        from transformers import AutoModelForCausalLM

        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except Exception as e:
            if attn_implementation == "flash_attention_2":
                logger.warning(
                    "huggingface_completions_agent: flash_attention_2 load failed "
                    "({}), falling back to sdpa",
                    e,
                )
                fallback = dict(model_kwargs)
                fallback["attn_implementation"] = "sdpa"
                return AutoModelForCausalLM.from_pretrained(model_id, **fallback)
            raise


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _resolve_torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported torch_dtype {name!r}; use one of {sorted(mapping)}"
        )
    return mapping[name]


def _truncate_at_stop_tokens(
    text: str,
    stop_tokens: list[str],
    *,
    include_stop_str_in_output: bool,
) -> tuple[str, bool]:
    """Return (truncated_text, stopped_by_token)."""
    if not stop_tokens:
        return text, False
    earliest_idx: int | None = None
    matched_stop = ""
    for stop in stop_tokens:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx
            matched_stop = stop
    if earliest_idx is None:
        return text, False
    end = earliest_idx + (len(matched_stop) if include_stop_str_in_output else 0)
    return text[:end], True


def _hf_local_generate(
    *,
    handle: _HFModelHandle,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    stop_tokens: list[str],
    skip_special_tokens: bool,
    include_stop_str_in_output: bool,
    seed: int | None,
) -> dict[str, Any]:
    import torch

    tokenizer = handle.tokenizer
    model = handle.model

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_tokens),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    greedy = temperature < 1e-5
    gen_kwargs["do_sample"] = not greedy
    if not greedy:
        gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    with handle._generate_lock, torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    new_ids = output_ids[0, input_ids.shape[1] :]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens)
    raw_text, stopped = _truncate_at_stop_tokens(
        raw_text,
        stop_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
    )

    completion_tokens = len(new_ids)
    if stopped:
        finish_reason = "stop"
    elif completion_tokens >= int(max_tokens):
        finish_reason = "length"
    else:
        finish_reason = "stop"

    prompt_tokens = int(input_ids.shape[1])
    return {
        "choices": [{"text": raw_text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _hf_completions_generate_with_logfire(
    *,
    handle: _HFModelHandle,
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
            "huggingface completions [assistant]",
            _span_name="huggingface.completions [assistant]",
            _tags=["LLM"],
            model=model_name,
            llm_system="huggingface",
            llm_model_name=model_name,
            gen_ai_operation_name="completions",
            gen_ai_request_model=model_name,
        )
    except Exception:
        logfire = None
        span_cm = nullcontext()

    pred = _hf_local_generate(
        handle=handle,
        prompt=str(payload["prompt"]),
        max_tokens=int(payload["max_tokens"]),
        temperature=float(payload.get("temperature", 0.0) or 0.0),
        top_p=payload.get("top_p"),
        top_k=payload.get("top_k"),
        stop_tokens=list(payload.get("stop") or []),
        skip_special_tokens=bool(payload.get("skip_special_tokens", False)),
        include_stop_str_in_output=bool(
            payload.get("include_stop_str_in_output", True)
        ),
        seed=payload.get("seed"),
    )

    choices = pred.get("choices") or []
    raw_text = (choices[0].get("text") or "") if choices else ""
    finish_reason = (choices[0].get("finish_reason") or "") if choices else ""
    usage = pred.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

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

    with span_cm as span:
        if logfire is not None and span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("llm.model_name", model_name)
            span.set_attribute("llm.system", "huggingface")
            span.set_attribute("gen_ai.operation.name", "completions")
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.response.model", model_name)
            span.set_attribute("gen_ai.system", "huggingface")
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
            span.set_attribute("llm.token_count.total", total_tokens)
            span.set_attribute("tool_round", tool_round)
            span.set_attribute("all_messages_events", all_messages_events)
            span.set_attribute("response_data", response_data)
            span.set_attribute("output.value", response_data)

        if logfire is not None:
            logfire.info(
                "huggingface.completions.render",
                model=model_name,
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


class HuggingFaceCompletionsAgent(LLMAgent):
    """
    Tau3 agent using local Hugging Face ``generate()`` with client-side Jinja2
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
        self._hf_handle: _HFModelHandle | None = None

    def _get_model_id(self) -> str:
        llm_args = self.llm_args or {}
        model_id = str(llm_args.get("model_id") or (self.llm or "").strip()).strip()
        if not model_id:
            raise ValueError(
                "model_id must be set in llm_args (or agent_llm) for "
                "huggingface_completions_agent."
            )
        return model_id

    def _get_handle(self) -> _HFModelHandle:
        if self._hf_handle is None:
            self._hf_handle = _HFModelHandle.from_llm_args(
                self.llm_args or {},
                self.llm or "",
            )
        return self._hf_handle

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
                "jinja_template_path must be set in agent_llm_args for "
                "huggingface_completions_agent."
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
            "skip_special_tokens": skip_special_tokens,
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
            "[huggingface_completions_agent] generating model={} temperature={} seed={} "
            "prompt_len={} active_prefix={} device={}",
            model,
            temperature,
            runtime_seed,
            len(prompt_str),
            repr(active_prefix[:40]) if active_prefix else None,
            handle.model.device,
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
            pred = _hf_completions_generate_with_logfire(
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
                        "huggingface.completions.length_truncation",
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
                "huggingface_completions_agent: length-truncated turn, wrapped as "
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
                "huggingface_completions_agent returned empty response; inserting placeholder."
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
            "hf_model_id": model,
            "hf_adapter_path": (self.llm_args or {}).get("adapter_path"),
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


def create_huggingface_completions_agent(tools, domain_policy, **kwargs):
    return HuggingFaceCompletionsAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
