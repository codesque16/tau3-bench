import base64
import json
import os
import time
import types as py_types
from typing import Any, Optional

from loguru import logger

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.user.user_simulator import UserSimulator
from tau2.utils.genai_logfire import (
    genai_generate_with_logfire,
    vertex_endpoint_generate_with_logfire,
)
from tau2.utils.vertex_content_replay import (
    build_model_parts_from_raw_data,
    enrich_raw_candidates_parts_from_response,
    register_tool_names_from_parts,
    should_replay_thought_parts,
)
from tau2.utils.vertex_endpoint_chat import (
    build_openai_chat_completions_body,
    build_vertex_predict_body,
    build_vertex_predict_url,
    normalize_vertex_openai_chat_url,
    prediction_assistant_text,
    prediction_message_reasoning_text,
    prediction_tool_calls_as_tau,
    prediction_usage_with_cost,
    tau_messages_to_openai_chat,
    uses_vertex_endpoint,
    uses_vertex_openai_chat,
    vertex_openai_chat_id_token_audience,
)


class VertexUserSimulator(UserSimulator):
    """
    Half-duplex user simulator using google.genai SDK directly on Vertex AI.
    """

    def __init__(
        self,
        llm: str,
        instructions: Optional[str] = None,
        tools: Optional[list] = None,
        llm_args: Optional[dict] = None,
        persona_config=None,
    ):
        super().__init__(
            llm=llm,
            instructions=instructions,
            tools=tools,
            llm_args=llm_args,
            persona_config=persona_config,
        )
        self._genai_client = None
        self._thought_signatures: dict[str, bytes] = {}

    def _get_client(self):
        if self._genai_client is not None:
            return self._genai_client

        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for vertex_user_simulator. "
                "Install extras with: uv sync --extra voice"
            ) from e

        project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        )
        location = os.environ.get("VERTEXAI_LOCATION") or "global"
        if not project:
            raise ValueError(
                "VERTEXAI_PROJECT (or GOOGLE_CLOUD_PROJECT) must be set for vertex_user_simulator."
            )

        self._genai_client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        return self._genai_client

    def _resolve_model_name(self) -> str:
        model = (self.llm or "").strip()
        if model.startswith("vertex_ai/"):
            return model.removeprefix("vertex_ai/")
        if model.startswith("gemini/"):
            return model.removeprefix("gemini/")
        return model

    def _to_gemini_tools(self, openai_tools: list[dict[str, Any]]) -> list[Any]:
        from google.genai import types

        declarations = []
        for tool in openai_tools:
            fn = tool.get("function", {})
            name = fn.get("name")
            if not name:
                continue
            params = fn.get("parameters") or {"type": "object", "properties": {}}
            schema = types.Schema.from_json_schema(
                json_schema=types.JSONSchema.model_validate(params)
            )
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=str(fn.get("description") or ""),
                    parameters=schema,
                )
            )
        if not declarations:
            return []
        return [types.Tool(function_declarations=declarations)]

    def _resolve_thinking_config(self, types_module: Any) -> Any | None:
        reasoning_level = self.llm_args.get("reasoning_level")
        if reasoning_level is None:
            return None
        level = str(reasoning_level).strip().upper()
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if level not in allowed:
            logger.warning(
                "Invalid reasoning_level={} for vertex_user_simulator; expected one of {}",
                reasoning_level,
                sorted(allowed),
            )
            return None
        include_thoughts = bool(self.llm_args.get("include_thoughts", True))
        return types_module.ThinkingConfig(
            include_thoughts=include_thoughts,
            thinking_level=level,
        )

    def _resolve_pricing(self) -> dict[str, float] | None:
        pricing = self.llm_args.get("pricing")
        if not isinstance(pricing, dict):
            return None
        try:
            input_cost = float(pricing["input_cost_per_million"])
            output_cost = float(pricing["output_cost_per_million"])
            cached_input_cost = float(
                pricing.get("cached_input_cost_per_million", input_cost)
            )
        except Exception:
            logger.warning("Invalid pricing config for vertex_user_simulator: {}", pricing)
            return None
        return {
            "input_cost_per_million": input_cost,
            "cached_input_cost_per_million": cached_input_cost,
            "output_cost_per_million": output_cost,
        }

    def _build_usage_and_cost(self, response: Any) -> tuple[dict[str, Any] | None, float | None]:
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return None, None

        prompt_tokens = int(
            getattr(usage_metadata, "prompt_token_count", None)
            or getattr(usage_metadata, "input_token_count", None)
            or 0
        )
        completion_tokens = int(
            getattr(usage_metadata, "candidates_token_count", None)
            or getattr(usage_metadata, "output_token_count", None)
            or 0
        )
        total_tokens = int(
            getattr(usage_metadata, "total_token_count", None)
            or (prompt_tokens + completion_tokens)
        )
        cached_input_tokens = int(
            getattr(usage_metadata, "cached_content_token_count", None) or 0
        )
        reasoning_tokens = int(getattr(usage_metadata, "thoughts_token_count", None) or 0)

        usage: dict[str, int] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cached_input_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

        pricing = self._resolve_pricing()
        if pricing is None:
            return usage, None

        uncached_input_tokens = max(prompt_tokens - cached_input_tokens, 0)
        input_cost_without_cache = (
            prompt_tokens * pricing["input_cost_per_million"] / 1_000_000
        )
        input_cost_with_cache = (
            uncached_input_tokens * pricing["input_cost_per_million"]
            + cached_input_tokens * pricing["cached_input_cost_per_million"]
        ) / 1_000_000
        output_cost = completion_tokens * pricing["output_cost_per_million"] / 1_000_000
        total_cost = input_cost_with_cache + output_cost

        usage.update(
            {
                "input_cost_with_cache_usd": input_cost_with_cache,
                "input_cost_without_cache_usd": input_cost_without_cache,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
            }
        )
        return usage, total_cost

    def _to_gemini_contents(self, messages: list[Any]) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        tool_name_by_id: dict[str, str] = {}
        pending_tool_response_parts: list[Any] = []

        def flush_pending_tool_responses() -> None:
            if not pending_tool_response_parts:
                return
            contents.append(
                types.Content(role="function", parts=pending_tool_response_parts.copy())
            )
            pending_tool_response_parts.clear()

        for msg in messages:
            role = getattr(msg, "role", None)
            if role == "system":
                continue

            if role == "user":
                flush_pending_tool_responses()
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=msg.content or "")])
                )
                continue

            if role == "assistant":
                flush_pending_tool_responses()
                parts: list[Any] = []
                raw = getattr(msg, "raw_data", None)
                if should_replay_thought_parts(self.llm_args):
                    replayed = build_model_parts_from_raw_data(raw, types)
                    if replayed:
                        parts = replayed
                        register_tool_names_from_parts(parts, tool_name_by_id)
                if not parts:
                    if msg.content:
                        parts.append(types.Part(text=msg.content))
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name_by_id[tc.id] = tc.name
                            part_kwargs: dict[str, Any] = {
                                "function_call": types.FunctionCall(
                                    id=tc.id or None,
                                    name=tc.name,
                                    args=tc.arguments or {},
                                )
                            }
                            thought_signature = self._thought_signatures.get(tc.id)
                            if thought_signature is not None:
                                part_kwargs["thought_signature"] = thought_signature
                            parts.append(types.Part(**part_kwargs))
                if not parts:
                    parts = [types.Part(text="")]
                contents.append(types.Content(role="model", parts=parts))
                continue

            if isinstance(msg, ToolMessage):
                tool_name = tool_name_by_id.get(msg.id, "unknown_tool")
                payload = msg.content or ""
                try:
                    parsed = json.loads(payload) if payload.strip() else ""
                except Exception:
                    parsed = payload
                response_payload = parsed if isinstance(parsed, dict) else {"result": parsed}
                pending_tool_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=msg.id or None,
                            name=tool_name,
                            response=response_payload,
                        )
                    )
                )

        flush_pending_tool_responses()
        return contents

    def _generate_next_message(self, message, state) -> UserMessage:
        if isinstance(message, AssistantMessage) and message.is_audio:
            raise ValueError(
                "Assistant message cannot be audio. Use VoiceUserSimulator instead."
            )

        if hasattr(message, "tool_messages"):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
        elif message.has_content() or message.is_tool_call():
            state.messages.append(message)

        from google.genai import types

        model = self._resolve_model_name()

        history = state.system_messages + state.flip_roles()
        openai_tools = [tool.openai_schema for tool in self.tools] if self.tools else []

        if uses_vertex_endpoint(self.llm_args):
            api_messages = tau_messages_to_openai_chat(
                self.system_prompt,
                history,
                vertex_include_reasoning_in_request=bool(
                    self.llm_args.get("vertex_include_reasoning_in_request", False)
                ),
            )
            if uses_vertex_openai_chat(self.llm_args):
                predict_url = normalize_vertex_openai_chat_url(
                    str(self.llm_args.get("vertex_openai_chat_url") or "")
                )
                openai_model = str(
                    self.llm_args.get("vertex_openai_chat_model") or model
                ).strip()
                if not openai_model:
                    raise ValueError(
                        "Set vertex_openai_chat_model in llm_args (or a non-empty user_llm) "
                        "when using vertex_openai_chat_url."
                    )
                body = build_openai_chat_completions_body(
                    self.llm_args,
                    api_messages,
                    openai_tools or None,
                    model=openai_model,
                )
                if self.llm_args.get("vertex_openai_chat_use_access_token"):
                    id_audience = None
                else:
                    aud = str(self.llm_args.get("vertex_openai_chat_audience") or "").strip()
                    id_audience = aud or vertex_openai_chat_id_token_audience(predict_url)
            else:
                predict_url = build_vertex_predict_url(self.llm_args)
                body = build_vertex_predict_body(
                    self.llm_args, api_messages, openai_tools or None
                )
                id_audience = None
            start = time.perf_counter()
            full_payload, pred = vertex_endpoint_generate_with_logfire(
                predict_url=predict_url,
                body=body,
                api_messages=api_messages,
                openai_tools=openai_tools or None,
                model_log_name=model,
                actor="user",
                call_name="vertex_user_simulator_response",
                logfire_render_config=py_types.SimpleNamespace(
                    include_thoughts_in_history=(self.llm_args or {}).get(
                        "include_thoughts_in_history"
                    )
                ),
                id_token_audience=id_audience,
            )
            elapsed = time.perf_counter() - start
            content = prediction_assistant_text(pred)
            reasoning_only = prediction_message_reasoning_text(pred)
            tool_calls = prediction_tool_calls_as_tau(pred, requestor="user")
            usage, cost = prediction_usage_with_cost(
                pred, pricing=self._resolve_pricing()
            )
            logger.debug(f"Vertex user response: {content}")
            raw_data = {
                "vertex_predict_response": full_payload,
                "vertex_prediction": pred,
            }
            return UserMessage(
                role="user",
                content=content if content else None,
                reasoning_content=(reasoning_only or None),
                tool_calls=tool_calls or None,
                cost=cost,
                usage=usage,
                raw_data=raw_data,
                generation_time_seconds=elapsed,
            )

        client = self._get_client()
        contents = self._to_gemini_contents(history)
        gemini_tools = self._to_gemini_tools(openai_tools)

        config_kwargs: dict[str, Any] = {
            "system_instruction": self.system_prompt,
            "temperature": self.llm_args.get("temperature", 0.0),
        }
        if self.llm_args.get("max_tokens") is not None:
            config_kwargs["max_output_tokens"] = int(self.llm_args["max_tokens"])
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
        if self.llm_args.get("seed") is not None:
            config_kwargs["seed"] = int(self.llm_args["seed"])
        thinking_config = self._resolve_thinking_config(types)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config

        start = time.perf_counter()
        response = genai_generate_with_logfire(
            client=client,
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
            actor="user",
            call_name="vertex_user_simulator_response",
        )
        elapsed = time.perf_counter() - start

        content = ""
        tool_calls: list[ToolCall] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                is_thought = bool(getattr(part, "thought", False))
                if isinstance(text, str) and text and not is_thought:
                    content += text
                fn_call = getattr(part, "function_call", None)
                if fn_call is not None:
                    # Preserve API function-call ids (do not synthesize call_{idx}).
                    raw_id = getattr(fn_call, "id", None)
                    call_id = raw_id if raw_id is not None else ""
                    thought_signature = getattr(part, "thought_signature", None)
                    if isinstance(thought_signature, str):
                        try:
                            thought_signature = base64.b64decode(thought_signature)
                        except Exception:
                            thought_signature = None
                    if isinstance(thought_signature, bytes):
                        self._thought_signatures[call_id] = thought_signature
                    args = getattr(fn_call, "args", None) or {}
                    if not isinstance(args, dict):
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            name=getattr(fn_call, "name", "") or "",
                            arguments=args,
                            requestor="user",
                        )
                    )

        usage, cost = self._build_usage_and_cost(response)

        logger.debug(f"Vertex user response: {content}")
        raw_data = None
        if hasattr(response, "model_dump"):
            raw_data = response.model_dump(mode="json")
            raw_data = enrich_raw_candidates_parts_from_response(raw_data, response)

        return UserMessage(
            role="user",
            content=content if content else None,
            tool_calls=tool_calls or None,
            cost=cost,
            usage=usage,
            raw_data=raw_data,
            generation_time_seconds=elapsed,
        )
