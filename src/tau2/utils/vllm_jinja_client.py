"""
LLM Client using a custom Jinja2 chat template.

Supports:
- Multi-turn conversation history
- Tool/function calling
- Tool responses
- Thinking mode (enable_thinking)
- System / developer prompts
- Image / audio / video content blocks
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import requests
from jinja2 import Environment

from tau2.utils.vertex_endpoint_chat import (
    fetch_google_identity_token_for_audience,
    gemma4_vllm_generation_params_from_llm_args,
    normalize_vertex_openai_chat_url,
    resolve_runtime_seed,
    vertex_openai_chat_id_token_audience,
)

# Gemma / vLLM OpenAI ``/v1/completions`` stop tokens (same convention as tau3 Vertex runs).
_TURN_OPEN = "<|turn>"
_TURN_CLOSE = "<turn|>"


def _vertex_openai_identity_headers(base_url: str) -> dict[str, str]:
    chat_url = normalize_vertex_openai_chat_url(base_url)
    audience = vertex_openai_chat_id_token_audience(chat_url)
    token = fetch_google_identity_token_for_audience(audience)
    return {"Authorization": f"Bearer {token}"}


def _vllm_openai_completions_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    stream: bool,
    stop: list[str] | None,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    include_stop_str_in_output: bool,
    seed: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": str(prompt).rstrip("\n"),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": bool(stream),
        "stop": list(stop) if stop is not None else [_TURN_CLOSE, _TURN_OPEN],
        "add_special_tokens": bool(add_special_tokens),
        "skip_special_tokens": bool(skip_special_tokens),
        "include_stop_str_in_output": bool(include_stop_str_in_output),
    }
    if top_p is not None:
        body["top_p"] = float(top_p)
    if seed is not None:
        body["seed"] = int(seed)
    return body


def _vllm_openai_completions_post(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    timeout_s: int = 180,
    stream: bool = False,
    stop: list[str] | None = None,
    add_special_tokens: bool = False,
    skip_special_tokens: bool = False,
    include_stop_str_in_output: bool = True,
    seed: int | None = None,
    log_raw_json: bool = False,
) -> tuple[str, dict[str, Any]]:
    """POST ``/v1/completions``; return ``(choices[0].text, usage_dict)``."""
    body = _vllm_openai_completions_body(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        stop=stop,
        add_special_tokens=add_special_tokens,
        skip_special_tokens=skip_special_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
        seed=seed,
    )
    url = f"{base_url.rstrip('/')}/v1/completions"
    resp = requests.post(
        url,
        headers={
            **_vertex_openai_identity_headers(base_url),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    if log_raw_json:
        _print_vllm_raw_response_json(
            payload=payload,
            status_code=resp.status_code,
            reason=resp.reason,
            url=url,
            model=model,
        )
    if not isinstance(payload, dict):
        return "", {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", usage
    first = choices[0]
    if isinstance(first, dict):
        t = first.get("text")
        return (t if isinstance(t, str) else ""), usage
    return "", usage


def _post_vllm_openai_completions_text(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    timeout_s: int = 180,
    stream: bool = False,
    stop: list[str] | None = None,
    add_special_tokens: bool = False,
    skip_special_tokens: bool = False,
    include_stop_str_in_output: bool = True,
    seed: int | None = None,
    log_raw_json: bool = False,
) -> str:
    """POST ``/v1/completions`` with Google identity token; return ``choices[0].text``."""
    text, _ = _vllm_openai_completions_post(
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_s=timeout_s,
        stream=stream,
        stop=stop,
        add_special_tokens=add_special_tokens,
        skip_special_tokens=skip_special_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
        seed=seed,
        log_raw_json=log_raw_json,
    )
    return text


def _print_vllm_raw_response_json(
    *,
    payload: Any,
    status_code: int,
    reason: str,
    url: str,
    model: str,
) -> None:
    """Pretty-print raw completion JSON with muted metadata."""
    try:
        from rich.console import Console
        from rich.json import JSON as RichJSON
        from rich.panel import Panel

        console = Console()
        meta = (
            f"[dim]POST[/dim] [cyan]{url}[/cyan]\n"
            f"[dim]status[/dim] [green]{status_code} {reason}[/green]    "
            f"[dim]model[/dim] [magenta]{model}[/magenta]"
        )
        console.print(Panel.fit(meta, title="Raw vLLM Response", border_style="blue"))
        console.print(RichJSON.from_data(payload))
    except Exception:
        print("\n--- raw vLLM response json ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_chat_template() -> str:
    """Load official Gemma chat template from sibling ``chat_template.jinja``."""
    template_path = Path(__file__).resolve().parent / "chat_template.jinja"
    return template_path.read_text(encoding="utf-8")


CHAT_TEMPLATE = _load_chat_template()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolParameter:
    """Describes one parameter of a tool."""
    type: str                          # 'string' | 'integer' | 'number' | 'boolean' | 'object' | 'array'
    description: str = ""
    enum: list[str] | None = None
    properties: dict[str, "ToolParameter"] | None = None
    required: list[str] | None = None
    items: dict | None = None
    nullable: bool = False

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type}
        if self.description:
            d["description"] = self.description
        if self.enum is not None:
            d["enum"] = self.enum
        if self.properties is not None:
            d["properties"] = {k: v.to_dict() for k, v in self.properties.items()}
        if self.required is not None:
            d["required"] = self.required
        if self.items is not None:
            d["items"] = self.items
        if self.nullable:
            d["nullable"] = True
        return d


@dataclass
class Tool:
    """Represents a callable tool (function)."""
    name: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str] = field(default_factory=list)
    response: dict | None = None          # Optional response schema
    handler: Callable | None = None       # Python callable invoked locally

    def to_dict(self) -> dict:
        func: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {k: v.to_dict() for k, v in self.parameters.items()},
                "required": self.required,
            },
        }
        if self.response:
            func["response"] = self.response
        return {"type": "function", "function": func}


def schema_to_tool_parameter(schema: dict[str, Any]) -> ToolParameter:
    """
    Build :class:`ToolParameter` from a JSON-schema-like dict (same rules as
    ``tools/test_vllm_jinja_tau3_vertex.py``).
    """
    t = str(schema.get("type") or "string")
    enum = schema.get("enum")
    enum_list = [str(v) for v in enum] if isinstance(enum, list) else None

    properties_raw = schema.get("properties")
    props: dict[str, ToolParameter] | None = None
    if isinstance(properties_raw, dict):
        props = {
            str(k): schema_to_tool_parameter(v)
            for k, v in properties_raw.items()
            if isinstance(v, dict)
        }

    items_raw = schema.get("items")
    items = items_raw if isinstance(items_raw, dict) else None
    required_raw = schema.get("required")
    required = [str(x) for x in required_raw] if isinstance(required_raw, list) else None

    return ToolParameter(
        type=t,
        description=str(schema.get("description") or ""),
        enum=enum_list,
        properties=props,
        required=required,
        items=items,
        nullable=bool(schema.get("nullable", False)),
    )


@dataclass
class Message:
    """A single conversation turn."""
    role: str                                   # 'system' | 'user' | 'assistant' | 'developer'
    content: str | list[dict] | None = None     # text or content blocks
    tool_calls: list[dict] | None = None        # outgoing tool calls (assistant)
    tool_responses: list[dict] | None = None    # incoming tool results (user/tool turn)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role}
        d["content"] = self.content or ""
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_responses:
            d["tool_responses"] = self.tool_responses
        # Ensure absent keys still exist so the template never KeyErrors
        d.setdefault("tool_calls", [])
        d.setdefault("tool_responses", [])
        return d


def openai_chat_dicts_to_messages(messages: list[dict[str, Any]]) -> list[Message]:
    """
    Convert OpenAI chat API messages (with interleaved ``role: tool``) into
    :class:`Message` list for :class:`ChatTemplateRenderer` / :class:`LLMClient`.
    """
    out: list[Message] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if not isinstance(m, dict):
            i += 1
            continue
        role = str(m.get("role") or "")
        if role == "tool":
            i += 1
            continue
        if role == "assistant":
            base_tc = list(m.get("tool_calls") or [])
            id_to_name: dict[str, str] = {}
            tool_calls_clean: list[dict[str, Any]] = []
            for tc in base_tc:
                if not isinstance(tc, dict):
                    continue
                tid = str(tc.get("id") or "")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = str(fn.get("name") or "unknown_tool")
                if tid:
                    id_to_name[tid] = name
                fn_copy = dict(fn)
                raw_args = fn_copy.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                        if isinstance(parsed, dict):
                            fn_copy["arguments"] = parsed
                    except Exception:
                        pass
                tool_calls_clean.append({**tc, "function": fn_copy})
            out.append(
                Message(
                    role="assistant",
                    content=m.get("content") if m.get("content") is not None else "",
                    tool_calls=tool_calls_clean or None,
                )
            )
            i += 1
            tool_responses: list[dict[str, Any]] = []
            while i < n and str((messages[i] or {}).get("role") or "") == "tool":
                tm = messages[i]
                tid = str(tm.get("tool_call_id") or "")
                tname = id_to_name.get(tid, "unknown_tool")
                raw_c = tm.get("content")
                if isinstance(raw_c, (dict, list)):
                    resp_body: Any = raw_c
                elif isinstance(raw_c, str):
                    try:
                        resp_body = json.loads(raw_c)
                    except Exception:
                        resp_body = {"result": raw_c}
                else:
                    resp_body = {"result": raw_c}
                if not isinstance(resp_body, dict):
                    resp_body = {"value": resp_body}
                tool_responses.append({"name": tname, "response": resp_body})
                i += 1
            if tool_responses:
                out.append(
                    Message(
                        role="user",
                        content="",
                        tool_responses=tool_responses,
                    )
                )
            continue
        out.append(
            Message(
                role=role,
                content=m.get("content") if m.get("content") is not None else "",
                tool_calls=list(m.get("tool_calls") or []) or None,
                tool_responses=list(m.get("tool_responses") or []) or None,
            )
        )
        i += 1
    return out


# ---------------------------------------------------------------------------
# Template renderer
# ---------------------------------------------------------------------------

def normalize_openai_messages_for_template(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure keys exist so ``chat_template.jinja`` never raises on missing fields."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        d = dict(m)
        d["role"] = str(d.get("role") or "user")
        if d.get("content") is None:
            d["content"] = ""
        d.setdefault("tool_calls", [])
        d.setdefault("tool_responses", [])
        out.append(d)
    return out


class ChatTemplateRenderer:
    """Renders messages + tools into a prompt string via the Jinja2 template."""

    def __init__(self, template_str: str = CHAT_TEMPLATE, bos_token: str = "<|begin_of_text|>"):
        self.env = Environment()
        self.template = self.env.from_string(template_str)
        self.bos_token = bos_token

    def render(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        raw_messages = [m.to_dict() for m in messages]
        raw_tools = [t.to_dict() for t in tools] if tools else []
        return self.template.render(
            messages=raw_messages,
            tools=raw_tools,
            bos_token=self.bos_token,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def render_from_openai_dicts(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        """Render from OpenAI-style ``messages`` / ``tools`` dicts (e.g. YAML-loaded)."""
        norm_messages = normalize_openai_messages_for_template(messages)
        raw_tools = list(tools) if tools else []
        return self.template.render(
            messages=norm_messages,
            tools=raw_tools,
            bos_token=self.bos_token,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

class ResponseParser:
    """
    Parses raw model output that uses the custom token format:
      <|tool_call>call:func_name{arg:val}<tool_call|>
    Tool arguments: unwrap ``<|"|>…<|"|>`` to the literal text inside, split ``key: value`` and
    top-level commas/brackets as in the model string — no int/bool coercion or other reinterpretation.
    """

    TOOL_CALL_PATTERN = re.compile(
        r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL
    )
    QUOTED_VALUE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
    # Full thought channel block (Gemma 4: private reasoning before user-visible text).
    _THOUGHT_STRIP_RE = re.compile(
        r"<\|channel>thought\s*\n?[\s\S]*?<channel\|>", re.MULTILINE
    )
    _THOUGHT_CAPTURE_RE = re.compile(
        r"<\|channel>thought\s*\n?([\s\S]*?)<channel\|>", re.MULTILINE
    )

    @staticmethod
    def strip_thought_blocks(text: str | None) -> str:
        """User-visible assistant text: strip thoughts and control tokens."""
        if not text:
            return ""
        s = ResponseParser._THOUGHT_STRIP_RE.sub("", str(text))
        if "<|channel>thought" in s:
            # Truncated generation: no closing ``<channel|>`` — drop the tail.
            s = s.split("<|channel>thought", 1)[0]
        # Remove non-user-visible control tokens that can leak into content traces.
        for tok in ("<|tool_response>", "<tool_response|>", "<|turn>", "<turn|>"):
            s = s.replace(tok, "")
        return s.strip()

    @staticmethod
    def extract_thought_blocks(text: str | None) -> str:
        """Extract thought-channel text only (without channel tokens)."""
        if not text:
            return ""
        s = str(text)
        blocks = [m.group(1).strip() for m in ResponseParser._THOUGHT_CAPTURE_RE.finditer(s)]
        if blocks:
            return "\n\n".join(b for b in blocks if b).strip()
        # Truncated thought (no closing token) fallback.
        if "<|channel>thought" in s:
            tail = s.split("<|channel>thought", 1)[1].strip()
            return tail
        return ""

    def parse(self, raw_output: str) -> dict:
        tool_calls = []
        text_parts = []
        last_end = 0

        for m in self.TOOL_CALL_PATTERN.finditer(raw_output):
            # text before this tool call
            text_before = raw_output[last_end : m.start()].strip()
            if text_before:
                text_parts.append(text_before)

            func_name = m.group(1)
            args_raw = m.group(2)
            args = self._parse_args(args_raw)
            tool_calls.append({"function": {"name": func_name, "arguments": args}})
            last_end = m.end()

        # remaining text
        tail = raw_output[last_end:].strip()
        if tail:
            text_parts.append(tail)

        return {
            "content": "\n".join(text_parts) or None,
            "tool_calls": tool_calls,
        }

    def _parse_args(self, raw: str) -> dict:
        """Decode ``{...}`` body: quoted spans → literal text; same comma/bracket structure as in ``raw``."""
        result: dict[str, Any] = {}
        placeholders: list[str] = []

        def replace_quoted(m: re.Match) -> str:
            placeholders.append(m.group(1))
            return f"__PLACEHOLDER_{len(placeholders) - 1}__"

        cleaned = self.QUOTED_VALUE.sub(replace_quoted, raw)

        for pair in self._split_top_level(cleaned, ","):
            if ":" not in pair:
                continue
            key, _, val = pair.partition(":")
            key = key.strip()
            val = val.strip()
            for i, ph in enumerate(placeholders):
                val = val.replace(f"__PLACEHOLDER_{i}__", ph)
            result[key] = self._parse_arg_value(val)

        return result

    @staticmethod
    def _parse_arg_value(val: str) -> Any:
        """Return one scalar string, or one list of strings per top-level comma inside ``[...]``."""
        s = val.strip()
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            inner = s[1:-1]
            if not inner.strip():
                return []
            parts = ResponseParser._split_top_level(inner, ",")
            return [e.strip() for e in parts]
        return s

    @staticmethod
    def _split_top_level(s: str, delimiter: str) -> list[str]:
        """Split on delimiter only at depth 0 (ignoring braces/brackets)."""
        parts, depth, current = [], 0, []
        for ch in s:
            if ch in "{[":
                depth += 1
            elif ch in "]}":
                depth -= 1
            if ch == delimiter and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Stateful LLM client that:
    - Maintains a conversation history
    - Manages a tool registry
    - Renders prompts via the Jinja2 chat template
    - Sends rendered prompts to an HTTP inference endpoint
    - Parses responses and auto-invokes registered tool handlers
    """

    def __init__(
        self,
        endpoint: str,
        *,
        bos_token: str = "<|begin_of_text|>",
        enable_thinking: bool = False,
        prompt_injection: str = "",
        system_prompt: str | None = None,
        http_headers: dict | None = None,
        vertex_openai_model: str | None = None,
    ):
        """
        :param endpoint: Base URL for inference. If ``vertex_openai_model`` is set, this must be
            the Vertex vLLM OpenAI base (same as ``vertex_openai_chat_url`` in tau3 configs);
            requests go to ``POST {endpoint}/v1/completions`` with a Google identity token.
        :param vertex_openai_model: When set, use Vertex vLLM OpenAI ``/v1/completions``
            (identity token) instead of the legacy generic JSON ``/generate`` shape.
        """
        self.endpoint = endpoint
        self._vertex_openai_model = vertex_openai_model
        self.enable_thinking = enable_thinking
        self.prompt_injection = str(prompt_injection or "")
        # Suffix appended by ``build_prompt`` when injection is active (for parsing: prefix + API text).
        self._last_prompt_injection_applied: str = ""
        self.renderer = ChatTemplateRenderer(bos_token=bos_token)
        self.parser = ResponseParser()
        self.history: list[Message] = []
        self.tools: list[Tool] = []
        self._tool_map: dict[str, Tool] = {}
        self.http_headers = {"Content-Type": "application/json", **(http_headers or {})}

        if system_prompt:
            self.history.append(Message(role="system", content=system_prompt))

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        """Register a tool. If tool.handler is set it will be auto-invoked."""
        self.tools.append(tool)
        self._tool_map[tool.name] = tool

    def register_tools(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register_tool(t)

    def clear_registered_tools(self) -> None:
        self.tools = []
        self._tool_map = {}

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        self.history.append(Message(role="user", content=text))

    def add_user_content(self, content_blocks: list[dict]) -> None:
        """Add a multi-modal user message (text + image/audio/video blocks)."""
        self.history.append(Message(role="user", content=content_blocks))

    def clear_history(self, keep_system: bool = True) -> None:
        if keep_system and self.history and self.history[0].role == "system":
            self.history = [self.history[0]]
        else:
            self.history = []

    def replace_history_from_openai_chat(self, messages: list[dict[str, Any]]) -> None:
        """
        Replace non-system turns from OpenAI ``messages`` (same folding as the Jinja
        integration test: interleaved ``role: tool`` → ``user`` + ``tool_responses``).
        Preserves the initial ``system`` message from :meth:`clear_history` /
        ``__init__`` (call :meth:`clear_history` first if you need a clean slate).
        """
        self.clear_history(keep_system=True)
        self.history.extend(openai_chat_dicts_to_messages(messages))

    # ------------------------------------------------------------------
    # Core send / receive
    # ------------------------------------------------------------------

    @staticmethod
    def _has_open_model_turn(prompt: str) -> bool:
        """True when prompt currently ends inside an assistant/model turn."""
        s = str(prompt or "")
        model_start = s.rfind("<|turn>model\n")
        if model_start < 0:
            return False
        # If a turn close appears after the most recent model start, model turn is not open.
        next_close = s.find("<turn|>", model_start)
        return next_close < 0

    def _prompt_injection_suffix_for_prompt(self, prompt: str) -> tuple[str, str]:
        """
        Return ``(full_prompt, applied_suffix)``.

        ``applied_suffix`` is exactly what we append before the model generates; the API
        returns only the continuation, so parsing should use ``applied_suffix + raw_text``.
        """
        inj = self.prompt_injection
        if not inj:
            return prompt, ""
        if not self._has_open_model_turn(prompt):
            return prompt, ""
        if prompt.endswith(inj):
            return prompt, ""
        if inj.startswith("\n") or prompt.endswith("\n"):
            suffix = inj
        else:
            suffix = "\n" + inj
        return prompt + suffix, suffix

    def build_prompt(self, add_generation_prompt: bool = True) -> str:
        prompt = self.renderer.render(
            messages=self.history,
            tools=self.tools or None,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )
        full, applied = self._prompt_injection_suffix_for_prompt(prompt)
        self._last_prompt_injection_applied = applied
        return full

    def _call_endpoint(self, prompt: str, **generation_kwargs) -> str:
        """
        POST the rendered prompt to the inference endpoint.

        If ``vertex_openai_model`` was set in ``__init__``, POSTs to Vertex vLLM
        ``POST {endpoint}/v1/completions`` with a Google identity token (see
        ``_post_vllm_openai_completions_text``).

        Otherwise expects a JSON response like: ``{"generated_text": "..."}``.
        Override this method to integrate your own inference server.
        """
        if self._vertex_openai_model is not None:
            g = generation_kwargs
            return _post_vllm_openai_completions_text(
                base_url=self.endpoint,
                model=self._vertex_openai_model,
                prompt=prompt,
                max_tokens=int(g.get("max_tokens", 512)),
                temperature=float(g.get("temperature", 0.0)),
                top_p=float(g["top_p"]) if g.get("top_p") is not None else None,
                timeout_s=int(g.get("timeout_s", 180)),
                stream=bool(g.get("stream", False)),
                stop=g.get("stop"),
                add_special_tokens=bool(g.get("add_special_tokens", False)),
                skip_special_tokens=bool(g.get("skip_special_tokens", False)),
                include_stop_str_in_output=bool(
                    g.get("include_stop_str_in_output", True)
                ),
                seed=g.get("seed"),
                log_raw_json=bool(g.get("log_raw_json", False)),
            )

        import urllib.request

        payload = json.dumps({"prompt": prompt, **generation_kwargs}).encode()
        req = urllib.request.Request(self.endpoint, data=payload, headers=self.http_headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())

        # Support common response shapes
        if isinstance(data, list):
            return data[0].get("generated_text", "")
        if isinstance(data, dict):
            return data.get("generated_text") or data.get("text") or data.get("response", "")
        return str(data)

    def _run_tool(self, call: dict) -> dict:
        """Execute a single tool call dict and return a tool_response dict."""
        func_name = call["function"]["name"]
        args = call["function"].get("arguments", {})
        tool = self._tool_map.get(func_name)

        if tool is None:
            return {"name": func_name, "response": {"error": f"Unknown tool: {func_name}"}}

        if tool.handler is None:
            return {"name": func_name, "response": {"error": "No handler registered"}}

        if isinstance(args, dict):
            args = self._coerce_tool_args(tool, args)

        try:
            result = tool.handler(**args)
            if not isinstance(result, dict):
                result = {"result": result}
            return {"name": func_name, "response": result}
        except Exception as exc:
            return {"name": func_name, "response": {"error": str(exc)}}

    def _coerce_tool_args(self, tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
        """Coerce parsed tool args to declared ToolParameter types."""
        out: dict[str, Any] = {}
        for key, value in args.items():
            schema = tool.parameters.get(key)
            out[key] = self._coerce_value_by_param(value, schema)
        return out

    def _coerce_value_by_param(self, value: Any, param: ToolParameter | None) -> Any:
        """Best-effort coercion for common scalar + container tool arg types."""
        if param is None:
            return value

        typ = str(param.type or "").lower()
        if typ == "string":
            return str(value)
        if typ == "integer":
            try:
                return int(value)
            except Exception:
                return value
        if typ == "number":
            try:
                return float(value)
            except Exception:
                return value
        if typ == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in {"true", "1", "yes", "on"}:
                    return True
                if v in {"false", "0", "no", "off"}:
                    return False
            return bool(value)
        if typ == "array":
            if isinstance(value, str):
                parsed = self._try_parse_json_like_container(value)
                if parsed is not None:
                    value = parsed
            if not isinstance(value, list):
                return value
            item_schema = (
                self._tool_param_from_json_schema(param.items)
                if isinstance(param.items, dict)
                else None
            )
            return [self._coerce_value_by_param(v, item_schema) for v in value]
        if typ == "object":
            if isinstance(value, str):
                parsed = self._try_parse_json_like_container(value)
                if parsed is not None:
                    value = parsed
            if not isinstance(value, dict):
                return value
            props = param.properties or {}
            return {
                k: self._coerce_value_by_param(v, props.get(k))
                for k, v in value.items()
            }
        return value

    @staticmethod
    def _try_parse_json_like_container(value: str) -> Any | None:
        """Parse list/object encoded as a string, e.g. ``\"[1,2]\"`` or ``\"{...}\"``."""
        s = value.strip()
        if not s:
            return None
        if not (
            (s.startswith("[") and s.endswith("]"))
            or (s.startswith("{") and s.endswith("}"))
        ):
            return None
        try:
            return json.loads(s)
        except Exception:
            # Fallback for single-quoted pseudo-JSON.
            try:
                return json.loads(s.replace("'", '"'))
            except Exception:
                return None

    def _tool_param_from_json_schema(self, schema: dict[str, Any]) -> ToolParameter:
        """Construct ToolParameter from a JSON-schema-like dict (subset used here)."""
        props_raw = schema.get("properties")
        props: dict[str, ToolParameter] | None = None
        if isinstance(props_raw, dict):
            props = {
                str(k): self._tool_param_from_json_schema(v)
                for k, v in props_raw.items()
                if isinstance(v, dict)
            }
        required_raw = schema.get("required")
        required = [str(x) for x in required_raw] if isinstance(required_raw, list) else None
        items = schema.get("items") if isinstance(schema.get("items"), dict) else None
        enum_raw = schema.get("enum")
        enum = [str(x) for x in enum_raw] if isinstance(enum_raw, list) else None
        return ToolParameter(
            type=str(schema.get("type") or "string"),
            description=str(schema.get("description") or ""),
            enum=enum,
            properties=props,
            required=required,
            items=items,
            nullable=bool(schema.get("nullable", False)),
        )

    def chat(
        self,
        user_message: str | None = None,
        *,
        auto_invoke_tools: bool = True,
        max_tool_rounds: int = 50,
        verbose: bool = True,
        trace_events: list[dict[str, Any]] | None = None,
        **generation_kwargs,
    ) -> str:
        """
        Send a user message, get a response. Optionally auto-invoke tools.

        Returns the final assistant text content.

        If ``trace_events`` is a list, one record is appended per **model completion**
        (each generation normally ends at ``<turn|>`` / stop — i.e. one template turn).
        All ``tool_calls`` from that completion and their ``tool_responses`` sit in the
        same record. A later HTTP round (next open model turn) appends a **new** record.
        """
        if user_message is not None:
            self.add_user_message(user_message)

        visible = ""
        for round_idx in range(max_tool_rounds):
            prompt = self.build_prompt()
            if verbose:
                print(f"\n{'='*60}")
                print("RENDERED PROMPT:")
                print(prompt)
                print(f"{'='*60}\n")

            raw_response = self._call_endpoint(prompt, **generation_kwargs)
            inj_prefix = self._last_prompt_injection_applied or ""
            # Model completes the prompt after our injected prefix; parse the effective turn.
            effective_assistant_text = inj_prefix + raw_response
            parsed = self.parser.parse(effective_assistant_text)
            # User-visible text only: never return raw CoT (matches Gemma guidance to strip
            # thoughts between turns; here we strip for display and for stored history).
            content_raw = parsed.get("content")
            thought = self.parser.extract_thought_blocks(content_raw)
            visible = self.parser.strip_thought_blocks(content_raw)
            if trace_events is not None:
                trace_events.append(
                    {
                        "round": round_idx,
                        "prompt": prompt,
                        "prompt_injection_prefix": inj_prefix,
                        "raw_response_text": raw_response,
                        "effective_text_for_parsing": effective_assistant_text,
                        "assistant_content_raw": content_raw,
                        "assistant_thought": thought,
                        "assistant_content_visible": visible,
                        "tool_calls": parsed.get("tool_calls") or [],
                        "tool_responses": [],
                    }
                )

            # Record assistant turn
            assistant_msg = Message(
                role="assistant",
                content=visible or None,
                tool_calls=parsed["tool_calls"] or None,
            )
            self.history.append(assistant_msg)

            # If no tool calls, we're done
            if not parsed["tool_calls"] or not auto_invoke_tools:
                return visible or ""

            # Auto-invoke tool handlers and add results
            tool_responses = [self._run_tool(tc) for tc in parsed["tool_calls"]]
            if trace_events is not None:
                trace_events[-1]["tool_responses"] = tool_responses
            self.history.append(Message(
                role="user",
                tool_responses=tool_responses,
                content=None,
            ))

        return visible or ""

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def print_history(self) -> None:
        print("\n" + "="*60 + " HISTORY " + "="*60)
        for i, msg in enumerate(self.history):
            print(f"[{i}] role={msg.role}")
            if msg.content:
                print(f"    content: {str(msg.content)[:200]}")
            if msg.tool_calls:
                print(f"    tool_calls: {msg.tool_calls}")
            if msg.tool_responses:
                print(f"    tool_responses: {msg.tool_responses}")
        print("="*129 + "\n")


def register_openai_function_schemas_on_client(
    client: LLMClient,
    openai_tools: list[dict[str, Any]] | None,
    handlers_by_name: Mapping[str, Callable[..., Any]] | None = None,
) -> None:
    """
    Register OpenAI-style tool dicts (``type`` / ``function`` / ``parameters``) on
    ``client``, same shape as ``tools/test_vllm_jinja_tau3_vertex.py`` retail registration.
    Handlers are optional (tau3 executes tools in the environment).
    """
    client.clear_registered_tools()
    hb = handlers_by_name or {}
    for schema in openai_tools or []:
        if not isinstance(schema, dict):
            continue
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        params = fn.get("parameters")
        param_props: dict[str, Any] = {}
        required_list: list[str] = []
        if isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                param_props = {
                    str(k): v for k, v in props.items() if isinstance(v, dict)
                }
            rq = params.get("required")
            if isinstance(rq, list):
                required_list = [str(x) for x in rq]
        param_defs = {
            str(k): schema_to_tool_parameter(v) for k, v in param_props.items()
        }
        client.register_tool(
            Tool(
                name=name,
                description=str(fn.get("description") or name),
                parameters=param_defs,
                required=required_list,
                handler=hb.get(name),
            )
        )


def _vertex_gemma_bos_token_from_llm_args(llm_args: dict[str, Any]) -> str:
    p = llm_args.get("vertex_endpoint_parameters")
    if isinstance(p, dict):
        b = p.get("bos_token")
        if isinstance(b, str) and b.strip():
            return b.strip()
    b2 = llm_args.get("gemma_bos_token")
    if isinstance(b2, str) and b2.strip():
        return b2.strip()
    return "<bos>"


def call_jinja_vllm_for_vertex_openai(
    *,
    base_url: str,
    model: str,
    llm_args: dict[str, Any],
    api_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    runtime_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Vertex vLLM Gemma path: same rendering and tool registration as
    ``tools/test_vllm_jinja_tau3_vertex.py`` — :class:`LLMClient` ``build_prompt`` +
    ``POST /v1/completions`` + :class:`ResponseParser`.

    Return shape matches ``gemma4_vllm_client_simple.call_gemma4_vllm_for_vertex_openai_simple``
    for Logfire / :func:`vertex_endpoint_generate_with_logfire`.

    When ``vertex_use_gemma4_vllm_loop_state`` is true, defers to the legacy simple client.
    """
    if bool((llm_args or {}).get("vertex_use_gemma4_vllm_loop_state")):
        from tau2.utils.gemma4_vllm_client_simple import (
            call_gemma4_vllm_for_vertex_openai_simple,
        )

        return call_gemma4_vllm_for_vertex_openai_simple(
            base_url=base_url,
            model=model,
            llm_args=llm_args,
            api_messages=api_messages,
            tools=tools,
            runtime_seed=runtime_seed,
        )

    gp = gemma4_vllm_generation_params_from_llm_args(llm_args)
    seed = (
        runtime_seed if runtime_seed is not None else resolve_runtime_seed(llm_args)
    )
    prompt_injection = str(gp.get("prompt_injection") or "")

    stripped_api: list[dict[str, Any]] = []
    for m in api_messages or []:
        if not isinstance(m, dict):
            continue
        stripped_api.append(
            {k: v for k, v in m.items() if k != "_logfire_mixed_content"}
        )

    system_parts: list[str] = []
    idx = 0
    while idx < len(stripped_api) and str(stripped_api[idx].get("role") or "") == "system":
        c = stripped_api[idx].get("content")
        if isinstance(c, str):
            system_parts.append(c)
        elif c is not None:
            system_parts.append(str(c))
        idx += 1
    system_prompt = "\n\n".join(p for p in system_parts if str(p).strip()).strip() or None

    client = LLMClient(
        endpoint=base_url.rstrip("/"),
        vertex_openai_model=model,
        bos_token=_vertex_gemma_bos_token_from_llm_args(llm_args),
        enable_thinking=bool(gp["enable_thinking"]),
        prompt_injection=prompt_injection,
        system_prompt=system_prompt,
    )
    register_openai_function_schemas_on_client(client, tools)
    client.replace_history_from_openai_chat(stripped_api[idx:])

    params = llm_args.get("vertex_endpoint_parameters") or {}
    if not isinstance(params, dict):
        params = {}
    gen_kwargs: dict[str, Any] = {
        "max_tokens": int(gp["max_tokens"]),
        "temperature": float(gp["temperature"]),
        "timeout_s": int(gp["timeout"]),
        "add_special_tokens": bool(params.get("add_special_tokens", False)),
        "skip_special_tokens": bool(params.get("skip_special_tokens", False)),
        "include_stop_str_in_output": bool(
            params.get("include_stop_str_in_output", True)
        ),
        "stream": bool(params.get("stream", False)),
        "seed": seed,
        "log_raw_json": bool(
            llm_args.get("vertex_gemma_vllm_log_raw_json", False)
        ),
    }
    if params.get("top_p") is not None:
        gen_kwargs["top_p"] = float(params["top_p"])
    if params.get("stop") is not None:
        gen_kwargs["stop"] = params["stop"]
    elif tools:
        gen_kwargs["stop"] = ["<turn|>", "<|turn>", "<|tool_response>"]

    prompt = client.build_prompt()
    raw, usage = _vllm_openai_completions_post(
        base_url=base_url,
        model=model,
        prompt=prompt,
        **gen_kwargs,
    )
    inj_prefix = client._last_prompt_injection_applied or ""
    effective = inj_prefix + raw
    parser = client.parser
    parsed = parser.parse(effective)
    content_raw = parsed.get("content")
    thought = ResponseParser.extract_thought_blocks(content_raw)
    visible = ResponseParser.strip_thought_blocks(content_raw)

    tool_calls_openai: list[dict[str, Any]] = []
    for tidx, tc in enumerate(parsed.get("tool_calls") or []):
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        fn_name = str(fn.get("name") or "")
        if not fn_name:
            continue
        args = fn.get("arguments")
        if not isinstance(args, dict):
            args = {}
        tool_calls_openai.append(
            {
                "id": f"call_{tidx}_{fn_name}",
                "type": "function",
                "function": {"name": fn_name, "arguments": json.dumps(args)},
            }
        )

    msg: dict[str, Any] = {"role": "assistant", "content": visible or ""}
    if str(thought or "").strip():
        msg["reasoning_content"] = str(thought)
    if tool_calls_openai:
        msg["tool_calls"] = tool_calls_openai

    body = _vllm_openai_completions_body(
        model=model,
        prompt=prompt,
        max_tokens=int(gen_kwargs["max_tokens"]),
        temperature=float(gen_kwargs["temperature"]),
        top_p=gen_kwargs.get("top_p"),
        stream=bool(gen_kwargs["stream"]),
        stop=gen_kwargs.get("stop"),
        add_special_tokens=bool(gen_kwargs["add_special_tokens"]),
        skip_special_tokens=bool(gen_kwargs["skip_special_tokens"]),
        include_stop_str_in_output=bool(gen_kwargs["include_stop_str_in_output"]),
        seed=gen_kwargs.get("seed"),
    )

    full_payload: dict[str, Any] = {
        "_tau2_gemma4_vllm_jinja_client": True,
        "_tau2_gemma4_vllm_client_simple": False,
        "_tau2_gemma4_vllm_request_url": f"{base_url.rstrip('/')}/v1/completions",
        "_tau2_gemma4_vllm_request_payload": body,
        "_tau2_gemma4_vllm_transcript": {
            "prompt": prompt,
            "completion": raw,
            "prompt_plus_completion": prompt + raw,
        },
        "_tau2_gemma4_vllm_response_after_chat_template": {
            "reasoning_content": msg.get("reasoning_content", ""),
            "content": msg.get("content", ""),
            "tool_calls": tool_calls_openai,
            "conversation_array": [
                {
                    "role": "assistant",
                    "content": content_raw or "",
                    "thought": thought,
                    "tool_calls": tool_calls_openai,
                }
            ],
        },
        "object": "text_completion",
        "choices": [{"text": raw, "finish_reason": "stop", "index": 0}],
        "usage": usage,
        "model": model,
    }
    pred: dict[str, Any] = {
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": usage,
    }
    return full_payload, pred


# ---------------------------------------------------------------------------
# Example usage / smoke-test (no real network call)
# ---------------------------------------------------------------------------

def _demo_render_only():
    """
    Renders a sample prompt without making any HTTP calls.
    Useful to verify the template output is correct.
    """
    client = LLMClient(
        endpoint="http://localhost:8000/generate",
        system_prompt="You are a helpful assistant.",
        enable_thinking=False,
    )

    # Define a weather tool
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "location": ToolParameter(type="string", description="City name"),
            "unit": ToolParameter(
                type="string",
                description="Temperature unit",
                enum=["celsius", "fahrenheit"],
            ),
        },
        required=["location"],
        handler=lambda location, unit="celsius": {
            "temperature": 22,
            "unit": unit,
            "condition": "Sunny",
            "location": location,
        },
    )
    client.register_tool(weather_tool)

    # Add a user message
    client.add_user_message("What's the weather in Paris?")

    # Render the prompt (no HTTP call)
    prompt = client.build_prompt()
    print("=" * 70)
    print("RENDERED PROMPT OUTPUT:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    # Simulate a model response with a tool call
    simulated_response = (
        '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>'
    )
    parsed = ResponseParser().parse(simulated_response)
    print("\nPARSED RESPONSE:")
    print(json.dumps(parsed, indent=2))

    # Record the assistant turn manually
    client.history.append(Message(
        role="assistant",
        content=parsed["content"],
        tool_calls=parsed["tool_calls"],
    ))

    # Run the tool handler
    tool_response = client._run_tool(parsed["tool_calls"][0])
    print("\nTOOL RESPONSE:")
    print(json.dumps(tool_response, indent=2))

    # Add tool response and render next turn
    client.history.append(Message(role="user", tool_responses=[tool_response]))
    client.add_user_message("")  # empty follow-up so template closes turn cleanly

    prompt2 = client.build_prompt()
    print("\n" + "=" * 70)
    print("PROMPT AFTER TOOL RESPONSE:")
    print("=" * 70)
    print(prompt2)

    client.print_history()


if __name__ == "__main__":
    _demo_render_only()