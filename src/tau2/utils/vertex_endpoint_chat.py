"""
Vertex AI dedicated endpoint (Model Garden deploy) using ``@requestFormat: chatCompletions``.

Configure via ``agent_llm_args`` / ``user_llm_args`` (same keys as the standalone
``agent_gemini`` YAML): ``vertex_endpoint_id``, optional ``vertex_http_predict_base``,
``vertex_project``, ``vertex_location``, ``vertex_http_predict_api_version``,
``vertex_endpoint_parameters``, optional ``vertex_include_reasoning_in_request`` (default
``true``: prior turns' chain-of-thought is included in ``messages[].content``; set ``false``
to send body-only text on the wire while Logfire can still show thoughts via
``reasoning_content`` / ``_logfire_mixed_content``).

**OpenAI-compatible HTTP** (e.g. vLLM on Cloud Run): set ``vertex_openai_chat_url`` to the
service base or full ``.../v1/chat/completions`` URL. Use ``vertex_openai_chat_model`` for the
JSON ``model`` field (often matches the server's ``SERVED_MODEL_NAME``). Authentication uses a
Google identity token whose audience defaults to ``scheme://host`` of that URL; override with
``vertex_openai_chat_audience``, or set ``vertex_openai_chat_use_access_token: true`` to send
the usual GCP access token instead (only if your ingress accepts it).

**Gemma 4 on vLLM (official ``chat_template.jinja`` + ``/v1/completions``)** — for Gemma-serving
URLs, the framework uses ``tau2.utils.vllm_jinja_client.call_jinja_vllm_for_vertex_openai``
(same Jinja template as HF / ``tools/test_vllm_jinja_tau3_vertex.py``). Set
``vertex_use_gemma4_vllm_client: false`` to force the older OpenAI ``/v1/chat/completions``
path. When the flag is omitted, the Gemma completions path is used if ``vertex_openai_chat_model``
contains ``"gemma"`` (case-insensitive). If ``vertex_use_gemma4_vllm_loop_state: true``, the
legacy string builder in ``gemma4_vllm_client_simple`` is used instead. Optional
``vertex_gemma_vllm_timeout_s`` overrides the HTTP timeout for that client.

For **vLLM** Gemma 4 thinking, raw ``curl`` and the Gemma 4 recipe use **top-level**
``"chat_template_kwargs": {"enable_thinking": true}`` on the JSON body — the same shape
``vertex_endpoint_parameters`` produces (no rewrite). The OpenAI Python SDK instead sends that
inside ``extra_body``; if your server only accepts that form, add an explicit ``extra_body`` key
under ``vertex_endpoint_parameters`` in YAML. Set ``TAU2_VLLM_MIRROR_CHAT_TEMPLATE_KWARGS_EXTRA_BODY=1``
to copy root ``chat_template_kwargs`` into ``extra_body`` as well (both keys present).

**Prompt injection** (Gemma ``/v1/completions`` path via ``vllm_jinja_client``): set
``chat_template_kwargs.prompt_injection`` to a string appended immediately after the model-turn
opener (``<|turn>model`` plus newline) on each new assistant turn (same idea as
``tau2.utils.vllm_jinja_client`` ``prompt_injection``). Legacy keys ``thought_injection_suffix``
and ``assistant_turn_injection_prefix`` are accepted as fallbacks and normalized to the same value.

Locally, ``google.oauth2.id_token.fetch_id_token`` often fails with user Application Default
Credentials; tau2 then runs ``gcloud auth print-identity-token`` **without** ``--audiences``
(the same as ``curl`` + ``$(gcloud auth print-identity-token)``). User accounts often cannot use
``--audiences=`` (gcloud requires a service account for that). To try audience-scoped tokens
first (e.g. with ``gcloud auth activate-service-account``), set
``TAU2_GCLOUD_IDENTITY_TOKEN_USE_AUDIENCES=1``. Disable the ``gcloud`` fallback entirely with
``TAU2_VERTEX_OPENAI_CHAT_SKIP_GCLOUD_ID_TOKEN=1``, or point ``TAU2_GCLOUD_BIN`` at a non-default
``gcloud`` binary. Tokens are cached for ``TAU2_IDENTITY_TOKEN_CACHE_S`` seconds (default 3000).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolRequestor,
    UserMessage,
)


def uses_vertex_openai_chat(llm_args: dict | None) -> bool:
    return bool(str((llm_args or {}).get("vertex_openai_chat_url") or "").strip())


def uses_vertex_predict_endpoint(llm_args: dict | None) -> bool:
    return bool(str((llm_args or {}).get("vertex_endpoint_id") or "").strip())


def uses_vertex_endpoint(llm_args: dict | None) -> bool:
    return uses_vertex_predict_endpoint(llm_args) or uses_vertex_openai_chat(llm_args)


# Logfire Model Run: replay assistant/user turns with ``type: thinking`` blocks on the *input* side.
# Attached to OpenAI-shaped dicts; stripped in ``build_vertex_predict_body`` before ``:predict``.
LOGFIRE_MIXED_CONTENT_KEY = "_logfire_mixed_content"

_OPENAI_MSG_KEYS_FOR_PREDICT = frozenset(
    {"role", "content", "name", "tool_calls", "tool_call_id"}
)


def _strip_leading_reasoning_from_merged(merged: str, reasoning_stripped: str) -> str:
    if not merged:
        return ""
    prefix = reasoning_stripped + "\n\n"
    if merged.startswith(prefix):
        return merged[len(prefix) :].lstrip()
    if merged.strip() == reasoning_stripped:
        return ""
    return merged


def openai_content_for_vertex_request(
    msg: Message,
    *,
    include_reasoning_in_request: bool,
) -> str:
    """
    ``content`` string to send on the wire in ``messages[]``.

    When ``include_reasoning_in_request`` is False and ``msg.reasoning_content`` is set,
    strip the leading reasoning block from ``msg.content`` (same prefix rule as merge).
    Logfire replay still uses full merged text via ``vertex_logfire_mixed_content``.
    """
    full = str(getattr(msg, "content", None) or "")
    reasoning = getattr(msg, "reasoning_content", None)
    if include_reasoning_in_request:
        return full
    if not isinstance(reasoning, str) or not reasoning.strip():
        return full
    return _strip_leading_reasoning_from_merged(full, reasoning.strip())


def vertex_logfire_mixed_content(
    merged_content: str | None,
    reasoning: str | None,
) -> list[dict[str, str]] | None:
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    r = reasoning.strip()
    merged = "" if merged_content is None else str(merged_content)
    body = _strip_leading_reasoning_from_merged(merged, r)
    parts: list[dict[str, str]] = [{"type": "thinking", "text": r}]
    if body:
        parts.append({"type": "text", "text": body})
    return parts


def sanitize_openai_messages_for_vertex_predict(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {k: m[k] for k in _OPENAI_MSG_KEYS_FOR_PREDICT if k in m}
        for m in messages
        if isinstance(m, dict)
    ]


def _load_chat_template_from_path(path_value: Any) -> str:
    """
    Resolve and read a chat template file path from YAML params.

    - Accepts absolute or relative paths.
    - Relative paths are resolved against current working directory.
    """
    path_raw = str(path_value or "").strip()
    if not path_raw:
        raise ValueError("chat_template_path is empty")
    p = Path(path_raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.is_file():
        raise ValueError(f"chat_template_path does not exist or is not a file: {p}")
    return p.read_text(encoding="utf-8")


def normalize_vertex_openai_chat_url(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("vertex_openai_chat_url is empty")
    if "://" not in s:
        s = "https://" + s
    s = s.rstrip("/")
    low = s.lower()
    if low.endswith("/v1/chat/completions"):
        return s
    if low.endswith("/v1"):
        return s + "/chat/completions"
    return s + "/v1/chat/completions"


def vertex_openai_chat_service_base_url(raw: str) -> str:
    """
    Service root for OpenAI-style base URLs (strip ``/v1/chat/completions`` or trailing ``/v1``).

    Used by the Gemma 4 ``/v1/completions`` client, which expects ``https://host`` without path.
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("vertex_openai_chat_url is empty")
    if "://" not in s:
        s = "https://" + s
    s = s.rstrip("/")
    low = s.lower()
    if low.endswith("/v1/chat/completions"):
        return s[: -len("/v1/chat/completions")]
    if low.endswith("/v1"):
        return s[: -len("/v1")]
    return s


def use_gemma4_vllm_completions_client(llm_args: dict | None, *, openai_model: str) -> bool:
    """
    Whether to use ``vllm_jinja_client`` (``chat_template.jinja`` + ``POST .../v1/completions``).

    Default: ``True`` when ``openai_model`` contains ``\"gemma\"`` (case-insensitive).
    Override with ``vertex_use_gemma4_vllm_client: true|false`` in ``llm_args``.

    Not used when ``vertex_openai_chat_use_access_token`` is set: that path uses OAuth access
    tokens for ``/v1/chat/completions``, while the Gemma client uses identity tokens only.
    """
    if (llm_args or {}).get("vertex_openai_chat_use_access_token"):
        return False
    raw = (llm_args or {}).get("vertex_use_gemma4_vllm_client")
    if raw is None:
        return "gemma" in (openai_model or "").lower()
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("1", "true", "yes")


def gemma4_vllm_generation_params_from_llm_args(llm_args: dict[str, Any]) -> dict[str, Any]:
    """Merge ``temperature`` / ``max_tokens`` / thinking flags like ``build_openai_chat_completions_body``."""
    temperature = float(llm_args.get("temperature", 0.0) or 0.0)
    max_tok = llm_args.get("max_tokens")
    max_tokens = int(max_tok) if max_tok is not None else 3072
    params = llm_args.get("vertex_endpoint_parameters") or {}
    if params is not None and not isinstance(params, dict):
        raise ValueError("vertex_endpoint_parameters must be a dict when set.")
    for k, v in dict(params or {}).items():
        if k == "temperature":
            temperature = float(v)
        elif k == "max_tokens":
            max_tokens = int(v)
    enable_thinking = True
    prepend_thought_channel = True
    force_model_turn_restart_after_tool_response = False
    # Appended after ``<|turn>model\n`` for Gemma vLLM completions (see module docstring).
    prompt_injection = ""
    ctk = params.get("chat_template_kwargs") if isinstance(params, dict) else None
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        enable_thinking = bool(ctk["enable_thinking"])
    if isinstance(ctk, dict) and "prepend_thought_channel" in ctk:
        prepend_thought_channel = bool(ctk["prepend_thought_channel"])
    if isinstance(ctk, dict) and "force_model_turn_restart_after_tool_response" in ctk:
        force_model_turn_restart_after_tool_response = bool(
            ctk["force_model_turn_restart_after_tool_response"]
        )
    if isinstance(ctk, dict):
        if "prompt_injection" in ctk and ctk["prompt_injection"] is not None:
            prompt_injection = str(ctk["prompt_injection"])
        elif "assistant_turn_injection_prefix" in ctk and ctk["assistant_turn_injection_prefix"] is not None:
            prompt_injection = str(ctk["assistant_turn_injection_prefix"])
        elif "thought_injection_suffix" in ctk and ctk["thought_injection_suffix"] is not None:
            prompt_injection = str(ctk["thought_injection_suffix"])
    timeout_s = int(
        llm_args.get("vertex_gemma_vllm_timeout_s")
        or os.getenv("TAU2_GEMMA4_VLLM_TIMEOUT_S")
        or "180"
    )
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "prepend_thought_channel": prepend_thought_channel,
        "force_model_turn_restart_after_tool_response": force_model_turn_restart_after_tool_response,
        "prompt_injection": prompt_injection,
        # Deprecated aliases (same string); kept for callers that still read these keys.
        "thought_injection_suffix": prompt_injection,
        "assistant_turn_injection_prefix": prompt_injection,
        "timeout": timeout_s,
    }


def resolve_runtime_seed(llm_args: dict | None) -> int | None:
    """
    Runtime seed resolution for model calls.

    - ``seed`` unset/None -> ``None`` (provider default randomness).
    - ``seed`` >= 0 -> deterministic fixed seed.
    - ``seed`` == -1 -> generate a fresh random seed per request.
    """
    raw = (llm_args or {}).get("seed")
    if raw is None:
        return None
    seed = int(raw)
    if seed == -1:
        return random.SystemRandom().randint(0, 2_147_483_647)
    return seed


def vertex_openai_chat_id_token_audience(chat_completions_url: str) -> str:
    p = urlparse(chat_completions_url)
    if not p.scheme or not p.netloc:
        raise ValueError(
            f"Cannot derive identity-token audience from vertex_openai_chat_url={chat_completions_url!r}"
        )
    return f"{p.scheme}://{p.netloc}"


# Audience-specific tokens from ``fetch_id_token`` (SA / metadata).
_id_token_by_audience: dict[str, tuple[str, float]] = {}
# One user token from ``gcloud auth print-identity-token`` (no ``--audiences``); works across URLs.
_gcloud_bare_id_token_cache: tuple[str, float] | None = None


def fetch_google_identity_token_for_audience(audience: str) -> str:
    """
    OIDC identity token for calling Cloud Run (audience is ``https://<host>`` for IAM checks).

    Uses ``google.oauth2.id_token.fetch_id_token`` when ADC supports it (service account, GCE,
    etc.). With **user** credentials, that call raises ``DefaultCredentialsError``; we then run
    ``gcloud auth print-identity-token`` with **no** ``--audiences`` flag, matching a working
    ``curl`` that uses ``$(gcloud auth print-identity-token)``.

    Successful tokens are cached for ``TAU2_IDENTITY_TOKEN_CACHE_S`` seconds (default 3000).
    """
    global _gcloud_bare_id_token_cache

    cache_ttl = float(os.getenv("TAU2_IDENTITY_TOKEN_CACHE_S", "3000"))
    now = time.monotonic()
    cached = _id_token_by_audience.get(audience)
    if cached is not None and now < cached[1]:
        return cached[0]

    from google.auth import exceptions as google_auth_exceptions
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token as google_id_token

    skip_gcloud = os.getenv("TAU2_VERTEX_OPENAI_CHAT_SKIP_GCLOUD_ID_TOKEN", "").lower() in (
        "1",
        "true",
        "yes",
    )

    adc_exc: BaseException | None = None
    try:
        tok = google_id_token.fetch_id_token(Request(), audience)
        if tok:
            _id_token_by_audience[audience] = (tok, now + cache_ttl)
            return tok
    except google_auth_exceptions.DefaultCredentialsError as e:
        adc_exc = e
    except Exception:
        raise

    if adc_exc is None:
        raise RuntimeError(
            f"google.oauth2.id_token.fetch_id_token returned an empty token "
            f"for audience={audience!r}"
        )

    if skip_gcloud:
        raise RuntimeError(
            "Could not mint a Google identity token with Application Default Credentials "
            "(user ADC cannot mint Cloud Run ID tokens). "
            "Either unset TAU2_VERTEX_OPENAI_CHAT_SKIP_GCLOUD_ID_TOKEN so tau2 can run "
            "`gcloud auth print-identity-token`, or use GOOGLE_APPLICATION_CREDENTIALS "
            "with a service account key / run on GCP."
        ) from adc_exc

    bare = _gcloud_bare_id_token_cache
    if bare is not None and now < bare[1]:
        return bare[0]

    gcloud_exe = os.getenv("TAU2_GCLOUD_BIN", "").strip() or shutil.which("gcloud")
    if not gcloud_exe:
        raise RuntimeError(
            "Could not mint a Google identity token: ADC failed and `gcloud` was not found "
            "on PATH. Install the Google Cloud SDK, run `gcloud auth login`, or set "
            "GOOGLE_APPLICATION_CREDENTIALS to a service account JSON key."
        ) from adc_exc

    use_audiences = os.getenv("TAU2_GCLOUD_IDENTITY_TOKEN_USE_AUDIENCES", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_audiences:
        r_aud = subprocess.run(
            [gcloud_exe, "auth", "print-identity-token", f"--audiences={audience}"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if r_aud.returncode == 0:
            t_aud = (r_aud.stdout or "").strip()
            if t_aud:
                _id_token_by_audience[audience] = (t_aud, now + cache_ttl)
                return t_aud

    result = subprocess.run(
        [gcloud_exe, "auth", "print-identity-token"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Could not mint a Google identity token for Cloud Run: "
            f"`gcloud auth print-identity-token` failed (exit {result.returncode}): "
            f"{detail[:1200]}. "
            "Ensure `gcloud auth login` works and your account can invoke the service."
        ) from adc_exc

    token = (result.stdout or "").strip()
    if not token:
        raise RuntimeError(
            "`gcloud auth print-identity-token` returned an empty token."
        ) from adc_exc
    exp = now + cache_ttl
    _gcloud_bare_id_token_cache = (token, exp)
    _id_token_by_audience[audience] = (token, exp)
    return token


def parse_openai_chat_completion_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level OpenAI ``/v1/chat/completions`` JSON → same inner shape as ``parse_vertex_prediction``."""
    err = payload.get("error")
    if isinstance(err, dict):
        m = err.get("message")
        raise RuntimeError(
            "OpenAI chat completion error: "
            f"type={err.get('type')} code={err.get('code')} message={m!r}"
        )
    if isinstance(err, str) and err.strip():
        raise RuntimeError(f"OpenAI chat completion error: {err!r}")
    choices = payload.get("choices")
    if isinstance(choices, list):
        return payload
    return {}


def is_openai_chat_completions_request_body(body: dict[str, Any]) -> bool:
    """True when ``body`` is a flat OpenAI chat payload (not Vertex ``instances`` wrap)."""
    return "instances" not in body and isinstance(body.get("messages"), list)


def parse_vertex_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    preds = payload.get("predictions")
    if preds is None:
        return {}
    if isinstance(preds, str):
        try:
            preds = json.loads(preds)
        except json.JSONDecodeError:
            return {}
    if isinstance(preds, dict):
        if preds.get("object") == "error":
            m = preds.get("message")
            raise RuntimeError(
                "Vertex endpoint error: "
                f"code={preds.get('code')} type={preds.get('type')} message={m!r}"
            )
        return preds
    if isinstance(preds, list) and preds:
        first = preds[0]
        if isinstance(first, str):
            try:
                first = json.loads(first)
            except json.JSONDecodeError:
                return {}
        if isinstance(first, dict):
            if first.get("object") == "error":
                m = first.get("message")
                raise RuntimeError(
                    "Vertex endpoint error: "
                    f"code={first.get('code')} type={first.get('type')} message={m!r}"
                )
            return first
    return {}


def _assistant_message_content_text(msg: dict[str, Any]) -> str:
    """OpenAI ``message.content`` only (string or multimodal list)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
        return "".join(parts)
    return ""


def _message_reasoning_chain_text(msg: dict[str, Any]) -> str:
    """
    Chain-of-thought from OpenAI-shaped ``message`` dict.

    - Raw vLLM / some clients: ``reasoning_content``
    - Vertex Model Garden Gemma 4 + vLLM: ``reasoning`` (same role; different key)
    """
    chunks: list[str] = []
    for key in ("reasoning_content", "reasoning"):
        raw = msg.get(key)
        if isinstance(raw, str):
            s = raw.strip()
            if s:
                chunks.append(s)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]
    return "\n\n".join(chunks)


def prediction_message_reasoning_text(pred: dict[str, Any]) -> str:
    """Reasoning-only string from ``predictions`` (empty if absent)."""
    choices = pred.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = c0.get("message") if isinstance(c0, dict) else {}
    if not isinstance(msg, dict):
        return ""
    return _message_reasoning_chain_text(msg)


def prediction_assistant_body_text(pred: dict[str, Any]) -> str:
    """``message.content`` only (no ``reasoning`` / ``reasoning_content``), for Logfire mixed events."""
    choices = pred.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = c0.get("message") if isinstance(c0, dict) else {}
    if not isinstance(msg, dict):
        return ""
    return _assistant_message_content_text(msg)


def prediction_assistant_text(pred: dict[str, Any]) -> str:
    """
    Assistant-visible text from a chat-completions-shaped ``predictions`` object.

    Thinking may live in ``message.reasoning`` (Vertex Gemma) or ``message.reasoning_content``
    (OpenAI / some vLLM builds); final reply stays in ``message.content``. We prepend reasoning
    so logs and downstream code see the full assistant turn.
    """
    choices = pred.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = c0.get("message") if isinstance(c0, dict) else {}
    if not isinstance(msg, dict):
        return ""
    reasoning = _message_reasoning_chain_text(msg)
    body = _assistant_message_content_text(msg)
    if reasoning and body:
        return f"{reasoning}\n\n{body}"
    if reasoning:
        return reasoning
    return body


def prediction_tool_calls_as_tau(
    pred: dict[str, Any],
    *,
    requestor: ToolRequestor = "assistant",
) -> list[ToolCall]:
    choices = pred.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = c0.get("message") if isinstance(c0, dict) else {}
    if not isinstance(msg, dict):
        return []
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    out: list[ToolCall] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn.get("name"), str) else ""
        args_raw = fn.get("arguments")
        args: dict[str, Any] = {}
        if isinstance(args_raw, str):
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        if not name:
            continue
        out.append(
            ToolCall(
                id=str(tc.get("id") or ""),
                name=name,
                arguments=args,
                requestor=requestor,
            )
        )
    return out


def prediction_usage_dict(pred: dict[str, Any]) -> dict[str, int] | None:
    usage = pred.get("usage")
    if not isinstance(usage, dict):
        return None
    pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    tt = int(usage.get("total_tokens") or (pt + ct))
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def prediction_usage_with_cost(
    pred: dict[str, Any],
    *,
    pricing: dict[str, float] | None,
) -> tuple[dict[str, Any] | None, float | None]:
    """Token usage from OpenAI-shaped ``pred`` plus optional USD cost from pricing dict."""
    base = prediction_usage_dict(pred)
    if base is None:
        return None, None
    usage: dict[str, Any] = dict(base)
    if pricing is None:
        return usage, None
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    cached_input_tokens = int(usage.get("cached_input_tokens") or 0)
    uncached_input_tokens = max(prompt_tokens - cached_input_tokens, 0)
    input_cost_without_cache = prompt_tokens * pricing["input_cost_per_million"] / 1_000_000
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


def tau_messages_to_openai_chat(
    system_prompt: str,
    messages: list[Message],
    *,
    vertex_include_reasoning_in_request: bool = True,
) -> list[dict[str, Any]]:
    """Tau conversation → OpenAI ``messages`` for ``@requestFormat: chatCompletions``.

    ``vertex_include_reasoning_in_request`` (from ``llm_args``): when False, assistant/user
    ``content`` in the payload omits chain-of-thought (body only); Logfire still gets
    ``_logfire_mixed_content`` from the full merged ``msg.content`` when present.
    """
    out: list[dict[str, Any]] = []
    sys = (system_prompt or "").strip()
    if sys:
        out.append({"role": "system", "content": sys})

    pending_tool: list[ToolMessage] = []

    def flush_tools() -> None:
        for tm in pending_tool:
            raw = tm.content
            if raw is None:
                body = ""
            elif isinstance(raw, (dict, list)):
                body = json.dumps(raw)
            else:
                body = str(raw)
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tm.id,
                    "content": body,
                }
            )
        pending_tool.clear()

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, UserMessage):
            flush_tools()
            merged = str(getattr(msg, "content", None) or "")
            req_text = openai_content_for_vertex_request(
                msg, include_reasoning_in_request=vertex_include_reasoning_in_request
            )
            umsg: dict[str, Any] = {"role": "user", "content": req_text}
            lf = vertex_logfire_mixed_content(merged, getattr(msg, "reasoning_content", None))
            if lf is not None:
                umsg[LOGFIRE_MIXED_CONTENT_KEY] = lf
            out.append(umsg)
            continue
        if isinstance(msg, AssistantMessage):
            flush_tools()
            merged = str(getattr(msg, "content", None) or "")
            api_msg: dict[str, Any] = {"role": "assistant"}
            if merged:
                api_msg["content"] = merged
            reasoning_content = getattr(msg, "reasoning_content", None)
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                api_msg["reasoning"] = reasoning_content
            tcs = getattr(msg, "tool_calls", None) or []
            if isinstance(tcs, list) and tcs:
                api_calls: list[dict[str, Any]] = []
                for tc in tcs:
                    if not isinstance(tc, ToolCall):
                        continue
                    if not tc.name:
                        continue
                    api_calls.append(
                        {
                            "id": str(tc.id or ""),
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments or {}),
                            },
                        }
                    )
                if api_calls:
                    api_msg["tool_calls"] = api_calls
            lf = vertex_logfire_mixed_content(merged, getattr(msg, "reasoning_content", None))
            if lf is not None:
                api_msg[LOGFIRE_MIXED_CONTENT_KEY] = lf
            out.append(api_msg)
            continue
        if isinstance(msg, ToolMessage):
            pending_tool.append(msg)
            continue
        if isinstance(msg, MultiToolMessage):
            pending_tool.extend(msg.tool_messages)
            continue

    flush_tools()
    return out


def build_vertex_predict_url(llm_args: dict[str, Any]) -> str:
    endpoint_id = str(llm_args.get("vertex_endpoint_id") or "").strip()
    if not endpoint_id:
        raise ValueError("vertex_endpoint_id is required for dedicated Vertex endpoint mode.")
    project = (
        str(llm_args.get("vertex_project") or "").strip()
        or (os.environ.get("VERTEXAI_PROJECT") or "").strip()
        or (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    )
    if not project:
        raise ValueError(
            "Set vertex_project in llm_args or VERTEXAI_PROJECT / GOOGLE_CLOUD_PROJECT "
            "for dedicated Vertex endpoint mode."
        )
    location = str(llm_args.get("vertex_location") or "").strip() or "us-central1"
    base = str(llm_args.get("vertex_http_predict_base") or "").strip().rstrip("/")
    if not base:
        dedicated_domain = (os.environ.get("DEDICATED_ENDPOINT_DOMAIN") or "").strip()
        if dedicated_domain:
            base = f"https://{dedicated_domain}"
    if not base:
        base = f"https://{endpoint_id}.{location}-{project}.prediction.vertexai.goog"
    api_ver = str(llm_args.get("vertex_http_predict_api_version") or "v1").strip().strip("/") or "v1"
    return (
        f"{base}/{api_ver}/projects/{project}/locations/{location}/endpoints/"
        f"{endpoint_id}:predict"
    )


def build_vertex_predict_body(
    llm_args: dict[str, Any],
    openai_messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Build ``instances[0]`` for ``@requestFormat: chatCompletions``.

    ``llm_args["temperature"]`` seeds ``temperature``; ``vertex_endpoint_parameters`` is merged
    afterward and overrides (so nested ``temperature`` / ``max_tokens`` / etc. win).

    Strips ``LOGFIRE_MIXED_CONTENT_KEY`` from each message (Logfire-only replay shape).
    """
    clean_messages = sanitize_openai_messages_for_vertex_predict(openai_messages)
    instance: dict[str, Any] = {
        "@requestFormat": "chatCompletions",
        "messages": clean_messages,
        "temperature": float(llm_args.get("temperature", 0.0) or 0.0),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    max_tok = llm_args.get("max_tokens")
    if max_tok is not None:
        instance["max_tokens"] = int(max_tok)
    params = llm_args.get("vertex_endpoint_parameters") or {}
    if params is not None and not isinstance(params, dict):
        raise ValueError("vertex_endpoint_parameters must be a dict when set.")
    chat_template_path = (
        params.get("chat_template_path") if isinstance(params, dict) else None
    )
    if chat_template_path is not None:
        instance["chat_template"] = _load_chat_template_from_path(chat_template_path)
    for k, v in dict(params or {}).items():
        if k not in ("messages", "@requestFormat", "chat_template_path"):
            instance[k] = v
    if tools:
        instance["tools"] = tools
    return {"instances": [instance]}


def _vllm_maybe_mirror_chat_template_kwargs_to_extra_body(body: dict[str, Any]) -> None:
    """Optional: duplicate root ``chat_template_kwargs`` into ``extra_body`` (OpenAI SDK shape)."""
    if os.getenv("TAU2_VLLM_MIRROR_CHAT_TEMPLATE_KWARGS_EXTRA_BODY", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    raw = body.get("chat_template_kwargs")
    if not isinstance(raw, dict):
        return
    eb_in = body.get("extra_body")
    eb: dict[str, Any] = dict(eb_in) if isinstance(eb_in, dict) else {}
    prev = eb.get("chat_template_kwargs")
    if isinstance(prev, dict):
        eb["chat_template_kwargs"] = {**prev, **raw}
    else:
        eb["chat_template_kwargs"] = dict(raw)
    body["extra_body"] = eb


def build_openai_chat_completions_body(
    llm_args: dict[str, Any],
    openai_messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    *,
    model: str,
) -> dict[str, Any]:
    """
    Flat JSON for ``POST .../v1/chat/completions`` (OpenAI-compatible servers).

    ``vertex_endpoint_parameters`` merges in like ``build_vertex_predict_body``, excluding keys
    that would overwrite ``messages`` or ``model``.

    For vLLM/OpenAI SDK compatibility, ``chat_template`` / ``chat_template_kwargs`` are sent
    under ``extra_body``:

    - ``chat_template_path`` (file path) -> ``extra_body.chat_template`` (file contents)
    - ``chat_template`` -> ``extra_body.chat_template``
    - ``chat_template_kwargs`` -> ``extra_body.chat_template_kwargs``
    """
    clean_messages = sanitize_openai_messages_for_vertex_predict(openai_messages)
    body: dict[str, Any] = {
        "model": model,
        "messages": clean_messages,
        "temperature": float(llm_args.get("temperature", 0.0) or 0.0),
    }
    max_tok = llm_args.get("max_tokens")
    if max_tok is not None:
        body["max_tokens"] = int(max_tok)
    params = llm_args.get("vertex_endpoint_parameters") or {}
    if params is not None and not isinstance(params, dict):
        raise ValueError("vertex_endpoint_parameters must be a dict when set.")
    extra_body_in = params.get("extra_body") if isinstance(params, dict) else None
    extra_body: dict[str, Any] = dict(extra_body_in) if isinstance(extra_body_in, dict) else {}
    chat_template_path = params.get("chat_template_path") if isinstance(params, dict) else None
    if chat_template_path is not None:
        extra_body["chat_template"] = _load_chat_template_from_path(chat_template_path)
    chat_template_inline = params.get("chat_template") if isinstance(params, dict) else None
    if chat_template_inline is not None:
        extra_body["chat_template"] = str(chat_template_inline)
    ctk = params.get("chat_template_kwargs") if isinstance(params, dict) else None
    if isinstance(ctk, dict):
        prev_ctk = extra_body.get("chat_template_kwargs")
        if isinstance(prev_ctk, dict):
            extra_body["chat_template_kwargs"] = {**prev_ctk, **ctk}
        else:
            extra_body["chat_template_kwargs"] = dict(ctk)
        # Compatibility: some vLLM deployments read chat_template_kwargs at root.
        body["chat_template_kwargs"] = dict(ctk)
    if "chat_template" in extra_body and "chat_template" not in body:
        # Compatibility mirror for servers that expect root-level chat_template.
        body["chat_template"] = extra_body["chat_template"]
    if extra_body:
        body["extra_body"] = extra_body
    reserved = frozenset({"messages", "model", "@requestFormat"})
    for k, v in dict(params or {}).items():
        if k not in reserved and k not in (
            "chat_template_path",
            "chat_template",
            "chat_template_kwargs",
            "extra_body",
        ):
            body[k] = v
    if tools:
        body["tools"] = tools
    return body


def tool_round_from_openai_messages(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "tool")
