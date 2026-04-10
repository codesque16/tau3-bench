"""vLLM client v0: build /v1/completions body, POST, print request + raw response."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import yaml

from tau2.utils.vertex_endpoint_chat import (
    fetch_google_identity_token_for_audience,
    normalize_vertex_openai_chat_url,
    vertex_openai_chat_id_token_audience,
)

TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"
BOS = "<bos>"
SYSTEM_TURN_OPEN = "<|turn>system\n"
USER_TURN_OPEN = "<|turn>user\n"
MODEL_TURN_OPEN = "<|turn>model\n"
SYSTEM_THINK_TOKEN = "<|think|>"

# Defaults match ``examples/retail_vertex_text.yaml`` ``agent_llm_args`` (26b fp8 block).
DEFAULT_VERTEX_OPENAI_CHAT_URL = "https://gemma4-26b-fp8-gcreopmlnq-uc.a.run.app"
DEFAULT_VERTEX_OPENAI_CHAT_MODEL = "gemma4-26b-fp8"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_MESSAGE = "Say hi in one word."

DEFAULT_REQUEST_PARAMETERS: dict[str, Any] = {
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": False,
    "stop": [TURN_CLOSE, TURN_OPEN],
    "add_special_tokens": False,
    "skip_special_tokens": False,
    "include_stop_str_in_output": True,
    "timeout_s": 180,
}


def default_request_config() -> dict[str, Any]:
    """Defaults when ``request.yaml`` is missing or invalid."""
    return {
        "prompt_mode": "simple",
        "messages": [],
        "tools": [],
        "bos_token": BOS,
        "add_generation_prompt": True,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_message": DEFAULT_USER_MESSAGE,
        "enable_thinking": True,
        "vertex_openai_chat_url": DEFAULT_VERTEX_OPENAI_CHAT_URL,
        "vertex_openai_chat_model": DEFAULT_VERTEX_OPENAI_CHAT_MODEL,
        "parameters": dict(DEFAULT_REQUEST_PARAMETERS),
    }


def load_request_yaml(path: Path) -> dict[str, Any]:
    """Load ``request.yaml``; merge ``parameters`` over defaults; missing file is OK."""
    cfg = default_request_config()
    if not path.is_file():
        return cfg
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        return cfg
    for key in (
        "prompt_mode",
        "messages",
        "tools",
        "bos_token",
        "add_generation_prompt",
        "system_prompt",
        "user_message",
        "enable_thinking",
        "vertex_openai_chat_url",
        "vertex_openai_chat_model",
        "prompt",
    ):
        if key in raw and raw[key] is not None:
            cfg[key] = raw[key]
    params_in = raw.get("parameters")
    if isinstance(params_in, dict):
        cfg["parameters"] = {**cfg["parameters"], **params_in}
    return cfg


def _str_overlay(env_val: str, cfg_val: Any, default: str) -> str:
    if env_val.strip():
        return env_val.strip()
    if isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val.strip()
    return default


def _enable_thinking_overlay(cfg: dict[str, Any]) -> bool:
    env_raw = os.environ.get("VLLM_ENABLE_THINKING", "").strip()
    if env_raw:
        return _env_bool("VLLM_ENABLE_THINKING", default=True)
    if "enable_thinking" in cfg:
        return bool(cfg["enable_thinking"])
    return True


def build_full_prompt(
    *,
    system_prompt: str,
    user_message: str,
    enable_thinking: bool,
) -> str:
    """Gemma transcript: ``<bos>`` + system turn, optional ``<|think|>``, user turn, open model turn.

    Matches ``build_prompt_from_api_messages_simple`` system block in
    ``gemma4_vllm_client_simple``.
    """
    parts: list[str] = [BOS, SYSTEM_TURN_OPEN]
    if enable_thinking:
        parts.append(SYSTEM_THINK_TOKEN)
    parts.append(str(system_prompt))
    parts.append(TURN_CLOSE + "\n")
    parts.append(USER_TURN_OPEN)
    parts.append(str(user_message))
    parts.append(TURN_CLOSE + "\n")
    parts.append(MODEL_TURN_OPEN)
    return "".join(parts)


def _env_bool(key: str, *, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def build_request(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 3072,
    temperature: float = 0.0,
    stream: bool = False,
    stop: list[str] | None = None,
    add_special_tokens: bool = False,
    skip_special_tokens: bool = False,
    include_stop_str_in_output: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build JSON body for ``POST .../v1/completions`` (Gemma-style stops)."""
    body: dict[str, Any] = {
        "model": model,
        "prompt": str(prompt).rstrip("\n"),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": bool(stream),
        "stop": list(stop) if stop is not None else [TURN_CLOSE, TURN_OPEN],
        "add_special_tokens": bool(add_special_tokens),
        "skip_special_tokens": bool(skip_special_tokens),
        "include_stop_str_in_output": bool(include_stop_str_in_output),
    }
    if seed is not None:
        body["seed"] = int(seed)
    return body


def _auth_headers(base_url: str) -> dict[str, str]:
    chat_url = normalize_vertex_openai_chat_url(base_url)
    audience = vertex_openai_chat_id_token_audience(chat_url)
    token = fetch_google_identity_token_for_audience(audience)
    return {"Authorization": f"Bearer {token}"}


def post_v1_completions_get_text(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_s: int = 180,
    stream: bool = False,
    stop: list[str] | None = None,
    add_special_tokens: bool = False,
    skip_special_tokens: bool = False,
    include_stop_str_in_output: bool = True,
    seed: int | None = None,
) -> str:
    """POST ``/v1/completions`` (Vertex identity token); return ``choices[0].text`` or ``\"\"``."""
    body = build_request(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
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
        headers={**_auth_headers(base_url), "Content-Type": "application/json"},
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if isinstance(first, dict):
        t = first.get("text")
        return t if isinstance(t, str) else ""
    return ""


def call_print_raw(
    *,
    base_url: str,
    body: dict[str, Any],
    timeout_s: int = 180,
) -> requests.Response:
    """POST to ``{base_url}/v1/completions``, print request JSON, then response, return response."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    # Same serialization ``requests`` uses for ``json=`` (stdlib json.dumps).
    raw_request_body = json.dumps(body)
    print("--- raw request ---")
    print(f"POST {url}")
    print("Content-Type: application/json")
    print(raw_request_body)
    print("--- raw response ---")
    resp = requests.post(
        url,
        headers={**_auth_headers(base_url), "Content-Type": "application/json"},
        json=body,
        timeout=timeout_s,
    )
    print(f"HTTP {resp.status_code} {resp.reason}")
    print(resp.text)
    return resp


def main() -> None:
    """Smoke test: read request YAML (``VLLM_REQUEST_YAML``); env overrides YAML.

    Prompt building:

    - ``prompt_mode: simple`` — ``system_prompt`` + ``user_message`` + ``enable_thinking``
      (hand-built transcript; ends with ``<|turn>model``).
    - ``prompt_mode: template`` — ``messages`` (+ optional OpenAI-style ``tools``) rendered
      with official ``chat_template.jinja`` (set ``bos_token``, ``add_generation_prompt``).

    If ``prompt_mode`` is omitted: use ``template`` when ``messages`` is non-empty, else
    ``simple``.

    Env (wins over YAML when set): ``VLLM_BASE_URL``, ``VERTEX_OPENAI_CHAT_URL``,
    ``VLLM_MODEL``, ``VERTEX_OPENAI_CHAT_MODEL``, ``VLLM_PROMPT``, ``VLLM_SYSTEM_PROMPT``,
    ``VLLM_USER_MESSAGE``, ``VLLM_ENABLE_THINKING``.
    """
    req_path = Path(
        os.environ.get(
            "VLLM_REQUEST_YAML",
            str(Path(__file__).resolve().parent / "request.yaml"),
        )
    )
    cfg = load_request_yaml(req_path)

    base_url = (
        os.environ.get("VLLM_BASE_URL", "").strip()
        or os.environ.get("VERTEX_OPENAI_CHAT_URL", "").strip()
        or _str_overlay(
            "", cfg.get("vertex_openai_chat_url"), DEFAULT_VERTEX_OPENAI_CHAT_URL
        )
    )
    model = (
        os.environ.get("VLLM_MODEL", "").strip()
        or os.environ.get("VERTEX_OPENAI_CHAT_MODEL", "").strip()
        or _str_overlay(
            "", cfg.get("vertex_openai_chat_model"), DEFAULT_VERTEX_OPENAI_CHAT_MODEL
        )
    )
    raw_override = os.environ.get("VLLM_PROMPT", "").strip()
    if not raw_override:
        p = cfg.get("prompt")
        if isinstance(p, str) and p.strip():
            raw_override = p.strip()
    if raw_override:
        prompt = raw_override
    else:
        messages_raw = cfg.get("messages")
        has_messages = isinstance(messages_raw, list) and len(messages_raw) > 0
        pm_raw = str(cfg.get("prompt_mode") or "").strip().lower()
        if not pm_raw:
            prompt_mode = "template" if has_messages else "simple"
        else:
            prompt_mode = pm_raw
        if prompt_mode not in ("simple", "template"):
            raise SystemExit(
                f"Unknown prompt_mode {prompt_mode!r}; use 'simple' or 'template'."
            )
        use_template = prompt_mode == "template"
        if use_template:
            if not has_messages:
                raise SystemExit(
                    "``prompt_mode: template`` requires a non-empty ``messages:`` list."
                )
            from tau2.utils.vllm_jinja_client import ChatTemplateRenderer

            bos = str(cfg.get("bos_token") or BOS)
            add_gp = bool(cfg.get("add_generation_prompt", True))
            tools_raw = cfg.get("tools")
            if not isinstance(tools_raw, list):
                tools_raw = []
            renderer = ChatTemplateRenderer(bos_token=bos)
            prompt = renderer.render_from_openai_dicts(
                messages_raw,
                tools_raw,
                add_generation_prompt=add_gp,
                enable_thinking=_enable_thinking_overlay(cfg),
            )
        else:
            system_prompt = _str_overlay(
                os.environ.get("VLLM_SYSTEM_PROMPT", ""),
                cfg.get("system_prompt"),
                DEFAULT_SYSTEM_PROMPT,
            )
            user_message = _str_overlay(
                os.environ.get("VLLM_USER_MESSAGE", ""),
                cfg.get("user_message"),
                DEFAULT_USER_MESSAGE,
            )
            enable_thinking = _enable_thinking_overlay(cfg)
            prompt = build_full_prompt(
                system_prompt=system_prompt,
                user_message=user_message,
                enable_thinking=enable_thinking,
            )

    params = cfg.get("parameters")
    if not isinstance(params, dict):
        params = DEFAULT_REQUEST_PARAMETERS
    max_tokens = int(params.get("max_tokens", DEFAULT_REQUEST_PARAMETERS["max_tokens"]))
    temperature = float(
        params.get("temperature", DEFAULT_REQUEST_PARAMETERS["temperature"])
    )
    seed_raw = params.get("seed")
    seed: int | None
    if seed_raw is None:
        seed = None
    else:
        seed = int(seed_raw)
    timeout_s = int(params.get("timeout_s", DEFAULT_REQUEST_PARAMETERS["timeout_s"]))
    stream = bool(params.get("stream", DEFAULT_REQUEST_PARAMETERS["stream"]))
    stop_raw = params.get("stop", DEFAULT_REQUEST_PARAMETERS["stop"])
    stop: list[str]
    if isinstance(stop_raw, list):
        stop = [str(x) for x in stop_raw]
    else:
        stop = list(DEFAULT_REQUEST_PARAMETERS["stop"])
    add_special_tokens = bool(
        params.get("add_special_tokens", DEFAULT_REQUEST_PARAMETERS["add_special_tokens"])
    )
    skip_special_tokens = bool(
        params.get("skip_special_tokens", DEFAULT_REQUEST_PARAMETERS["skip_special_tokens"])
    )
    include_stop_str_in_output = bool(
        params.get(
            "include_stop_str_in_output",
            DEFAULT_REQUEST_PARAMETERS["include_stop_str_in_output"],
        )
    )

    body = build_request(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        stop=stop,
        add_special_tokens=add_special_tokens,
        skip_special_tokens=skip_special_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
        seed=seed,
    )
    call_print_raw(base_url=base_url, body=body, timeout_s=timeout_s)


if __name__ == "__main__":
    main()
