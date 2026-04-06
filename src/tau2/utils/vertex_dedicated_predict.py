"""HTTP POST to Vertex AI dedicated endpoint ``...:predict`` (chatCompletions JSON)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def vertex_predict_post(
    url: str,
    bearer_token: str,
    body: dict[str, Any],
    *,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """
    POST JSON to the Vertex ``:predict`` URL; return parsed JSON
    (e.g. ``{"predictions": ...}``).
    """
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Vertex endpoint predict failed HTTP {e.code} for {url!r}: {detail[:4000]}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Vertex endpoint predict failed for {url!r}: {type(e).__name__}: {e}"
        ) from e
