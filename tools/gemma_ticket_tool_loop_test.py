"""Run Gemma /v1/completions with tool loop until no tool calls.

This script:
1) Starts from a raw request transcript text file.
2) Calls /v1/completions repeatedly.
3) Executes emitted tool calls against a local mock retail tool runtime.
4) Appends <|tool_response> blocks back into the transcript.
5) Stops when the latest model completion emits no tool calls.
6) Dumps a readable JSON trace and final parsed messages.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tau2.utils.gemma4_vllm_client_simple import (
    build_v1_completions_request,
    extract_completion_text,
    parse_request_response_to_messages,
    post_v1_completions_simple,
)

TOOL_CALL_RE = re.compile(r"<\|tool_call>call:(\w+)\{([\s\S]*?)\}<tool_call\|>")
TOOL_RESPONSE_RE = re.compile(r"<\|tool_response>response:(\w+)\{value:<\|\"\|>([\s\S]*?)<\|\"\|>\}<tool_response\|>")
THOUGHT_BLOCK_RE = re.compile(r"<\|channel>thought\s*\n?([\s\S]*?)<channel\|>")
TURN_CLOSE = "<turn|>"
MODEL_TURN_OPEN = "<|turn>model\n"
DEFAULT_THOUGHT_PRIMING = (
    "<|channel>thought\n"
    "I need to first start by mapping the current state to a node which policies are currently relvant for the state the tool hints attached to the node. I need to first retrieve the current context and cite the current node , the relevant policies in that node , and y other relvant rtules and then reason on top of it for the next steps and present it in proper yaml format\n"
    "```yaml\n"
    "context:\n"
    "    current_node:\n"
    "    relevant_node_policies:\n"
    "    tool_hints:\n"
    "   relevant_rules:\n"
    "\n"
    "next_steps:\n"
    "    node_transition: <required/not>\n"
    "    next_node: <if node transition required then which node>\n"
    "    reasoning: <tool calls next steps, rationale behind node transition>\n"
    "```\n"
    "<channel|>\n"
    "```yaml\n"
    "context:\n"
    "    current_node:\n"
)


def _parse_args(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads("{" + s + "}")
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    normalized = s.replace('<|"|>', '"')
    try:
        obj = json.loads("{" + normalized + "}")
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    out: dict[str, Any] = {}
    for m in re.finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', s):
        out[m.group(1)] = m.group(2)
    for m in re.finditer(r'(\w+):([^,}<|\n"]+)', s):
        k, v = m.group(1), m.group(2).strip()
        if k not in out and v:
            out[k] = v
    return out


def extract_tool_calls_from_chunk(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, m in enumerate(TOOL_CALL_RE.finditer(text or "")):
        name = m.group(1)
        args = _parse_args(m.group(2))
        out.append(
            {
                "id": f"call_{i}_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return out


class MockRetailRuntime:
    def __init__(self) -> None:
        self.user = {
            "user_id": "yusuf_rossi_9620",
            "first_name": "Yusuf",
            "last_name": "Rossi",
            "zip": "19122",
            "orders": ["#W1001001"],
        }
        self.order = {
            "order_id": "#W1001001",
            "status": "delivered",
            "payment_method_id": "credit_card_1111",
            "items": [
                {"item_id": "2001", "name": "cleaner", "price": 19.99},
                {"item_id": "2002", "name": "headphone", "price": 129.00},
                {"item_id": "2003", "name": "smart watch", "price": 199.00},
            ],
            "returns": [],
        }
        self.products = {
            "tshirt": {
                "product_id": "P-TSHIRT",
                "items": [
                    {"item_id": "T1", "variant": "S / Blue", "available": True},
                    {"item_id": "T2", "variant": "M / Blue", "available": True},
                    {"item_id": "T3", "variant": "L / Blue", "available": False},
                    {"item_id": "T4", "variant": "S / Black", "available": True},
                    {"item_id": "T5", "variant": "M / Black", "available": True},
                    {"item_id": "T6", "variant": "L / Black", "available": True},
                ],
            }
        }

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(args)
        except Exception as e:
            return {"error": str(e)}

    def tool_find_user_id_by_name_zip(self, args: dict[str, Any]) -> dict[str, Any]:
        if (
            str(args.get("first_name") or "").lower() == "yusuf"
            and str(args.get("last_name") or "").lower() == "rossi"
            and str(args.get("zip") or "") == "19122"
        ):
            return {"value": self.user["user_id"]}
        return {"error": "not found"}

    def tool_get_user_details(self, args: dict[str, Any]) -> dict[str, Any]:
        uid = str(args.get("user_id") or "")
        if uid != self.user["user_id"]:
            return {"error": "user not found"}
        return {
            "user_id": uid,
            "first_name": self.user["first_name"],
            "last_name": self.user["last_name"],
            "zip": self.user["zip"],
            "orders": self.user["orders"],
        }

    def tool_get_order_details(self, args: dict[str, Any]) -> dict[str, Any]:
        oid = str(args.get("order_id") or "")
        if oid != self.order["order_id"]:
            return {"error": "order not found"}
        return self.order

    def tool_list_all_product_types(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"products": [{"name": "tshirt", "product_id": "P-TSHIRT"}]}

    def tool_get_product_details(self, args: dict[str, Any]) -> dict[str, Any]:
        pid = str(args.get("product_id") or "")
        if pid != "P-TSHIRT":
            return {"error": "product not found"}
        return self.products["tshirt"]

    def tool_return_delivered_order_items(self, args: dict[str, Any]) -> dict[str, Any]:
        oid = str(args.get("order_id") or "")
        if oid != self.order["order_id"]:
            return {"error": "order not found"}
        item_ids = [str(x) for x in (args.get("item_ids") or [])]
        self.order["returns"].extend(item_ids)
        self.order["status"] = "return requested"
        return {"ok": True, "returned_item_ids": item_ids, "order_id": oid}

    def tool_calculate(self, args: dict[str, Any]) -> dict[str, Any]:
        expr = str(args.get("expression") or "").strip()
        if not re.match(r"^[0-9\.\+\-\*\/\(\) ]+$", expr):
            return {"error": "invalid expression"}
        value = eval(expr, {"__builtins__": {}}, {})
        return {"value": value}


def _format_tool_response(name: str, payload: dict[str, Any]) -> str:
    val = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return (
        "<|tool_response>response:"
        + name
        + "{value:<|\"|>"
        + val
        + "<|\"|>}<tool_response|>"
    )


def _extract_tool_responses(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, m in enumerate(TOOL_RESPONSE_RE.finditer(text or "")):
        name = m.group(1)
        raw_val = m.group(2).strip()
        parsed: Any = raw_val
        try:
            parsed = json.loads(raw_val)
        except Exception:
            parsed = raw_val
        out.append({"id": f"toolresp_{i}_{name}", "name": name, "value": parsed})
    return out


def _extract_text_part(text: str) -> str:
    s = str(text or "")
    s = THOUGHT_BLOCK_RE.sub("", s)
    s = TOOL_CALL_RE.sub("", s)
    s = TOOL_RESPONSE_RE.sub("", s)
    s = s.replace(MODEL_TURN_OPEN, "")
    s = s.replace("<|turn>user\n", "")
    s = s.replace("<|turn>system\n", "")
    s = s.replace(TURN_CLOSE, "")
    return s.strip()


def _augment_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        content = str(mm.get("content") or "")
        mm["tool_responses"] = _extract_tool_responses(content)
        mm["text_part"] = _extract_text_part(content)
        out.append(mm)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--request-file", required=True)
    ap.add_argument("--out-json", default="tau3-bench-fork/tools/gemma_ticket_tool_loop_trace.json")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=int, default=180)
    ap.add_argument(
        "--restart-model-turn-after-tools",
        action="store_true",
        default=True,
        help="After attaching tool responses, close turn and start a fresh model thought turn.",
    )
    ap.add_argument(
        "--model-thought-prefix",
        default=DEFAULT_THOUGHT_PRIMING,
        help="Thought priming inserted after starting the fresh model turn.",
    )
    ap.add_argument(
        "--model-thought-prefix-file",
        default=None,
        help="Optional file path; if set, overrides --model-thought-prefix.",
    )
    ap.add_argument(
        "--out-rich-log",
        default="tau3-bench-fork/tools/gemma_ticket_tool_loop_rich.log",
    )
    args = ap.parse_args()
    if args.model_thought_prefix_file:
        args.model_thought_prefix = Path(args.model_thought_prefix_file).read_text()

    request_text = Path(args.request_file).read_text()
    response_text = ""
    runtime = MockRetailRuntime()
    steps: list[dict[str, Any]] = []
    rich_log_lines: list[str] = []

    for step in range(args.max_steps):
        prompt = request_text + response_text
        body = build_v1_completions_request(
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=None,
        )
        payload = post_v1_completions_simple(
            base_url=args.base_url,
            body=body,
            timeout_s=args.timeout_s,
        )
        chunk = extract_completion_text(payload)
        response_text += chunk
        chunk_calls = extract_tool_calls_from_chunk(chunk)

        rich_log_lines.append(f"\n===== STEP {step} =====")
        rich_log_lines.append(f"PROMPT_LEN={len(prompt)}")
        rich_log_lines.append("PROMPT_BEGIN")
        rich_log_lines.append(prompt)
        rich_log_lines.append("PROMPT_END")
        rich_log_lines.append("REQUEST_BODY_BEGIN")
        rich_log_lines.append(json.dumps(body, indent=2, ensure_ascii=True))
        rich_log_lines.append("REQUEST_BODY_END")
        rich_log_lines.append("RAW_COMPLETION_CHUNK_BEGIN")
        rich_log_lines.append(chunk)
        rich_log_lines.append("RAW_COMPLETION_CHUNK_END")

        executed: list[dict[str, Any]] = []
        for tc in chunk_calls:
            fn = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            result = runtime.call(fn, fn_args)
            tool_response_block = _format_tool_response(fn, result)
            response_text += tool_response_block
            executed.append({"name": fn, "args": fn_args, "result": result})
            rich_log_lines.append("TOOL_EXECUTION")
            rich_log_lines.append(
                json.dumps(
                    {
                        "tool_name": fn,
                        "tool_args": fn_args,
                        "tool_result": result,
                        "tool_response_block": tool_response_block,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )

        if chunk_calls and args.restart_model_turn_after_tools:
            # Force explicit turn boundary after tool responses, then reopen model
            # with thought prefix so generation doesn't continue inside tool block.
            response_text += TURN_CLOSE + MODEL_TURN_OPEN + args.model_thought_prefix
            rich_log_lines.append("TURN_RESTART_AFTER_TOOLS")
            rich_log_lines.append(TURN_CLOSE + MODEL_TURN_OPEN + args.model_thought_prefix)

        parsed = _augment_messages(
            parse_request_response_to_messages(request_text, response_text)
        )
        rich_log_lines.append(f"CUMULATIVE_RESPONSE_LEN={len(response_text)}")
        rich_log_lines.append("CUMULATIVE_RESPONSE_BEGIN")
        rich_log_lines.append(response_text)
        rich_log_lines.append("CUMULATIVE_RESPONSE_END")
        rich_log_lines.append("PARSED_MESSAGES_BEGIN")
        rich_log_lines.append(json.dumps(parsed, indent=2, ensure_ascii=True))
        rich_log_lines.append("PARSED_MESSAGES_END")
        steps.append(
            {
                "step": step,
                "completion_chunk": chunk,
                "tool_calls_in_chunk": chunk_calls,
                "executed_tools": executed,
                "parsed_messages": parsed,
            }
        )

        # Stop when latest model completion chunk contains no tool calls.
        if not chunk_calls:
            break

    final_messages = _augment_messages(
        parse_request_response_to_messages(request_text, response_text)
    )
    out = {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "steps": len(steps),
        },
        "request_text": request_text,
        "response_text": response_text,
        "final_messages": final_messages,
        "steps_trace": steps,
    }

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    rich_path = Path(args.out_rich_log)
    if not rich_path.is_absolute():
        rich_path = Path.cwd() / rich_path
    rich_path.parent.mkdir(parents=True, exist_ok=True)
    rich_path.write_text("\n".join(rich_log_lines))

    print(f"Wrote trace: {out_path}")
    print(f"Wrote rich log: {rich_path}")


if __name__ == "__main__":
    main()
