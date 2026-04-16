"""Unit tests for Gemma 4 vLLM tool-call extraction (balanced braces + JSON args)."""

import json

from tau2.utils.gemma4_vllm_client import (
    _extract_tool_calls,
    _matching_brace_end,
    _parse_args,
    apply_chat_template,
    conversation_array_from_request_response,
    merge_prompt_model_slot_with_completion,
    parse_raw_completion,
)


def test_matching_brace_nested_json() -> None:
    inner = '{"order_id": "#W9911714", "nested": {"x": 1}}'
    blob = "{" + inner + "}"
    end = _matching_brace_end(blob, 0)
    assert end is not None
    assert end == len(blob) - 1
    assert blob[1:end] == inner


def test_parse_args_json_object() -> None:
    assert _parse_args('{"product_id": "1656367028"}') == {"product_id": "1656367028"}


def test_parse_args_gemma_wrapped_values() -> None:
    raw = 'first_name:<|"|>Ethan<|"|>,last_name:<|"|>Garcia<|"|>,zip:<|"|>80280<|"|>'
    d = _parse_args(raw)
    assert d["first_name"] == "Ethan"
    assert d["zip"] == "80280"


def test_extract_tool_call_double_brace_json() -> None:
    text = (
        '<|tool_call>call:get_product_details{{"product_id": "1656367028"}}<tool_call|>'
    )
    calls = _extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_product_details"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "product_id": "1656367028",
    }


def test_extract_tool_call_empty_json() -> None:
    text = "<|tool_call>call:get_order_details{{}}<tool_call|>"
    calls = _extract_tool_calls(text)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_extract_tool_call_gemma_native_no_double_brace() -> None:
    text = (
        '<|tool_call>call:get_user_details{user_id:<|"|>ethan_garcia_1261<|"|>}<tool_call|>'
    )
    calls = _extract_tool_calls(text)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "user_id": "ethan_garcia_1261",
    }


def test_extract_multiple_tool_calls() -> None:
    text = (
        '<|tool_call>call:get_order_details{{"order_id": "#W1"}}<tool_call|>'
        '<|tool_call>call:get_item_details{{"item_id": "4107812777"}}<tool_call|>'
    )
    calls = _extract_tool_calls(text)
    assert len(calls) == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"order_id": "#W1"}
    assert json.loads(calls[1]["function"]["arguments"]) == {"item_id": "4107812777"}


def test_parse_raw_completion_prefers_tool_calls() -> None:
    raw = (
        "<|channel>thought\nplan<channel|>\n"
        '<|tool_call>call:calculate{{"expression": "1+1"}}<tool_call|>'
    )
    out = parse_raw_completion(raw)
    assert out["tool_calls"]
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {
        "expression": "1+1",
    }


def test_parse_raw_completion_applies_assistant_prefix() -> None:
    raw = "<|channel>thought\nplan<channel|>\nDone."
    out = parse_raw_completion(
        raw,
        assistant_turn_injection_prefix="ASSISTANT_PREFIX: ",
    )
    assert out["content"] == "ASSISTANT_PREFIX: Done."


def test_parse_raw_completion_applies_thought_injection_to_reasoning() -> None:
    raw = "<|channel>thought\nplan<channel|>\nDone."
    out = parse_raw_completion(
        raw,
        thought_injection_suffix="INJECTED_CONTEXT",
    )
    assert out["reasoning_content"].startswith("INJECTED_CONTEXT")


def test_apply_chat_template_appends_thought_prefill_from_assistant_prefix() -> None:
    p = apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        enable_thinking=True,
        assistant_turn_injection_prefix="PREFILL_LINE\n<channel|>\n",
    )
    assert "<|channel>thought\nPREFILL_LINE\n<channel|>\n" in p


def test_apply_chat_template_can_disable_prepended_thought_channel() -> None:
    p = apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        enable_thinking=True,
        prepend_thought_channel=False,
        assistant_turn_injection_prefix="PREFILL_LINE\n<channel|>\n",
    )
    assert "<|channel>thought\nPREFILL_LINE\n<channel|>\n" not in p
    assert "PREFILL_LINE\n<channel|>\n" in p


def test_apply_chat_template_does_not_inject_thought_while_continuing_tool_response() -> None:
    messages = [
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": '{"temp":72}'},
    ]
    p = apply_chat_template(
        messages,
        enable_thinking=True,
        assistant_turn_injection_prefix="PREFILL_LINE\n<channel|>\n",
    )
    assert p.endswith("<|tool_response>response:get_weather{value:<|\"|>{\"temp\":72}<|\"|>}<tool_response|>")
    assert "PREFILL_LINE\n<channel|>\n" not in p
    assert not p.endswith("<|turn>model\n")


def test_apply_chat_template_force_restart_after_tool_response_injects_prefill() -> None:
    messages = [
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": '{"temp":72}'},
    ]
    p = apply_chat_template(
        messages,
        enable_thinking=True,
        force_model_turn_restart_after_tool_response=True,
        assistant_turn_injection_prefix="PREFILL_LINE\n<channel|>\n",
    )
    assert "<|tool_response>response:get_weather" in p
    assert "<turn|>\n<|turn>model\n<|channel>thought\nPREFILL_LINE\n<channel|>\n" in p


def test_parse_raw_completion_thought_like_assistant_prefix_to_reasoning() -> None:
    raw = "<|channel>thought\nmore<channel|>\nDone."
    inj = "PREFILL\n<channel|>\n"
    out = parse_raw_completion(raw, assistant_turn_injection_prefix=inj)
    assert out["reasoning_content"].startswith("PREFILL")
    assert out["content"] == "Done."


def test_parse_raw_completion_strips_thought_prefill_from_visible_content() -> None:
    prefill = (
        "<|channel>thought\n"
        "PRIMING_LINE_1\n"
        "PRIMING_LINE_2\n"
        "<channel|>\n"
    )
    raw = "PRIMING_LINE_1\nPRIMING_LINE_2\nVisible output"
    out = parse_raw_completion(raw, assistant_turn_injection_prefix=prefill)
    assert out["content"] == "Visible output"


def test_merge_prompt_slot_includes_prefill_in_reasoning_for_tools() -> None:
    prompt = (
        "<bos><|turn>user\nhi<turn|>\n"
        "<|turn>model\n<|channel>thought\nINJECTED_BLOCK\n<channel|>\n``yaml\nk:"
    )
    raw = " v\n<channel|>\n<|tool_call>call:calculate{{\"expression\": \"1\"}}<tool_call|>"
    virtual = merge_prompt_model_slot_with_completion(prompt, raw)
    out = parse_raw_completion(
        virtual,
        thought_injection_suffix="",
        assistant_turn_injection_prefix="",
    )
    assert "INJECTED_BLOCK" in out["reasoning_content"]
    assert "k:" not in out["reasoning_content"]
    assert "k:" in out["content"] or "v" in out["content"]
    assert out["tool_calls"]


def test_tool_round_does_not_leak_thought_into_assistant_content() -> None:
    prompt = (
        "<bos><|turn>system\nsys<turn|>\n"
        "<|turn>user\nu<turn|>\n"
        "<|turn>model\n<|channel>thought\nINJ\n<channel|>\n```yaml\ncontext:\n  current_node:"
    )
    raw = (
        " START\n  relevant_node_policies: []\n```"
        "\n<|channel>thought\n<channel|>"
        '<|tool_call>call:find_user_id_by_name_zip{{"first_name":"Yusuf","last_name":"Rossi","zip":"19122"}}<tool_call|>'
        "<|tool_response>"
    )
    virtual = merge_prompt_model_slot_with_completion(prompt, raw)
    out = parse_raw_completion(
        virtual,
        thought_injection_suffix="",
        assistant_turn_injection_prefix="",
    )
    assert out["tool_calls"]
    assert "START" in out["content"]
    assert "INJ" in out["reasoning_content"]
    assert "START" not in out["reasoning_content"]


def test_tool_round_splits_multiple_thought_blocks_from_visible_text() -> None:
    raw = (
        "<|turn>model\n"
        "<|channel>thought\nINJECTED\n<channel|>\n"
        "VISIBLE_1\n"
        "<|channel>thought\n\n<channel|>\n"
        '<|tool_call>call:find_user_id_by_name_zip{{"first_name":"Yusuf","last_name":"Rossi","zip":"19122"}}<tool_call|>'
    )
    out = parse_raw_completion(
        raw,
        thought_injection_suffix="",
        assistant_turn_injection_prefix="",
    )
    assert out["tool_calls"]
    assert "INJECTED" in out["reasoning_content"]
    assert out["content"] == "VISIBLE_1"


def test_conversation_array_from_request_response_openai_shape() -> None:
    request = (
        "<bos><|turn>system\nSYS<turn|>\n"
        "<|turn>user\nUSR<turn|>\n"
        "<|turn>model\n<|channel>thought\nplan<channel|>\n"
        '<|tool_call>call:calculate{{"expression":"1+1"}}<tool_call|>'
    )
    response = (
        "<|tool_response>response:calculate{value:<|\"|>2<|\"|>}<tool_response|>"
        "<|channel>thought\nnext<channel|>\nVisible done<turn|>"
    )
    arr = conversation_array_from_request_response(request, response)
    assert arr[0]["role"] == "system"
    assert arr[1]["role"] == "user"
    assert arr[2]["role"] == "assistant"
    assert arr[2]["tool_calls"]
    assert arr[2]["tool_responses"]
    assert "Visible done" in arr[2]["text_part"]
    assert "plan" in arr[2]["thought"]
    assert "<|tool_call>call:calculate" in arr[2]["content"]
    assert "<|tool_response>response:calculate" in arr[2]["content"]
    assert "Visible done" in arr[2]["content"]
