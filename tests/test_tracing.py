"""Tests for Logfire run id / span labeling helpers."""

from tau2.data_model.simulation import TextRunConfig
from tau2.runner.tracing import build_run_id


def _minimal_text_config(**kwargs):
    base = dict(
        domain="retail",
        agent="vertex_agent",
        llm_agent="gemini-a",
        llm_user="gemini-u",
        user="user_simulator",
    )
    base.update(kwargs)
    return TextRunConfig(**base)


def test_build_run_id_config_name_only():
    c = _minimal_text_config(config_name="run_a", fresh=False, save_to="artifact")
    rid = build_run_id(c, policy_name="policy.md")
    assert rid == "[retail][policy.md][gemini-a][gemini-u][run_a]"


def test_build_run_id_fresh_appends_save_basename():
    c = _minimal_text_config(
        config_name="run_a",
        fresh=True,
        save_to="retail_full_04_02_12_00_00",
    )
    rid = build_run_id(c, policy_name="policy.md")
    assert rid == "[retail][policy.md][gemini-a][gemini-u][run_a|retail_full_04_02_12_00_00]"


def test_build_run_id_save_only_when_no_config_name():
    c = _minimal_text_config(config_name=None, fresh=False, save_to="my_save")
    rid = build_run_id(c, policy_name="policy.md")
    assert rid == "[retail][policy.md][gemini-a][gemini-u][my_save]"
