"""--tree モードの記録・引用抽出・回答検証のテスト（API キー不要）。"""

import json
import sys
from dataclasses import dataclass

import pytest
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.test import TestModel

from app.agent import (
    TreeDeps,
    browse_tree_with_deps,
    extract_citations,
    extract_lectures,
    main,
    tree_agent,
    validate_tree_answer,
)


@dataclass
class _DummyCtx:
    deps: TreeDeps


def test_browse_tree_records_listed_and_read():
    deps = TreeDeps()
    ctx = _DummyCtx(deps=deps)

    # セクション一覧を開くと候補は listed にだけ入る
    sec_res = json.loads(browse_tree_with_deps(ctx, "lec01-main-s029"))
    assert "lec01-main-p034" in deps.listed
    assert "lec01-main-p034" not in deps.read
    assert sec_res["children"][0]["read"] is False
    assert "cite_after_read" in sec_res["children"][0]

    # ページIDで開くと read に入る
    page_res = json.loads(browse_tree_with_deps(ctx, "lec01-main-p034"))
    assert deps.read["lec01-main-p034"] == "第1回・PDF p.34"
    assert page_res["node"]["read"] is True
    assert page_res["node"]["cite"] == "第1回・PDF p.34"


def test_extract_citations_range_and_list():
    assert extract_citations("説明（第1回・PDF p.34-35, p.52）") == [
        (1, 34),
        (1, 35),
        (1, 52),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("第1回・PDF p.34、p.35", [(1, 34), (1, 35)]),
        ("第1回・PDF p.34, 35", [(1, 34), (1, 35)]),
        ("第1回 PDF p.34", [(1, 34)]),
        ("第1回・PDF p.34〜35", [(1, 34), (1, 35)]),
        ("第1回・PDF p.34 - p.35", [(1, 34), (1, 35)]),
        ("第1回・PDF p.34 p.35", [(1, 34), (1, 35)]),
        ("第１回・PDF ｐ.３４", [(1, 34)]),
    ],
)
def test_extract_citations_variants(text, expected):
    assert extract_citations(text) == expected


def test_validate_range_with_p_prefix_checks_start_page():
    deps = TreeDeps(
        question="第1回の評価は？",
        read={"lec01-main-p035": "第1回・PDF p.35"},
    )
    with pytest.raises(ModelRetry, match="lec01-main-p034"):
        validate_tree_answer("評価が重要です（第1回・PDF p.34 - p.35）。", deps)


def test_tree_list_tools_with_question_skips_validation(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["app.agent", "--tree", "--list-tools", "第1回の評価は？"])
    main()
    out = capsys.readouterr().out
    assert "[browse_tree]" in out


def test_extract_lectures():
    assert extract_lectures("第1〜3回のそれぞれで「評価」はどう扱われている？") == {1, 2, 3}
    assert extract_lectures("第2・3回の違いは？") == {2, 3}
    assert extract_lectures("LoRAとは何ですか？") == set()


def test_validate_unread_page_raises_with_page_id():
    deps = TreeDeps(question="第1回の評価は？")
    with pytest.raises(ModelRetry, match="lec01-main-p035"):
        validate_tree_answer("評価が重要です（第1回・PDF p.35）。", deps)


def test_validate_unlisted_but_existing_page_vs_nonexistent_page():
    deps = TreeDeps(question="第1回の評価は？")
    # 一覧に出ていなくても資料にあればページIDを示す
    with pytest.raises(ModelRetry, match="lec01-main-p035"):
        validate_tree_answer("評価（第1回・PDF p.35）。", deps)

    # 資料にない番号は「資料にない」と返す
    with pytest.raises(ModelRetry, match="第1〜3回の資料にないページ番号です"):
        validate_tree_answer("架空（第1回・PDF p.999）。", deps)


def test_validate_missing_citation_for_mentioned_lecture():
    deps = TreeDeps(question="評価について")
    with pytest.raises(ModelRetry, match="第2回"):
        validate_tree_answer("第2回ではモデル選定を扱います。", deps)

    # 「確認できない」と書いてあれば通る
    out = validate_tree_answer("第2回については確認できない。", deps)
    assert "確認できない" in out


def test_validate_missing_lecture_from_question():
    deps = TreeDeps(
        question="第1〜3回のそれぞれで「評価」はどう扱われている？",
        read={"lec01-main-p035": "第1回・PDF p.35", "lec03-main-p042": "第3回・PDF p.42"},
    )
    answer = "第1回では評価が重要（第1回・PDF p.35）。第3回では停止条件（第3回・PDF p.42）。"
    with pytest.raises(ModelRetry, match="第2回"):
        validate_tree_answer(answer, deps)


def test_validate_unconfirmed_in_lec03_does_not_excuse_lec02():
    deps = TreeDeps(
        question="第1〜3回のそれぞれで「評価」はどう扱われている？",
        read={"lec01-main-p035": "第1回・PDF p.35"},
    )
    answer = (
        "第1回では評価が重要（第1回・PDF p.35）。\n"
        "第2回ではモデル選定を扱います。\n"
        "第3回については確認できない。"
    )
    with pytest.raises(ModelRetry, match="第2回"):
        validate_tree_answer(answer, deps)


def test_validate_passes_when_all_cited_pages_read():
    deps = TreeDeps(
        question="第1回の評価は？",
        read={"lec01-main-p035": "第1回・PDF p.35"},
    )
    answer = "評価と改善が重要です（第1回・PDF p.35）。"
    assert validate_tree_answer(answer, deps) == answer


class _RetryThenReadTestModel(TestModel):
    """1回目は未読ページを引用して差し戻され、2回目でページを読み、3回目で合格する TestModel。"""

    def __init__(self):
        super().__init__()
        self._step = 0

    async def request(self, messages, model_settings, model_request_parameters):
        self._step += 1
        if self._step == 1:
            return ModelResponse(parts=[TextPart("評価が重要です（第1回・PDF p.34）。")])
        if self._step == 2:
            return ModelResponse(
                parts=[ToolCallPart("browse_tree", {"node_id": "lec01-main-p034"})]
            )
        return ModelResponse(parts=[TextPart("評価が重要です（第1回・PDF p.34）。")])


def test_tree_agent_retry_reread_pass_cycle():
    deps = TreeDeps(question="第1回の評価は？")
    model = _RetryThenReadTestModel()
    result = tree_agent.run_sync("第1回の評価は？", model=model, deps=deps)
    assert "第1回・PDF p.34" in result.output
    assert "lec01-main-p034" in deps.read


class _AlwaysFailTestModel(TestModel):
    async def request(self, messages, model_settings, model_request_parameters):
        return ModelResponse(parts=[TextPart("未読引用（第1回・PDF p.34）。")])


def test_main_tree_exceeds_retries_exits_with_trace_and_read(monkeypatch, capsys):
    monkeypatch.setattr("app.agent.build_model", lambda: _AlwaysFailTestModel())
    monkeypatch.setattr(sys, "argv", ["app.agent", "--tree", "--trace", "第1回の評価は？"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "--- trace ---" in out
    assert "[retry]" in out
    assert "[read] 全文を読んだページ:" in out


def test_validate_ignores_lectures_outside_materials():
    deps = TreeDeps(
        question="第1回の評価は？",
        read={"lec01-main-p035": "第1回・PDF p.35"},
    )
    answer = "第4回の評価を踏まえると、評価と改善が重要です（第1回・PDF p.35）。"
    assert validate_tree_answer(answer, deps) == answer


def test_tree_agent_without_deps_does_not_crash():
    model = TestModel(call_tools=[], custom_output_text="回答")
    result = tree_agent.run_sync("第1回の評価は？", model=model)
    assert result.output == "回答"
