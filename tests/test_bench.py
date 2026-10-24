import pytest

from app.bench import aggregate_results, calculate_group_summary


def test_exclude_error_rows_from_elapsed():
    rows = [
        {
            "status": "ok",
            "elapsed_sec": "1.5000",
            "input_tokens": 10,
            "output_tokens": 20,
            "thinking_tokens": 0,
            "total_tokens": 30,
        },
        {
            "status": "ok",
            "elapsed_sec": "2.5000",
            "input_tokens": 20,
            "output_tokens": 40,
            "thinking_tokens": 0,
            "total_tokens": 60,
        },
        {
            "status": "error",
            "elapsed_sec": "99.0000",  # エラー行の秒数は除外されるべき
            "input_tokens": "",
            "output_tokens": "",
            "thinking_tokens": "",
            "total_tokens": "",
        },
    ]

    summary = calculate_group_summary(rows)

    assert summary["success_count"] == 2
    assert summary["elapsed_mean"] == "2.00"
    assert summary["elapsed_min"] == "1.50"
    assert summary["elapsed_max"] == "2.50"


def test_distinguish_429_and_503_in_summary():
    rows = [
        {"status": "ok", "elapsed_sec": "1.0", "input_tokens": 10},
        {"status": "error", "error": "429: quota exceeded"},
        {"status": "error", "error": "503: model overloaded"},
        {"status": "error", "error": "500: internal error with limit: 1429"},
    ]

    summary = calculate_group_summary(rows)
    assert summary["success_count"] == 1
    assert summary["error_429_count"] == 1
    assert summary["error_other_count"] == 2


def test_token_partial_none_averaged_on_available_only():
    rows = [
        {
            "status": "ok",
            "elapsed_sec": "1.0",
            "input_tokens": 100,
            "output_tokens": None,  # None
            "thinking_tokens": "",
            "total_tokens": 100,
        },
        {
            "status": "ok",
            "elapsed_sec": "2.0",
            "input_tokens": 200,
            "output_tokens": 50,  # 値あり
            "thinking_tokens": "",
            "total_tokens": 250,
        },
    ]

    summary = calculate_group_summary(rows)

    assert summary["input_tokens_avg"] == "150.0"
    # output_tokens は 2行目の 50 だけが対象
    assert summary["output_tokens_avg"] == "50.0"


def test_all_errors_returns_na_without_exception():
    rows = [
        {
            "status": "error",
            "elapsed_sec": "",
            "input_tokens": "",
            "output_tokens": "",
            "thinking_tokens": "",
            "total_tokens": "",
        },
        {
            "status": "error",
            "elapsed_sec": "",
            "input_tokens": None,
            "output_tokens": None,
            "thinking_tokens": None,
            "total_tokens": None,
        },
    ]

    summary = calculate_group_summary(rows)

    assert summary["success_count"] == 0
    assert summary["error_429_count"] == 0
    assert summary["error_other_count"] == 2
    assert summary["elapsed_mean"] == "N/A"
    assert summary["elapsed_min"] == "N/A"
    assert summary["elapsed_max"] == "N/A"
    assert summary["input_tokens_avg"] == "N/A"
    assert summary["output_tokens_avg"] == "N/A"
    assert summary["thinking_tokens_avg"] == "N/A"
    assert summary["total_tokens_avg"] == "N/A"


def test_specific_token_all_none_becomes_na():
    rows = [
        {
            "status": "ok",
            "elapsed_sec": "1.0",
            "input_tokens": 10,
            "output_tokens": 20,
            "thinking_tokens": None,  # 思考トークン非対応モデル
            "total_tokens": 30,
        },
        {
            "status": "ok",
            "elapsed_sec": "1.2",
            "input_tokens": 12,
            "output_tokens": 24,
            "thinking_tokens": "",
            "total_tokens": 36,
        },
    ]

    summary = calculate_group_summary(rows)

    assert summary["input_tokens_avg"] == "11.0"
    assert summary["output_tokens_avg"] == "22.0"
    assert summary["thinking_tokens_avg"] == "N/A"
    assert summary["total_tokens_avg"] == "33.0"


def test_aggregate_results_groups_properly():
    rows = [
        {"model": "m1", "question_id": "q1", "status": "ok", "elapsed_sec": "1.0", "input_tokens": 10},
        {"model": "m1", "question_id": "q1", "status": "ok", "elapsed_sec": "2.0", "input_tokens": 20},
        {"model": "m2", "question_id": "q1", "status": "ok", "elapsed_sec": "0.5", "input_tokens": 15},
    ]

    summaries = aggregate_results(rows)
    assert len(summaries) == 2
    m1_q1 = next(s for s in summaries if s["model"] == "m1" and s["question_id"] == "q1")
    assert m1_q1["success_count"] == 2
    assert m1_q1["elapsed_mean"] == "1.50"

    m2_q1 = next(s for s in summaries if s["model"] == "m2" and s["question_id"] == "q1")
    assert m2_q1["success_count"] == 1
    assert m2_q1["elapsed_mean"] == "0.50"


def test_resolve_out_path_custom_and_default():
    from datetime import datetime
    from pathlib import Path
    from app.bench import resolve_out_path

    custom = resolve_out_path("custom/bench.csv")
    assert custom == Path("custom/bench.csv")

    dt = datetime(2026, 10, 23, 15, 30, 45)
    auto = resolve_out_path(None, dt)
    assert auto == Path("results/bench_20261023_153045.csv")


def test_run_bench_prevents_overwrite_unless_specified(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from app.bench import run_bench

    out_file = tmp_path / "bench.csv"
    out_file.write_text("existing content", encoding="utf-8")

    # 既存ファイルがあり overwrite=False の場合は FileExistsError
    with pytest.raises(FileExistsError):
        run_bench(
            models=["dummy"],
            questions={"q1": "test"},
            repeat=1,
            interval=0,
            out_path=out_file,
            overwrite=False,
            client=MagicMock(),
        )

    # overwrite=True の場合は正常に実行され上書きされる
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.text = "ok"
    fake_resp.usage_metadata = None
    fake_client.models.generate_content.return_value = fake_resp

    rows = run_bench(
        models=["dummy"],
        questions={"q1": "test"},
        repeat=1,
        interval=0,
        out_path=out_file,
        overwrite=True,
        client=fake_client,
    )
    assert len(rows) == 1
    assert out_file.read_text(encoding="utf-8") != "existing content"


def test_run_bench_continues_on_429_api_error(tmp_path):
    from unittest.mock import MagicMock
    from google.genai import errors
    from app.bench import run_bench, aggregate_results

    out_file = tmp_path / "bench_429.csv"

    fake_client = MagicMock()
    # 1回目は429エラー、2回目は成功
    fake_resp = MagicMock()
    fake_resp.text = "ok on run 2"
    fake_resp.usage_metadata = None

    api_error = errors.APIError(429, {"error": {"message": "Resource exhausted"}})
    fake_client.models.generate_content.side_effect = [api_error, fake_resp]

    rows = run_bench(
        models=["test-model"],
        questions={"short-in": "LoRAとは？"},
        repeat=2,
        interval=0,
        out_path=out_file,
        overwrite=True,
        client=fake_client,
    )

    # 429が起きても中断されずに全2回実行される
    assert len(rows) == 2
    assert rows[0]["status"] == "error"
    assert "429" in rows[0]["error"]
    assert rows[1]["status"] == "ok"

    # CSVファイルにも書かれている
    csv_content = out_file.read_text(encoding="utf-8")
    assert "429" in csv_content
    assert "ok" in csv_content

    # 集計で 429 が1回、成功が1回として区別される
    summaries = aggregate_results(rows)
    assert len(summaries) == 1
    s = summaries[0]
    assert s["success_count"] == 1
    assert s["error_429_count"] == 1
    assert s["error_other_count"] == 0


def test_run_bench_raises_on_unexpected_exception(tmp_path):
    from unittest.mock import MagicMock
    from app.bench import run_bench

    out_file = tmp_path / "bench_bug.csv"
    fake_client = MagicMock()
    # コードのバグ（TypeErrorなど）が発生した場合は握りつぶさずに例外送出する
    fake_client.models.generate_content.side_effect = TypeError("Unexpected bug in code")

    with pytest.raises(TypeError, match="Unexpected bug in code"):
        run_bench(
            models=["test-model"],
            questions={"short-in": "LoRAとは？"},
            repeat=1,
            interval=0,
            out_path=out_file,
            overwrite=True,
            client=fake_client,
        )





def test_long_in_includes_glossary_regardless_of_cwd(tmp_path, monkeypatch):
    from app.bench import DATA_GLOSSARY_PATH, get_default_questions

    monkeypatch.chdir(tmp_path)  # リポジトリの外から実行した場合
    questions = get_default_questions()

    glossary_text = DATA_GLOSSARY_PATH.read_text(encoding="utf-8")
    assert glossary_text in questions["long-in"]
    assert len(questions["long-in"]) > len(questions["short-in"]) + len(glossary_text)


def test_missing_glossary_raises(tmp_path, monkeypatch):
    from app import bench

    monkeypatch.setattr(bench, "DATA_GLOSSARY_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        bench.get_default_questions()
