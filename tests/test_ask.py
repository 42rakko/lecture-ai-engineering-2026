from dataclasses import dataclass
from unittest.mock import MagicMock

from app.ask import AskResult, ask, ask_with_metrics


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None
    thoughts_token_count: int | None = None
    total_token_count: int | None = None


@dataclass
class FakeResponse:
    text: str
    usage_metadata: FakeUsageMetadata | None = None


class FakeClient:
    def __init__(self, response: FakeResponse):
        self.models = MagicMock()
        self.models.generate_content.return_value = response


def test_ask_with_metrics_extracts_tokens():
    usage = FakeUsageMetadata(
        prompt_token_count=12,
        candidates_token_count=34,
        thoughts_token_count=5,
        total_token_count=51,
    )
    fake_client = FakeClient(FakeResponse(text="test answer", usage_metadata=usage))

    result = ask_with_metrics("テスト質問", model="dummy-model", client=fake_client)

    assert isinstance(result, AskResult)
    assert result.model == "dummy-model"
    assert result.question == "テスト質問"
    assert result.text == "test answer"
    assert result.elapsed_sec >= 0
    assert result.input_tokens == 12
    assert result.output_tokens == 34
    assert result.thinking_tokens == 5
    assert result.total_tokens == 51


def test_ask_with_metrics_usage_metadata_none():
    fake_client = FakeClient(FakeResponse(text="no metadata", usage_metadata=None))

    result = ask_with_metrics("テスト質問", client=fake_client)

    assert result.text == "no metadata"
    assert result.elapsed_sec >= 0
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.thinking_tokens is None
    assert result.total_tokens is None


def test_ask_with_metrics_partial_none_fields():
    usage = FakeUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=None,
        total_token_count=30,
    )
    fake_client = FakeClient(FakeResponse(text="partial", usage_metadata=usage))

    result = ask_with_metrics("テスト質問", client=fake_client)

    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.thinking_tokens is None
    assert result.total_tokens == 30


def test_ask_returns_string(monkeypatch):
    usage = FakeUsageMetadata(prompt_token_count=5, candidates_token_count=10, total_token_count=15)
    fake_client = FakeClient(FakeResponse(text="回答文字列", usage_metadata=usage))

    monkeypatch.setattr("app.ask.get_api_key", lambda: "fake-key")
    monkeypatch.setattr("app.ask.genai.Client", lambda api_key: fake_client)

    res = ask("テスト")
    assert isinstance(res, str)
    assert res == "回答文字列"


def test_main_displays_na_when_tokens_are_none(monkeypatch, capsys):
    from app.ask import main

    fake_client = FakeClient(FakeResponse(text="テスト回答", usage_metadata=None))
    monkeypatch.setattr("app.ask.get_api_key", lambda: "fake-key")
    monkeypatch.setattr("app.ask.genai.Client", lambda api_key: fake_client)
    monkeypatch.setattr("sys.argv", ["ask.py", "質問"])

    main()
    captured = capsys.readouterr().out
    assert "[トークン] 入力 N/A / 出力 N/A / 思考 N/A / 合計 N/A" in captured


def test_main_displays_counts_when_tokens_are_present(monkeypatch, capsys):
    from app.ask import main

    usage = FakeUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=0,
        total_token_count=30,
    )
    fake_client = FakeClient(FakeResponse(text="テスト回答", usage_metadata=usage))
    monkeypatch.setattr("app.ask.get_api_key", lambda: "fake-key")
    monkeypatch.setattr("app.ask.genai.Client", lambda api_key: fake_client)
    monkeypatch.setattr("sys.argv", ["ask.py", "質問"])

    main()
    captured = capsys.readouterr().out
    assert "[トークン] 入力 10 / 出力 20 / 思考 0 / 合計 30" in captured

