"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from dataclasses import dataclass
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass
class AskResult:
    model: str
    question: str
    text: str
    elapsed_sec: float  # time.perf_counter() で generate_content の前後を測る
    input_tokens: int | None  # usage_metadata.prompt_token_count
    output_tokens: int | None  # usage_metadata.candidates_token_count
    thinking_tokens: int | None  # usage_metadata.thoughts_token_count（思考するモデルのみ）
    total_tokens: int | None  # usage_metadata.total_token_count


def ask_with_metrics(
    question: str,
    model: str = MODEL,
    client: genai.Client | None = None,
) -> AskResult:
    if client is None:
        client = genai.Client(api_key=get_api_key())

    start = time.perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    elapsed = time.perf_counter() - start

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    thinking_tokens = getattr(usage, "thoughts_token_count", None) if usage else None
    total_tokens = getattr(usage, "total_token_count", None) if usage else None

    return AskResult(
        model=model,
        question=question,
        text=response.text or "",
        elapsed_sec=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
    )


def ask(question: str) -> str:
    return ask_with_metrics(question).text


def _format_token(val: int | None) -> str:
    return str(val) if val is not None else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        result = ask_with_metrics(args.question)
        print(result.text)
        print(f"\n[時間] {result.elapsed_sec:.2f} 秒")
        tok_in = _format_token(result.input_tokens)
        tok_out = _format_token(result.output_tokens)
        tok_think = _format_token(result.thinking_tokens)
        tok_total = _format_token(result.total_tokens)
        print(f"[トークン] 入力 {tok_in} / 出力 {tok_out} / 思考 {tok_think} / 合計 {tok_total}")
    except errors.APIError as e:
        detail = f"{e.message} {getattr(e, 'details', '')}".strip()
        print(api_error_message(e.code, detail))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
