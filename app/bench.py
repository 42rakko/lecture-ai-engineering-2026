"""第1回 宿題・発展: 複数のモデルと質問で応答時間とトークン数を比較するスクリプト。

使い方:
    uv run python -m app.bench
    uv run python -m app.bench --models gemini-3.5-flash-lite gemini-3.5-flash --repeat 3 --interval 5
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

from google import genai
from google.genai import errors

from app.ask import DEFAULT_QUESTION, ask_with_metrics
from app.config import get_api_key

DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
# 実行したディレクトリによらず同じファイルを読むよう、このファイルの位置から決める
DATA_GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "glossary.json"

CSV_FIELDNAMES = [
    "timestamp",
    "model",
    "question_id",
    "run",
    "elapsed_sec",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "total_tokens",
    "status",
    "error",
]


def get_default_questions() -> dict[str, str]:
    # 見つからなければ FileNotFoundError で止める（空のまま続けると long-in が短い質問になり、比較が成り立たない）
    glossary_text = DATA_GLOSSARY_PATH.read_text(encoding="utf-8")

    return {
        "short-in": "LoRAとは何ですか？3文以内で簡潔に説明してください。",
        "long-in": (
            "以下の参考資料を読み、質問に答えてください。\n\n"
            f"【参考資料】\n{glossary_text}\n\n"
            "【質問】\nLoRAとは何ですか？3文以内で簡潔に説明してください。"
        ),
        "long-out": "LoRAとは何ですか？仕組み、利点、従来のファインチューニングとの違いを含めて、1000字程度で詳しく解説してください。",
        "course": DEFAULT_QUESTION,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLMベンチマーク計測スクリプト")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="計測対象のモデル一覧",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="各質問の繰り返し回数（既定: 3）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="API呼び出し間の待機秒数（既定: 5.0）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="CSV出力先パス（未指定時は results/bench_YYYYMMDD_HHMMSS.csv）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の出力先ファイルを上書きする",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="既定の質問を上書きする場合に指定（質問IDは q1, q2, ... になります）",
    )
    return parser.parse_args()


def resolve_out_path(out_arg: str | None, now: datetime | None = None) -> Path:
    """出力先パスを決定する。未指定時は日時入りのファイル名を生成する。"""
    if out_arg:
        return Path(out_arg)
    if now is None:
        now = datetime.now().astimezone()
    return Path("results") / f"bench_{now.strftime('%Y%m%d_%H%M%S')}.csv"


def make_success_row(
    timestamp: str,
    model: str,
    question_id: str,
    run: int,
    res: Any,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "model": model,
        "question_id": question_id,
        "run": run,
        "elapsed_sec": f"{res.elapsed_sec:.4f}",
        "input_tokens": res.input_tokens if res.input_tokens is not None else "",
        "output_tokens": res.output_tokens if res.output_tokens is not None else "",
        "thinking_tokens": res.thinking_tokens if res.thinking_tokens is not None else "",
        "total_tokens": res.total_tokens if res.total_tokens is not None else "",
        "status": "ok",
        "error": "",
    }


def make_error_row(
    timestamp: str,
    model: str,
    question_id: str,
    run: int,
    error: str,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "model": model,
        "question_id": question_id,
        "run": run,
        "elapsed_sec": "",
        "input_tokens": "",
        "output_tokens": "",
        "thinking_tokens": "",
        "total_tokens": "",
        "status": "error",
        "error": error,
    }


def calculate_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """1つの (model, question_id) グループに対する集計を行う。"""
    success_rows = [r for r in rows if r.get("status") == "ok"]
    error_rows = [r for r in rows if r.get("status") == "error"]

    error_429_count = sum(1 for r in error_rows if str(r.get("error", "")).startswith("429:"))
    error_other_count = len(error_rows) - error_429_count

    elapsed_list: list[float] = []
    for r in success_rows:
        val = r.get("elapsed_sec")
        if val is not None and val != "":
            try:
                elapsed_list.append(float(val))
            except (ValueError, TypeError):
                pass

    if elapsed_list:
        mean_elapsed: str = f"{sum(elapsed_list) / len(elapsed_list):.2f}"
        min_elapsed: str = f"{min(elapsed_list):.2f}"
        max_elapsed: str = f"{max(elapsed_list):.2f}"
    else:
        mean_elapsed = "N/A"
        min_elapsed = "N/A"
        max_elapsed = "N/A"

    def calc_token_avg(key: str) -> str:
        vals: list[int] = []
        for r in rows:
            val = r.get(key)
            if val is not None and val != "":
                try:
                    vals.append(int(val))
                except (ValueError, TypeError):
                    pass
        if not vals:
            return "N/A"
        return f"{sum(vals) / len(vals):.1f}"

    return {
        "success_count": len(success_rows),
        "error_429_count": error_429_count,
        "error_other_count": error_other_count,
        "elapsed_mean": mean_elapsed,
        "elapsed_min": min_elapsed,
        "elapsed_max": max_elapsed,
        "input_tokens_avg": calc_token_avg("input_tokens"),
        "output_tokens_avg": calc_token_avg("output_tokens"),
        "thinking_tokens_avg": calc_token_avg("thinking_tokens"),
        "total_tokens_avg": calc_token_avg("total_tokens"),
    }


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """モデル×質問ごとに集計したリストを返す。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r.get("model", ""), r.get("question_id", ""))
        groups.setdefault(key, []).append(r)

    summaries = []
    for (model, q_id), group_rows in groups.items():
        summary = calculate_group_summary(group_rows)
        summary["model"] = model
        summary["question_id"] = q_id
        summaries.append(summary)

    return summaries


def format_markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| モデル | 質問ID | 成功 | 429 | その他エラー | 応答時間 平均(s) | 最小(s) | 最大(s) | 入力トークン平均 | 出力トークン平均 | 思考トークン平均 | 合計トークン平均 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['question_id']} | {s['success_count']} | {s['error_429_count']} | {s['error_other_count']} | "
            f"{s['elapsed_mean']} | {s['elapsed_min']} | {s['elapsed_max']} | "
            f"{s['input_tokens_avg']} | {s['output_tokens_avg']} | {s['thinking_tokens_avg']} | {s['total_tokens_avg']} |"
        )
    return "\n".join(lines)


def run_bench(
    models: list[str],
    questions: dict[str, str],
    repeat: int,
    interval: float,
    out_path: str | Path,
    overwrite: bool = False,
    client: genai.Client | None = None,
) -> list[dict[str, Any]]:
    out_file = Path(out_path)
    if out_file.exists() and not overwrite:
        raise FileExistsError(f"出力先 '{out_file}' は既に存在します。上書きする場合は overwrite=True または --overwrite を指定してください。")

    if client is None:
        client = genai.Client(api_key=get_api_key())

    total_requests = len(models) * len(questions) * repeat
    print(f"[bench] 総リクエスト数: {total_requests} 回 ({len(models)}モデル × {len(questions)}質問 × {repeat}回)")
    print(f"[bench] 呼び出し間隔: {interval} 秒 / 出力先: {out_file}\n")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    first_call = True

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        f.flush()

        current_request = 0
        for model in models:
            for q_id, q_text in questions.items():
                for run_idx in range(1, repeat + 1):
                    current_request += 1
                    if not first_call and interval > 0:
                        time.sleep(interval)
                    first_call = False

                    timestamp = datetime.now().astimezone().isoformat()
                    print(f"[{current_request}/{total_requests}] model={model} question={q_id} run={run_idx} ... ", end="", flush=True)

                    try:
                        res = ask_with_metrics(q_text, model=model, client=client)
                        row = make_success_row(timestamp, model, q_id, run_idx, res)
                        print(f"ok ({res.elapsed_sec:.2f}s, in={res.input_tokens}, out={res.output_tokens})")
                    except errors.APIError as e:
                        row = make_error_row(timestamp, model, q_id, run_idx, f"{e.code}: {e.message}")
                        print(f"error (APIError {e.code})")

                    writer.writerow(row)
                    f.flush()
                    rows.append(row)

    return rows


def main() -> None:
    args = parse_args()

    start_dt = datetime.now().astimezone()
    out_path = resolve_out_path(args.out, start_dt)

    if out_path.exists() and not args.overwrite:
        print(
            f"エラー: 出力先 '{out_path}' は既に存在します。\n"
            "上書きする場合は --overwrite を指定するか、--out で別のパスを指定してください。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.questions:
        questions = {f"q{idx}": q for idx, q in enumerate(args.questions, start=1)}
    else:
        questions = get_default_questions()

    client = genai.Client(api_key=get_api_key())

    rows = run_bench(
        models=args.models,
        questions=questions,
        repeat=args.repeat,
        interval=args.interval,
        out_path=out_path,
        overwrite=args.overwrite,
        client=client,
    )

    summaries = aggregate_results(rows)
    table_md = format_markdown_table(summaries)

    print("\n" + "=" * 50)
    print("### ベンチマーク実行条件")
    print(f"- 実行日時: {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"- モデル一覧: {', '.join(args.models)}")
    print(f"- 質問ID一覧: {', '.join(questions.keys())}")
    print(f"- 繰り返し回数: {args.repeat}")
    print(f"- 待機時間: {args.interval} 秒")
    print(f"- 出力先: {out_path}")
    print()
    print("### ベンチマーク集計表")
    print(table_md)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

