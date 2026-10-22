"""講座全体で共通の設定。

APIキーやモデル名はコードに書かず、環境変数から読み込む。
"""

import os

# 既定モデル。環境変数 AIE_MODEL を設定すると切り替えられる。
#   例: export AIE_MODEL=gemini-3.5-flash
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL = os.getenv("AIE_MODEL", DEFAULT_MODEL)


# 1日あたりの上限（RPD）は待っても回復せず、太平洋時間の午前0時にリセットされる
DAILY_LIMIT_MESSAGE = (
    "1日あたりの上限に達した場合は、1分待っても回復しません。\n"
    "AI Studio の使用状況（https://aistudio.google.com/usage）で確認し、"
    "リセット（太平洋時間の午前0時 = 日本時間の16時、米国の冬時間中は17時）を待つか、"
    "AIE_MODEL で別のモデルに切り替えてください。"
)


def api_error_message(status: int, detail: str = "") -> str:
    """HTTPステータスコードを、受講生向けの日本語の説明にする（ask.py と agent.py で共用）。"""
    if status == 429:
        if "PerDay" in detail:
            return DAILY_LIMIT_MESSAGE
        return (
            "レート制限に達しました。1分ほど待ってから再実行してください。\n"
            "待っても続く場合は1日あたりの上限の可能性があります。\n" + DAILY_LIMIT_MESSAGE
        )
    if status == 503:
        return "モデルが混み合っています（503）。少し待ってから再実行してください。"
    if status in (400, 401, 403, 404):
        return f"APIキーかモデル名に問題があります（{status}）: {detail}"
    return f"APIエラー（{status}）: {detail}"


def get_api_key() -> str:
    """環境変数 GEMINI_API_KEY を返す。未設定なら手順を表示して終了する。"""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "環境変数 GEMINI_API_KEY が見つかりません。\n"
            "1. GitHub の Settings → Codespaces → Secrets に GEMINI_API_KEY を登録し、\n"
            "   使う repository に自分の fork を選んでください。\n"
            "2. Codespaces を起動したあとに登録した場合は、Codespaces を再起動してください。\n"
            "   （左下の「Codespaces」→「Stop Current Codespace」→ 再度開く）"
        )
    return key
