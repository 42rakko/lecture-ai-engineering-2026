"""APIエラーの日本語メッセージのテスト。"""

from app.config import DAILY_LIMIT_MESSAGE, api_error_message


def test_rate_limit_mentions_both_minute_and_daily_limits():
    msg = api_error_message(429, "Resource has been exhausted")
    assert "1分ほど待って" in msg
    assert DAILY_LIMIT_MESSAGE in msg


def test_daily_limit_does_not_say_wait_one_minute():
    msg = api_error_message(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert msg == DAILY_LIMIT_MESSAGE
    assert "1分ほど待って" not in msg
