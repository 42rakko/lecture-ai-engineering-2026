# 計画書: ask.py に応答時間・トークン数の計測と比較機能を追加する（第1回 宿題・発展）

## 目的

第1回 宿題の「発展」レベルに取り組む。

> 応答時間とトークン数を計測し、複数のモデルや質問で比較する。無料枠のレート制限との関係も考察する

成果物は次の3つ。

1. `app/ask.py` で、1回の呼び出しの応答時間とトークン数を取れるようにする（演習の TODO も兼ねる）
2. 複数のモデルと質問を総当たりで実行し、結果を表にまとめるスクリプト `app/bench.py`
3. 結果と考察を書いた `lectures/01/answer.md`

## 現状

- `app/ask.py` の `ask()` は `client.models.generate_content()` を1回呼び、`response.text` だけを返す
- `TODO(第1回 演習)` に「`response.usage_metadata` から入力・出力トークン数を表示する」とある
- モデルは `app/config.py` の `MODEL`（環境変数 `AIE_MODEL`、既定は `gemini-3.5-flash-lite`）で決まり、
  実行中に切り替える手段はない
- APIエラーは `api_error_message()` で日本語にしている（429 = レート制限。1分あたりと1日あたりの上限を区別して案内する）
- テストは `tests/` にあり、実際のAPIは呼ばない方針（`TestModel` など）

## 方針

- `ask.py` は「1回質問する」役割のまま、計測結果を返せるようにする。比較の繰り返し処理は別ファイル
  `app/bench.py` に分け、`ask.py` の関数を呼ぶ。第3回以降も `ask.py` が素の呼び出しの例として読めるようにするため
- 既存の使い方（`uv run python -m app.ask "質問"`）は変えない
- 無料枠の上限値はコードに書かない。上限は変わることがあるので、実行時点の値を
  公式ページ（https://ai.google.dev/gemini-api/docs/rate-limits ）と AI Studio で確認し、answer.md に記録する
- テストはAPIを呼ばない（偽のクライアントを差し込む）

## 変更内容

### 1. `app/ask.py`

計測結果を入れるデータクラスと、計測つきの関数を追加する。

```python
@dataclass
class AskResult:
    model: str
    question: str
    text: str
    elapsed_sec: float        # time.perf_counter() で generate_content の前後を測る
    input_tokens: int | None  # usage_metadata.prompt_token_count
    output_tokens: int | None # usage_metadata.candidates_token_count
    thinking_tokens: int | None  # usage_metadata.thoughts_token_count（思考するモデルのみ）
    total_tokens: int | None  # usage_metadata.total_token_count


def ask_with_metrics(question: str, model: str = MODEL, client: genai.Client | None = None) -> AskResult:
    ...


def ask(question: str) -> str:
    return ask_with_metrics(question).text  # 既存の呼び出し元を壊さない
```

- `model` を引数で渡せるようにする（ベンチマークで1プロセス内から複数モデルを試すため）
- `client` を引数で渡せるようにする（テストで偽クライアントを差し込むため。省略時は今まで通り作る）
- APIエラー（`errors.APIError`）はここでは捕まえず、そのまま呼び出し元に送出する。記録するかどうかは `main()` と `bench.py` が決める
- 計測するのは `generate_content` の呼び出しの前後だけ。クライアントの生成は計測に含めない
- `usage_metadata` やその各フィールドは `None` のことがあるので、`None` のまま返す（0 にしない）
- 思考トークンは出力トークンとは別に数えられ、課金の対象になるので分けて記録する。
  一方、レート制限の TPM は公式ページで「Tokens per minute (input)」と書かれており、**入力トークン数**の上限である
  （https://ai.google.dev/gemini-api/docs/rate-limits 、2026-10-15 更新版で確認）。
  そのため TPM の見積もりは `input_tokens` を基準にする。ただし出力を含めて数えるという説もあるため、
  `total_tokens` で計算した回数も「控えめな見積もり」として併記し、実測で確かめる（考察の観点 4）
- `main()` で回答のあとに次の形で表示する（演習の TODO を解消。メタデータがない項目は 0 ではなく `N/A` と表示して実測値の 0 と区別する）

  ```
  [時間] 1.23 秒
  [トークン] 入力 25 / 出力 310 / 思考 0 / 合計 335
  ```

### 2. `app/bench.py`（新規）

```bash
uv run python -m app.bench \
    --models gemini-3.5-flash-lite gemini-3.5-flash \
    --repeat 3 \
    --interval 5
```

- 出力先 `--out` は未指定時、実行ごとに消えないよう日時入り（`results/bench_YYYYMMDD_HHMMSS.csv`）で保存される。既存ファイルを上書きする場合は `--overwrite` を明示する

- 質問は既定で次の4種類を持つ。`--questions "質問1" "質問2" ...` で上書きでき、そのときの質問IDは `q1`, `q2`, ... とする
  - `short-in`: 短い入力で、回答を「3文以内」に指定した質問（例: LoRA とは何かを3文以内で）
  - `long-in`: `short-in` と同じ質問と回答形式（3文以内）の前に、参考資料として `data/glossary.json` の全文
    （約31KB）を付けたもの。**出力長をそろえて入力長だけを変える**ための組。
    受講生の間で同じ条件になるよう、ファイルは固定し、加工せずにそのまま付ける。
    用語集に LoRA の説明が入っていると回答の中身は変わりうるが、比べるのは時間とトークン数なので、出力長がそろっていればよい
  - `long-out`: 短い入力で、長い回答（例: 1000字程度の解説）を求める質問。`short-in` と比べて出力長だけを変える
  - `course`: 講座に固有の質問（`DEFAULT_QUESTION`）
- `genai.Client` は開始時に1回だけ作り、すべての呼び出しで使い回す（`ask_with_metrics(..., client=client)`）。
  毎回作ると、接続の確立（TLS ハンドシェイクなど）の時間が応答時間に混ざるため。
  それでも全体の最初の1回には接続の時間が乗るので、考察ではその1回が外れ値になっていないか確かめる
- モデル × 質問 × 繰り返し回数 を順に実行し、1回ごとに1行を CSV に書く（成功なら `AskResult` の値、エラーなら下記のエラー行）。
  列は `timestamp, model, question_id, run, elapsed_sec, input_tokens, output_tokens, thinking_tokens, total_tokens, status, error`
- 呼び出しの間に `--interval` 秒待つ（既定5秒）。レート制限を避けるためと、考察で「待ち時間を変えると 429 が出るか」を試すため
- 429 などの APIエラーは `bench.py` 側で `try`/`except errors.APIError` で受け、止めずに記録して続ける。
  エラー行は `status=error`、`error` に「ステータスコード: メッセージ」を入れ、`elapsed_sec` とトークン数の列は空欄にする
  （エラーまでの時間は集計に使わないので記録しない）。429 の回数がレート制限の考察の材料になる
- 最後に、モデル × 質問ごとの集計表（平均・最小・最大の応答時間、平均トークン数、成功数、429数、その他エラー数）を標準出力に Markdown 表で出す。
  429 と 503 等の混雑・その他エラーを区別してレート制限の考察をしやすくし、answer.md にそのまま貼れるようにする。表の直前に実行条件（実行日時、モデル一覧、質問ID一覧、repeat 回数、interval 秒、出力先）を
  箇条書きで出し、表とまとめて転記できるようにする。集計の規則は次のとおり
  - 応答時間は**成功した行だけ**で集計する（エラーになった呼び出しの時間は含めない）
  - トークン数の平均は、項目（入力・出力・思考・合計）ごとに**値がある行だけ**で計算する。`None` を 0 として数えない
  - 集計対象の値が1つもない場合（全件 429 など）は、その欄を `N/A` と表示する。0 や空欄にしない
- 実行前に総リクエスト数を表示する（例: 2モデル × 4質問 × 3回 = 24回）。無料枠の1日あたり上限を使い切らないよう確認するため
- `results/` は `.gitignore` に追加する（生データはコミットしない。集計表だけ answer.md に載せる）

### 3. テスト `tests/test_ask.py` と `tests/test_bench.py`（新規）

どちらもAPIを呼ばない。

`tests/test_ask.py`: `generate_content` が決まった `text` と `usage_metadata` を返す偽クライアントを作り、
`ask_with_metrics(..., client=fake)` に渡して確かめる。

- トークン数が `usage_metadata` の各フィールドから正しく取り出される
- `usage_metadata` が `None`、または一部のフィールドが `None` でも落ちない
- `elapsed_sec` が 0 以上
- `ask()` が今まで通り文字列を返す

`tests/test_bench.py`: 集計関数（平均・最小・最大・成功数・エラー数）に、手で作った行を渡して確かめる。
  - エラー行を応答時間の集計から除く
  - トークン数の一部が `None` の行があっても、項目ごとに値がある行だけで平均する
  - 全件エラーのとき、応答時間とトークン数の欄が `N/A` になり、例外にならない
  - ある項目（例: 思考トークン）がすべて `None` のとき、その欄だけ `N/A` になる

### 4. `lectures/01/answer.md`

宿題欄に次を書く。

- 取り組んだレベル: 発展
- 実行条件: 日時、モデル、質問、繰り返し回数、待ち時間
- 結果: `bench.py` の集計表
- 確認した無料枠の上限（モデルごとの RPM / TPM / RPD）
- 考察（下記の観点）

## 考察の観点

計測しながら、次の問いに答えられるようにする。

1. **モデル間の比較**: flash-lite と flash で応答時間とトークン数はどう違うか。思考トークンはどちらで発生するか
2. **質問間の比較**: 応答時間は入力長と出力長のどちらに強く効くか。
   `short-in` と `long-in`（出力長をそろえて入力長だけ変える）の差と、
   `short-in` と `long-out`（入力長をそろえて出力長だけ変える）の差を比べる。
   回答形式の指定どおりに出力長がそろっているかも `output_tokens` で確かめる
3. **ばらつき**: 同じ質問・同じモデルでも応答時間と出力トークン数はどれくらい変わるか
4. **レート制限との関係**
   - 1回あたりの**入力**トークン数と、確認した TPM（1分あたり入力トークン数）から、1分間に何回まで呼べるか見積もる。
     RPM（1分あたりリクエスト数）とどちらが先に上限になるか。`short-in` と `long-in` で結論が変わるかも見る。
     `total_tokens` で計算した控えめな見積もりも併記する
   - TPM に出力が含まれるかを実測で確かめる: `long-out` を続けて送ったあと、AI Studio のレート制限の画面
     （https://aistudio.google.com/rate-limit ）で TPM の使用量を見て、`input_tokens` の合計と `total_tokens` の合計の
     どちらに近いかを比べる
   - RPD（1日あたりリクエスト数）から、このベンチマークを1日に何回実行できるか
   - `--interval` を短くして（例: 0〜1秒）同じモデルに続けて送り、429 がいつ出るかを試す。
     見積もりと実際が合うか（RPD を使い切らないよう、回数を絞って安いモデルで1回だけ行う）
   - 第3回以降のエージェントは1回の質問でモデルを複数回呼ぶ。今回の計測結果から、
     エージェントが無料枠でどれくらい動かせそうか見積もる

## 作業手順

1. `ask.py` に `AskResult` と `ask_with_metrics()` を追加し、`main()` で時間とトークン数を表示する
2. `tests/test_ask.py` を書き、`uv run pytest -q` が通ることを確かめる
3. `uv run python -m app.ask "LoRAとは何ですか？"` で表示を確かめる
4. `app/bench.py` を作り、`tests/test_bench.py` で集計関数を確かめる
5. まず `--repeat 1` で小さく動かし、総リクエスト数と所要時間を確かめる
6. 本番の計測（2モデル × 4質問 × 3回）を行う
7. レート制限の試験を1回だけ行う
8. answer.md に結果と考察を書き、`lec01` ブランチで PR を出す（base は自分の fork）

## 完了条件

- `uv run python -m app.ask` が従来どおり回答を表示し、加えて時間とトークン数を表示する
- `uv run python -m app.bench` が CSV と集計表を出す。429 が起きても最後まで動く
- `uv run pytest -q` が通る（新しいテストはAPIキーなしで通る）
- answer.md に集計表、無料枠の上限値、考察が書かれている

## 注意

- 無料枠の上限はモデルとアカウントごとに異なり、変更されることもある。計測日と確認した値を必ず記録する
- 応答時間はネットワークやサーバーの混み具合でも変わる。比較は同じ時間帯にまとめて実行する
- 503（混雑）が出たときは 429 と分けて記録し、レート制限と混同しない
