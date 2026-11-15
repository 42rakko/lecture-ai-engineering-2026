# 第4回 解答

## 演習

- 基準別の一致率（直す前）：根拠　7/9　、範囲　10/10　、ヒント　10/10　、分かりやすさ　10/10　、回の表示　10/10　 
- 要確認の件数（直す前）： 1
- 直した基準と、書き足した方針： 用語集にない語について、「用語集にはない」と伝えたうえで一般的な説明をすることは根拠がないので fail
- 基準別の一致率と要確認の件数（直した後）：根拠　9/10　、範囲　10/10　、ヒント　10/10　、分かりやすさ　10/10　、回の表示　10/10　 
- 講師のラベルと自分のラベルで判定が違った項目と、そう判定した理由：{"id": "06", "grounding": "fail", "理由":"LLMの提示した一般論が必ずしも正しいとは限らないから"}
- 気づき・疑問： 要確認だけでなく、不一致に関する事項も変化させるためにいろいろ試してみたが、LLMのこたえが毎回微妙に違うため、不一致を0にするのは非常に難しいと感じた。LLMの出すテスト結果を毎回同じにする必要があるようだ。

## 宿題

- 取り組んだレベル：発展
- やったこと：SWE-bench Verifiedをhuggingfaceのdataset viewerでissueを検索
    https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
- 結果：
	django__django-13158:
        Django の QuerySet.union() 等で結合されたクエリに対して .none() を呼び出した際、空の EmptyQuerySet にならず元の結合結果がそのまま返ってしまうバグの修正
        Django ORM の内部構造（QuerySet と SQLCompiler、特に combinator 属性の扱い）を追う必要があるが、再現コードが明確なので時間はかければ解けそう
        Issue の本文の大半が ModelForm や ModelMultipleChoiceField、ManyToManyField のコード例で占められているため、表層的に読むと「フォーム側のバリデーションや ModelMultipleChoiceField のバグ」だと勘違いしやすい
    scikit-learn__scikit-learn-25232
        scikit-learnのIterativeImputerにfill_valueパラメータを追加し、initial_strategy="constant"時に任意の定数（np.nan を含む）で欠損値を初期補完できるようにする機能拡張
        やることが明確で対象クラスも IterativeImputer と特定されているため、比較的解きやすい
        scikit-learn では全推定器の__init__引数に対して厳格な型・値チェックを定義する独自ルールがあるため、そこの対応が難しそう
    django__django-11239
        Djangoのmanage.py dbshellコマンドにおいて、相互TLS認証（mTLS）に必要なクライアント証明書・秘密鍵のオプション（sslcert, sslkey）がpsqlに引き渡されない。
        おそらく、単純な実装漏れだと思われるので該当箇所を探索した上で他のDBに対する実装と合わせて実装すれば良さそう
        Pythonのコード単体を読むだけでは「PostgreSQLのpsqlコマンドがどの環境変数名を受け取るか」は分からないため、LLM自身の事前知識（PostgreSQL の仕様）と既存コードの実装パターンを正確に結びつける必要がある。
    プリンストン大学の研究チーム 9名が集めたデータの収集システムのなかから、良問500件を人間が目で見て選び抜くためにOpen AIが雇用したプロのソフトウェアエンジニア93名が各問題を3名チームになって厳選した
- 気づき・疑問：LLMに渡されるissueの内容を要約されたものと問題の修正とが合致するとは限らないし、それとは別に修正の難易度もあるため、それぞれを満遍なくテストに入れることがベンチマークとして重要だと考え、そのバランスが難しいと感じた。厳選する人たちが専門家であることが重要。
    
