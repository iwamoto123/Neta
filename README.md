# Neta: Daily Trend Collector

毎日自動でトレンド情報を収集し、`ideas/daily/YYYYMMDD-trend.md` を生成します。

## 運用方法（現在）

- **毎日の自動実行**: GitHub Actions（`.github/workflows/daily-trend.yml`）
  - **実行時刻**: 毎日 **JST 08:00**（workflowのcronは **UTC 23:00**）
  - **実行内容**: `python scripts/collect_trend.py`
  - **保存**: 差分がある場合のみ **自動commit & push**
- **成果物（収集データ）**
  - **日次Markdown**: `ideas/daily/YYYYMMDD-trend.md`
  - **まとめページ（ローカル/任意）**: `ideas/daily/index.html`（注目トピックのみ）
- **情報源の追加/変更（手で触る場所）**: `scripts/trend_sources.json`
  - **note**: `note.tags` / `note.users` を編集
  - **資金調達/Exitサイト**: `rss_sites` を編集（RSSが見つかれば自動取得、見つからなければスキップ）
  - **X（記事）**: `x.article_urls`（`https://x.com/i/articles/...`）または `x.article_authors`（`@id`）を編集
- **好みに寄せる（URL貼り学習）**: `scripts/feedback.json`
  - `liked_urls` に「良かったURL」を追加 → 次回実行でキーワード重みを更新し、興味度（★）に反映
- **注意**: Cookieなどの機密情報は **絶対にGitにコミットしない**（`.gitignore` で `cookies_x.json` / `*.cookies.json` を無視）

## 構成（最小）

- `scripts/collect_trend.py`: 収集 & Markdown生成
- `scripts/trend_sources.json`: 追跡する情報源（noteタグ/ユーザー、RSSサイト、Xなど）
- `scripts/feedback.json`: 「良かったURL」を貼って学習（簡易キーワード重み）
- `.github/workflows/daily-trend.yml`: 毎朝8:00(JST)に実行してpush
- `ideas/daily/`: 出力先

## ローカル実行

```bash
python3 -m pip install requests
python scripts/collect_trend.py
```

## ローカル用ダッシュボード（GitHub Pages不要）

`collect_trend.py` は `ideas/daily/index.html`（注目トピックまとめ）も毎回生成します。

ローカルでブラウザを開く:

```bash
python scripts/open_dashboard.py
```

## note の「フォロー中」初回取り込み

`scripts/trend_sources.json` の `note.followings_url` が設定されていて `note.users` が空の場合、初回だけ `followings_url` を参照して `note.users` を自動生成します。

対象ページ例（あなたの指定）: `https://note.com/takechannel/followings`

以後は `scripts/trend_sources.json` の `note.users` を手動で増やしてください。

## X（記事URL）を取る（Playwright + Cookie）

GitHub Actions上でXを安定取得するため、CookieをSecretで渡します。

> 重要: Cookieはログイン情報そのものです。**絶対にGitにコミット/Pushしないでください。**

### 1) Cookieの用意（例）

Chromeで `x.com` にログインした状態で、CookieをJSONとしてエクスポートしてください（拡張機能などを利用）。

必要な形式は Playwright の `context.add_cookies()` に渡す配列（list）です。
拡張機能の出力が `expirationDate` / `sameSite: no_restriction` などの形式でも、`collect_trend.py` 側で可能な範囲で自動変換します。

ローカルで明示的に変換したい場合は、次を使えます:

```bash
python3 scripts/convert_cookies.py
```

### 2) GitHub Secrets

リポジトリの Secrets に以下を追加:

- `X_COOKIES_JSON`: Cookie JSON（文字列）

Secretが入っている場合のみ、Workflowが Playwright をインストールしてX収集を試みます。

## X（直近24h）を取る（Grok / xAI API）

`XAI_API_KEY` が設定されている場合、`collect_trend.py` は **Grok（xAI API）を優先**して「直近24時間のX投稿」を興味領域に合わせて収集し、`X` セクションに出力します。

### GitHub Secrets

リポジトリの Secrets に以下を追加:

- `XAI_API_KEY`: xAI APIキー

※キーは絶対にコードやリポジトリにコミットしないでください。

### 3) 追跡する記事URL/著者

`scripts/trend_sources.json` の以下を編集してください：

- `x.article_urls`: 追跡したい `https://x.com/i/articles/...` のURLを列挙
- `x.article_authors`: 著者の `@id`（例: `"jack"`）を列挙（`https://x.com/<id>/articles` を見に行きます）

## 学習（URL貼り方式）

`scripts/feedback.json` に `liked_urls` を追加すると、次回実行時に（可能なら）各URLの `<title>` を取りに行き、
出現キーワードを `keyword_weights` として蓄積します。これを「興味度」算定に加点します。

## GitHub Actions 実行時刻

`JST 08:00` で動かすため、workflowは `UTC 23:00` にスケジュールしています。


