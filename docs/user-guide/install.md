# インストール

## Windows(インストーラ)

1. 最新版のインストーラをダウンロードする:
   **<https://github.com/kinjo-tohoku/kaisetsu/releases/latest/download/kaisetsu-windows-setup.exe>**
   (過去の版は <https://github.com/kinjo-tohoku/kaisetsu/releases>)
2. ダウンロードしたファイルを実行する。管理者権限は要らない(自分のユーザーのフォルダ
   `%LOCALAPPDATA%\Programs\kaisetsu` に入る)。
3. スタートメニューの「kaisetsu」から起動する。初回はウィンドウが出るまで 10〜20 秒かかることがある
   (数値計算ルーチンのコンパイル。2 回目からは速い)。

### 「Windows によって PC が保護されました」と出たとき

kaisetsu のインストーラにはまだコード署名を付けていないので、初回に SmartScreen の青い画面が出ることがある。

1. 「詳細情報」をクリックする(画面の文章の中の小さなリンク)。
2. 「実行」ボタンが出るので押す。

ブラウザ(Edge / Chrome)が「このファイルは一般的にダウンロードされていません」と止めることもある。
その場合はダウンロード一覧でそのファイルの「…」→「保存」(Edge)または「継続」(Chrome)を選ぶ。

どちらも「配布元が Microsoft に登録されていない」という意味で、ウイルス検出ではない。
心配なら実行前に VirusTotal(<https://www.virustotal.com/>)にファイルを上げて確認できる。

### 更新

起動時に 1 日 1 回、新しい版があるか GitHub に問い合わせる(ヘルプ → 「更新を確認…」でいつでも)。
新しい版があれば変更点が出るので「ダウンロードしてインストール」を押す。インストーラを
ダウンロードしたあと kaisetsu が終了してインストーラが起動するので、「次へ」で進める
(設定とプロジェクトファイルはそのまま残る)。「この版は知らせない」を選ぶとその版については
起動時に出なくなる。自動確認はヘルプ → 「起動時に更新を確認する」で止められる。

自分で新しいインストーラをダウンロードして実行しても同じ(上書きされる)。

確認は GitHub にアクセスするだけで、こちらから送る情報は無い。プロキシが要るネットワークでは
環境変数 `HTTPS_PROXY`(例 `http://proxy.example.co.jp:8080`)を設定しておく。

### アンインストール

Windows の「設定 → アプリ → インストールされているアプリ」から kaisetsu を削除する。

### コマンドライン版(kaisetsu-cli)

インストーラは GUI の `kaisetsu.exe` と一緒にコマンドライン版 `kaisetsu-cli.exe` も入れます。
インストール時の「コマンドライン版 (kaisetsu-cli) を PATH に追加する」を付けておくと、
コマンドプロンプトや PowerShell を開き直してから `kaisetsu-cli --help` で使えます
(アンインストールで PATH からも外れます)。使い方は利用者ガイドの 7h 節。

### zip 版

インストーラが使えない環境(共有 PC など)には `kaisetsu-windows.zip` もある。
展開して `kaisetsu.exe` を実行する。更新は新しい zip を展開してフォルダごと差し替える
(「更新を確認…」は新しい版を知らせ、ダウンロードページを開くところまで)。

## macOS / Linux / Python が入っている Windows

Python 3.12 以上と [uv](https://docs.astral.sh/uv/) があれば、PyInstaller 版を使わずに入れられる。

```bash
# uv が無ければ(Windows は PowerShell で `winget install astral-sh.uv`)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install "kaisetsu[gui] @ git+https://github.com/kinjo-tohoku/kaisetsu"
kaisetsu-gui
```

コマンドラインは `kaisetsu --help`(GUI は `kaisetsu-gui`)。
更新は `uv tool upgrade kaisetsu`(「更新を確認…」は新しい版を知らせるだけ)。Linux では Qt の実行に `libxcb-cursor0`、`libegl1`、
`libxkbcommon0` などが要る(Ubuntu: `sudo apt install libxcb-cursor0 libegl1 libxkbcommon0`)。

## 開発版(ソースから)

```bash
git clone https://github.com/kinjo-tohoku/kaisetsu
cd kaisetsu
uv sync --extra gui
uv run kaisetsu-gui
```

## 動作環境

- Windows 10/11 64 bit(インストーラ)。macOS と Linux は Python 経由。
- メモリ 4 GB 以上。MEM の大きな格子(0.05 Å 間隔など)は 8 GB 以上を推奨。
- 必要なディスク容量: 約 500 MB。

## 困ったとき

- 起動しない: 一度アンインストールして入れ直す。それでも駄目なら
  `%LOCALAPPDATA%\Programs\kaisetsu\kaisetsu.exe` をコマンドプロンプトから実行し、出たメッセージを
  Issue(<https://github.com/kinjo-tohoku/kaisetsu/issues>)に貼る。
- 日本語が□になる: Windows の「設定 → 時刻と言語 → 言語と地域」で日本語の言語パックが入っているか確認する。
