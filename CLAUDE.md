# kaisetsu 開発規約

初心者向け・日本語対応の粉末回折解析プログラム(Rietveld / PDF / MEM)。
計画と設計判断は `docs/ROADMAP.md` を正とする。

## 言語・コミュニケーション
- ユーザーとのやり取り、docstring、コメント、ユーザー向けメッセージは日本語。
- 識別子(変数・関数・クラス・モジュール名)は英語。
- GUI 文字列は将来の英語化に備えて `tr()` で囲む(Phase 4 以降)。

## 環境
- Python >= 3.12、パッケージ管理は uv。実行は `uv run ...`。
- 依存を追加するときはライセンスを確認し、商用利用可能(BSD / MIT / MPL / LGPL / PSF)なものに限る。GPL は不可。
- `docs/ROADMAP.md` 4 節の依存表を更新する。

## アーキテクチャの約束
- `src/kaisetsu/gui/` 以外から Qt を import しない。コアは GUI なしで動く。
- Rietveld / PDF / MEM は同じ `Structure` と `ParameterSet` を共有する。
- 数値計算は NumPy ベクトル化が基本。numba はホットループだけ、`fastmath=False`。
- 物理式を実装する関数の docstring には出典(文献、式番号)を書く。

## 品質
- 型ヒント必須。`uv run ruff check . && uv run ruff format --check . && uv run mypy` をクリーンに保つ。
- テストは pytest。`uv run pytest` が常に通る状態でコミットする。
  速く回したいときは `uv run pytest -m "not slow"`(精密化の統合テストと GSAS-II 精密化比較を除く)。
- Windows のコンソールで日本語を出すコマンドには `PYTHONUTF8=1` を付ける(cp932 で落ちる)。
- 外部ソフト(GSAS-II 等)との突き合わせは `tests/validation/` に置き、`@pytest.mark.validation` を付ける。
  参照値 JSON は `validation/gsas2/` のスクリプトで生成し、`validation/reference/` に保存する。
- 検証結果の記録(バージョン・日付・差分)は `docs/validation/` に残す。

## 検証用の外部ツール
- GSAS-II 5.7.4: `C:\Users\KatsukiKinjo\gsas2main`(Python 3.13)。
  参照値生成は `uv run python validation/gsas2/run_gsas2.py ...` が内部で gsas2main の Python を subprocess で呼ぶ。
  GSAS-II をこのパッケージの依存にしてはいけない。

## 配布
- 自作コードは MIT(`LICENSE`)。配布物には `packaging/collect_licenses.py` が作る依存のライセンス一覧を同梱する。
- Windows 配布物は `.\packaging\build_windows.ps1`(PyInstaller → `--smoke-test` で起動確認 → Inno Setup → zip)。
- GitHub(kinjo-tohoku/kaisetsu)にタグ `v<version>` を push すると `.github/workflows/release.yml` が Release を作る。
  タグは `kaisetsu.__version__` と一致させる。

## コミット
- 変更はまとまりごとにコミットする。メッセージは日本語でよい。
