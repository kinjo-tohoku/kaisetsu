"""GUI の文字列の翻訳(日本語 → 英語)。

日本語の文字列をキーにした辞書で引く簡易方式。`tr("保存")` のように使う。
言語は `set_language("en")`(設定 "language"、または起動時の --lang)で切り替える。
辞書に無い文字列はそのまま返すので、未翻訳の箇所は日本語のまま表示される。
"""

from __future__ import annotations

_LANG = "ja"

EN: dict[str, str] = {
    # メニュー
    "ファイル(&F)": "&File",
    "新規プロジェクト(&N)": "&New project",
    "プロジェクトを開く(&O)…": "&Open project…",
    "プロジェクトを保存(&S)": "&Save project",
    "名前を付けて保存(&A)…": "Save &as…",
    "計算パターンを CSV に書き出す…": "Export calculated pattern as CSV…",
    "反射リストを CSV に書き出す…": "Export reflection list as CSV…",
    "相を CIF に書き出す(VESTA で開けます)…": "Export phase as CIF (opens in VESTA)…",
    "図を画像で保存 (PNG/SVG)…": "Save plot as image (PNG/SVG)…",
    "終了(&X)": "E&xit",
    "データ(&D)": "&Data",
    "回折データを読み込む(&L)…": "&Load diffraction data…",
    "CIF から相を追加(&P)…": "Add &phase from CIF…",
    "装置と試料の設定(&I)…": "&Instrument and sample…",
    "背景の設定(&B)…": "&Background…",
    "相の設定(強度の扱い・選択配向軸・異方性)…": "Phase settings (intensities, preferred orientation, anisotropy)…",
    "PDF データ G(r) を読み込む…": "Load PDF data G(r)…",
    "I(Q) から G(r) を作る…": "Transform I(Q) to G(r)…",
    "はじめての精密化(ウィザード)(&W)…": "First refinement (&wizard)…",
    "連続データを順に精密化…": "Sequential refinement…",
    "Le Bail で格子定数だけ(放射光マイクロ XRD 向け)…": "Lattice parameters only (Le Bail)…",
    "Le Bail で格子定数だけ": "Lattice parameters only (Le Bail)",
    "精密化(&R)": "&Refine",
    "パターンを計算(&C)": "&Calculate pattern",
    "精密化を実行(&R)": "&Run refinement",
    "標準の手順で精密化(&S)…": "Refine with &standard recipe…",
    "PDF の標準手順で精密化…": "Refine PDF with standard recipe…",
    "中止(&T)": "S&top",
    "すべて固定(&F)": "&Fix all",
    "直前の精密化の前に戻す(&Z)": "&Undo last refinement",
    "すべて初期値に戻す(&I)": "Reset all to &initial values",
    "表示中の値を初期値に戻す": "Reset shown values to initial",
    "データ範囲と除外領域(&E)…": "Fit range and &excluded regions…",
    "ピーク幅が負にならないよう拘束する(罰則)": "Constrain peak widths to be non-negative (penalty)",
    "詳細も表示": "Show advanced",
    "試料": "Sample",
    "結果の要約を表示(&U)": "Show s&ummary",
    "残差の点検(未同定ピーク・こぶ)…": "Inspect residuals (unindexed peaks, humps)…",
    "残差の点検": "Residual inspection",
    "選んだピークを除外領域に追加": "Exclude selected peaks",
    "選んだこぶを背景のハンプに追加": "Add selected humps to background",
    "再検出": "Detect again",
    "閉じる": "Close",
    "MEM 電子密度 / 核密度(&M)…": "&MEM density…",
    "表示(&V)": "&View",
    "ヘルプ(&H)": "&Help",
    "パラメータ名の規則": "Parameter naming",
    "バージョン情報(&A)": "&About",
    "更新を確認(&U)…": "Check for &updates…",
    "起動時に更新を確認する": "Check for updates at startup",
    "利用者ガイド(&G)": "User &guide",
    "インストールと更新の説明": "Install and update notes",
    "オンラインの説明書(GitHub)…": "Online documentation (GitHub)…",
    # 利用者ガイドの窓
    "利用者ガイド": "User guide",
    "はじめに(使い方)": "Getting started",
    "インストールと更新": "Install and update",
    "説明書": "Documents",
    "目次": "Contents",
    "本文を検索(Enter で次へ)": "Search (Enter for next)",
    "前を検索": "Find previous",
    "次を検索": "Find next",
    "ブラウザで開く(GitHub)": "Open in browser (GitHub)",
    "説明書がこの配布物に入っていません。": "The user guide is not bundled with this build.",
    "GitHub の説明書を開いてください:": "Please open the guide on GitHub:",
    "説明書が見つかりません:": "Document not found:",
    "説明書がこの配布物に入っていないので、GitHub の説明書をブラウザで開きます": (
        "The guide is not bundled with this build; opening the GitHub page in the browser"
    ),
    "更新の確認": "Software update",
    "ダウンロードしてインストール": "Download and install",
    "ダウンロードページを開く": "Open download page",
    "この版は知らせない": "Skip this version",
    "あとで": "Later",
    # ツールバー
    "ウィザード": "Wizard",
    "データ読込": "Load data",
    "CIF 追加": "Add CIF",
    "装置設定": "Instrument",
    "背景設定": "Background",
    "相の設定": "Phase",
    "計算 (F5)": "Calc (F5)",
    "精密化 (F9)": "Refine (F9)",
    "標準の手順で精密化": "Standard recipe",
    "中止": "Stop",
    "PDF 読込": "Load PDF",
    "MEM": "MEM",
    "保存": "Save",
    # ドック・ラベル
    "パラメータ": "Parameters",
    "構造": "Structure",
    "χ² の推移": "χ² history",
    "プロジェクト": "Project",
    "相": "Phases",
    "ヒストグラム": "Histograms",
    "PDF データ": "PDF data",
    "データと相を読み込んでください": "Load data and a phase to begin",
    "精密化中…": "Refining…",
    "精密化中のみ": "Refined only",
    "表示中の項目に全部チェック": "Check all shown",
    "表示中のチェックを全部外す": "Uncheck all shown",
    "全て": "All",
    "格子": "Cell",
    "原子": "Atoms",
    "装置": "Instrument",
    "背景": "Background",
    "スケール": "Scale",
    # ダイアログ題名
    "背景の設定": "Background settings",
    "PDF の標準手順で精密化": "Refine PDF with standard recipe",
    "PDF データの条件": "PDF data settings",
    "I(Q) から G(r) を作る": "Transform I(Q) to G(r)",
    "MEM 電子密度 / 核密度": "MEM electron / nuclear density",
    "相の名前": "Phase name",
    "結果の要約": "Summary",
    "確認": "Confirm",
    "読み込みエラー": "Load error",
    "はじめての精密化": "First refinement",
}


def set_language(lang: str) -> None:
    global _LANG
    _LANG = "en" if str(lang).lower().startswith("en") else "ja"


def current_language() -> str:
    return _LANG


def tr(text: str) -> str:
    """日本語の文字列を現在の言語に翻訳する(辞書に無ければそのまま)。"""
    if _LANG == "en":
        return EN.get(text, text)
    return text
