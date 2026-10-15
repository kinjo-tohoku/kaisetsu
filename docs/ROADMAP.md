# Rietveld解析プログラム 開発ロードマップ

作成日: 2026-10-11
最終更新: 2026-10-11

## 進捗

| フェーズ | 状態 | 備考 |
|---|---|---|
| Phase 0 基盤 | 完了 (2026-10-11) | uv / ruff / mypy / pytest、GSAS-II 参照値スクリプト、PbSO4 公開データ |
| Phase 1 結晶学コア | 完了 (2026-10-11) | 反射リスト完全一致、\|F\|^2 一致。記録: `docs/validation/phase1_structure_factors.md` |
| Phase 2 パターン計算 | 完了 (2026-10-11) | 5 ケースで GSAS-II の ycalc と相対差 1e-4 以下。記録: `docs/validation/phase2_pattern.md` |
| Phase 3 精密化エンジン | 完了 (2026-10-11) | PbSO4 X線・中性子・同時で GSAS-II と Rwp 差 0.2% 以内、構造パラメータ 3σ 以内。記録: `docs/validation/phase3_refinement.md`。ピーク計算は numba 並列の一括カーネル + 特異点除去した FCJ 積分(1 パターン 10〜20 ms)。共分散は列を正規化した J^T J の固有値分解による擬似逆行列(特異でも \|ρ\| ≤ 1。感度の無い変数と、組み合わせでしか決まらない変数の組を報告。`np.linalg.inv` が特異な行列で桁違いの値を返し、CaF2 で \|ρ\| = 8.9 の相関が表示されていた、2026-10-14)。残課題: ヤコビアンの並列化・解析微分、Pawley 法 |
| Phase 4 GUI | v2 (2026-10-11) | PySide6: プロジェクトツリー・パターン図・パラメータ表・ログ、装置プリセット、精密化スレッド(中止可)、標準レシピ、ウィザード、保存・CSV/CIF 出力。v2 で追加: PDF 表示と読み込み/変換/標準手順、MEM ダイアログ(断面、VESTA 出力、MPF)、相の設定(Le Bail/Pawley、選択配向軸、異方性 U、Brindley)、構造 3D 表示(QPainter)、χ² 推移、図の PNG/SVG 保存、背景の種類(Chebyshev/固定点)とハンプ、Kβ。PyInstaller の spec と build スクリプト(`packaging/`)。`docs/user-guide/getting-started.md`。逐次精密化のダイアログ、英語 UI(`--lang en`、主要メニューのみ辞書翻訳)、.npz 分離保存。残課題: 身内での試用、説明文の英語化、PyInstaller 配布物の実機確認 |
| Phase 5 PDF | 完了 (2026-10-11) | 5b 実空間フィット(PDFfit2 の定式化: δ1/δ2、Q_damp/Q_broad、球状粒子、Q_max 終端効果、異方性 U)は diffpy.pdffit2 1.6.0 と Ni の X 線・中性子で全パラメータ 0.2σ 以内、Rw 一致。5a I(Q)→G(r) は PDFgetX3 型のアドホック多項式補正(容器差引・偏光・吸収・Lorch 窓)。Rietveld と同じ `Project` / `ParameterSet` に `g{i}:` のデータセットとして入り、粉末との同時精密化も可。記録: `docs/validation/phase5_pdf.md`。残課題: Compton 散乱の明示的な補正(表が必要)、GSAS-II の PDF 出力との比較 |
| Phase 6 MEM | 完了 (2026-10-11) | `kaisetsu.mem`: FFT で F を計算する MEM(L-BFGS、変数 v = 2√m、λ は割線法で C = 1 に合わせる)、Sakata-Sato の ZSPA(検算用)、負の散乱長の 2 チャンネル(正負エントロピーで 1 変数に還元)、対称操作で反射を展開、CCP4/XPLOR/VESTA grd 書き出し、断面・線分。Rietveld 分配の F_obs と σ から MEM を作る `mem_from_project`、MPF(`run_mpf`: MEM の |F| を固定した表でパターンフィッティング→再分配→再 MEM)。Pawley 法(`start_pawley`: 残差評価ごとに強度を非負線形最小二乗で解く、完全重なりは現在比で分配、σ と相関)。検証: NaCl・ルチルの合成 F で原子位置の極大と電子数を回収、PbSO4 実データで Pb に最大密度。記録: `docs/validation/phase6_mem.md`。重なり反射の G 拘束(`overlap_groups`、GUI の選択肢)も実装。残課題: Dysnomia との数値比較 |
| Phase 7(将来項目) | 一部完了 (2026-10-12) | TOF 中性子(GSAS 型プロファイル、numba 一括計算、波長依存吸収、POWGEN NAC の精密化で GSAS-II と Rwp 8.40 vs 8.47%・全パラメータ 3σ 以内、`docs/validation/phase7_tof.md`)、Sabine 消衰補正(GSAS-II と一致)、Stephens 異方的歪み、一軸異方的な結晶子サイズ(GSAS-II の uniaxial と同じ)、球面調和関数の集合組織(円筒対称、ラウエ群不変な調和関数を射影で自動生成)、逐次精密化、英語 UI(主要メニュー)を実装。Windows インストーラ(Inno Setup、`--smoke-test` による起動確認付き)と GitHub Actions の CI / Release も整備(2026-10-12)。Z-Rietveld の Type0m プロファイル(iMATERIA BS・SE の計算曲線から逆算、時定数の単位 = 擬 Voigt の FWHM、2026-10-13)、opXRD リーダーと spglib による対称性検出、公開データ 13 本での「一発の標準解析」と「真面目な解析」の比較(`docs/validation/opxrd_cases.md`、2026-10-13。Le Bail 分配の背景混入を修正。格子定数・ピーク幅が発散したときに hkl 候補とピーク窓の点数の上限で止める `PatternSizeError`、格子定数の範囲は初期値の 0.5〜2 倍)。Le Bail で格子定数だけを速く出す入口(`refine.lattice`、GUI の「Le Bail で格子定数だけ」、装置テンプレート *.kaisetsu-inst.json、一括処理 + CSV、Fit2D .chi。放射光マイクロ XRD 向け、合成データで a を 1e-4 Å、距離ずれ 0.1 mm を回収、2026-10-14)。アプリ内の更新確認(`kaisetsu.update`: GitHub Releases の最新版を 1 日 1 回確認、レート制限時はリダイレクト先のタグだけ読む。GUI はヘルプ → 更新を確認、インストーラ版はダウンロードして終了 → インストーラ起動、zip 版・Python 版は案内のみ、2026-10-15)。標準手順の改善(ウィザードの背景 12 項、Kα2 なしの自動試行、重原子が支配する相の軽原子 U_iso 共通化、放射光で SH/L、対象の無い段階の省略、残差の点検 `refine.residual` + `gui.residual_dialog`(未同定ピーク・こぶの候補、自動除外はしない)、ウィザードの Le Bail 並走、2026-10-15)。残差の広いこぶを背景のガウスのハンプにして試す段階と pdCIF リーダー(COD 2240592 の Ba(ClO4)2 で Rwp 11.5 → 7.0%、公表 6.0%。`docs/validation/baclo4_cod2240592.md`、2026-10-15)。ヘルプ → 利用者ガイド(同梱の説明書を窓で表示)。未着手: 磁気構造、2D 画像積分(pyFAI)、コード署名 |

---

## 1. 目的

初心者でも使える日本語対応の粉末回折解析プログラムを作る。
GSAS-II / FullProf は強力だが、初心者には操作が難しく日本語にも対応していない。
最終目標は Rietveld 解析だけでなく、**PDF(二体分布関数)解析**と **MEM(最大エントロピー法)電子/核密度解析**まで一貫して扱えること。

### 差別化の方針

- 日本語のGUIと、各パラメータの意味・推奨手順をその場で説明する「ガイド付き精密化」。
- Rietveld・PDF・MEM が同じ結晶構造モデルと同じ精密化エンジンを共有する(将来は Rietveld + PDF の同時精密化も可能な設計)。
- Python スクリプトからも同じ機能が使える(自動化・バッチ処理・教育用)。
- 計算式はすべて文献を明記し、GSAS-II との突き合わせ検証を記録として残す。

---

## 2. 決定事項(2026-10-11 ヒアリング結果)

| 項目 | 決定 |
|---|---|
| 対象データ | 実験室X線(CuKα 等、Kα1/Kα2二重線)、放射光X線、定波長中性子。**TOF中性子は将来対応** |
| 技術スタック | Python + NumPy/SciPy(+numba で高速化)、GUI は PySide6 + pyqtgraph |
| 計算エンジン | Rietveld 計算コア(構造因子・プロファイル・最小二乗)は自前実装。CIF・空間群・散乱因子などは商用利用可能なライブラリを利用 |
| ライセンス方針 | 依存ライブラリは商用利用可能なもの(BSD/MIT/MPL/LGPL)に限定。GPL は使わない |
| 品質水準 | 企業の研究所でも使えるレベル(検証記録・テスト・再現性・ドキュメント) |
| 最初の目標 | エンジン先行。スクリプトで実データを精密化できる状態を先に作り、GSAS-II と結果を突き合わせてから GUI に進む |
| 検証データ | 手元に実データなし。公開データ(GSAS-II 同梱の PbSO4 ラウンドロビンデータ等)と合成データで進める |
| 比較ソフト | GSAS-II 5.7.4 が `C:\Users\KatsukiKinjo\gsas2main` に導入済み(Python 3.13)。`GSASIIscriptable` で同じデータ・同じモデルを精密化し、結果を突き合わせる |
| 配布 | GitHub(kinjo-tohoku/kaisetsu)で公開。タグ `v*` を push すると GitHub Actions が Windows のインストーラ(Inno Setup)と zip を Release に載せ、`releases/latest/download/kaisetsu-windows-setup.exe` の固定 URL で配る。macOS/Linux は `uv tool install`。コード署名は利用者が増えてから |
| 名前・ライセンス | パッケージ名 `kaisetsu`(PyPI でも空いている)。自作コードは MIT(2026-10-12 決定)。依存ライブラリのライセンス一覧は `packaging/collect_licenses.py` で配布物に同梱 |
| OS | Windows 中心で開発・テスト。OS固有機能は使わず Mac/Linux でも動くよう書く |

### まだ決めていないこと(必要になった時点で確認)

- 研究室の回折装置のファイル形式(Rigaku .ras/.rasx、Bruker .raw/.brml、PANalytical .xrdml のどれを優先するか)
- docstring の言語(提案: 識別子は英語、docstring・コメントは日本語可、GUI文字列は日本語+翻訳機構)

---

## 3. 全体アーキテクチャ

```
+-----------------------------------------------------+
|  GUI (PySide6 + pyqtgraph)              Phase 4-    |  日本語UI、ガイド付き精密化
+-----------------------------------------------------+
|  Scripting API  (Project / Phase / Histogram)       |  Phase 3   スクリプト・バッチ・テスト
+----------------+----------------+-------------------+
|  Rietveld      |  PDF           |  MEM / MPF        |  Phase 2-3 / 5 / 6
|  パターン計算  |  G(r) 計算     |  密度再構成       |
+----------------+----------------+-------------------+
|  Refinement Engine (パラメータ・拘束・最小二乗・R因子・esd)   Phase 3
+-----------------------------------------------------+
|  Crystallography Core (格子・空間群・原子・構造因子・散乱因子)  Phase 1
+-----------------------------------------------------+
|  I/O (回折データ各形式・CIF・プロジェクトファイル)            Phase 1-
+-----------------------------------------------------+
```

原則:

- **コアは GUI に依存しない。** `gui/` 以外から Qt を import しない。すべての機能はスクリプトから呼べる。
- **1つのパラメータモデルを全手法で共有する。** Rietveld / PDF / MEM が同じ `Structure` と `ParameterSet` を使う。
- **数値計算は NumPy ベクトル化を基本とし、ホットループ(ピーク積算、PDF の原子対和、MEM の反復)のみ numba 化。**
- **物理式は docstring に出典(文献・式番号)を書く。**

### ディレクトリ構成(案)

```
RietveldProgram/
+- pyproject.toml / uv.lock / README.md / CLAUDE.md / LICENSE
+- src/kaisetsu/
|   +- crystal/      lattice.py, spacegroup.py, atoms.py, structure.py, cif.py
|   +- scattering/   xray.py (f0, f', f''), neutron.py (b), tables/
|   +- reflections/  hkl.py (生成・消滅則・多重度), structure_factor.py
|   +- profile/      pseudo_voigt.py, tchz.py, fcj.py (軸発散非対称), background.py
|   +- pattern/      instrument.py, histogram.py, calculator.py (y_calc の組み立て)
|   +- refine/       parameters.py, constraints.py, lsq.py, rfactors.py, recipe.py
|   +- io/           readers/ (xy, xye, gsa/fxye, xrdml, ras, brml ...), writers/ (CIF, CSV), project.py
|   +- pdf/          reduction.py (I(Q)→S(Q)→G(r)), calculator.py, fit.py
|   +- mem/          mem.py, mpf.py, grid_io.py (VESTA用出力)
|   +- api/          project.py (スクリプト用の窓口)
|   +- gui/          app.py, widgets/, models/, i18n/
+- tests/            unit/, integration/, validation/
+- validation/gsas2/ GSAS-II 側で参照値を作るスクリプト(gsas2main の Python で実行)
+- data/public/      公開テストデータ(出典・ライセンスを README に明記)
+- docs/             ROADMAP.md, theory/ (式と出典), validation/ (突き合わせ記録), user-guide/
+- examples/         スクリプト例
```

---

## 4. 技術スタックと依存ライブラリ

| 用途 | ライブラリ | ライセンス | 備考 |
|---|---|---|---|
| 数値計算 | numpy, scipy | BSD-3 | 最小二乗は scipy.optimize.least_squares から始め、必要なら自前 LM に置換 |
| 高速化 | numba | BSD-2 | Python 3.14 で 0.68 が解決済み |
| 対称性の検出(P1 → 空間群と非対称単位) | spglib | BSD-3 | `crystal/symmetry.py`。opXRD や DFT 出力の全原子セルから空間群を求める |
| CIF・空間群・異常分散 | gemmi | MPL-2.0 | `read_small_structure`, `SpaceGroup.operations()`, `cromer_liberman(z, energy)` を実機確認済み |
| 対称性(補助) | spglib | BSD-3 | 対称性検出・標準化が必要になった場合 |
| 原子散乱因子 f0 | 自前テーブル | - | Waasmaier-Kirfel (1995) / Cromer-Mann 係数を BSD 系ソース(cctbx 等)から出典明記で取り込む。gemmi 0.7.5 の Python API には IT92 テーブルが露出していないため |
| 中性子散乱長 | 自前テーブル | - | Sears (1992) 表を取り込む(periodictable(BSD/PD)からの転載も可) |
| GUI | PySide6 | LGPL-3 | 動的リンクなので商用利用可。必要なら商用ライセンスもある |
| 高速プロット | pyqtgraph | MIT | 数万点のパターン描画・ズーム |
| 静止画出力 | matplotlib | PSF系(BSD互換) | 論文用図 |
| データ保存 | h5py / JSON | BSD-3 | プロジェクトファイル |
| 検証(依存ではない) | GSAS-II 5.7.4 | 独自OSS | 別環境で実行し参照値を得るだけ。配布物には含めない |
| PDF検証(任意) | diffpy.pdffit2 | BSD系(要確認) | Phase 5 の比較用 |
| 2D画像積分(将来) | pyFAI | MIT | Phase 7 以降 |

使わないもの: GPL ライブラリ全般、RIETAN-FP / Dysnomia(クローズド。比較対象としてのみ参照)。

Python: `requires-python >= 3.12`。ローカル開発は 3.14(全依存が解決済み)。パッケージ管理は uv、ロックファイルで再現性を保証。

---

## 5. フェーズ別ロードマップ

各フェーズに「成果物」と「受け入れ基準」を置く。基準を満たすまで次へ進まない。

### Phase 0: 基盤整備

- uv プロジェクト初期化(src レイアウト)、pyproject、ruff(lint/format)、mypy、pytest + coverage。
- git 初期化、.gitignore、CLAUDE.md(規約)、LICENSE(仮)。
- `data/public/` に GSAS-II 同梱の PbSO4 ラウンドロビンデータ(`PBSO4.XRA`, `PBSO4.CWN`, `INST_XRY.PRM`, `inst_d1a.prm`, `PbSO4-Wyckoff.cif`)と `diamond.cif` を出典付きでコピー。
- `validation/gsas2/` に GSAS-II で参照値を書き出す仕組み(gsas2main の Python を subprocess で呼ぶ)。

受け入れ基準: `uv run pytest` が通る。GSAS-II 側スクリプトが PbSO4 プロジェクトを作って反射リストを JSON で出せる。

### Phase 1: 結晶学コア

- 格子: 格子定数 → 計量テンソル・逆格子・d(hkl)・体積。晶系による格子定数拘束。
- 空間群: gemmi の対称操作を利用。一般位置展開、多重度、**特殊位置の座標拘束の自動導出**(サイト対称から自由な座標成分を決める)、異方性 U_ij の対称拘束。
- 原子: 元素/イオン種、分率座標、占有率、Uiso / Uaniso。
- 反射生成: d_min 以上の全 hkl → ラウエ群で独立反射に縮約、消滅則判定、多重度。
- 散乱因子: X線 f0(sinθ/λ)、f'/f''(波長依存、gemmi)、中性子 b。
- 構造因子 F(hkl)(温度因子込み、ベクトル化)。
- CIF 読み込み(gemmi)と簡易書き出し。

受け入れ基準:
- 独立反射の hkl・d・多重度が GSAS-II の反射リストと完全一致(PbSO4, diamond, 立方/六方/三斜の各1例)。
- |F_calc|^2 が GSAS-II と相対誤差 1e-4 以内で一致(X線・中性子の両方)。
- 特殊位置拘束のテスト(例: Fd-3m の 8a, 16d、P6_3/mmc の 2c)が通る。

### Phase 2: 粉末パターン計算(フォワードモデル)

- ピーク位置: Bragg 式 + ゼロ点 + 試料変位 + 透過補正。Kα1/Kα2 二重線(強度比はパラメータ)。
- プロファイル: pseudo-Voigt → Thompson-Cox-Hastings 型(U,V,W,P / X,Y)→ Finger-Cox-Jephcoat 軸発散非対称(SH/L = S/L + H/L を 1 つの量として扱う)。
- 強度補正: Lorentz-偏光(実験室: モノクロ偏光係数、放射光: 偏光度、中性子: Lorentz のみ)、March-Dollase 選択配向、円筒試料の吸収(中性子)、表面粗さ(Bragg-Brentano)。
- バックグラウンド: Chebyshev 多項式、固定点補間、(将来)非晶質ハンプ。
- 多相混合、スケール因子。ピークごとに ±N·FWHM の窓で積算(numba)。

受け入れ基準:
- 同じモデル・同じ装置パラメータで GSAS-II にシミュレーションさせたパターンと点ごとに比較し、ピーク窓内で相対差 1e-3 以内(非対称関数の実装差は別途記録)。
- 理論検証: 積分強度が m·|F|^2·LP に比例、FWHM が Caglioti 式に一致、など単体テスト。

### Phase 3: 精密化エンジンとスクリプト API

- パラメータモデル: 名前・値・精密化フラグ・上下限・esd。相ごと/ヒストグラムごとのグループ。等値拘束・線形拘束。対称拘束は自動適用。占有率和の拘束。
- 目的関数: 重み付き残差(w = 1/σ^2)。ヤコビアンは数値微分から始め、線形パラメータ(スケール・バックグラウンド)は解析的に。
- ソルバー: Levenberg-Marquardt(scipy)。減衰・発散検出・収束判定。
- 出力: Rp, Rwp, Rexp, χ^2, 相ごとの R_Bragg, R_F、esd、相関行列、相関が高い組の警告。
- Le Bail / Pawley 法(格子・プロファイルの先行決定と、MEM 用の強度抽出に必要)。
- 「レシピ」: 精密化手順を宣言的に書く仕組み(スケール+BG → 格子+ゼロ点 → プロファイル → 座標 → 温度因子 …)。初心者向け自動手順の基盤になる。
- プロジェクトファイル(JSON + 数値配列はHDF5/npz、スキーマにバージョン)。
- 回折データ読み込み: .xy/.xye/.dat、GSAS .gsa/.fxye、PANalytical .xrdml、Rigaku .ras(研究室の形式は確認後に優先)。

受け入れ基準(GSAS-II 突き合わせ、`docs/validation/` に記録):
- PbSO4 実験室X線、PbSO4 D1A中性子、両者の同時精密化の3ケースで、
  Rwp の差が 0.5%(絶対値)以内、格子定数・原子座標・Uiso が互いの 3σ 以内。
- 合成データのラウンドトリップ: 既知パラメータで生成+ノイズ → 初期値をずらして精密化 → 真値を 3σ 以内で回収。
- 放射光データ1例(GSAS-II チュートリアル公開データから選定)でも同様に一致。

### Phase 4: GUI v1(日本語)

- 画面: プロジェクトツリー(相・ヒストグラム)、パターン図(観測/計算/差/BG/ティック、ズーム、カーソル位置の d 値表示)、パラメータ表(チェックで精密化ON/OFF)、R因子の推移、ログ。
- 初心者モード: 「データを読む → 相を追加(CIF) → 装置を選ぶ(プリセット: CuKα Bragg-Brentano / 放射光 / 中性子) → 推奨手順で精密化」のウィザード。各パラメータに日本語の説明ツールチップ。発散・負値・高相関の警告を平易な日本語で表示。
- 精密化はワーカースレッドで実行し、途中停止できる。
- 出力: CIF(pdCIF)、図(PNG/SVG)、CSV、VESTA 用ファイル。
- 配布: PyInstaller の one-folder で Windows 配布。`uv run` でも起動可。
- 文字列は `tr()` で囲み、英語化できる構造にしておく(翻訳は後)。

受け入れ基準: 初心者が標準試料(Si/LaB6 相当の公開データ)を 10 操作以内で精密化し、Rwp と格子定数を得られる。身内 2〜3 名の試用でつまずいた点を記録・修正。

### Phase 5: PDF 解析

**5a. 全散乱データ処理**(I(Q) → S(Q) → F(Q) → G(r))
- 補正: 容器・空セル差し引き、コンプトン散乱(X線)、吸収、偏光、Faber-Ziman 規格化、PDFgetX3 型の多項式アドホック補正、Q_max 打ち切り、Lorch 窓(任意)、正弦フーリエ変換。
- 検証: 同じ I(Q) に対して GSAS-II の PDF 機能(GSASIIpwd)または PDFgetX3 の出力と G(r) を比較。

**5b. 実空間フィット**(PDFfit2 / PDFgui と同じ定式化)
- G_calc(r): 原子対和(スーパーセル近傍探索、numba)、ADP 由来のガウス幅、相関運動補正 δ1/δ2、Q_damp・Q_broad、球状粒子形状因子、スケール。
- Rietveld と同じ `Structure` / `ParameterSet` / 精密化エンジンを使う。
- 検証: diffpy.pdffit2 の公開 Ni データで、G_calc と精密化後パラメータを比較。

受け入れ基準: Ni(fcc)の格子定数・Uiso・Q_damp が pdffit2 の結果と 3σ 以内。G(r) の計算値の残差が数値精度レベル。

### Phase 6: MEM 解析と MPF

- 入力: Rietveld から得た F_obs(hkl)(重なり反射はモデルで分配)と σ、または Pawley/Le Bail 抽出強度 + モデル位相。グリッド(例 128^3)、空間群で対称化、F000。
- アルゴリズム: Sakata-Sato / Collins の 0次単一ピクセル近似(ZSPA)を実装して正しさを確認 → L-BFGS 型(Dysnomia 相当)で高速化。F 拘束(χ^2 = 1)、重なり反射の G 拘束、不確かさ重み。中性子の負の散乱長に対応する二チャンネル MEM。
- 出力: VESTA で読める 3D グリッド(.ccp4 / .xplor / .pgrid 相当)、GUI での 2D 断面図。
- MPF(MEM-based pattern fitting): Rietveld → MEM → MEM 密度から F を再計算 → 重なり反射を再分配 → 再 MEM、を収束まで繰り返すレシピ。
- 検証: 既知構造から計算した F にノイズを加えた合成データで密度ピーク位置・積分電荷を回収。Dysnomia(無償配布)の例題があれば密度分布を比較。

受け入れ基準: 合成データで原子位置に密度極大が再現され、残差 χ^2 が 1 に収束する。実データ(PbSO4 等)で化学的に妥当な密度図が得られ、MPF で Rwp が Rietveld 単独より改善する。

### Phase 7 以降(将来)

- TOF 中性子: 実装済み(`pattern/tof.py`、`profile/tof_profile.py`。GSAS type 3 相当の閉形式、GSAS-II の POWGEN データで積分強度 1% 一致。記録: `docs/validation/phase7_tof.md`)。波長依存の円筒吸収(μR ∝ λ)と Sabine 消衰(CW でも使える、GSAS-II の `GetPwdrExt` と一致)も実装済み。複数バンクの一括読み込み(GUI の「すべてのバンク」)も可能。
- 磁気構造(中性子)。
- 微細構造: 歪みの異方性(Stephens 型)は実装済み(`profile/stephens.py`、ラウエ群の拘束は射影で自動)。
  結晶子サイズの一軸異方性も実装済み(`PhaseContribution.size_axis` / `size_axial_um`、
  1/p(φ) = sqrt(sin²φ/p_i² + cos²φ/p_a²)、CW と TOF の両方)。楕円体モデル、Williamson-Hall は未実装。
- 定量相分析、非晶質量。
- 球面調和関数による集合組織: 実装済み(`profile/texture.py`。試料は円筒対称を仮定し、結晶座標系の
  ラウエ群不変な実球面調和関数(l ≤ 10)を回転平均の射影から数値的に作る。係数は `h{i}:相:sh{l}_{j}`、
  集合組織指数 J = 1 + ΣC² を要約に表示)。試料方位を陽に持つ一般の場合(GSAS-II の SH)は未実装。
- 逐次精密化(温度・時間系列)。
- 2D 検出器画像の積分(pyFAI)。
- 英語 UI、インストーラ、コード署名。

---

## 6. 検証戦略

1. **単体テスト**: 純粋関数(d 値、多重度、f0、プロファイル関数の積分=1 など)。
2. **性質テスト**: 対称操作で F が不変、Friedel 対、格子定数を変えたときの d の単調性。
3. **回帰テスト**: 固定入力に対する数値を固定し、意図しない変化を検出。
4. **ラウンドトリップ**: 合成パターンから真値を回収。
5. **外部突き合わせ**: GSAS-II(Rietveld)、pdffit2(PDF)、Dysnomia(MEM)。比較スクリプトとその結果(バージョン・日付・差分表)を `docs/validation/` に残す。
6. **公開実データ**: GSAS-II 同梱 PbSO4(実験室X線・D1A中性子)、GSAS-II チュートリアル公開データ(放射光・中性子)、diffpy の Ni PDF データ。出典とライセンスを `data/public/README.md` に明記する。
7. **初心者の使い方の再現**: opXRD(Zenodo、CC BY 4.0)の構造ラベル付きパターンに GUI の既定で一発の標準解析を当て、人が順に確かめる手順(異物除外・ハンプ・Kα2・背景・配向・Stephens)と Le Bail を並べて、「一発で何がダメか」を記録する(`docs/validation/opxrd_cases.md`)。標準手順やウィザードの既定を変えたら再実行する。

---

## 7. 品質基準(企業の研究所で使えるレベル)

- 物理式には必ず出典を書く(Young『The Rietveld Method』、Pecharsky & Zavalij、GSAS-II マニュアル、FullProf マニュアル、Finger-Cox-Jephcoat 1994、Thompson-Cox-Hastings 1987、Dollase 1986、Sakata & Sato 1990、Takata 2008、Juhás 2013、Proffen & Billinge 1999、Farrow 2007 など)。
- 型ヒント必須、ruff + mypy をクリーン、コアのテストカバレッジ 80% 以上。
- 再現性: uv.lock で依存固定、乱数シード固定、numba は fastmath を既定で無効。
- 数値: 倍精度、微分は有限差分との突き合わせテスト、桁落ちしやすい式(TCH の η・Γ)は安定形で実装。
- ユーザー向けメッセージ(エラー・警告・ログ)は日本語。内部ログは構造化(logging)。
- プロジェクトファイルにスキーマバージョンを持ち、旧バージョンを読めるようにする。
- 変更は必ず git 管理。検証記録は結果と一緒にコミット。

---

## 8. 次の一歩

Phase 0〜6 は完了(進捗表参照)。この先の優先順位:

1. 身内 2〜3 名での試用と、つまずいた点の修正(入門ガイドの改訂)。研究室の回折装置のファイルを 1 つ受け取って
   リーダーを確認する(Bruker RAW4.00 は合成テストのみ)。
2. TOF 中性子は実装済み(吸収・消衰・複数バンク・GSAS-II との精密化比較も)。残りは磁気構造など。
3. Dysnomia との MEM の数値比較(例題データが要る)。磁気構造。結晶子サイズの楕円体モデル。
4. 説明文・ダイアログ本文の英語化(主要メニューは `--lang en` で済んでいる)。GitHub に push 済み。Release は
   `v0.1.0`(2026-10-12)、`v0.1.1`(2026-10-13: Type0m、opXRD、Le Bail 分配の修正、発散ガード)、
   `v0.1.2`(2026-10-13: Type0m τ=Γ、Chebyshev 背景の定義域)、`v0.1.3`(2026-10-15: Le Bail 格子定数、
   初期値に戻す、偽の相関の修正)、`v0.1.5`(2026-10-15: 標準手順の改善、下記 5)、`v0.1.6`(2026-10-15:
   利用者ガイド F1、残差のこぶ → 背景ハンプの自動試行、pdCIF リーダー、Ba(ClO4)2 サンプル)、
   `v0.1.7`(2026-10-15: パラメータ欄を狭くできるように、窓の配置の記憶)。アプリ内の
   更新確認は v0.1.4 から。説明書(`docs/user-guide`)は配布物に同梱し、ヘルプ → 「利用者ガイド」(F1)で
   アプリの中から読める(2026-10-15)。コード署名は利用者が増えてから。
5. (2026-10-15 に実装)opXRD 13 本の比較(`docs/validation/opxrd_cases.md`)から出た標準手順の改善:
   ウィザードの背景 12 項、Kα2 なしの自動試行(0.3 ポイント改善で採用)、重原子が散乱能の 70% 以上を占める相の
   軽原子 U_iso 共通化、放射光で SH/L、未同定ピークとこぶの候補提示(「残差の点検」、自動除外はしない)、
   ウィザードの Le Bail 並走。Stephens と選択配向の自動試行は過剰パラメータの懸念があるので手動のまま。
   同日追加: 「残差のこぶを背景に試す」段階(ピーク幅のあとで残差の広いこぶをガウスのハンプにして
   Rwp 0.5 ポイント改善で採用。ガラスキャピラリ・非晶質向け。PbSO4 に幅 12° のこぶを合成した試験で
   Chebyshev 12 項 Rwp 28% → ハンプ 8%、20 項でも 11%)。人が足したハンプ(h*:hump*:*)も
   「スケールと背景」で動く(以前は標準手順で動かなかった。Ba(ClO4)2 のガラスのこぶが合わない報告から)。
   こぶの候補の高さの下限は最強ピークの 1%(5% では COD 2240592 の Ba(ClO4)2 で多項式が吸い残した
   2% のこぶを拾えなかった。`docs/validation/baclo4_cod2240592.md`)。同じ CIF に粉末データが入っている
   pdCIF(COD、Acta E)を `io/readers/pdcif.py` で読む。
   パラメータ欄はノート PC で狭くできなかった(ドックの最小幅 400 px と横並びのボタン 9 + 3 個で
   600 px 以上)ので、ボタンの行を折り返す `gui/flow_layout.py` にして最小幅を外し(約 220 px)、
   窓の大きさとドックの配置を QSettings に記憶する(2026-10-15、v0.1.6 の後)。テストが利用者の
   レジストリを汚さないように `KAISETSU_SETTINGS_FILE` で設定の置き場所を ini に切り替えられる。
