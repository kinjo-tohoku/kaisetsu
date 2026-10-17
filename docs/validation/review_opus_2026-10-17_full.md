# Opus 5.5(xhigh)によるレビュー 2026-10-17

対象: kaisetsu v0.1.13(HEAD 7f4f69d)。4 本の独立レビュー(差分 1 本 + 全体 3 本)。
リポジトリは変更していない。実験スクリプトは scratchpad の `review_opus\`(差分)と
`review_opus_all\{physics,refine,gui_io}\`(全体)にある。

## 統合した順位(初心者の実験室データへの実害で並べ替え)

### A. 黙って間違う・結果を失う(すべて CONFIRMED)

| # | 件名 | 場所 | 出典 |
|---|---|---|---|
| A1 | 平板の標準手順がゼロ点しか動かさず、試料高さのずれを格子定数が吸う(0.15 mm で −213 ppm = esd の 575 倍、χ² 1.27 で気づけない) | refine/recipe.py:647,702、gui/wizard.py:155 | refine #1 |
| A2 | 3 桁に丸めた特殊位置 (0.333, 0.667, z) が認識されず多重度 3 倍・|F|² 9 倍(ZnO) | refine/constraints.py:122、crystal/spacegroup.py:252,277 | physics #1 |
| A3 | 窓を閉じても未保存の確認が出ない。変更フラグ自体が無い | gui/main_window.py:1523 | gui #1 |
| A4 | 保存の失敗が見えない(配布物は console 無し)。先に切り詰めてから書くので途中失敗で元ファイルが壊れる | gui/main_window.py:676、io/project.py:514 | gui #2 |
| A5 | 精密化の実行中に他の操作がほぼ全部有効。除外領域追加でワーカー失敗、CIF 追加/装置設定で ParameterSet 差し替え→表とモデルが食い違い保存で結果消失、numba workqueue 層ではプロセス即終了 | gui/main_window.py:1274 | gui #3 |
| A6 | 「連続データを順に精密化」が本体 Project を最後のファイルで書き換え、Ctrl+S で元ファイルが混ざる | gui/main_window.py:1397、refine/sequential.py:134 | gui #4 |
| A7 | ヘッダ無し 3 列 .xye が RIETAN 形式と誤判定(1 行目が整数 2θ で始まると 10 点のゴミ)。小数点カンマも同様 | io/readers/rietan.py:252、readers/__init__.py:69 | gui #5 |
| A8 | CW の試料広がりでガウス成分に 8 ln2 が余分(ローレンツ割合 < 1 で幅 2.355 倍。既定 1.0 では無影響。TOF 側は正しい) | pattern/calculator.py:167,394 | physics #2 |
| A9 | 対称操作の無い CIF で原点選択が入口で違う(H-M だけ→:2、番号だけ→:1)。スピネル手書き CIF で多重度が入れ替わる | crystal/cif.py:341-355 | physics #3 |
| A10 | PDF の Q_max がデータの上限で切られない(Cu Kα を既定 25 で変換すると U 4 倍、Rw 38%) | pdf/transform.py:252,266、gui/pdf_dialogs.py:167 | refine #2 |
| A11 | Lorch 窓を掛けた G(r) を窓を知らないモデルでフィット(U 2.2 倍) | pdf/transform.py:179、pdf/calculator.py:255 | refine #3 |
| A12 | MEM(X 線)の F_obs に f′/f″ が残ったまま F000 は Σf0(PbSO4 で Pb 球内電子数 −4%、最大密度 −9%) | mem/mpf.py:54-65,109 | refine #4 |
| A13 | 結晶子サイズを 0(= 無限大)から動かすと 0.0001 μm に潰れて success=True | refine/project.py:172,1023 | refine #5 |
| A14 | 同名ヒストグラムで R 因子の辞書が衝突(PBSO4.XRA + PBSO4.CWN、同じファイル 2 回、ウィザード 2 回) | refine/project.py:1713 | refine #6、gui #15 |
| A15 | ウィザードを 2 回完了すると前のデータに足されて同時精密化 | gui/wizard.py:262 | gui #6 |
| A16 | 別プロジェクトを開いた後も Ctrl+Z が前のプロジェクトの値を書き込む | gui/main_window.py:635,657 | gui #7 |
| A17 | CIF の座標 '1/3' や '?' が NaN のまま通る | crystal/cif.py:286、crystal/atoms.py:306 | physics #4 |

### B. v0.1.13 の直し残し(差分レビュー)

| # | 件名 | 判定 |
|---|---|---|
| B1 | MEM λ 探索: 遠い λ で打ち切った C が lo/hi を汚染する根本原因は残る(予算 400/150 で 27 ケース中 旧 17 → 新 21 成功)。既定予算では全件収束するが 2 チャンネルで内側反復 1.4〜2.7 倍、test_two_channel_neutron 5.0 → 10.7 s | CONFIRMED |
| B2 | 未収束表示が逐次精密化の表・CSV・summary・CLI・diagnostics・結果の要約に出ない。ガイド「表の備考に出ます」が実装と食い違う(ダイアログのログ行だけ) | CONFIRMED |
| B3 | v0.1.12 以前で保存したプロジェクトの壊れた座標(I4/mcm 8h が (x,x,0)、多重度 16)は v0.1.13 で読んでもそのまま、警告無し。文書にも無い | CONFIRMED |
| B4 | 逐次精密化の振り直しで例外が出ると成功していた 1 回目まで失敗扱い(restore されない) | CONFIRMED |
| B5 | #3 の判定をすり抜ける拘束(逆順の連鎖、同相の格子→構造従属)。API 経由のみ | CONFIRMED・低 |
| B6 | 文書: max_nfev の式、MEM の変更が docs/validation に無い。lsq は scipy のメッセージ文字列判定(status を持つ方が堅い) | 低 |

差分レビューの結論: 外部レビュー 5 件は直っている。#1 は gemmi 全 560 設定・特殊位置 6,135 個で失敗 0(旧版は 1,076 失敗)。527 本全部通過、ruff/mypy クリーン。

### C. セキュリティ・配布・ライセンス

| # | 件名 | 判定 |
|---|---|---|
| C1 | 更新のインストーラをハッシュでも Authenticode でも検証せず起動。Releases API の asset digest で照合可 | CONFIRMED(読み) |
| C2 | 同梱ライセンス一覧に Qt(PySide6)の LGPL-3.0 / GPL-3.0 本文が無い(LGPLv3 §4 の要件)。OpenSSL も 0 件。dev グループ(coverage、hypothesis、setuptools)が配布物に入っているのに一覧に無い | CONFIRMED |
| C3 | アンインストールが `{app}` を丸ごと消すのに、保存の初期フォルダが `_internal/data/public` の中 | PLAUSIBLE |
| C4 | Release を公開状態で作ってから asset を上げる(その間 404)。Actions がタグ固定。release.yml で pytest 無し | 読み |

### D. その他(頻度低・性能・使いにくさ)

- ウィザードの既定装置がモノクロメータ(P = 0.556)。Ni フィルタ機で U_iso +0.0013 Å²(physics #5)
- ウィザードのファイル選択に .ras/.rasx/.xrdml/.csv が無い(gui #8)
- XRDML の Kα1 単色でも Kα2 を拾う、beamAttenuationFactors 無視、RAS の減衰器係数が重みに反映されない、cps 未対応(gui #9)
- 重み 1/y_obs で低計数の背景が −25% 偏る(1/max(y_calc,1) で解消)、背景差引データで 1/3 の点が黙って捨てられる(refine #10)
- CIF の異方性 U を X 線の標準手順がそのまま動かす。正定値の点検無し(refine #11)
- PDF: U = 0 のサイトでピーク面積が保存されない(配位数 12 が 0.2〜8.7)(refine #7)
- 半解析ヤコビアンで数値微分列の後の解析列が摂動後の state で作られ esd 0.3% ずれ(refine #8)
- refine_lattice の体積 esd が格子定数の相関を無視(0.6〜0.7 倍)(refine #9)
- MEM: 格子点数の検査が独立反射しか見ない、Pawley 経路の two_channel=None、N_F の数え方(refine #12)
- 逐次精密化: 範囲が狭まる一方、失敗時に悪い値が次の初期値(refine #13)
- CIF のラベル重複で落ちる、B 2/m 等で crystal_system が落ちる(physics #7, #8)
- March-Dollase が反射ごとの Python ループ(PbSO4 で 33 ms/回)(physics #9)
- 更新確認/精密化スレッドが残ったまま終了すると異常終了(exit 127)(gui #13)
- CSV: 相名のカンマで列ずれ。配布物の PYTHONUTF8 が効いておらず cp932(gui #14)
- Ctrl+Z が Kα2 の試行・ハンプ追加を戻さない(gui #16)
- BOM 付きで最初の 1 点が落ちる、UTF-16、未知バイナリ、3 列目を無条件に σ、複数レンジの先頭だけ、装置設定 OK で桁丸め、CLI の生エラー、tr() 漏れ(gui #17)
- 軽微: CW d_max の低角余白、TOF/CW の歪み定義 2 倍差、profile→refine の層逆転、U_eq の対角平均、aniso_B 未対応(physics #10)

### 合っていたと確認されたもの

構造因子(手計算 1e-16)、異方 ADP の等価反射不変、消滅則・多重度、f′/f″(IT Vol.C)、中性子散乱長の符号、LP 因子、試料変位・透過、円筒吸収・Suortti、pseudo-Voigt 面積、TCH、FCJ、非標準設定の組み立て、重量分率、esd(モンテカルロ 0.74〜1.18)、R 因子の定義、解析ヤコビアンの式(1e-13)、拘束(昨日の修正含む)、Le Bail/Pawley、cell_scan、MEM の FFT・対称展開・勾配・2 チャンネル、PDF の Faber-Ziman と PDFfit2 式、Qt の import 境界、プロジェクト往復、日本語パス、更新の版比較・オフライン・.part、release.yml のタグ照合。

### 共通して挙がったテストの穴

GSAS-II との比較が PbSO4(Pnma)と NAC TOF だけ。2 相・重量分率、Le Bail/Pawley、選択配向、混合サイト、異方性 U、平板の変位/透過、非中心対称、CW の LGmix < 1、六方晶/単斜晶の反射リストが比較されていない。統合テストは自分の合成データの往復のみ。GUI は閉じる時の確認・保存失敗・実行中ガード・逐次精密化後の状態・ウィザード 2 回目を見ていない。

---

# 以下、4 本の報告の全文

## 1. 差分レビュー(bba63a6 → 7f4f69d)

## 結論

外部レビューの 5 件は直っています。v0.1.12 向けの再現スクリプトを現在のコードで流すと、5 件とも修正後の値になりました(並進付き特殊位置の座標 (0.2, 0.6, 0) と |F|² が不変、CIF の占有率 [1, 0] が保持、跨ぎ拘束の背景が 2.0、ZnS 111 の相対誤差 2e-16、反転したときのパターン差 3e-16、未収束の注意文あり)。

#1 は数学的に正しく、全空間群で確かめました。全テスト 527 本(既存 497 + 新規 30)が通り、ruff / mypy もクリーンです。今回の修正で既存の結果が壊れた箇所は見つかりませんでした。

ただし次の 3 点が残っています。
- #6 の MEM の修正は症状を抑えただけで、根本原因は残っています。コミットメッセージの「内側の予算 2000 / 400 / 150 のどれでも収束」は 1 ケースでしか成り立ちません。
- #5 の未収束表示は、逐次精密化の表と CSV、CLI、「結果の要約」に出ません。利用者ガイドの「表の備考に出ます」はコードと食い違っています。
- v0.1.12 以前に保存したプロジェクトでは、壊れた座標がそのまま残ります。そのことがどこにも書かれていません。

## 指摘(重い順)

### 1. [中] MEM の λ 探索: 括りの汚染が直っておらず、2 チャンネルでは遅くなった
- **場所**: `src/kaisetsu/mem/core.py:496`(hard_cap / chunk)、`:546`(near_target の更新)、`:548`(lo / hi の更新)、`:558`(上限到達時に best を返す)
- **何が起きるか**:
  - 目標から遠い λ では、予算 100〜400 回で打ち切った C が lo / hi(λ を挟む上下の値)と割線法の点に使われます。打ち切りの C が偶然「目標を少し超えた」値になると、その λ が lo に入ります。本当は、そこで収束させた C は目標よりずっと小さいのに、です。
  - 以後の探索は lo に向かって縮み、C は 0.54 などに張り付いたまま上限(25 回)に達します。
  - 上限に達すると「目標に最も近い解」として、打ち切り途中の密度(C = 1.48、100 反復ぶん)を返します。
  - 深い反復のあとで C が 1/3 未満になると near_target が False に戻り、また打ち切りに戻る経路もあります。
  - 今回の修正は「目標の近くで深く回す」だけなので、この失敗はそのまま残ります。
- **根拠(実行)**: `probe_mem.py` と `probe_mem_trace.py` で、`tests/unit/test_mem.py` と同じ合成入力を使いました。目標 C = 1 から ±0.05 以内に入れば成功とし、外側の探索が上限に達した件数も数えました。

  | 条件 | 旧版 | 新版 |
  |---|---|---|
  | 予算 400 / 150(開発者と同じ再現方法)、3 ケース × seed 0〜2 | 成功 17/27、上限到達 13 | 成功 21/27、上限到達 8 |
  | 既定の予算 2000、seed 0〜12 | ルチル中性子 seed 7 で C = 1.77 のまま上限到達 | 全件収束 |
  | 既定の 2 チャンネル(ルチル中性子)の内側反復数 | — | 平均約 1.4 倍、seed 5 で 5,613 → 15,213(2.7 倍) |
  | `test_two_channel_neutron` | 5.0 s | 10.7 s |
  | `test_mem.py` の高速分 | 12.4 s | 18.5 s |

  - 新版でも上限に達した 8 件: 中性子ルチル seed 1 の 400 / 150 と、X 線ルチル seed 0〜2 の 400 / 150。
  - 新版の X 線ルチル seed 0(予算 400)のトレース: 外側 4 回目で λ = 0.1513、打ち切り後の C = 1.4825 となり、これが lo に入りました。5 回目以降は収束した C が 0.32〜0.54 で、λ が 0.1513 に縮んだまま 25 回で終わりました。
  - 既定の予算では新版の方が改善しています。macOS の CI が通ったのは、経路が変わったためかもしれません。
  - CONFIRMED
- **直し方の案**:
  - lo / hi は収束した内側の結果(status 0、または C が落ち着いたもの)からだけ更新し、打ち切りの点は外挿にだけ使う。
  - または、一度 near_target になったら戻さない。
  - または、括りが十分に狭まっても |C − 目標| が大きいときは、lo を深い予算で解き直して確かめる。
  - 上限到達時に返す best も、収束した解の中から選ぶ。

### 2. [中] 未収束の表示(#5)が届かない出口がある
- **場所**:
  - `src/kaisetsu/gui/sequence_dialog.py:145`: 表の列は データ / Rwp / χ² / パラメータだけで、備考の列がありません。
  - `src/kaisetsu/refine/sequential.py:61`(`to_csv` のヘッダ)と `summary()`: message を出しません。
  - `src/kaisetsu/refine/project.py:1776`: `diagnostics()` に収束の判定がありません。
  - `src/kaisetsu/cli.py` の `_report`: diagnostics しか出しません。
  - `docs/user-guide/getting-started.md:399`
- **何が起きるか**:
  - 逐次精密化では、注意が出るのはダイアログのログ行だけです。表にも CSV にも残らないので、逐次精密化の本来の成果物である CSV では未収束の行が分かりません。CLI に `--quiet` を付けると完全に消えます。
  - `kaisetsu refine --recipe current --cycles 1` は success=False でも注意を出さず、exit 0 で保存します。
  - 「結果の要約」と、Le Bail 一括処理の warnings 列にも出ません。
  - GUI のステータス行は 1 行目(「完了: …」)だけを表示します。注意はログにだけ出ます。
- **根拠**: `probe_cli_unconverged.py` で、CLI の出力に注意が無いこと、`diagnostics()` に収束の文言が無いこと、CSV と `summary()` に「未収束」が無いことを確かめました。CONFIRMED
- **直し方の案**:
  - `convergence_note` を `Project.diagnostics()` に入れる。これで要約、CLI、lattice の warnings にまとめて出ます。
  - `SequenceResult.to_csv` と `summary()` と GUI の表に message(備考)列を足す。
  - ガイドの記述を実装に合わせる。

### 3. [中〜低] v0.1.12 以前で保存したプロジェクトは壊れた座標のまま
- **場所**: `docs/validation/review_2026-10-17.md:30`(影響範囲の節)、リリース記録
- **何が起きるか**:
  - 旧版で拘束を一度でも評価して保存したプロジェクトは、ずれた座標をファイルに持っています。
  - I4/mcm の 8h(傾斜ペロブスカイトの O)では (x, x+1/2, 0) が (x, x, 0) になり、多重度が 8 から 16 に倍増していました。
  - v0.1.13 で読み込んでもそのままで、警告も出ません。ずれた点も特殊位置として受け入れられるからです。
  - 文書には「COD で 13 件」とあるだけで、既存プロジェクトの扱いが書かれていません。
- **根拠**: `probe_oldproject.py` で、旧版で保存した結果は O2 = (0.7741, 0.7741, 0)、多重度 [4, 4, 4, 16] でした。新版で読み込んでも同じでした。CONFIRMED
- **直し方の案**:
  - リリースノートとガイドに「該当空間群(Ia-3d、I4/mcm、P4bm、I41/amd、P-421m、P213、Pn-3m など)のプロジェクトは CIF から読み直す」と書く。
  - 保存された version が 0.1.13 未満で、座標の従属拘束を持つサイトがあれば、読み込み時に注意を出す。

### 4. [低] #3 の判定をすり抜ける拘束(API・プロジェクトファイル経由のみ)
- **場所**: `src/kaisetsu/refine/jacobian.py:169`、`src/kaisetsu/refine/parameters.py:188`
- **何が起きるか**:
  - (a) `ParameterSet.tie` の連鎖チェックは「主がすでに従属か」しか見ていません。逆順に作ると連鎖ができます: `tie(h1:r:scale, r:O:uiso)` のあとに `tie(r:O:uiso, r:Ti:uiso)`。`dependents` は直接の従属しか見ないので、r:Ti:uiso の列は解析のままです。
  - (b) 同じ相の格子定数を構造パラメータに従属させると、`startswith(f"{phase}:")` の判定を通って解析のままになります。
- **根拠**: `probe_jac_ties.py` で、中心差分の数値微分との列の相対誤差を見ました。新旧とも (a) 1.09、(b) 1.00 です。どちらも病的なケースで、GUI からは作れません。CONFIRMED
- **直し方の案**:
  - `tie()` で `name` がすでに他の拘束の主なら拒否する。
  - jacobian の判定は従属を `classify()` して、`("intensity", None, phase)` 以外があれば数値微分に落とす。
  - `dependents` は推移閉包で作る。

### 5. [低・既存] 逐次精密化の振り直しで例外が出ると、成功していた 1 回目まで失敗になる
- **場所**: `src/kaisetsu/refine/sequential.py:153`、`:158-159`、`:165`
- **何が起きるか**:
  - 1 回目は成功したが Rwp が悪く、振り直しの `_refine_once` が例外を出すと `failure = exc` になります。行は失敗扱いになり、`restore(end)` もされません。
  - `scan_all_cells` は try の外にあるので、それが例外を出しても同じです(こちらは読んだだけ)。
  - 今回の差分で手を入れたブロックの中です。
- **根拠**: `probe_seq_retry.py` で T1 FAILED "retry blew up" を確認しました。CONFIRMED(scan_all_cells の分は PLAUSIBLE)
- **直し方の案**: 1 回目が成功していれば、振り直しの例外は備考に記録するだけにして、`end` に戻す。

### 6. [低] 細かい点
- **`src/kaisetsu/refine/lsq.py:322`**: scipy のメッセージ文字列で判定しています。scipy 1.18.1 の `TERMINATION_MESSAGES[0]` と一致することは確認しました。`FitResult` に `res.status`(0 = max_nfev)を持たせる方が確実です。
- **消衰との組み合わせ**: Sabine 消衰を使うときは E(平均 |F|²) で近似しています。厳密には両反射の補正後の強度を足すことになります。レビュー #2 で言及されていた点で、文書には書かれていません。PLAUSIBLE
- **MEM の |F_obs|**: `mem/mpf.py` の |F_obs| は `tbl.f2`(f'' 成分 |B|² を含む)から作っていますが、位相は f'' を除いて計算しています(既存の不整合)。PLAUSIBLE
- **文書の式**: `docs/validation/review_2026-10-17.md:93` は max_nfev = 3 × max_cycles と書いていますが、コードは max(3·max_cycles, 5) です。
- **記録の置き場所**: MEM の変更は ROADMAP にしかありません。`docs/validation/` には記録がありません。

## 確認したこと
- **#1(特殊位置の定数項)**
  - xs は、サイト対称操作の並進を格子並進で補正した上での軌道平均です。これは安定化群の不動点になるので、`project_values=False` は正しいです。
  - gemmi の全 560 設定(原点選択・菱面体設定を含む)で、6,135 個の特殊位置(定数項が 0 でないもの 1,069)を調べました。従属関係が xs で成り立つこと、主座標を動かしても全サイト対称操作で不変なこと、多重度が変わらないこと、n_free = rank(P) を確かめ、失敗は 0 件でした。tol 内の ±2e-4 の摂動を入れても 0 件です。同じ検査を旧版にかけると 1,076 件が失敗し、検査そのものが機能していることも確かめました。
  - ADP と Stephens は既定の True のままで、挙動は変わっていません。同梱 CIF 9 本は新旧とも座標が動きません。
- **#2(Friedel 対の平均)**
  - F(−h) = conj(A) + i·conj(B) なので、平均は |A|² + |B|² になります(交差項は消える)。f' は A 側、f'' は B 側で、占有率・温度因子・位相の掛け方と chunk の境界も正しいです。
  - 中心対称(Ba(ClO4)2)では旧式と 7e-15 で一致し、速度も同じでした(9.6 ms と 10.0 ms)。
  - 呼び出し側はパターン計算(CW・TOF)、ヤコビアン、Le Bail の初期強度で統一されています。残る `structure_factors` の使用は MPF の位相計算だけで、f'' を意図して除いています。
  - GSAS-II の `GSASIIstrMath.py` の 1247 行(平均形がコメントアウト)と 1248 行(代表 1 本の式)は、文書の記述どおりでした。
- **#3(跨ぎ拘束のヤコビアン)**: 中心差分の数値微分との列の相対誤差で比べました。
  - 旧版で 0.42〜0.81 ずれていた、ヒストグラム跨ぎのスケール、相跨ぎの U_iso、相が別のヒストグラムにある場合、構造パラメータに従属するヒストグラムパラメータ、跨ぎの md_r は、新版では前進差分の誤差の範囲(3e-3 以下)に収まります。
  - 同じ相の拘束は解析のままで、相対誤差 1e-13 です。
  - 数値微分の列は `_perturbed`(拘束の評価と書き込み)をしてから残差を計算しています。
- **#4(CIF の占有率 0)**: 列なし・`?`・`.` は 1、0 は 0、0.25 は 0.25、1.2 は 1、負はエラーになり、往復でも保たれることを、テストと再現スクリプトで確認しました。
- **#5(未収束の表示)**
  - `real_fit` の流れ(1 回目の結果、振り直しを採用した場合、不採用で esd を付け直す 1 サイクル)は正しいです。
  - 標準の手順を PbSO4、Ba(ClO4)2、FAP にかけると全段階が収束し、最後の注意は出ませんでした。つまり誤報はありません。
- **新規テスト 30 本**: 旧版で動かすと 18 本が失敗し、2 ファイル(6 本)は新しい関数が無いため読み込み時点で失敗します。旧版でも通る 6 本は、占有率の既定値 5 本と、収束したときに静かなことを見る 1 本で、守りのテストとして妥当です。既存テストの変更はありません。
- **ruff / mypy**: `uv run ruff check .`、`ruff format --check .`(236 files)、`mypy`(103 files)はすべてクリーンです。
- **実行したテスト**(いずれも `PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR=C:/Windows/Fonts uv run pytest -q -p no:cacheprovider` に以下を付けて実行):

  | 対象 | 付けた引数 | 結果 | 時間 |
  |---|---|---|---|
  | 高速分 | `-m "not slow"` | 488 passed、39 deselected | 147.9 s |
  | 検証(slow 含む全部) | `tests/validation` | 36 passed | 199 s |
  | slow の unit と integration | `-m slow tests/unit tests/integration` | 10 passed | 311 s |
  | slow の GUI | `-m slow tests/gui` | 23 passed | 139 s |
  | 新規テスト 30 本 | 該当 6 ファイル | 30 passed | 6.5 s |

  合計 527 本で、文書の「497 + 30、高速 488」と一致します。
- **`test_mem_pbso4` の時間**: 旧版 259 s、新版 214 s で、遅くはなっていません。

## 回さなかった・分からなかったこと
- macOS の CI は再現できないため、新版で macOS が安定して通るかは分かりません。
- 文書にある「COD 1,100 件中 13 件 25 サイト」は、コーパスがリポジトリに無いため確かめていません。
- 非中心対称の X 線で GSAS-II とどれだけずれるかは、GSAS-II を再実行していないので実測していません。
- GUI の手操作、配布物のビルド、実データでの MEM に Friedel 平均が与える影響は見ていません。
- 片付け: 旧版の worktree は `git worktree remove` で削除し、リポジトリは 7f4f69d のまま clean です。実験スクリプトは scratchpad の `review_opus\` にあります(`probe_affine.py`、`probe_mem.py`、`probe_mem_trace.py`、`probe_jac_ties.py`、`probe_cli_unconverged.py`、`probe_oldproject.py`、`probe_seq_retry.py`、`probe_recipe_conv.py`、ログの `*.txt` など)。

---

## 2. 全体レビュー: 物理計算の中核(crystal / reflections / scattering / profile / pattern)

## 結論
担当範囲(crystal / reflections / scattering / profile / pattern)のうち、構造因子の計算そのもの、多重度・消滅則、Lorentz-偏光、FCJ、TCH、吸収・表面粗さ・試料変位の式は、手計算・GSAS-II のソース・性質テストのどれと比べても合っていました。問題は入力の解釈の側にあり、黙って間違った答えが出るものが 4 件あります。
1. 3 桁に丸めた特殊位置(0.333, 0.667)を特殊位置として認識できない。
2. CW の試料広がりで、ガウス成分に 8 ln2 が余分に掛かっている。
3. 対称操作の無い CIF で原点選択を取り違える。
4. CIF の座標 '1/3' や '?' が NaN のまま通る。

この 4 件はいずれも、CIF に入っている `_atom_site_symmetry_multiplicity` と照合すれば検出できます。

## 指摘(重い順)

**1. 3 桁に丸めた (1/3, 2/3, z) が特殊位置に戻らず、多重度が 3 倍・|F|² が約 9 倍になる(CONFIRMED)**
- 場所:
  - `src/kaisetsu/refine/constraints.py:122-143`(`position_constraint`、tol=5e-4)
  - `src/kaisetsu/crystal/spacegroup.py:277-285`(`site_symmetry`、分率座標で成分ごとに比較)
  - `src/kaisetsu/crystal/spacegroup.py:252`(`equivalent_positions`、tol=1e-4)
- 何が起きるか:
  - 判定が分率座標の成分ごとです。六方晶の操作 (-y, x-y, z) では x−y の形で誤差が 2 倍になり、0.333/0.667(ずれ 3.3e-4)でも 1e-3 ずれて判定から外れます。
  - その結果、乗せ直しが行われません。ウルツ鉱 ZnO の CIF(`_atom_site_symmetry_multiplicity 2` と書いてある)を `Project.add_phase` に入れると、多重度 [6, 6]、組成 Zn6O6 になります。
  - |F|²(100, 002, 101) は 876 / 2349 / 1071 から 7888 / 21144 / 9639 になります。約 0.003 Å 離れた原子が 3 個重なっている状態です。
  - 組成と単位胞質量が狂うので、重量分率も狂います。警告は出ません。
  - 論文の表から CIF を手で打つ初心者や、古い文献の 3 桁座標で起きます。4 桁(0.3333)なら正しく扱えます。
- 再現: `uv run python <scratch>/exp8_project_snap.py`、`exp2_special_positions.py`
- 直し方の案:
  - サイト対称の判定を直交座標の距離 |M(Rx+t−x−n)| < 0.01〜0.02 Å にする。
  - CIF に `_atom_site_symmetry_multiplicity` があれば計算値と照合し、合わなければ距離の許容で乗せ直して警告する。
  - 展開後に同じサイトの像どうしが 0.3 Å 以内に残っていたら警告する。

**2. CW の試料広がりで、ガウス成分の FWHM が sqrt(8 ln2) = 2.355 倍に大きい(CONFIRMED)**
- 場所: `src/kaisetsu/pattern/calculator.py:167`

  ```python
  add_g2 = EIGHT_LN2 * (((1.0 - fs) * g_size) ** 2 + ((1.0 - fm) * g_strain) ** 2)
  ```

- 何が起きるか:
  - GSAS-II(`GSASIIstrMath.GetSampleSigGam` 3280-3282 行)は σ² = ((1−mx)Γ)²/(8 ln2) を足し、FWHM² に直すと ((1−mx)Γ)² になります。
  - kaisetsu の `add_g2` は FWHM²(`fwhm_gauss**2` に足している)なのに、さらに 8 ln2 を掛けています。docstring(65-66 行)の「GSAS-II と同じ扱い」と矛盾します。
  - TOF 側(`pattern/tof.py:337`)は /8ln2 で正しいので、CW と TOF で食い違っています。
  - 実測: Scherrer 幅 0.4498° に対し、ローレンツ割合 0 で FWHM_G = 1.0591°(比 2.355)。割合 0.5 では 0.5296°(期待値 0.2249°)。
  - GUI の相の設定(`gui/phase_dialog.py:123-124`)で割合を 1 未満にすると、結晶子サイズが最大 2.35 倍大きく、歪みが最大 2.35 倍小さく出ます。Stephens・一軸サイズも同じ行を通ります。既定値 1.0 では影響しません。
- 検証の穴: GSAS-II との CW 比較は全ケース LGmix=1 です(`--lgmix` を使ったのは TOF の nac_tof_fit だけ)。
- 再現: `exp1_size_gauss.py`
- 直し方の案: 8 ln2 を外す。GSAS-II の参照に CW + `--lgmix 0.5` のケースを足す。

**3. 対称操作の無い CIF で原点選択が入口によって違い、照合も無い(CONFIRMED)**
- 場所: `src/kaisetsu/crystal/cif.py:341-355`
- 何が起きるか:
  - H-M 記号だけの 'F d -3 m' は gemmi の small structure の解釈で :2 になります。一方、国際番号だけ(`from_number`)と `SpaceGroup.from_symbol` は :1 です。
  - 原点選択 1 の座標で書いたスピネル(Mg 0,0,0 / Al 5/8 / O 0.3874)を H-M だけの CIF で読むと、:2 として解釈されます。多重度 [16, 8, 32](CIF には 8, 16, 32 と書いてある)、組成 Mg16Al8O32 になり、警告は出ません。番号だけの CIF に原点選択 2 の座標があれば、逆向きに壊れます。
  - ICSD 流の 'F d -3 m Z' / 'S' は、対称操作の無い CIF で ValueError になります。
  - 市販 DB の CIF は対称操作を持つので頻度は低めです。ただし手書きの CIF や Fd-3m・Pn-3n・Pnnn・I41/amd などでは現実に起きます。
- 再現: `exp3_settings.py`、`exp9_cif_edge.py`
- 直し方の案: 原点選択が 2 つある群で対称操作が無ければ両方を試す。CIF の多重度か Wyckoff 記号と合う方を選び(無ければ最短原子間距離で選ぶ)、どちらを選んだかをログに出す。

**4. 座標 '1/3' や '?' が NaN のまま通る(CONFIRMED)**
- 場所: `src/kaisetsu/crystal/cif.py:286`、`src/kaisetsu/crystal/atoms.py:306-312`
- 何が起きるか:
  - gemmi は '1/3' と '?' を NaN として返し、そのまま AtomSite になります。
  - 多重度は群の位数(P63/mmc で 24)になり、|F|² は NaN です。精密化の途中で分かりにくいエラーになるはずです。
  - 初心者の手書き CIF で起きます。
- 再現: `exp10_cif_fraction.py`
- 直し方の案: `AtomSite.__post_init__` で有限値か検査し、「0.333333 のように小数で」と日本語で返す。単純な分数 'a/b' は読み取って警告する手もある。

**5. ウィザードの既定の装置がモノクロメータ(P = 0.556)になっている(CONFIRMED、影響は小)**
- 場所:
  - `src/kaisetsu/gui/wizard.py:118`(`setCurrentIndex(1)`。index 0 が「(変更しない)」なので、PRESETS の先頭のモノクロメータが選ばれる)
  - `src/kaisetsu/pattern/presets.py:118`(CLI の別名 `cu` もモノクロメータ)
  - `gui/lattice_dialog.py:148` も先頭のモノクロメータが既定
- 何が起きるか:
  - Ni/Kβ フィルタ + 1 次元検出器の普通の卓上機を既定のまま使うと、LP の比が 20° / 60° / 90° で 1.007 / 1.067 / 1.111 ずれます。
  - 15〜130° で U_iso が約 +0.0013 Å² 大きく出ます(B で約 +0.10 Å²)。偏光係数は標準手順では精密化しません。
- 再現: `exp7_pol_and_perf.py`
- 直し方の案: 既定を Ni フィルタにするか、未選択のままにして選ばせる。結果の要約に偏光係数を表示する。

**6. 自動発散スリット・アテネータ・低角のビームはみ出しの扱いが無い(PLAUSIBLE、io との境界)**
- 場所: `src/kaisetsu/io/readers/vendors.py:169-208`(xrdml の `beamAttenuationFactors` やスリットモードを読まない)、強度補正の側にも対応項が無い
- 何が起きるか:
  - 自動発散スリット(照射長一定)のデータは強度が sinθ に比例し、20° と 90° で約 4 倍の傾きになります。標準手順では U が負に走るはずです。
  - アテネータの係数を掛けないと、最強ピークが削れます。
  - GSAS-II の xrdml リーダーも係数を無視しているので珍しい欠落ではありませんが、初心者の Rigaku / PANalytical のデータで起こりえます。
- 直し方の案: 読み込み時にメタデータから検出して補正するか、警告を出す。

**7. CIF のラベル重複で落ちる(CONFIRMED)**
- 場所: `src/kaisetsu/crystal/structure.py:78-80`
- 何が起きるか: 'O O O' のようにラベルが重複した CIF で ValueError になります。手書きの CIF でありえます。
- 直し方の案: O_1, O_2 のように自動で名前を付け直して警告する。

**8. gemmi の表に無い対称操作の組で Project が落ちる(CONFIRMED)**
- 場所: `src/kaisetsu/crystal/spacegroup.py:166-170`、`refine/project.py:670`
- 何が起きるか: B 2/m(B 底心の単斜晶)や原点をずらした P-1 で、`crystal_system` が ValueError を投げます。
- 直し方の案: 回転部分から晶系を自前で判定する(3 回軸・4 回軸・6 回軸の有無)。

**9. March-Dollase が遅い(CONFIRMED、性能)**
- 場所: `src/kaisetsu/profile/corrections.py:76-82`(反射ごとの Python ループ + `np.unique`。`axis_cos2` も同じ)
- 何が起きるか: PbSO4 の 384 反射で 1 回 33 ms。パターン計算が r=1 で 25 ms、r=0.9 で 71 ms になり、数値ヤコビアンではその倍数で効きます。
- 直し方の案: 反射リストごとに等価反射を一度だけ求めてキャッシュし、ベクトル化する。TCH の FWHM を反射ごとの Python ループで求めている 2 箇所(`calculator.py:401`、`fcj_batch.py:268`)は `tch_fwhm_array` に置き換えられる。

**10. 軽微なもの(PLAUSIBLE、読んだだけ)**
- `calculator.py:325` / `project.py:296`: CW の d_max はデータの先頭点 x[0] で切っていて余白がありません。x[0] の直下にある反射の高角側の裾と Kα2 が入りません。高角側には余白 2° があります。
- `pattern/tof.py:331`: TOF の歪みは Δd/d = ε、CW は ε/2 です(GSAS-II から引き継いだ差)。CW + TOF の同時精密化で、歪みパラメータの意味がずれます。
- `profile/stephens.py:26`: profile が refine.constraints を import しています(層の逆転)。
- `crystal/lattice.py:204-226`: `Lattice.constrained` は呼び出し元が無く、三方晶の判定 α≈90° も危ういです。
- `crystal/cif.py:418-421`: CIF に書き出す U_eq が対角成分の平均です(単斜晶・三斜晶では不正確)。
- 読み込み: aniso_B / aniso_beta のループは gemmi が読まず、等方 U に落ちます。ADP の無い CIF は U=0 になり、警告がありません。同位体ラベル(Li7、B11)は解釈できません。

## 確認したこと
- NaCl の X 線(f0 + Cromer-Liberman の f'、f''、DW)と中性子の |F|² が手計算と一致(相対 6e-16)。
- 単斜晶・六方晶で、等方 U と等価な異方 U^ij の F が一致(1e-15)。β = 2π² a*_i a*_j U^ij と U' = R U Rᵀ の変換は、計量が対称性を満たす限り正しい。
- 異方 ADP の一般位置(P21/c、P63/m、I41/a、R-3c)で、ラウエ等価反射の |F| が一致(3e-14)。
- 消滅則は h·R = h かつ h·t が整数でない、で正しい。多重度は |ラウエ群| / |安定化部分群|。hkl の候補の上限 |h| ≤ a/d_min は正しい。
- f'、f''(Fe・Co・Ni・Cu・Pb・Gd、Cu/Co Kα)が IT Vol.C の値と整合。Na、Cl も表の値と一致。
- 中性子散乱長は H・Ti・Li が負、D が正。B の Im b の符号は Friedel 平均で効かない。
- 特性 X 線の波長(Hölzer 1997)は妥当。
- Lorentz 1/(2 sin²θ cosθ)、偏光 (1−P)cos²2θ+P、モノクロメータの P = 1/(1+cos²2θm) が GSAS-II の Polarization と同じ。
- BB の試料変位・透過、DS の試料ずれの式と単位が GSAS-II の GetReflPos と一致。
- 円筒吸収(Lobanov)と Suortti が GSAS-II と係数まで一致。μR=3 で連続。
- pseudo-Voigt の面積は 1、TCH の η 係数は正しい。FCJ の面積は 2θ = 5〜160° で 0.9987〜0.9997。窓 50/75 幅の打ち切りは純ローレンツで約 0.5% の欠け(GSAS-II と同じ規則)。
- Pbnm、P21/n、R-3c(H 軸・R 軸)を H-M 記号だけの CIF から正しく組める。対称操作があればそれを優先する。
- 重量分率 W ∝ S·M_cell は、1/V を陽に含むスケールの定義と整合。Brindley の μ̄ は体積分率で平均。
- 一軸サイズ、Stephens、等方サイズ・歪みのローレンツ部分の換算が GSAS-II と同じ。
- numba は fastmath を使っていない(既定 False)。
- 実行したコマンド:
  - `PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR=C:/Windows/Fonts uv run pytest -q -m "not slow" -p no:cacheprovider` に担当領域の unit テスト 20 ファイル → 192 passed(8 s)
  - `uv run python` で下記 exp1〜exp11 を実行(結果は上の各項)

## 回さなかった・分からなかったこと
- GSAS-II との比較テスト(validation、slow)は回していません。GSAS-II 本体も実行していません(式はソースで確認)。
- 検証の網羅について: GSAS-II との比較は直方晶(PbSO4)と立方晶(ダイヤモンド、NAC)だけで、六方晶・単斜晶・三斜晶の反射リスト比較(ROADMAP Phase 1 の受け入れ基準)と、X 線の非中心対称 + 異常分散の比較はありません(後者は unit テストのみ)。ただし多重度の和は全格子点の数え上げと照合するテストがあります。
- TOF(`tof.py`、`tof_profile.py`、`tof_batch.py`)と Z-Rietveld の換算は流し読みです。GUI のダイアログは動かしていません。配布物(PyInstaller)で numba の `cache=True` がどう振る舞うかは確認していません。
- 中性子の吸収の強い核種のエネルギー依存は見ていません。
- 実験スクリプトはすべて scratchpad の `review_opus_all\physics\` にあります(exp1_size_gauss.py、exp2_special_positions.py、exp3_settings.py、exp4_fcalc.py、exp5_profile_area.py、exp6_cif_adp.py、exp7_pol_and_perf.py、exp8_project_snap.py、exp9_cif_edge.py、exp10_cif_fraction.py、exp11_aniso_invariance.py)。リポジトリは変更していません。

---

## 3. 全体レビュー: 精密化・MEM・PDF(refine / mem / pdf)

## 結論
対象は HEAD 7f4f69d の refine / mem / pdf です。MEM と PDF は並行して別に調べ、その結果もここに入れました。リポジトリは変更していません(`git status` はクリーン)。

初心者の実験室データで、黙って間違った値が出る経路を実行で確認しました。いちばん重いのは次の 1 件です。

- **Bragg-Brentano(平板)の標準手順の既定(GUI ウィザード・CLI とも)は、ゼロ点しか動かしません。** 試料高さが 0.05〜0.15 mm ずれた平板データでは、格子定数が −71〜−213 ppm ずれます(esd の 200〜575 倍)。それでも χ² は 1.03〜1.27 で、利用者には気付けません。

ほかに確認した重い問題:
- PDF の Q_max が実データの上限で切られていない。
- Lorch 窓を掛けた G(r) を、窓を知らないモデルでフィットしている。
- MEM の F_obs に異常分散が残ったままになっている。
- 結晶子サイズを 0 から動かすと 0.0001 μm に潰れても success=True になる。
- 同じ名前のヒストグラムが 2 つあると、R 因子が別のヒストグラムの値に置き換わる。

問題が無かったもの:
- 最小二乗の esd は、モンテカルロのばらつきと 0.74〜1.18 倍で一致した(N = 20)。
- 半解析ヤコビアンの構造・背景・スケールの式は正しい。ただし評価順の不具合が 1 件ある(指摘 8)。
- R 因子の定義、MEM の FFT・対称展開・勾配・2 チャンネル化は正しい。

## 指摘(重い順)

### 1. 平板試料で、標準手順がゼロ点しか動かさず、格子定数が数百 ppm ずれる【CONFIRMED・P1】
- **場所**:
  - `refine/recipe.py:647`(`sample_displacement: bool = False`)と `:702`(cell + `inst:zero` だけ)
  - `gui/wizard.py:155`(チェックの既定はオフ)
  - CLI の `--displacement` も既定はオフ
  - Le Bail で格子定数を出す経路も同じ(GUI は `position="zero"` が既定)
  - `_shift_matches_geometry` はゼロ点と変位を同時には動かさない
- **何が起きるか**: 実験室の平板試料では、主な系統誤差は試料高さのずれ(Δ2θ ∝ cosθ)です。ゼロ点(定数)しか動かさないと、そのずれを格子定数が吸います。
- **再現**: `exp1_displacement.py`。CeO2、Cu Kα1+2、R = 240 mm、2θ 15–120°、最大 58 万カウント。初期値は a を +0.05% ずらし、`standard_recipe()` を回した。

  | 真のずれ | 動かしたもの | a の誤差 | Rwp | χ² |
  |---|---|---|---|---|
  | 変位 0.15 mm | ゼロ点(既定) | −213 ppm(−575 esd) | 2.28% | 1.267 |
  | 変位 0.15 mm | 変位 | 0 ppm | 2.03% | 1.001 |
  | 変位 0.05 mm | ゼロ点(既定) | −71 ppm(−215 esd) | — | 1.026 |
  | ゼロ点 0.05° | 変位 | −140 ppm | — | 1.095 |

  最後の行のとおり、逆の場合も同じようにずれます。固定でどちらか一方を選ぶこと自体が問題です。
- **直し方の案**:
  - 平板では、既定を「変位」にする。
  - 加えて、Kα2 の試行と同じ方式で「ゼロ点のみ / 変位のみ / 両方」を試し、χ² の差で採用する。
  - 2θ 範囲が広いときは両方を動かし、相関が 0.95 を超えたら警告する。
  - 軽元素の試料では透過(`sample:transparency`)も同じ形で効くが、標準手順では一度も動かさない。この点も利用者ガイドに書くとよい。

### 2. PDF の Q_max をデータの上限 Q で切っていない【CONFIRMED・P1(PDF を使う場合)】
- **場所**: `pdf/transform.py:252, 266-267, 158-166`、`gui/pdf_dialogs.py:167`(既定 25)、`gui/main_window.py:928-932`。
- **何が起きるか**: Cu Kα(Q = 7.88)や Mo Kα(Q = 17.08)のデータを既定の Q_max = 25 のまま変換すると、次のずれが重なります。
  - フーリエ変換はデータの上限で止まる。
  - Lorch 窓、多項式の次数、metadata の qmax、フィットの終端効果は 25 のまま。
- **再現**: Ni、真の帯域で制限し、ノイズ 0.02。

  | 線源 | モデルの Q_max | Rw | U(真値 0.006) | scale(真値 1) |
  |---|---|---|---|---|
  | Cu | 25 | 38% | 0.025 | 1.44 |
  | Cu | 正しい値 | 0.54% | 0.00599 | — |
  | Mo | 25 | 15.8% | 0.00725 | 1.08 |

- **直し方の案**: `qmax_eff = min(settings.qmax, q[use].max())` を上の 4 か所すべてに使い、ログに出す。`add_pdf` は metadata の qmin も拾う。

### 3. Lorch 窓を掛けた G(r) を、窓を入れていないモデルでフィットしている【CONFIRMED】
- **場所**: `pdf/transform.py:179-183`、`pdf/calculator.py:255-269`(`bandlimit` は矩形で切るだけ)。
- **再現**: Ni、Mo Kα、Lorch をオンにした場合:
  - U = 0.0133(真値 0.006 の 2.2 倍)、Rw 4.6%
  - Lorch をオフにすると U = 0.00598
- **直し方の案**: metadata と PDFEntry に `lorch` を持たせ、`bandlimit` で窓 M(Q) を掛ける。最低でも、Lorch ありのデータを追加するときに警告する。

### 4. MEM(X 線)の F_obs に異常分散 f′, f″ が残っているのに、F000 は Σf0 になっている【CONFIRMED】
- **場所**: `mem/mpf.py:109-110`(|F_obs|² = `tbl.f2` × I_obs/I_calc。`tbl.f2` は f0 + f′ + if″ の Friedel 平均)、`mem/mpf.py:54-65`(F000 は f0 だけ)。
- **再現**: PbSO4、Cu Kα1/2、y_obs = y_calc の合成データ。振幅を |F(f0)|/|F(f0+f′+if″)| で補正したものと比べた。

  | | 現状 | 補正後 |
  |---|---|---|
  | Pb 球内の電子数 | 73.3 e | 76.4 e(現状は −4%) |
  | 最大密度 | — | 現状は −9% |
  | 密度の中央値 | — | 現状は +15% |

  Cu Kα で Fe、Co、Mn、ランタノイドを測る場合も同じ向きにずれます。
- **直し方の案**: X 線では `f_abs *= |F_calc(f′ = f″ = 0)| / sqrt(tbl.f2)`。MPF の 2 周目以降は 1 周目の比を使い回す。

### 5. 結晶子サイズ `h{i}:相:size` を既定値 0 から動かすと 0.0001 μm に潰れ、success=True のまま【CONFIRMED・P2】
- **場所**:
  - `refine/project.py:1023` 付近: 値 0 = 広がりなし、step 1e-3 μm、min 0
  - `refine/project.py:172`(`_size_setter`)
- **何が起きるか**: 0 は「サイズ無限大」の意味です。前進差分の 1 歩目が「1 nm の結晶子」になり、微分が意味を失います。下限 0 ちょうどではないので、`diagnostics()` も知らせません。
- **再現**: `exp6_size_zero.py`(CeO2)。

  | 真値 | 初期値 | 結果 | Rwp |
  |---|---|---|---|
  | 0.03 μm | 0 | 0.0001 μm | 81% |
  | 0.2 μm | 0 | 0.0000 μm | 87% |
  | 両方 | 1 μm | 真値を回収 | 4.5% |

- **直し方の案**: 次のどれか。
  - vary を立てたときに 0 なら 1 μm から始める(`enable_uniaxial_size` と同じ扱い)。
  - 1/size で変数化する。
  - min を 1e-3 μm にして警告する。

### 6. 同じ名前のヒストグラムが 2 つあると、R 因子の辞書が衝突する【CONFIRMED・P2】
- **場所**: `refine/project.py:1713`(`out[h.name] = ...`)。名前の既定は `data.name`、つまりファイル名の stem。
- **影響範囲**:
  - 読む側: `gui/main_window.py:566` のステータス表示、`cli.py:47` の `_report`、`Project.summary`、`mem/mpf.py:287,318`、`sequential.py`
  - 名前が "total" や "g0" のヒストグラムも、全体の値や PDF の値と衝突する
- **再現**: `exp2_hist_name.py`。PBSO4.XRA と PBSO4.CWN はどちらも名前が 'PBSO4' になる。
  - 辞書のキーは `['PBSO4', 'total']` の 2 つだけ。
  - h0 の表示は 6001 点の X 線ではなく、中性子の 2919 点の R 因子になる。
- **直し方の案**: キーを `h{i}` にする。または `add_histogram` で名前を一意にする。

### 7. PDF: U ≈ 0 のサイトはピーク面積が保存されない【CONFIRMED】
- **場所**: `pdf/calculator.py:243`(σ の下限 1e-3 Å、内部グリッドは 0.01 Å)。ADP の無い CIF は U = 0 で読まれる。
- **再現**: Ni、U = 0 で第 1 殻の配位数が a = 3.524 → 3.5245 → 3.526 Å で 8.74 → 4.28 → 0.24 と変わる(正しくは 12)。U = 5e-5 なら 12.000。
- **直し方の案**: 各ビンへの寄与を erf の差で積分する。または幅の下限を dr 程度にする。または U = 0 のサイトに既定値を入れて知らせる。

### 8. 半解析ヤコビアン: 数値微分の列が `h.result` を摂動後の状態のまま残し、その後の解析列が汚れる【CONFIRMED・P3】
- **場所**: `refine/jacobian.py:255-262`。`fn()` で全ヒストグラムの result が摂動後の値になり、`_perturbed(-step)` は値を戻すだけで再計算しません。最後に戻すのは `:264` だけです。
- **何が起きるか**: 自由パラメータの順は「相(格子 → サイト)→ ヒストグラム(装置 → 背景 → スケール)」です。このため、格子の列の後の構造の列、装置の列の後のスケールと md_r の列が、摂動後のプロファイル c_k と y_phases で作られます。
- **再現**: `exp5_jacobian.py`、`exp5c.py`、`exp5d.py`。中心差分と比べた相対誤差:
  - `h0:inst:zero` の後の両相の scale と md_r: 2.6e-3
  - `rutile:cell:a` の後の構造の列: 3e-3
  - 数値微分の列が前に無い列: 1e-13
  - 消衰のある相の構造の列は別に 1e-4 ずれる(Kα2 の λ を λ1 で近似しているため)
- **影響**: 解は残差で決まるので値は変わりません。esd・相関が約 0.3% ずれ、収束の経路が変わります。結晶子サイズの step が値に比べて大きい場合(例: 20 nm に対して 1 nm)は数 % になります。
- **直し方の案**: 数値微分の列の後に基準の result を戻す(基準の結果を保存して代入するか、`fn()` を呼び直す)。または数値微分の列を最後に回す。
- **性能(読んだだけ)**: `lsq.py:169-177` の `jac()` が `fun(x)` を呼び直し、`project_jacobian` の最後でもう一度 `fn()` を呼びます。反復ごとに 2 回、全パターン計算が余計に走ります。

### 9. `refine_lattice` の体積 esd が格子定数どうしの相関を無視している【CONFIRMED・P3】
- **場所**: `refine/lattice.py:283-284`。
- **再現**: `exp8_volume_esd.py`。PbSO4、Le Bail、ゼロ点あり。

  | 2θ 範囲 | 共分散込みの esd に対する比 | 相関 |
  |---|---|---|
  | 15–70° | 0.74 | +0.4 |
  | 15–40° | 0.62 | +0.8 |

  一括処理の CSV にもこの esd が出ます。
- **直し方の案**: `last_fit.covariance` から gᵀCg を計算する。
- **同類(読んだだけ)**: `parameters.py` の `set_esd` は、主パラメータが複数ある拘束(3 種類以上の混合サイトの占有率など)で共分散を無視して二乗和にしています。

### 10. 重みの扱い(io / pattern との境界)【CONFIRMED・P3】
- **場所**: `pattern/histogram.py:41-44`(w = 1/y_obs、y ≤ 0 は重み 0 で除外)。
- **低計数の場合**: `exp4_lowcount.py`、背景 4 カウント、N = 16。

  | 重み | 背景 | U(Ti) | 平均 χ² |
  |---|---|---|---|
  | 1/y_obs(現状) | 3.0(真値 4、−25%) | +26 ± 9% | 1.064 |
  | 1/max(y_calc, 1) | 偏りなし | 偏りなし | 1.003 |

  GSAS-II の xrdml リーダーは y = 0 の点も重み 1 で使います。
- **背景を引いて書き出したデータの場合**: `exp7_bkgsub.py`。
  - 3 分の 1 の点が黙って捨てられる。
  - Rwp 14.5%、χ² 12.6(正しい σ なら 5.1%、0.97)。構造パラメータはほぼ正しい。
- **直し方の案**: 1/max(y, 1) にし、0 カウントも使う。y ≤ 0 の点が多いデータは「背景を引いたデータでは?」と警告する。

### 11. CIF の異方性 U を、実験室データの標準手順がそのまま動かす。正定値の点検も無い【PLAUSIBLE・読んだだけ】
- **場所**:
  - `recipe.py:750`: 「変位パラメータ」が u11〜u23 を含む
  - `crystal/cif.py:53-56`: CIF の U_ij を保持する
  - `project.py:1780`: 診断は対角が負かどうかしか見ない
- **何が起きるか**: 単結晶由来の CIF(ICSD、COD)では、普通に起きます。
- **直し方の案**: X 線の標準手順では、異方性 U は固定するか U_eq の等方性にする。診断に Cartesian U の固有値の点検を加える。
- **境界**: `gui/phase_dialog.py:242` は等方性に戻すときに U_iso = mean(U11, U22, U33) にしています。斜交軸では U_eq と異なります(`cif.py` の `_u_equiv` を使うべき)。

### 12. MEM のその他【指摘ごとに CONFIRMED / PLAUSIBLE を付記】
- **格子点数の検査**: `mem/core.py:98-103` は独立反射の |h| しか見ていない【CONFIRMED・低】。
  - 石英 P3₁21 で検査を通った (9,5,11) の格子で 96 点が重複し、C = 371 で上限に達した。
  - GUI の提案する格子では起きない。
- **Pawley 経路の 2 チャンネル**: `gui/mem_dialog.py:85-88` で `two_channel=None` が False 扱いになり、負の b を持つ中性子のデータで 1 チャンネルになる【PLAUSIBLE】。
- **N_F の数え方**: 範囲外の反射や σ = ∞ の反射も N_F に数えていて、C = 1 の目標が緩む【PLAUSIBLE・軽】(`mpf.py:89-116`)。
- **λ 探索の区間**: 打ち切った内側で決めた lo / hi を見直さない【PLAUSIBLE・軽】(`core.py:547-552`)。終了は保証されている。
- **PDF の原子対キャッシュ**: キーに元素と空間群が入っていない【PLAUSIBLE・軽】(`project.py:300-308`)。

### 13. 逐次精密化【PLAUSIBLE・読んだだけ】
- `sequential.py:135-136`: フィット範囲が狭くなる一方で、広がらない。
- `:166, 171`: 2 回とも失敗したり悪化したりすると、発散したか悪い値のまま次のファイルの初期値になる。前のデータの値(`start`)に戻すのが安全。

### 14. テストの穴
- **GSAS-II との突き合わせ**: PbSO4(Pnma、特殊位置は定数だけ)と NAC TOF しか無い。2 相・重量分率、Le Bail / Pawley、選択配向、混合サイト / 占有率、異方性 U、平板の変位 / 透過、非中心対称は比較していない(2026-10-17 の #1 が見つからなかった理由と同じ)。
- **統合テスト**: 自分の計算で作ったデータを自分で精密化して一致を見るものだけ。指摘 1 は、合成ラウンドトリップ(`test_refine_synthetic.py`)がゼロ点と変位を両方動かしているので見えない。
- **MEM / PDF**: 異常分散ありの X 線、六方晶、Pawley 経路、「変換 → 追加 → フィット」の通し(Q_max、Lorch)、U = 0 の場合が無い。

## 確認したこと
- **esd**: (JᵀJ)⁻¹ × χ²_red を、列を正規化した固有値分解で計算している。`res.jac` は解の点のもの。モンテカルロ(`exp3_esd_mc.py`、ルチル、N = 20)で、ばらつきと esd の比が a 1.01、c 0.74、O x 1.18、U(Ti) 1.02、U(O) 0.83、変位 0.89 だった。
- **R 因子**: Rp、Rwp、Rexp、χ²、背景差引の Rwp′、R_B、R_F の定義は Young / Toby と一致。重み 0 の点を除外。全体の値は `weight_factor` を分子・分母の両方に入れている。
- **重量分率**: 強度 ∝ S/V なので、W ∝ S·M で正しい。esd の式も正しい(相関は無視と明記されている)。
- **半解析ヤコビアン**: 構造、背景、スケール、md_r、混合サイトの拘束、異方性 U の拘束、同じ相内の拘束の解析列は、数値微分の列が前に無ければ 1e-13 で一致(`exp5d.py`)。
- **拘束**: 特殊位置の定数項の修正(`constraint_from_projector(project_values=False)`)を読んで確認した。ADP の射影、格子の晶系拘束(単斜晶の unique axis、菱面体)、混合サイトの鏡像と占有率の和、拘束の連鎖を禁止している点も確認した。
- **lsq**: 前進差分の上限側の処理、非有限の残差を 1e10 にする処理、`PatternSizeError` の扱い、bounds、x0 の clip、`convergence_note` の判定文字列("function evaluations" は scipy 1.18.1 の文言と一致)。
- **Le Bail**: 範囲外の反射を中央値にする、下限を最大値の 1e-4 にする、最初の周期だけ面積合わせで以後は重み付き。Pawley の完全重なりのまとめ方と NNLS も確認した。
- **cell_scan**: 刻みを幅の 0.4 倍にすること、採用の条件、±3% への拡張が初期値が既に谷の中なら害を出さないことを読んで確認した。
- **MEM**: FFT の符号と規格化、対称展開の位相(P2₁2₁2₁ で 2e-13)、勾配(5.9e-9)、2 チャンネル化の式、Kα2 込みの分配で |F_obs|² が一致すること(3e-8)、σ の伝播、λ の内側ループが必ず終わること。2026-10-17 の対策は妥当。
- **PDF**: Faber–Ziman の規格化、不等間隔の台形則、PDFfit2 の式(δ1/δ2、Q_broad、Q_damp、球状粒子、ρ0、重み)、`bandlimit` の pad、原子対を r_max + 6 Å まで取ること。
- **既存テスト**(`PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen ... uv run pytest -q -m "not slow" -p no:cacheprovider ...`):
  - refine 関連 11 ファイル: 66 passed(88 秒)
  - mem / pdf 関連 5 ファイル: 26 passed

## 回さなかった・分からなかったこと
- slow テスト(GSAS-II 比較、精密化の統合テスト、`test_mem_pbso4.py`)は回していない。Dysnomia / RIETAN-FP が MEM 用の F_o から異常分散をどう除いているかは文献で確かめていない(指摘 4 の不整合自体は実行で確認済み)。
- Kα2 なしの試行: ピークが広く Kα2 が分離しないデータで誤って採用されると、格子定数が約 +0.08% ずれる可能性がある。採用の閾値が 0.3 ポイントなので可能性は低いと見て、試していない。
- 採用したハンプの中心・幅が座標や U の段階でも動き続け、U と相関する可能性。build_parameters のたびに 5e-4 以内の一般位置が特殊位置に乗せ直されて固定される件。これらは読んだだけで試していない。
- CIF のサイトラベルの重複(パラメータ名の重複で読み込みエラー)は crystal 担当の範囲なので確かめていない。

### 実行したスクリプト
実行は `PYTHONUTF8=1 uv run python <script>`。

refine 用(scratchpad の `review_opus_all\refine\` の中):
- `exp1_displacement.py`: 引数 0.15 0 / 0 0.05 / 0.05 0(指摘 1 の表)
- `exp2_hist_name.py`(指摘 6)
- `exp3_esd_mc.py 20`(esd の検証)
- `exp4_lowcount.py 16 4.0 0.02`、`exp7_bkgsub.py`(指摘 10)
- `exp5_jacobian.py`、`exp5b.py`、`exp5c.py`、`exp5d.py`(指摘 8)
- `exp6_size_zero.py`(指摘 5)
- `exp8_volume_esd.py`(指摘 9)

MEM / PDF 用(同じフォルダの下の `memfork\` の中):
- `exp1_expand.py`、`exp2_anomalous.py`、`exp2b_f2check.py`、`exp3_pdf.py`、`exp3b_pdf_mo.py`、`exp4_grad.py`、`exp5_transform_qmax.py`、`exp6_lorch_fit.py`

---

## 4. 全体レビュー: GUI・入出力・CLI・配布(gui / io / api / packaging / workflows)

## 結論

担当範囲(GUI・入出力・CLI・配布)で一番重いのは「初心者が普通に操作しただけで結果を失う」経路で、少なくとも 7 本あります。どれも実行して確かめました。

- 窓を閉じても未保存の確認が出ない。
- 保存に失敗しても何も表示されない。
- 精密化の実行中も、精密化ボタン以外の操作がほぼ全部有効のままになっている。その結果、精密化が失敗する、表の値とモデルの値が食い違う、numba の構成によってはプロセスが即座に終了する。
- 「連続データを順に精密化」がメインのプロジェクトのデータを最後のファイルで上書きする。
- ヘッダの無い 3 列の .xye が RIETAN 形式と誤判定され、10 点のゴミとして読まれる。
- ウィザードを 2 回完了すると、前のデータに新しいデータが足されて同時に精密化される。
- 別のプロジェクトを開いた後に Ctrl+Z を押すと、前のプロジェクトの値が書き込まれる。

ほかに次の問題があります。
- **更新**: ダウンロードしたインストーラをハッシュでも署名でも検証していない。
- **ライセンス一覧**: Qt(PySide6)の LGPL/GPL の本文、OpenSSL、配布物に入ってしまっている開発用パッケージが漏れている。
- **アンインストール**: `{app}` を丸ごと消す。一方で保存ダイアログの初期フォルダがインストール先の中なので、そこに保存したプロジェクトが消える。

リポジトリは変更していません(`git status` はクリーン)。

## 指摘(重い順)

### 1. 窓を閉じるとき、未保存の確認が出ない(データを失う)
- **場所**: `src/kaisetsu/gui/main_window.py:1523` の `closeEvent`。`_confirm_discard`(1491)を呼んでいない。そもそも変更の有無(dirty フラグ)を持っていない。
- **何が起きるか**: 長い精密化の後に × か「終了(Ctrl+Q)」を押すと、何も聞かれずに全部消える。
- **再現**: 相を追加した MainWindow で `close()` を呼ぶ。`QMessageBox.question` は 1 回も呼ばれず、窓が閉じた。
  - スクリプト: scratchpad `review_opus_all\gui_io\t_slot_exc.py`
  - 結果: `close: question asked? [] visible after close: False`
- **判定**: CONFIRMED
- **直し方**: `Project` に変更カウンタを持たせ、精密化・編集・読み込みで上げ、保存で 0 に戻す。`closeEvent` では「保存する / 保存しない / キャンセル」の 3 択を出し、窓のタイトルに `*` を付ける。`_confirm_discard` も同じ判定を使い、保存直後は聞かないようにする。

### 2. 保存の失敗が利用者に見えない。書き込みもアトミックでない
- **場所**:
  - `main_window.py:676` の `save_project` に try/except が無い。
  - `src/kaisetsu/io/project.py:514` は `p.write_text(...)` で直接書いている(先に切り詰めてから書く)。
  - `gui/app.py` で `sys.excepthook` を設定していない。
  - 配布物は `console=False`(`packaging/kaisetsu.spec`)なので `sys.stderr` が None。
- **何が起きるか**:
  - 書けない場所に保存した場合(OneDrive のロック、読み取り専用、権限、ディスクいっぱい)、例外はスロットの中で stderr に出るだけになる。配布物では完全に見えない。ログにも「保存しました」が出ないだけで、利用者は保存できたと思って閉じる(→ 指摘 1 と重なってデータを失う)。
  - 書き込みの途中で失敗すると、既存のプロジェクトファイルが切り詰められたまま壊れる。
- **再現**: `project_path = Z:/no/such/dir/x.kaisetsu.json` で `act_save.trigger()` を呼んだ。
  - 結果: `trigger returned normally; message boxes: []`、traceback は stderr にだけ出た。
  - スクリプトは上と同じ `t_slot_exc.py`。
- **判定**: 失敗が見えない点は CONFIRMED、切り詰めは PLAUSIBLE(読んだだけ)。
- **直し方**:
  - `save_project` を try/except で囲み、`QMessageBox.critical` を出す。
  - io 側は同じフォルダの一時ファイルに書いてから `os.replace` で置き換える(.npz も同様)。
  - `app.py` で `sys.excepthook` を設定し、ダイアログと `%LOCALAPPDATA%\kaisetsu\log` へのファイルログを出す。CSV や CIF の書き出し、`recalculate`、`refine_recipe` など、例外を受けていない他のスロットの失敗もこれで全部見えるようになる。

### 3. 精密化の実行中に UI 側から同じ Project を計算・変更できてしまう(落ちる・黙って食い違う)
- **場所**: `main_window.py:1274` の `_start_worker`。無効にしているのは `act_refine`、`act_recipe`、`act_pdf_recipe` の 3 つだけ。次の操作はすべて実行中も動く。
  - 保存 / 開く / 新規
  - データ読込 / CIF 追加 / 装置設定 / 背景設定 / 範囲設定 / 相の設定
  - ウィザード / 逐次精密化 / MEM / 残差の点検
  - F5、ツリーのクリック(`_refresh_plot` → `project.calculate`)
  - 図の上での Shift+ドラッグ(`_on_exclude_requested`:1121)
  - パラメータ表の値の編集(`param_model.setData` → `apply`)
- **再現と結果**(core で GUI と同じ呼び方をした):
  - **除外領域を追加**(Shift+ドラッグと同じ `set_excluded`)すると、ワーカーが `ValueError: operands could not be broadcast together with shapes (5650,) (5751,)` で失敗した。GUI には「エラー: operands could not…」と出る。
    - スクリプト: `review_opus_all\gui_io\t_mutate.py exclude`
  - **`build_parameters()`**(装置設定 OK・CIF 追加・背景設定の後に GUI が呼ぶ)を走らせると、ParameterSet が差し替わる。精密化は古いセットで続き、表と保存に使われる新しいセットは途中の値のまま残る。
    - 結果: `param table a = 9.39422 / model lattice a = 9.38768`、`w = 0.00429 / 0.00718`。
    - 保存すると params の古い値がモデルを上書きするので、結果が黙って失われる。
    - スクリプト: `t_mutate.py rebuild`(上と同じフォルダ)
  - **UI スレッドで `calculate` を並行に回した場合**、既定の threading layer(この PC では `omp`)では落ちなかった。ただし `NUMBA_THREADING_LAYER=workqueue` では次のメッセージを出してプロセスが終了した(exit 3)。
    - メッセージ: `Numba workqueue threading layer is terminating: Concurrent access has been detected.`
    - スクリプト: `review_opus_all\gui_io\t_concurrent.py`
    - 配布物の omp 層は VCOMP140.DLL に依存する。この DLL は `_internal/shiboken6/` にしか入っておらず、見つからなければ workqueue に落ちる可能性がある。
  - **実行中にプロジェクトを開く・新規・ウィザードを使った場合**、`_on_refine_finished` と `_undo_values` が新しいプロジェクトに適用される。
- **判定**:
  - 除外による失敗と ParameterSet の差し替えは CONFIRMED。
  - workqueue での即終了は CONFIRMED(環境変数で強制した場合)。配布物で workqueue になるかは PLAUSIBLE。
  - 開く・新規・ウィザードの件は PLAUSIBLE。
- **直し方**:
  - 実行中はモデルを変えるアクション全部と表の編集を無効にする(`_set_running(bool)` に集約する)。
  - 除外領域の帯と表は読み取り専用にし、ツリーのクリックは再計算せずワーカーが送る曲線だけを表示する。
  - 逐次精密化・MEM・ウィザードも同じガードを通す。
  - 補足として、numba の threading layer を明示し(tbb を同梱する、または起動時に `numba.threading_layer()` をログに出す)、計算をロックで直列化する。

### 4. 「連続データを順に精密化」がメインのプロジェクトをそのまま書き換える(データを失う)
- **場所**:
  - `main_window.py:1389` が `SequenceDialog(self.project, …)` で本体のプロジェクトを渡している。
  - `refine/sequential.py:134` で `h.data = data` としている。x_min/x_max は狭まる一方で、背景も推定し直される。
- **何が起きるか**: 実行後、ヒストグラムの名前は元のまま、中身は最後のファイルのデータになる。続けて Ctrl+S を押すと、元のプロジェクトファイルが混ざった内容で上書きされる。ダイアログにはその旨の注意が無い。
- **再現**: 結果は次のとおり。
  - `before: name FAP path …fap\FAP.XRA npts 5753`
  - `after : name FAP path …pbso4\PBSO4.XRA npts 6001`
  - スクリプト: `review_opus_all\gui_io\t_seq.py`
- **その他**:
  - 「閉じる」ボタンは `reject`(`sequence_dialog.py:97`)で、`closeEvent` を通らない。そのため実行中のワーカーを残したままダイアログだけが隠れ、メイン側で `_refresh_all()` が並行に走る。中止ボタンも無い。MEM ダイアログも同じ構造になっている。
- **判定**: 書き換えは CONFIRMED、閉じた後も走り続ける件は PLAUSIBLE。
- **直し方**: `copy.deepcopy(project)` を渡して元を守る(最後に「結果をメインに反映する」ボタンを付けてもよい)。実行中は閉じられないようにするか中止できるようにし、`reject` を上書きして待つ。

### 5. ヘッダの無い 3 列 .xye が RIETAN 形式と誤判定される(黙って間違ったデータを読む)
- **場所**:
  - `src/kaisetsu/io/readers/rietan.py:252` の `is_rietan_block`。
  - `readers/__init__.py:69` で `read_xy` より前に判定している。
- **何が起きるか**: 1 行目が「10.000 1234 35.1」のように、2θ の整数値(10, 20, …)で始まり σ < 強度になっていると、「点数 10、開始角 1234、刻み 35.1」と解釈される。結果は 10 点(または 20 点)、x = 100〜190° のデータになる。エラーは出ず、ログには「10 点」と出るだけ。
- **小数点カンマのデータ**:
  - タブ区切りは同じく RIETAN と誤判定される(10 点)。
  - セミコロン区切りは `read_xy` の区切りにカンマが入っているので、x=10, y=0, σ=100 のように分解され、81 点のゴミになる。
- **再現**: `review_opus_all\gui_io\t_readers.py`
  - `xye no header, start 10.000 → fmt=rietan n=10 x=[100.000,190.000]`
  - `start 20.000 → n=20`
  - 開始角を 10.010 にすると正しく 4001 点になる
  - `decimal comma, tab → fmt=rietan n=10`、`semicolon → fmt=xy n=81`
- **判定**: CONFIRMED
- **直し方**:
  - RIETAN 判定に「1 行目の点数 n と、以降の数値の総数が一致する」「2 行目以降の列数がばらつかない」「1 行目の 2 列目が 0〜180 の角度」を加える。曖昧な場合は xy を優先する。
  - 読み込み後に点数が極端に少なければ(例: 50 点未満)エラーにする。
  - x が単調でない、重複が多いなら小数点カンマを疑うメッセージを出す。

### 6. ウィザードを 2 回完了すると、既存のプロジェクトに足されて 2 つのデータを同時精密化する
- **場所**: `src/kaisetsu/gui/wizard.py:262` の `accept`。新規プロジェクトにも確認ダイアログにもならない。
- **何が起きるか**: 「データを間違えたのでやり直す」とき、histograms が `['PBSO4', 'FAP']`、相は 1 つ(2 回目は「相 'Fluorapatite' はすでにあります」の警告が出るだけ)のまま、標準手順(`h*`)が両方に対して走る。
- **再現**: `review_opus_all\gui_io\t_wizard_twice.py`
- **判定**: CONFIRMED
- **直し方**: プロジェクトが空でなければ「新しいプロジェクトとして始める / 今のプロジェクトに追加する」を聞く。初期値は前者にする。

### 7. 別のプロジェクトを開いた・新規にした後でも Ctrl+Z が有効で、前のプロジェクトの値を書き込む
- **場所**: `main_window.py:657` の `open_project_file` と `:635` の `new_project` が `_undo_values` を消さず、`act_undo` も有効なまま残る(`adopt_project` だけは消している)。
- **何が起きるか**: 同じ名前のパラメータ(h0:inst:*, h0:bkg:*, 同名の相)に古い値が入る。
- **再現**: 結果は `B before Ctrl+Z: zero = 0.0 w = 0.004 → after: zero = 0.123 w = 0.05`。
  - スクリプト: `review_opus_all\gui_io\t_undo_after_open.py`
- **判定**: CONFIRMED
- **直し方**: プロジェクトを差し替える処理を 1 つの関数にまとめ、`_undo_values = None` と `act_undo.setEnabled(False)` を必ず通す。

### 8. ウィザードのファイル選択で Rigaku / PANalytical のファイルが表示されない(使いにくさ。ただし影響は大きい)
- **場所**: `wizard.py:78`。フィルタは `*.xy *.xye *.chi *.dat *.txt *.gsa *.gsas *.fxye *.raw *.xra *.cwn *.cif` で、`.ras .rasx .xrdml .csv .asc .int .histogramIgor` が無い。メインの `DATA_FILTER`(`main_window.py:72`)にも `.int` と `.histogramIgor` が無い。
- **何が起きるか**: 国内の実験室で一番多い .ras を持った初心者が、ウィザードで自分のファイルを見つけられない。
- **判定**: CONFIRMED(フィルタを表示して確認)
- **直し方**: `SUPPORTED_EXTENSIONS` からフィルタ文字列を生成して、両方で共用する。

### 9. 装置メーカー形式の解釈(黙って違う装置設定・重みになる)
- **場所と内容**:
  - **XRDML の Kα1 単色**(`vendors.py:175`): `<usedWavelength intended="K-Alpha 1">` でも kAlpha2 と ratio=0.5 を拾うので、`apply_hints` が Kα2 を有効にする。合成ファイルで `{'wavelength2': 1.544426, 'ratio2': 0.5}` を確認した(CONFIRMED)。標準手順の「Kα2 なしを試す」で救われることはある。
  - **XRDML の `<beamAttenuationFactors>`**: 無視している。強度が未補正の生計数なら、減衰器が入った最強ピークが 1/100 になる。合成ファイルでは無視されることを確認した。強度の意味は PLAUSIBLE。
  - **RAS / RASX の 3 列目(減衰器係数)**(`vendors.py:83-90`, `148-154`): 強度に掛けるのに、重みは `1/(補正後 y)` のまま。Poisson なら `1/(f²·c)` なので、減衰した点(=最強ピーク)の重みが f 倍(例では 100 倍)過大になる(PLAUSIBLE)。
  - **`*MEAS_SCAN_UNIT_Y "cps"`**: 見ていないので、cps のデータに `1/y` の重みが付き、χ² と Rexp が計数時間ぶんずれる(PLAUSIBLE)。
  - **計数 0 以下の点**: 重み 0(`vendors.py:30` ほか)。GSAS-II は `1/max(1,y)`(Rigaku)/ 1(PANalytical)で、低計数の背景では偏りうる。物理側との境界の問題。
- **再現**: `review_opus_all\gui_io\t_xrdml_ras.py`
- **直し方**:
  - XRDML は `intended` に "1" が含まれれば Kα2 を外す。
  - XRDML の減衰器係数を掛け、σ も補正後に揃える。
  - RAS は重みを `1/(f·y_corr)` にし、cps なら速度と刻みから計数時間を出して計数に戻す(またはメタデータに残して警告する)。
  - 実ファイル(SmartLab の自動 ATT、Empyrean)で確かめる必要がある。

### 10. アプリ内更新の完全性(セキュリティ)
- **場所**:
  - `src/kaisetsu/update.py:165`(`download_file`)と `:215`(`launch_installer`)
  - `gui/update_dialog.py:116`(大きさの比較だけ)
  - `.github/workflows/release.yml:62-64`
- **何が起きるか**:
  - ダウンロードしたインストーラを、ハッシュでも Authenticode 署名でも検証せずに起動している。安全性は TLS と GitHub アカウント(`contents: write` を持つ workflow)の安全性だけに依存する。
  - レート制限時の経路では `installer_size=0` になり、大きさの確認も無くなる。
  - Release は公開状態で作ってから asset を上げているので、その間に確認した利用者はインストーラの URL が 404 になる。
  - Actions はタグで固定している(SHA ではない)。
  - release.yml では pytest を走らせていない。
- **判定**: CONFIRMED(読んだ)
- **直し方**:
  - Releases API の asset の `digest`(`sha256:…`)と照合する。
  - コード署名を入れる(OSS なら SignPath Foundation など)。
  - 起動前に `WinVerifyTrust` で発行者を確認する。
  - `gh release create --draft` → upload → `gh release edit --draft=false` の順にする。
  - release.yml で `uv run pytest -m "not slow"` を必須にする。
  - Actions を SHA で固定する。

### 11. 同梱するライセンス一覧の漏れ(配布)
- **場所**: `packaging/collect_licenses.py:20, 100`、`build_windows.ps1`、`release.yml`(`uv sync --extra gui --group build`)
- **漏れているもの**:
  - **PySide6 / shiboken6**: dist-info には `LicenseRef-Qt-Commercial.txt` しか入っていない。生成された一覧にも LGPL-3.0 / GPL-3.0 の本文が無い。LGPLv3 第 4 条の「GPL と LGPL の写しを添付」を満たしていない。
  - **開発用パッケージ**: `uv sync` は既定で dev グループも入れる。そのため手元で 10-15 にビルドした `dist/kaisetsu/_internal` に `coverage`(Apache-2.0)、`hypothesis`(MPL-2.0)、`setuptools` が同梱されているが、一覧には無い。
  - **OpenSSL**: `libssl-3*.dll` と `libcrypto-3*.dll` を同梱しているのに、一覧に "OpenSSL" が 0 件。Qt 内部の third-party(harfbuzz、freetype、pcre2 など)の表示も無い。
- **判定**: CONFIRMED(生成物 `review_opus_all\gui_io\tpl.txt` と、手元の dist を確認)。CI で作った Release 物は未確認だが、コマンドが同じなので同様のはず。
- **直し方**:
  - ビルドを `uv sync --no-default-groups --extra gui --group build` にし、spec の `excludes` に hypothesis、coverage、pytest、setuptools を入れる。
  - 一覧は依存の閉包ではなく、PyInstaller の Analysis の TOC(実際に同梱したもの)から作る。
  - LGPL-3.0、GPL-3.0、OpenSSL の本文と Qt の third-party 一覧を固定ファイルとして同梱する。

### 12. アンインストールで `{app}` を丸ごと削除し、しかも保存の初期フォルダがその中にある(データを失う)
- **場所**:
  - `packaging/kaisetsu.iss:111-113`(`[UninstallDelete] Type: filesandordirs; Name: "{app}"`)
  - `gui/paths.py:120-136`。frozen では `_internal/data/public` が初期フォルダになり、サンプルを開いた後の「名前を付けて保存」の初期フォルダもそこになる。
- **何が起きるか**:
  - サンプルで練習してそのまま保存したプロジェクトが、アンインストール(再インストールのための削除を含む)で消える。
  - インストール先に `D:\データ` のような既存フォルダを指定すると、そのフォルダごと消える。
  - 補足: `[InstallDelete]` が無いので、更新のたびに古い版のファイル(Qt プラグインなど)が残る。
- **判定**: PLAUSIBLE
- **直し方**:
  - `[UninstallDelete]` は numba のキャッシュなど既知のパスだけにする。
  - 保存の初期フォルダは「ドキュメント\kaisetsu」にする。インストール先の中には保存させない(警告を出す)。
  - `[InstallDelete] Type: filesandordirs; Name: "{app}\_internal"` を追加する。

### 13. 更新確認や精密化のスレッドが残ったまま終了すると異常終了する
- **場所**: `main_window.py:1528-1529`(5 秒 / 3 秒待って `accept`)、`update_dialog.py:361`
- **再現**: fetch を 6 秒眠る関数に差し替えて 0.2 秒で閉じると、終了コードは 127 になった(待ち時間を 0 にした対照では 0)。精密化の 1 反復が 5 秒を超える場合も同じ経路になる。配布物では Windows の「動作を停止しました」が出る。
  - スクリプト: `review_opus_all\gui_io\t_close_during_check.py`(対照は同じフォルダの `t_close_control.py`)
- **判定**: CONFIRMED
- **直し方**:
  - 更新確認のスレッドは親を持たせずに切り離し、終了時は結果を捨てる。urllib のタイムアウトを短くする。
  - 精密化は中止を要求したうえで「中止を待っています」と表示し、終わるまで閉じない。

### 14. CSV 書き出しの列ずれと文字コード
- **場所**: `src/kaisetsu/io/export.py:28-34, 52-66`、`packaging/launch.py:10`、`launch_cli.py:10`
- **何が起きるか**:
  - **列ずれ**: 相の名前にカンマがあると(例: `Y3(Fe,Al)5O12`)、計算パターンの CSV はヘッダ 7 列・データ 6 列、反射リストはヘッダ 12 列・行 13 列にずれる。
  - **文字コード**: 起動スクリプトの `os.environ.setdefault("PYTHONUTF8","1")` は起動後に設定しても効かない(実測で `utf8_mode = 0  preferred = cp932`)。その結果、配布物では次のようになる。
    - `np.savetxt`(計算パターン)は cp932 で書かれる。cp932 に無い文字(例: `Ca₅(PO₄)₃F`)があると `UnicodeEncodeError` になり、スロットの中なので見えない。
    - 反射リストと説明できないピークの一覧は BOM 無しの UTF-8 で、日本語の Excel では文字化けする。
- **再現**:
  - `review_opus_all\gui_io\t_roundtrip.py`
  - 文字コードは `python -c` で確認した。
- **判定**: CONFIRMED
- **直し方**:
  - CSV は `csv` モジュールで書き、名前をクォートする。
  - 文字コードは明示的に `utf-8-sig` に統一する。
  - UTF-8 モードは spec の `EXE(..., options=[('X utf8_mode=1', None, 'OPTION')])` で有効にする。

### 15. ヒストグラム名の重複で R 因子の辞書が上書きされる
- **場所**: `refine/project.py` の `add_histogram`(名前の一意性を確認していない)と `rfactors`(`out[h.name]`)。GSAS の「すべてのバンク」は全部ファイル名(`gsas.py:210`)になる。同じファイルを 2 回読んだ場合や、ウィザードを 2 回使った場合も重複する。
- **何が起きるか**: ステータスバー、CLI の `_report`、逐次精密化が別のヒストグラムの値を表示する。
- **再現**: 結果は `histogram names: ['FAP', 'FAP']  rfactors keys: ['FAP', 'total']`。
  - スクリプト: `review_opus_all\gui_io\t_dupname.py`
- **判定**: CONFIRMED
- **直し方**: `add_histogram` で重複したら ` (2)` を付ける。バンクの名前は `"{stem} bank{k}"` にする。

### 16. Ctrl+Z と「すべて初期値に戻す」が Kα2 の試行・ハンプの追加を戻さない
- **場所**: `snapshot`/`restore` はパラメータの値だけを扱う。一方で `refine/recipe.py:294` は `radiation.ratio2=0` に書き換え、`:356` でハンプを足す。
- **何が起きるか**: 戻した後で標準手順をやり直すと、`ratio2>0` の条件が成り立たないので Kα2 の試行が二度と走らず、Kα2 なしのまま固定される。
- **判定**: PLAUSIBLE
- **直し方**: 取り消し用の記録に、装置の放射線設定とハンプも含める。

### 17. 軽微なもの
- **BOM 付きテキスト**: 最初のデータ点を黙って落とす(4001 → 4000 点)。CONFIRMED、`xy.py:26`。直し方: `encoding="utf-8-sig"`。
- **UTF-16 のテキスト**(Excel の「Unicode テキスト」): 「数値データが読めませんでした」になり、理由が分からない。CONFIRMED。BOM を見て判定する。
- **未知のバイナリ**: ASCII の行が 2 行あるだけで 2 点のデータとして読み込まれる。CONFIRMED(`t_binary.py`)。
- **3 列目**: 背景など σ 以外の列でも σ と見なす。CONFIRMED。`√y` との比が極端なら警告する。
- **複数の範囲・スキャン**: Bruker .raw の複数レンジ、RAS の複数スキャンは、最初の 1 つだけを通知なしに読む(読んだだけ)。
- **装置設定ダイアログ**: 開いて OK を押すだけで、精密化した値が表示桁に丸まり、esd も消える。CONFIRMED(`t_inst_dialog_roundtrip.py`)。実験室のデータでは Rwp は変わらない。放射光の U(〜1e-5)は小数 5 桁では崩れる。
- **CLI**:
  - `--hist 3` は英語の traceback(IndexError)が出る。
  - データファイルを渡すと「エラー: Extra data: line 1 column 8」になり、意味が通じない。
  - `--cycles 0` を受け付ける。
  - いずれも CONFIRMED。
- **古い・壊れたプロジェクトファイル**: `KeyError: 'x_min'` のような生のメッセージが出る(読んだだけ)。
- **`tr()`**: 囲み漏れが多い(wizard.py は 0 件、dialogs.py では生の文字列 27 か所に対して 10 件)。
- **`_confirm_discard`**: 「保存して閉じる」の選択肢が無い。保存の直後でも毎回聞かれる。

## 確認したこと
- **Qt の import**: `src/kaisetsu/gui/` の外では Qt を import していない(grep)。
- **プロジェクトの往復**: 精密化後に save → load して `project_to_dict` の差は 0 件、Rwp も一致した。
- **新しいスキーマ**: 新しい形式のファイルは日本語のメッセージで拒否される。
- **日本語のフォルダ名**: CIF の読み書き、XRA、Bruker raw、pdCIF、プロジェクトの保存・読み込みがすべて通る。UTF-8 モードの有無の両方で確認した。
- **テキスト形式の普通のケース**: CRLF、cp932 / UTF-8 の日本語ヘッダ、2θ の降順、重複(平均される)、負のカウント、空ファイル(日本語のエラー)はいずれも正しく扱える。
- **更新**:
  - 版の比較は 0.1.10 > 0.1.9、rc < 正式版。
  - オフライン時は URLError がそのまま呼び出し側に返り、「確認できませんでした」が出る。タイムアウトは 10 秒。
  - ダウンロードは .part に書いてから改名し、Content-Length と API の size で確認している。
  - `KAISETSU_NO_UPDATE_CHECK` で自動確認を止められる。
- **frozen 環境の numba キャッシュ**: numba が `sys.frozen` を見て利用者ごとのキャッシュに切り替える(`caching.py:288`)。手元の配布物 `dist\kaisetsu\kaisetsu-cli.exe selftest` は exit 0。
- **Inno Setup**: `PrivilegesRequired=lowest`、`CloseApplications=yes`。PATH の追加と削除は対称。
- **release.yml**: タグと `__version__` の一致を確認している。
- **CLI の基本**: ファイルが無い場合は日本語のエラーで終了コード 1。`reconfigure(errors="replace")` で文字化けによる停止を避けている。
- **ワーカー**: 例外は捕まえてメッセージで返している(握りつぶしていない)。中止は反復の区切りで効く。
- **テスト**: `PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen … uv run pytest -q -m "not slow" -p no:cacheprovider tests/gui` と io / update / cli / bruker / gsas の単体テストで 78 passed。
  - 穴: 閉じるときの確認、保存の失敗、実行中の操作のガード、逐次精密化後の状態、ウィザードの 2 回目を見るテストが無い。

## 回さなかった・分からなかったこと
- **実機の Rigaku .ras / .rasx と PANalytical .xrdml**: 減衰器係数と cps の意味(指摘 9)は合成ファイルとコードからの推定。SmartLab の自動 ATT や Empyrean の実ファイルで確かめる必要がある。
- **配布物の numba threading layer**: VCOMP140.DLL の解決に依存する。VC++ ランタイムの入っていない PC で omp と workqueue のどちらになるかは確かめていない。
- **CI で作った Release 物**: 手元の dist(10-15 にビルド)だけを見た。CI 版の中身と THIRD_PARTY_LICENSES は取得していない(ネットワーク禁止のため)。
- **配布物の挙動**: インストーラの実行、アンインストール、アプリ内更新の実動作は試していない。
- **未読の部分**: PDF / TOF のダイアログ、`structure_view`、`help_viewer`、`plot_widget` の細部(ホバー、画像の書き出し)はざっと読んだだけ。
- **macOS / Linux**: CI 以外の環境での GUI の挙動は見ていない。
