# GLM / DeepSeek レビュー 2026-10-17(原文)

ユーザーが別のモデル 2 つに読ませたレビューの原文。評価と対応は `review_glm_deepseek_2026-10-17.md`。

---

# kaisetsu コードレビューレポート（2026-10-17）

## レビューの範囲と方法

- **対象**: `src/kaisetsu/` 以下の Python ソースコード（`tests/`、`docs/`、`examples/`、`validation/`、`data/` は除外）。`src/kaisetsu/gui/` は分量が大きいため `worker.py` のみ目視し、残りは末尾の「調べきれなかったファイル」に列挙。
- **方法**: エントリポイント（`__main__.py`, `cli.py`, `api/`）→ コア計算（`pattern/`, `profile/`, `reflections/`, `scattering/`, `crystal/`）→ 精密化（`refine/`, `mem/`, `pdf/`）→ I/O（`io/`）の順に手動で通読。
- **前提**: コード内に「レビュー 2026-10-17 #N」由来の根拠コメントが多数ある。コメントで設計意図が明示されている箇所は既知の選択と見なし、重複指摘はしていない。物理定数（Lorentz 因子・偏光補正・Sabine/Lobanov 吸収・消衰）については GSAS-II との整合を確認済みのため対象外。
- **結論の先取り**: 精密化フローの本体（`lsq.py` の Levenberg-Marquardt、`rfactors.py` の R 因子計算、`project.py` の保存・読込）や Rietveld の物理係数まわりに「確実に壊れる」「黙って間違った結果を出す」レベルの問題は見つからなかった。以下の指摘はすべて条件次第で現れるもの、または性能・統計量の精度に影響するもの。水増しはしていない。

---

## 検出された問題

### [1] 可能性を絞る位置座標トライアルで、換算 χ² に Hamilton の F 検定を適用している

- **場所**: `src/kaisetsu/refine/recipe.py:546-556`（`_position_trial`）
- **重要度**: 中
- **確信度**: 中
- **内容**: `_position_trial` は変数の組み合わせ違いのトライアルを試し、`gain = (best.chi2 - c.chi2) / c.chi2` が `POSITION_F_CRIT * extra / dof` 以上のとき追加変数側（`c`）を採用する。この比較は Hamilton 型の F 検定（`(χ²₁−χ²₂)/χ²₂ ≥ F·Δp/dof`）を意図しているが、比較に使っている `chi2` は `powder_rfactors`（`rfactors.py:69`）で **`sum(w·diff²) / dof` と自由度で割り戻した値（換算 χ²）** であり、分子・分母のモデルで自由度が異なるため式の前提が崩れる。導出すると、採用条件は実質的に `(A/B) ≥ (1 + Fcrit·extra/d0)·(d1/d0)`（A=少数変数側の χ²、B=多数変数側の χ²、d1>d0）となり、正しい基準 `A/B ≥ 1 + Fcrit·extra/d0` より**厳しめ**、すなわち追加変数を採りにくい保守的な側に偏る。バイアス自体は小さいが、トライアルの採否が境界値で分かれるケースで結果が変わる可能性がある。`gain` が負のときは条件を満たさないので破綻はしない。
- **根拠**:
  ```python
  # recipe.py:546-555
  best = tried[0]
  for c in tried[1:]:
      extra = len(c.names) - len(best.names)
      if extra <= 0:
          if c.chi2 < best.chi2:
              best = c
          continue
      dof = max(c.fit.n_data - c.fit.n_params, 1)
      gain = (best.chi2 - c.chi2) / max(c.chi2, 1e-300)
      if gain >= POSITION_F_CRIT * extra / dof:
          best = c
  ```
  ```python
  # rfactors.py:67-69  — chi2 は dof で割った換算値
  dof = max(n - n_params, 1)
  rexp = float(np.sqrt(dof / sum_wyo2)) if sum_wyo2 > 0 else float("nan")
  chi2 = float(np.sum(w * diff ** 2) / dof)
  ```
- **再現条件**: `.pos` トライアルを含むレシピで、追加変数モデルの χ² 改善が `POSITION_F_CRIT × extra / dof` 境界付近のとき。
- **修正案（方向性のみ）**: 比較に生の `sum(w·diff²)`（自由度未除算）を使うか、換算 χ² のまま比較するなら閾値側の式を換算値に合わせて補正する。あるいは係数 `POSITION_F_CRIT = 10.0` を経験値として据え置く判断をコメントで明示する。

### [2] `numerical_jacobian` で値が下限に一致 + step が上限超のとき h=0 による 0/0 列

- **場所**: `src/kaisetsu/refine/lsq.py:102-111`（`numerical_jacobian`）
- **重要度**: 低
- **確信度**: 中
- **内容**: 前進差分で `xk[k] + h > upper[k]` のときは後退差分 `h = -h` に反転するが、その後 `xk[k] < lower[k]` になった場合の救済として `xk[k] = lower[k]; h = xk[k] - x[k]` としている。ここで `x[k]` がすでに下限と一致していると `h = 0` となり `(fun(xk) - r0) / h` は 0/0 となり NaN（または inf）の列がヤコビアンに入る。LM 側に NaN 防護があるかどうか（`refine_least_squares` 内の取り扱い）までは追い切れていないため、致命的に至るかは未確認。反転後 `h < 0` かつ `x[k] == lower[k]` という条件は「range < step かつ現在値が下限」という稀な組み合わせで起こる。
- **根拠**:
  ```python
  # lsq.py:102-111
  for k in range(len(x)):
      h = steps[k]
      xk = x.copy()
      if xk[k] + h > upper[k]:
          h = -h
      xk[k] += h
      if xk[k] < lower[k]:
          xk[k] = lower[k]
          h = xk[k] - x[k]      # x が下限一致だと h == 0
      jac[:, k] = (fun(xk) - r0) / h   # 0/0 → NaN
  ```
- **再現条件**: 下限クリップされたパラメータ（例: サイト占有率の 0 下限や分率座標の境界）で、`step` が上限との距離を超えている反復があれば発生し得る。
- **修正案（方向性のみ）**: `h == 0` になったら刻みを 1 段小さくして前進差分に戻す、または `abs(h) < ε` のとき列を 0 で埋めて NaN が LM に漏れないようにする。

### [3] `jacobian.py` の解析的差分に続く数値微分フォールバックが下限を考慮せず、反転後のクリップでスケールがずれる

- **場所**: `src/kaisetsu/refine/jacobian.py:266-274`（`project_jacobian`）
- **重要度**: 低
- **確信度**: 中
- **内容**: 強度・テクスチャ系の解析差分が適用できない場合のフォールバック数値微分は、上限接近のみを検出して後退差分に反転するが、反転後の値が下限を下回るケース（range < 2·step）をチェックしていない。`_perturbed` 側で下限にクリップされたとしても、除算には反転後の元の刻みを使って割るため、ヤコビアン列に実際の変位と食違うスケール（または 0）が入る。`lsq.numerical_jacobian`（[2]）と同系の境界処理の未整備だが、こちらは列が 0 または誤スケールになる点が異なる。
- **根拠**:
  ```python
  # jacobian.py:266-274
  # それ以外は数値微分(前進差分、上限に近ければ後退)
  step = p.step
  if math.isfinite(p.max) and p.value + step > p.max:
      step = -step          # 下限側は無チェック
  _perturbed(project, p, step)
  r1 = fn()
  _perturbed(project, p, -step)
  restore_results()
  jac[:, col] = (r1 - base) / step
  ```
- **再現条件**: 強度/テクスチャ系パラメータで、min と max の間隔が刻み未満に収まっている、または現在値が下限近傍の反復。
- **修正案（方向性のみ）**: `lsq.numerical_jacobian` と共通の境界処理ルーチンに統一し、反転後に下限超過を確認して刻みを縮める。

### [4] ヒストグラム単位の R 因子に全自由パラメータ数を使い、rexp・χ² が偏る

- **場所**: `src/kaisetsu/refine/project.py:1784-1818`（`rfactors`）
- **重要度**: 低
- **確信度**: 中
- **内容**: `rfactors()` は `n_par = len(self.params.free())` をプロジェクト全体の自由パラメータ数として取得し、各ヒストグラム・各 PDF データセット・total すべてに同じ値を渡す。total としては正しいが、個別ヒストグラムの `rexp` と `χ²` は `dof = n_data(ヒストグラム) − n_par(全体)` で計算されるため、複数ヒストグラムのとき点数に対し過大な n_par となり大幅にずれる。`Rwp` は自由度を含まないので影響なし。複数ヒストグラム Butt χ² を閾値判定に用いている箇所（`project.py:1951` の `if rf.chi2 > 10.0`）はメッセージ表示のみで実害は小さい。
- **根拠**:
  ```python
  # project.py:1784-1797
  def rfactors(self) -> dict[str, RFactors]:
      n_par = len(self.params.free())
      ...
      for i, h in enumerate(self.histograms):
          res = h.result if h.result is not None else self.calculate(i)
          m = h.mask()
          ...
          out[h.name] = powder_rfactors(
              h.data.y[m], res.y_calc[m], h.data.weight[m], n_par, res.y_background[m]
          )
  ```
- **再現条件**: 複数ヒストグラムまたは PDF プロジェクトで `rfactors()` のヒストグラム単位の値（ログ表示・レポート）を参照する場合。
- **修正案（方向性のみ）**: ヒストグラム単位では当該ヒストグラムに寄与するパラメータ数（ヒストグラム共通パラメータ + 位相パラメータ）を個別に数えて渡す。total は現状維持。

### [5] `project_jacobian` の `done` フラグが常に True で、数値微分へのフォールバックが発動しないデッド変数

- **場所**: `src/kaisetsu/refine/jacobian.py:207-265`
- **重要度**: 低
- **確信度**: 高（デッドコードであること自体は確実）
- **内容**: 強度・テクスチャ系の解析差分ブランチに入ると `done = True` が代入され、その後に False に戻す経路がない。一方ループ末尾は `if done: continue` なので、このブランチに入った列は必ずそこで打ち切られ、「それ以外は数値微分」の共通部に到達しない。意図としては「ヒストグラムで解析差分が計算できなければ数値微分に落ちる」だったと推測されるが、その場合 `targets` ループ内で該当相がどのヒストグラムにも含まれない（全 `continue`）ケースでヤコビアン列がゼロのまま残る。`done` 自体は不要な変数なので、フォールバックが設計どおりには動いていないデッドコード。
- **根拠**:
  ```python
  # jacobian.py:207-211, 264-274
  targets = [(hi, hists[hi])] if hi is not None else list(enumerate(hists))
  done = True                      # ← 常に True。後続で False にする箇所はない
  for i, h in targets:
      if phase not in h.contributions or h.result is None:
          continue
      ...
      if done:
          continue
  # それ以外は数値微分(前進差分、上限に近ければ後退)
  step = p.step
  ...
  ```
- **再現条件**: 強度またはテクスチャパラメータの位相が、有効なヒストグラムのいずれにも含まれないケース。`done` が常に True であることは静的に確実。
- **修正案（方向性のみ）**: 「解析差分が 1 つでも成功したら True」のセマンティクスにする（初期 False、成功ごとに True）か、`done` を削除して現在の動作を意図として明示する。

### [6] 格子定数走査で `span ≥ CELL_SCAN_WIDE_SPAN` のとき同一範囲の走査を 2 回実行する

- **場所**: `src/kaisetsu/refine/cell_scan.py:214-233`（`scan_cell`）
- **重要度**: 低（性能のみ、結果は変わらない）
- **確信度**: 高
- **内容**: `for sp in (span, max(CELL_SCAN_WIDE_SPAN, span))` の 2 周ループは、1 周目の等方走査で改善が得られなかった場合に広げた範囲で再走査する意図だが、最初から `span >= CELL_SCAN_WIDE_SPAN` の呼び出しでは `max(...)` は `span` と等価であり、2 周目が 1 周目と完全に同じ `step_sp, half_sp` で同じ `_scan_1d` を再計算する。分岐 `if rwp0 <= CELL_SCAN_WIDE_RWP or sp > span: break`（231 行目）は 1 周目の `sp == span` では break しないため、重複走査が実行される。結果は同一なので出力への影響はないが、走査点数分のパターン計算を無意味に 2 回行い、大きな構造では体感時間にそのまま響く。
- **根拠**:
  ```python
  # cell_scan.py:214-233
  for sp in (span, max(CELL_SCAN_WIDE_SPAN, span)):
      step_sp, half_sp = scan_step(project, sp)
      f_iso, rwp_iso = _scan_1d(project, params, step_sp, half_sp, progress)
      ...
      if rwp0 <= CELL_SCAN_WIDE_RWP or sp > span:
          break
      span_used = max(CELL_SCAN_WIDE_SPAN, span)
  ```
- **再現条件**: `scan_cell(project, phase, span=CELL_SCAN_WIDE_SPAN 以上)` で、1 周目の走査が `min_gain` を満たさなかったとき。
- **修正案（方向性のみ）**: ループ前に `spans = [span] if span >= CELL_SCAN_WIDE_SPAN else [span, CELL_SCAN_WIDE_SPAN]` のように候補を畳む。

---

## 確信度:低 の項目（問題か設計か判断できなかったもの）

### [7] 連続精密化の `x_min`/`x_max` 更新が単調に狭まり続ける

- **場所**: `src/kaisetsu/refine/sequential.py:141-142`
- **重要度**: 低 / **確信度**: 低
- **内容**: `h.x_min = max(h.x_min, float(data.x[0]))`、`h.x_max = min(h.x_max, float(data.x[-1]))` は「逐次処理を通じて共通範囲に正規化する意図」とすれば正しい。ただし更新はinder 依存で単調な縮小しかしないため、意図（共通領域の交差）なのか通過範囲のトラッキングなのかをコードからだけで確定できない。結果が安全側（狭い範囲）に倒れるため実害は限定的と見ているが、未確認の挙動として挙げる。
  ```python
  h.x_min = max(h.x_min, float(data.x[0]))
  h.x_max = min(h.x_max, float(data.x[-1]))
  ```

### [8] `_run_zspa` で λ 半減の反復が `history` に記録されない

- **場所**: `src/kaisetsu/mem/core.py:763-778`（`_run_zspa`）
- **重要度**: 低 / **確信度**: 低（記録されない事実は確実だが、記録しないのが意図の可能性がある）
- **内容**: χ² が悪化して `lam *= 0.5` で `continue` する分岐では `history.append((lam, chi2))` が行われず、`history`（λ・C の履歴）には成功した反復と初期値のみが入る。λ の絞り込み過程を後から確認したい用途（MEM 収束の診断）では履歴が不完全になる。収束判断に history を使っているかどうかまで追い切れておらず、診断上の問題の可能性が高い。
  ```python
  if chi2_new > chi2:
      lam *= 0.5
      if lam < 1e-8:
          return m, chi2, it, "刻みが小さくなりすぎました"
      continue          # ← ここでは history.append しない
  m, fc, chi2 = m_new, fc_new, chi2_new
  history.append((lam, chi2))
  ```

### [9] 多波長パターン計算の `inten_total` が「窓が切れた」成分のみ加算される

- **場所**: `src/kaisetsu/pattern/calculator.py:439-443, 467-468`
- **重要度**: 低 / **確信度**: 低
- **内容**: 波長ループ（特性 X 線の Kα₁/Kα₂ 等）で各反射ごとに `inten_total[i] += inten[i]` を行っているが、対象が `np.nonzero(lengths > 0)[0]`（計算窓が非空の反射）に限られる。実装を見ると `use = ok & (inten > 0.0) & (i1 > i0)` の条件は「Pattern に含まれる成分」と(`inside` 判定込みで)概ね一致するため無害化されていると判断したが、窓端ケースで `table.intensity_total` と `intensity` の和が完全に一致しない可能性の有無まで確認しきれていない。実害の検証は行っていない。
  ```python
  for i in np.nonzero(lengths > 0)[0]:
      ...
      inten_total[i] += inten[i]
  ...
  if table is not None:
      table.intensity_total = inten_total
  ```

---

## 調べきれなかったファイル・情報不足の箇所

以下は今回のレビューで読めていない、または決定的な検証ができなかったため、問題の有無を**報告できない**箇所（未確認であり、問題がないという意味ではない）。

- **`src/kaisetsu/gui/` の 24 ファイル**（`worker.py` / `__init__.py` を除く）: `main_window.py`（約 82 KB）、`app.py`, `wizard.py`, `dialogs.py`, `phase_dialog.py`, `param_model.py`, `plot_widget.py`, `structure_view.py`, `sequence_dialog.py`, `mem_dialog.py`, `pdf_dialogs.py`, `tof_dialog.py`, `lattice_dialog.py`, `background_dialog.py`, `range_dialog.py`, `residual_dialog.py`, `update_dialog.py`, `help_viewer.py`, `settings.py`, `paths.py`, `dialog_utils.py`, `flow_layout.py`, `i18n.py`, `__main__.py`。GUI は状態遷移とイベント処理が多く、目視だけでは「黙って間違う」系の検出精度が出ないため対象外とした。
- **`tests/` 全体**（unit / gui / integration / validation）: テストコード自体のレビューはしていない。「テストがある箇所は信頼度を上げる」方向の補正のみ行った。
- **`tools/`, `packaging/`, `examples/` のスクリプト**: 未読。
- **数値実験が必要な領域**: [1] の F 検定バイアスの具体的な大きさ、[9] の `intensity_total` の一致条件は静的解析のみで判断しており、実行検証はしていない（依存パッケージのインストール・スクリプト実行は不可のため）。
- **`io/readers/` のベンダ個別パーサ**（`gsas.py`, `bruker.py`, `fullprof.py` 等）: 構造は確認したが、実ファイルとの突合（endian・ヘッダ変種の網羅）はしていないため、パーサ全体が正しいとは言い切れない。

---

## まとめ

- 今回見つけた中で最も影響が大きいのは **[1] `_position_trial` の F 検定の統計的基準**。結果の採否（位置座標トライアルのモデル選択）に直接関わるが、偏りの方向が保守的側であり致命的ではない。
- その他は条件が揃ったときの NaN 生成（[2][3]）、表示統計のずれ（[4]）、デッドコードと性能（[5][6]）。
- 「確実に壊れる」「黙って物理的に間違った結果を出す」レベルの問題は、レビューした範囲では見つからなかった。GUI と外部ファイルパーサの実データ突合は未実施で、未知の領域が残っている。

---

# コードレビュー(バグ探索)DeepSeek V4.1 Flash 版 — kaisetsu v0.1.14

- 対象: このリポジトリの `src/kaisetsu/`(**読み取りのみ**。ファイルの作成・変更・削除は行っていない)
- 手法: 入口(`cli.py` / `gui/` / `api/__init__.py`)からデータフローをたどり、カテゴリ
  (1) 確実に壊れる / (2) 条件付きで壊れる / (3) 黙って誤った結果 / (4) 物理(単位・角度・規格化・フィット)
  の観点で確認。コードから確認できない点は「未確認」と明記。
- 既存の外部レビュー(`review_2026-10-17.md`, `review_opus_2026-10-17*.md`, v0.1.13→0.1.14)で
  修正済みの項目は再掲しない。本レビューは 0.1.14 に対する残存/新規分。
- 注意: すべて「読み取り」に基づく指摘。修正は行っていない。

---

## 全体構造(データフロー順)

- 入口: `cli.py` / `gui/` / `api/__init__.py`。中心は `refine/project.py` の `Project`。
- 入力: `io/readers/`(xy, gsas, vendors, bruker, fullprof, rietan, pdcif, opxrd, zrietveld)
  → `pattern/histogram.py` の `PowderData`。
- 構造: `crystal/`(lattice, spacegroup, structure, atoms, cif, symmetry)。
- 散乱: `scattering/`(xray, neutron, radiation, tables)。
- 反射: `reflections/`(hkl 生成, structure_factor)。
- パターン計算: `pattern/`(calculator, instrument, tof, histogram)
  → `profile/`(pseudo_voigt, fcj, fcj_batch, corrections, texture, stephens, background)。
- 精密化: `refine/`(lsq, jacobian, parameters, constraints, rfactors, residual, project,
  recipe, lattice, sequential, cell_scan, width_check, pawley)。
- 解析出力: `refine/pawley.py` → `mem/`(core, mpf, io)。`pdf/`(transform, calculator, data)。
- 永続化: `io/project.py`、エクスポート `io/export.py`。

---

## 指摘一覧(重要度の高い順)

### [1] 選択配向(球面調和)が係数長の不一致で黙って無効化される

- 場所: `src/kaisetsu/pattern/calculator.py:123-127`(`PhaseContribution.orientation_factor`)
- 重要度: 中
- 確信度: 高
- 内容: `sh_order > 0` でも、`sh_coefficients` の長さが基底次元と一致しないときは警告なしに
  `texture_factor` を掛けずに返す。`io/project.py:398-401` はプロジェクト読み込み時に
  `sh_order` を先に入れてから係数をそのまま代入し、長さを検証していないため、順序や空間群が
  変わったプロジェクトファイルでは選択配向が無言で消え、強度が系統的にずれる。
- 根拠:

  ```python
  if self.sh_order > 0:
      coef = self.sh_coefficients if sh_coefficients is None else sh_coefficients
      basis = self.texture_basis()
      if coef is not None and len(coef) == sum(u.shape[1] for _, u in basis):
          po = po * texture_factor(hkl, st.lattice, basis, coef)
  return np.asarray(po)
  ```

- 再現条件: `sh_order>0` の相で `sh_coefficients` の長さが基底次元と異なる
  (例: 旧プロジェクト読み込み、GUI 外の API 利用、空間群変更後)。
- 修正案: 不一致時は例外にするか `logging.warning` を出し、`sh_order=0` に落とすことを明示する。
  読み込み時に `refine/project.py:666-667` と同じ整合チェックを入れる。

### [2] hkl キー符号化が桁あふれを起こしうる(代表選択・重複除去)

- 場所: `src/kaisetsu/reflections/hkl.py:82-84`(`_hkl_key`)、`:89-100`(`canonical_hkl`)、`:152-164`(`generate_reflections`)
- 重要度: 中
- 確信度: 中
- 内容: `(h+off)*base²+(k+off)*base+(l+off)`, `base=2m+1`, `off=m` は各成分が `[-m,m]` に
  入る前提の位取り。`canonical_hkl` は入力 hkl の `hmax` だけから `base` を決めるため、回転で
  成分が `m` を超えると桁が `[0,base)` を外れ、キーの大小が辞書式順序と一致しなくなる
  (`(1,2,0)` と `(2,-3,0)` が同じキーになる、等)。ただし `generate_reflections` 側は、採用済み
  反射 `d≥d_min` に対しコーシー・シュワルツで `|h'|≤a/d_min≤hmax` が保証されるため桁あふれは
  起きず、影響は `canonical_hkl`(実質テストと未使用の `Project.canonical` のみ)に限定される。
  誤った代表が選ばれる具体例は未確認。
- 根拠:

  ```python
  def _hkl_key(hkl, base, offset):
      return (hkl[:,0]+offset)*base*base + (hkl[:,1]+offset)*base + (hkl[:,2]+offset)
  ...
  hmax = math.floor(lattice.a / d_min)
  ...
  m = max(hmax, kmax, lmax); base = 2*m+1
  ```

- 再現条件: `canonical_hkl` に、回転像が入力 hmax の外に出る hkl を渡す。
- 修正案: 符号化を使わず `np.lexsort` のタプル比較で代表を選ぶか、`base=4*m+2`, `offset=2*m+1`
  にして全回転像 `[-2m,2m]` を収める。

### [3] `find_symmetry` が異方性 ADP(U_ij)を捨てる

- 場所: `src/kaisetsu/crystal/symmetry.py:78-85` と `:113-119`(`find_symmetry`)
- 重要度: 中
- 確信度: 高
- 内容: `kinds` のキーが `(species, occupancy, u_iso)` のみで、出力 `AtomSite(..., u_iso=uiso)` に
  `u_aniso` を渡さない。異方性サイトは「等方・u_iso=0」として返るため、その構造を
  `Structure.expand()` → 構造因子に使うと温度因子が 1 になり強度が系統的にずれる。ただし現状
  `find_symmetry` は `tests/unit/test_symmetry_opxrd.py` からのみ呼ばれており、本体フロー
  (`io/readers/opxrd.py` はコメントのみ)では未使用。
- 根拠:

  ```python
  key = (site.species, round(site.occupancy, 6), round(site.u_iso, 6))
  ...
  species, occ, uiso = kinds[std_types[i] - 1]
  sites.append(AtomSite(f"{el}{counts[el]}", species, std_pos[i] % 1.0, occupancy=occ, u_iso=uiso))
  ```

- 再現条件: 異方性 U_ij を持つ構造を `find_symmetry` に通し、戻り値の `sites[*].u_aniso` を見る
  (常に `None`)。
- 修正案: `u_aniso`(丸めたタプル)を `kinds` のキーに含め、`AtomSite(..., u_aniso=u_aniso)` を
  渡す。少なくとも `u_iso==0 and u_aniso is not None` なら警告する。

### [4] GSAS 旧 `.prm` の偏光係数が空欄で 0.0(完全偏光)になる

- 場所: `src/kaisetsu/pattern/instrument.py:186-191`(`CWInstrument.from_gsas_prm`)
- 重要度: 低
- 確信度: 中
- 内容: ICONS 行の偏光フィールドが空/短いと `_sfloat` が 0.0 を返し、`Polariz.=0.0`(完全偏光)
  になる。`from_gsas2` は同キーの既定を 0.5 にしており、挙動が非対称。なお直上の
  `I(L2)/I(L1)` は `or 0.5` で補正しているのに偏光だけ補正していない。GSAS の実ファイルで
  当該フィールドが空になるか(桁位置)は未確認。
- 根拠:

  ```python
  d.update({"Lam1": lam1, "Lam2": lam2, "I(L2)/I(L1)": _sfloat(s[40:50]) or 0.5})
  d["Polariz."] = _sfloat(s[55:65])
  ...
  d["Polariz."] = _sfloat(s[40:50])
  ```

- 再現条件: 偏光フィールドが空欄の X 線 CW `.prm` を `from_gsas_prm` で読むと `polarization==0.0`。
- 修正案: `0.5` を既定にする(`_sfloat(...) or 0.5`)。中性子分岐の `=0.0` はそのままでよい。

### [5] `fcj_pseudo_voigt` の `n_quad` が `accurate` 法では無視される

- 場所: `src/kaisetsu/profile/fcj.py:138-192`(`fcj_pseudo_voigt`)
- 重要度: 低
- 確信度: 高
- 内容: `n_quad` は `method=="gsas"` の分岐でしか使われず、既定の `accurate` では
  `fcj_profile_batch(..., method="accurate")` に渡されない。`n_quad` を指定しても求積点数は
  変わらず、引数が黙って無効になる。ただしリポジトリ内に `n_quad` を渡す呼び出しは存在せず、
  直接 API を叩いた場合に限る。
- 根拠:

  ```python
  if method == "accurate":
      ...
      out, _, _ = fcj_profile_batch(..., method="accurate")   # n_quad 未使用
  ...
  n = n_quad if n_quad is not None else quadrature_order(...)
  ```

- 再現条件: `fcj_pseudo_voigt(..., n_quad=64, method="accurate")` とすると結果が変わらない。
- 修正案: `accurate` 側でも `n_quad` を尊重する(対応する引数を `fcj_profile_batch` に追加)か、
  `accurate` では `n_quad` を無視することを docstring/引数から明示する。

### [6] `ChebyshevBackground` の係数が空だと `basis` が例外

- 場所: `src/kaisetsu/profile/background.py:71-75`(`ChebyshevBackground.basis`)
- 重要度: 低
- 確信度: 中
- 内容: `order==0`(係数長 0)のとき `chebvander(t, -1)` が `ValueError` になる。既定は長さ 1
  なので、`ChebyshevBackground(np.zeros(0), ...)` のような直接構築や、外部からの復元で係数が
  空になった場合に `basis`/最小二乗構築で落ちる。
- 根拠:

  ```python
  return np.asarray(np.polynomial.chebyshev.chebvander(t, self.order - 1))
  ```

- 再現条件: `ChebyshevBackground(np.array([]), 0.0, 1.0).basis(np.array([0.5]))`。
- 修正案: `order == 0` を弾く(`__post_init__` で係数長 ≥ 1 を要求)か、`max(self.order - 1, 0)`
  にする。

### [7] 散乱因子テーブルの `Hiso` / `Cval` / `Sival` に到達できない

- 場所: `src/kaisetsu/scattering/xray.py:49-53`(`XrayScatteringFactors.resolve`)、
  `src/kaisetsu/crystal/atoms.py:23,32-34`(`normalize_species`/`split_species`)
- 重要度: 低
- 確信度: 高
- 内容: `resolve` は先に `normalize_species` を通すが、`_SPECIES_RE` は「元素+電荷」しか
  許さないため `"Cval"`, `"Hiso"`, `"Sival"` で `ValueError` になる。`it1992`/`wk1995` に実際に
  定義があるキー(`it1992.py:17,34,53`, `wk1995.py:21,526,561`)が API から選べず、呼び出し側が
  期待する `KeyError` ではなく `ValueError` が出る。
- 根拠:

  ```python
  key = normalize_species(species)
  if key in self._table:
      return key
  ```

- 再現条件: `XrayScatteringFactors().f0("Cval", np.array([0.1]))` → `ValueError`。
- 修正案: `resolve` の先頭で生ラベル(`strip`)を直接テーブル照合する
  (`key = species.strip(); if key in self._table: return key`)、または未使用キーをテーブルから削る。

### [8] `Lattice.reciprocal` の `acos` にクリップがない

- 場所: `src/kaisetsu/crystal/lattice.py:137-140`(`Lattice.reciprocal`)
- 重要度: 低
- 確信度: 中
- 内容: 軸角が 0°/180° に極端に近い、または計量テンソルがほぼ特異な格子では、丸め誤差で
  `acos` の引数が `1` をわずかに超え `math domain error` になる。`Structure.expand()` から毎回
  呼ばれるため入口で例外になる。`__post_init__` は体積が正なら通すので弾かれない。発火条件は
  未確認。
- 根拠:

  ```python
  alpha_r = math.degrees(math.acos(gs[1, 2] / (br * cr)))
  ```

- 再現条件: 退化に近い格子(軸角 ≈0°/180°)で `Lattice.reciprocal`。
- 修正案: `math.acos(min(1.0, max(-1.0, c)))` にする。

### [9] `Radiation` が `wavelength3` / `ratio2` / `ratio3` を検査しない

- 場所: `src/kaisetsu/scattering/radiation.py:58-62`(`Radiation.__post_init__`)
- 重要度: 低
- 確信度: 高
- 内容: `wavelength`, `wavelength2` は `>0` を検査するが、`wavelength3` と比率は検査しない。
  `Radiation.xray(1.54, wavelength3=-1.0, ratio3=0.1)` が通り、負波長や負比がそのまま
  `wavelengths` に入る。
- 根拠:

  ```python
  if self.wavelength2 is not None and self.wavelength2 <= 0:
      raise ValueError(...)
  ```

- 再現条件: `Radiation.xray(1.54, wavelength3=-1.0, ratio3=-0.1).wavelengths`。
- 修正案: `wavelength3` は `wavelength2` と同様に `>0` を、`ratio2`/`ratio3` は `>=0`(有限)を
  検査する。

### [10] 反射候補の `hmax` に許容誤差がなく境界の反射を落としうる

- 場所: `src/kaisetsu/reflections/hkl.py:124-127`(`generate_reflections`)
- 重要度: 低
- 確信度: 中
- 内容: 採用判定は `inv_d2 <= 1/d_min² * (1+1e-12)` と境界込みなのに、候補範囲は
  `math.floor(a/d_min)` で誤差を見ずに切る。`a/d_min` が数学的に整数で浮動小数の商が僅かに
  下回ると、その指数の反射が永久に候補から外れ、採用側の `1e-12` が無意味になる。具体例は
  未確認。
- 根拠:

  ```python
  hmax = math.floor(lattice.a / d_min)
  keep = inv_d2 <= 1.0 / d_min**2 * (1.0 + 1e-12)
  ```

- 再現条件: `a/d_min` がほぼ整数になる格子・範囲。
- 修正案: `hmax = math.floor(lattice.a / d_min * (1.0 + 1e-12))` のように余裕を持たせ、最終判定は
  既存 `keep` に委ねる。

### [11] `ReflectionList.index_of` の docstring と実装が食い違う

- 場所: `src/kaisetsu/reflections/hkl.py:71-74`(`ReflectionList.index_of`)
- 重要度: 低
- 確信度: 高
- 内容: 「代表でなくてもよい」と書いてあるが、実装は格納行(代表 hkl)との完全一致のみ。代表
  `(1,0,0)` しか無いとき `index_of(0,1,0)` は `-1`。
- 根拠:

  ```python
  """指定 hkl(代表でなくてもよい)の添字。無ければ -1。"""
  match = np.where(np.all(self.hkl == np.array([h, k, l]), axis=1))[0]
  ```

- 再現条件: 等価だが代表でない hkl を渡す。
- 修正案: docstring を実装に合わせる(代表完全一致のみ)か、`canonical_hkl` で正規化してから
  照合する。

### [12] 数値ヤコビアンが上下限でゼロ除算しうる

- 場所: `src/kaisetsu/refine/lsq.py:102-111`(`numerical_jacobian`)
- 重要度: 低
- 確信度: 中
- 内容: パラメータが下限ちょうどで、刻み `h` が `upper-lower` より大きいと、`h=-h`→`xk=h` が
  下限未満→`h = xk - x = 0` となり `(...)/0` で inf/nan が入る。通常は刻みが十分小さいので
  発火しないが、拘束や狭い上下限を組み合わせると起こりうる。
- 根拠:

  ```python
  if xk[k] + h > upper[k]:
      h = -h
  xk[k] += h
  if xk[k] < lower[k]:
      xk[k] = lower[k]
      h = xk[k] - x[k]
  jac[:, k] = (fun(xk) - r0) / h
  ```

- 再現条件: 下限一致パラメータで `step > (upper-lower)`。
- 修正案: `h` が 0(または極小)なら片側差分を諦める(`jac[:, k] = 0` にする、または警告)。

---

## 確信度=低 の項目

### [13] `InterpolatedBackground` の既定 `x_min/x_max` が `points_x` と同期しない

- 場所: `src/kaisetsu/profile/background.py:96-109`(`InterpolatedBackground.__post_init__`)
- 重要度: 低
- 確信度: 低
- 内容: 既定は `x_min=0.0, x_max=1.0` で、`__post_init__` の同期条件 `x_max <= x_min` を
  満たさないため同期されない。`InterpolatedBackground(points_x=[10,20,80])` では `x_min/x_max` が
  0/1 のまま残る。`x_min/x_max` は `with_domain` を持たない本クラスでは `__call__`/`basis` に
  効かず、実害が出る経路(シリアライズ・`refine/project.py:589-590` の再構築)はいずれも明示値を
  渡すため、実害を確認できなかった。
- 根拠:

  ```python
  x_min: float = 0.0
  x_max: float = 1.0
  ...
  if self.x_max <= self.x_min:
      self.x_min, self.x_max = float(self.points_x[0]), float(self.points_x[-1])
  ```

- 再現条件: 明示 `points_x` で構築し `x_min`/`x_max` を参照。
- 修正案: `points_x` から常に同期する(引数で与えられていない場合)。

---

## レビューできなかったファイル・情報不足

- サブエージェント分担: `profile`・`crystal`(結晶/散乱)担当は完了し内容を裏取り済み。
  `refine`・`io/cli`・`gui` 担当はキャンセル/未完のため、該当領域は直接読解のみ。
- 未読または部分読解(不具合の有無を断定できない):
  - `src/kaisetsu/refine/recipe.py`(全体)、`src/kaisetsu/refine/project.py`(大規模・部分)、
    `src/kaisetsu/refine/lattice.py`(部分)、`src/kaisetsu/refine/__init__.py`
  - `src/kaisetsu/io/readers/{bruker,fullprof,vendors,rietan,pdcif,zrietveld}.py`
  - `src/kaisetsu/io/{export,project}.py`(部分)
  - `src/kaisetsu/gui/`(全体)、`src/kaisetsu/cli.py`(部分)
  - `src/kaisetsu/selftest.py`、`src/kaisetsu/update.py`
  - `src/kaisetsu/pattern/{presets,histogram}.py`(部分)
- 情報不足(コードから確認できなかった点):
  - `CWInstrument.from_gsas_prm` の ICONS 桁位置(`s[40:50]`, `s[55:65]`)と、実 GSAS `.prm` で
    空欄になるか(指摘 [4])。
  - `_hkl_key` の桁あふれで代表選択が実際に誤る/反射が重複・欠落する具体例(指摘 [2])。
  - `generate_reflections` の `hmax` floor で取りこぼす具体例(指摘 [10])。
  - `Lattice.reciprocal` の `acos` が `domain error` になる実格子(指摘 [8])。

## 参考(バグではない)

- `examples/mem_pbso4.py` のコメントは「z = 1/4」だが実際は `axis=1`(y 軸)を使っており、
  ドキュメント側の食い違い(コードの動作は仕様どおり)。
- `fcj_pseudo_voigt` の `n_quad`([5])、`find_symmetry`([3])、`canonical_hkl`([2])はいずれも
  リポジトリ内の呼び出しが限定的(テスト/未使用 API)のため、実害は小さいと評価している。
