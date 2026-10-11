# Phase 7 検証記録: TOF 中性子

- 日付: 2026-10-12
- 実装: `kaisetsu.profile.tof_profile`(back-to-back 指数 × 擬 Voigt の閉形式)、`kaisetsu.pattern.tof`
  (`TOFInstrument`、`calculate_pattern_tof`)、GSAS リーダーの TOF(SLOG/FXYE、μs)対応、
  `TOFInstrument.from_instprm`(GSAS-II の .instprm)、Project / 保存 / GUI(TOF 装置ダイアログ、軸ラベル、d 値表示)
- データ: GSAS-II チュートリアル "TOF-CW Joint Refinement" の POWGEN NAC(`data/public/nac/PG3_22048.gsa`、
  `POWGEN_1066.instprm`)。参照値は `uv run python validation/gsas2/run_gsas2.py nac_tof`(0 サイクルの ycalc)。
- テスト: `tests/unit/test_tof_profile.py`、`tests/unit/test_tof.py`、`tests/validation/test_gsas2_tof.py`

## 1. ピーク形状(閉形式の正しさ)

G ⊗ h は erfcx、L ⊗ h は複素指数積分 E₁(scipy.special.exp1)で閉形式にした。

| 確認 | 結果 |
|---|---|
| 細かいグリッドでの数値畳み込み(TCH 擬 Voigt ⊗ 両側指数)との比較、4 組のパラメータ | 最大 0.5%(数値側の離散化誤差程度) |
| ローレンツ項だけを scipy.integrate.quad で直接積分(α = 0.045, β = 0.11, γ = 87 μs) | 7 桁一致 |
| 面積 | 1(有限範囲のローレンツの裾 γ/(πR) を除く) |
| α, β → 大 | 擬 Voigt に戻る(2% 以内) |

GSAS-II の Fortran `pyepsvoigt` と同じ引数で直接比べると、ピーク中心では一致するが裾で差がある:
α = 0.5, β = 0.05 で最大 0.5%、α = 0.2, β = 0.02 で 1%、α = 0.045(POWGEN の d = 3 Å)で裾が最大 10%
(GSAS-II が小さい側)。GSAS-II の EXPINT は単精度の級数・漸近展開で複素指数積分を近似しているためと考えられる
(こちらの値は quad と一致する)。

## 2. パターン(POWGEN NAC、GSAS-II との比較)

同じ装置パラメータ(.instprm)、同じ構造(GSAS-II が読んだ U を注入)、同じ背景(Chebyshev 3 項)、
試料の広がりは GSAS-II の既定(サイズ 1 μm、歪み 1000)。

| 項目 | 結果 |
|---|---|
| ピーク位置(反射リストの pos) | 0.05 μs 以内で一致 |
| 孤立した強い反射の積分強度(窓内の和の比) | 0.993〜1.008(7 本) |
| 全体のスケール比(最小二乗) | 0.9995 |
| 点ごとの差 | 中央値 9%、最大|Δ|/max 5%(上記の裾の差。積分強度は合う) |

TOF の試料広がりは GSAS-II の定義に合わせた: サイズ ΔT = DIFC d²/L(L: μm × 10⁴)、歪み ΔT = DIFC d ε × 10⁻⁶。
歪みの定義は CW(Δd/d = ε × 10⁻⁶ / 2)と 2 倍違う(GSAS-II 自身の不整合をそのまま踏襲)。
Lorentz 因子は d⁴ sinθ、入射スペクトルはデータ側で規格化済みとする(Mantid の FXYE と同じ)。

## 3. 吸収と消衰(2026-10-12 追加)

- TOF の吸収: 円筒試料の μR を λ = 1 Å で与え(`h{i}:sample:mu_r`)、反射ごとに λ = 2 d sinθ に比例させて
  Lobanov の近似式で透過率を求める(GSAS-II の `GetAbsorb` と同じ)。ベクトル化した
  `cylinder_absorption_array` はスカラー版と一致(μR = 0〜8、2θ = 20〜150°)。
- Sabine の粉末消衰(`profile.corrections.sabine_extinction`、`h{i}:相:ext`、CW と TOF の両方):
  GSAS-II の `GSASIIstrMath.GetPwdrExt` を直接呼んだ参照値(X 線・中性子 6 ケースずつ、
  `validation/reference/sabine_extinction.json`)と相対差 2e-16 で一致。E の単位は GSAS-II の
  Extinction と同じ(中性子の |F|² は fm² → (10⁻¹² cm)² に換算)。
  解析ヤコビアンは消衰込みの |F|² E(|F|²) で差分を取り、数値微分と 0.2% 以内で一致
  (`tests/unit/test_extinction.py`)。

## 4. GSAS 形式の TOF データの読み方(2026-10-12 修正)

GSAS 形式(FXYE/FXY、SLOG 等)の TOF データは x がビンの左端、強度はビンあたりの計数。
ビン幅が変わるデータをプロファイル(μs あたりで規格化)と合わせるには、GSAS-II と同じく
ビン中心に移し(最後の点は落ちる)、強度と esd をビン幅で割って counts/μs にする必要がある
(`io/readers/gsas.py`、metadata `yunit = "counts/us"`)。修正前は POWGEN NAC の精密化で
Rwp が 12.3% 止まり(GSAS-II 9.4%)だったのがこの読み方の違いで、修正後は下の表の通り。

## 5. 計算速度(2026-10-12 追加)

ピーク形状の計算を numba の一括カーネル(`profile/tof_batch.py`、ピークごとに並列)に置き換えた。
複素指数積分 e^w E₁(w) は級数・連分数(修正 Lentz 法)・漸近展開で自前評価し、
scipy.special.exp1 と相対差 3e-12 以内(|w| = 10⁻³〜10³、全方位)。

| ケース | 置き換え前 | 置き換え後 |
|---|---|---|
| POWGEN NAC(6104 点、1698 反射、Lorentz 成分あり)の 1 回のパターン計算 | 2.26 s | 0.26 s(16 スレッド) |

## 6. 精密化の比較(POWGEN NAC、2026-10-12)

GSAS-II 側は `uv run python validation/gsas2/run_gsas2.py nac_tof_fit`
(背景 3 項 + スケール → 格子 + ゼロ点 → sig-1 → beta-0 → 座標 → U_ij、各 3 サイクル)。
試料広がりはガウスのみ(LGmix = 0、X = Y = Z = 0)。ローレンツ成分があると GSAS-II の複素指数積分の
近似のため同じ極小にならず(a が 0.003 Å、β が 3 倍違う)、GSAS-II は原子の段階で
Levenberg-Marquardt が「stuck」になって原子を動かせなかった。alpha も装置校正値に固定
(GSAS-II で動かすと 2e8 に発散)。テスト: `tests/validation/test_gsas2_tof_fit.py`(約 2 分)。

| 項目 | kaisetsu | GSAS-II 5.7.4 |
|---|---|---|
| Rwp / GOF | 8.400% / 5.389 | 8.471% / 5.434 |
| a [Å] | 10.248458(13) | 10.248496 |
| Zero [μs] | −10.31(20) | −10.32 |
| スケール(GSAS-II 単位 /100) | 32.68(19) | 32.50 |
| sig-1 | −226.3(28) | −225.6 |
| beta-0 | 0.0763(13) | 0.0770 |
| 背景 3 項 | 27.45, −1.45, 11.49 | 27.71, −1.09, 11.80 |
| 座標(自由成分 12 個) | すべて 3σ 以内 | |
| U_ij(自由成分 24 個) | すべて 3σ 以内(Ca1 の U33 は両者とも負) | |

## 7. 残課題

- 複数バンクの同時精密化は、GUI の「すべてのバンク」でバンクごとにヒストグラムを追加して行う
  (バンク間でセル定数などは相パラメータとして自動的に共有される)。
- ローレンツ成分を含む精密化の GSAS-II との比較は、GSAS-II 側の複素指数積分の近似のため同じ極小に
  ならない(6 節)。
