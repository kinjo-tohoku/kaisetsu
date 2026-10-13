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

## 7. Z-Rietveld の配布サンプル(J-PARC)での標準手順(2026-10-12 深夜)

z-code-software.com のサンプル(iMATERIA BL20 の BS 155° / SE 90° バンク、SuperHRPD BL08)を、
同梱の .zDiffractometer を「装置ファイル」として読み、同梱の CIF から `standard_recipe("*", "h0")`
だけで精密化した(背景 Chebyshev 8 項、フィット範囲は装置ファイルの [Exclude range] の通り)。

装置ファイルからの取り込み: 換算係数 c₀, c₁, c₂ → zero, DIFC, DIFA、バンク角、Type0m / Type0 の
σ² = SigmaSquare0 + SigmaSquare1 d² + SigmaSquare2 d⁴ [μs²] と γ = Gamma0 + Gamma1 d + Gamma2 d² [μs]
(GSAS type 3 と同じ形。クイックガイドの iMATERIA BS「s1² = 45」と配布ファイルの 33.7 が整合)。
同じ名前のプロファイルが [Profile function](現在値)と [Profile function initial value] の両方に
あるので前者を優先する(iMATERIA の配布ファイルでは α, β が両者でまったく違う)。
立ち上がり・減衰(Alpha1, Alpha2, Beta0_0, Beta0_1, Beta1_0, r01〜r03)は Ikeda-Carpenter 系で
本プログラムの back-to-back 指数関数と対応しないので取り込まない。

標準手順の変更: TOF では「ピーク幅 W」で sig1 と α, β0, β1 を同時に動かし、次の段階で sig0, sig2, X, Y。
α を固定(既定 0.1/d)したまま sig1 だけ動かすと、幅の辻褄を σ² < 0 で合わせようとして
σ² = −4000 μs² まで流れ(CeO2 で Rwp 24.8%)、α を動かすと 12.5% になる。CeO2 BS で手順を 6 通り
比べ(α を後の段階で動かす 12.5〜13.6%、sig1 と同時 12.6%、W 段階を飛ばす 12.5%)、物理的に
素直な「主要な形状 3 つを先に」を採った。σ² が負になったときのガード(`Recipe._guard_widths`)は
TOF にも効かせ、動いている sig0, sig2(無ければ sig1)を段階前の値に固定してやり直す。
α は iMATERIA のように立ち上がりが分解能より速い装置では上限 10 に張り付いて esd が大きくなるが、
フィットには害がない(POWGEN NAC では 0.13 付近に決まる)。

| データ(バンク) | 相 | Rwp(変更前) | Rwp(変更後) |
|---|---|---|---|
| CeO2 MAT005500 BS 155° | CeO2 | 24.8% | 12.6% |
| CeO2 MAT005500 SE 90° | CeO2 | — | 8.2% |
| Al2O3 MAT005506 BS 155° | Al2O3 | — | 9.6% |
| Al2O3 MAT005506 SE 90° | Al2O3 | — | 5.9% |
| TiO2 MAT002294 BS 155° | アナターゼ + ルチル(2 相) | — | 9.5% |
| LiCo0.2Ni0.8O2 MAT002357 BS | 1 相 | — | 11.0% |
| LiCo0.8Ni0.2O2 MAT002358 BS | 1 相 | — | 15.6% |
| MnF2 120 K SuperHRPD(σ² は d⁴ 項だけ、γ が主) | MnF2 | — | 7.5% |
| POWGEN NAC(6 節と同じデータ、ローレンツ成分も動かす) | NAC | — | 7.7%(GSAS-II の段階手順 8.5%) |

ErF3(#14-4、LA / QA バンク)は 22% で止まった。磁気構造の例題(磁気散乱は未対応)で、核散乱だけでは
合わない。

注意(格子定数): Z-Rietveld の c₁(DIFC)は Ikeda-Carpenter 型の関数でピーク位置を定義して校正されて
いるため、本プログラムの back-to-back 指数関数で使うと格子定数が小さめに出る
(CeO2 BS: 5.4093 Å、Z-Rietveld の解 5.41197 Å、相対 −5×10⁻⁴。Al2O3 も 4.7572 / 12.9867 Å で
文献値 4.7589 / 12.991 Å より小さい)。減衰の裾 1/β が d とともに伸びる分がピーク重心をずらし、
ゼロ点では吸収しきれない。格子定数を精密に求めるには標準試料で DIFC とゼロ点を本プログラムで
校正し直すか、Type0m そのもの(定義は Z-Rietveld のマニュアルにしかない)を実装する必要がある。

## 8. Z-Rietveld の Type0m プロファイルの実装(2026-10-13)

Z-Rietveld の TOF プロファイル Type0m の定義は公開文書に無い(配布 zip にマニュアルは無く、
クイックガイドにも式は無い)。そこで Z-Rietveld 2.9.1(Windows、解析コア 2026072902)で
サンプル #2_CeO2_iMATERIA の BS 155° バンク、次いで SE 90° バンクの装置ファイルを読み、
結晶モデルを入れずに 0 サイクル実行し、保存されたプロジェクト `.zrietveldn` から計算曲線を
取り出して関数形を逆算した。

- `.zrietveldn` は zstd 圧縮フレーム(マジック 28 B5 2F FD)が並んだコンテナで、各フレームは CBOR。
  ヒストグラム(x, y, σ)、装置、結晶モデル(結晶を入れないと既定相 Si、a = 5.43 Å、Fd-3m、B = 0.34)、
  計算曲線 `yCals`、背景 `yBackgrounds`、反射ごとの局所値 `locals`(d、位置 p、強度 i、
  σ² = g、γ = g2、α' = g3、β₁' = g4、β₀' = g5、R = g6、窓 pi/pa)、ログが入る。
  Python 3.14 の `compression.zstd` と `cbor2`(MIT)で読める。SE の保存(結晶モデル無し)には
  `yBackgrounds` と窓 pi/pa が無く、背景は装置の Legendre 係数(10 項)から求めた。その定義域は
  フィット範囲(除外範囲の端 4000〜70000 μs)で、計算曲線と 1e-13 で一致する(データ全体や
  peakRange では 1e-3 ずれる)。
- ログの「Global parameters」が構成を教える: `(σ_Gauss)²`、`H_Lorentz`、`α'`(2 個)、`Constant`
  (Beta1_0)、`β'₀`(2 個)、`Switch Function R`(3 個)。
- 169 反射の局所値と装置ファイルの値を突き合わせ(相対差 1e-8):
  α' = 1 / (Alpha1 + Alpha2 d)、β₁' = Beta1_0、β₀' = Beta0_0 + Beta0_1 / d、
  R = r01 exp(−r02 / d^r03)、σ² = S0 + S1 d² + S2 d⁴ [μs²]、γ = G0 + G1 d + G2 d² [μs、FWHM]、
  位置 p = c₀ + c₁ d + c₂ d²。
- 関数形は候補を数値畳み込みで当てて決めた。Ikeda-Carpenter(α, β, R を率として)は合わない
  (残差 7〜9%)。合ったのは「TCH 擬 Voigt ⊗ [(1 − R) B2B(α, β₁) + R B2B(α, β₀)]」
  (B2B は接合点を位置とする両側指数、Si 111 で残差 2e-7、真の Voigt だと 2e-4 なので擬 Voigt は TCH)。
  α', β₀', β₁' は率ではなく時定数の係数で、実時間の時定数 = 係数 × τ。
- **τ = 擬 Voigt の全 FWHM Γ(TCH の式)**。BS だけでは τ(d) = (TOF/1000) × ρ(d)、
  ρ = 1.0544 − 0.0306 ln d という経験式(18 反射で 0.08%)しか得られず、d と t のどちらに
  依存するか決められなかった。SE 90° バンク(DIFC 10050、Alpha1 = 124.5、Alpha2 = 379、
  Beta1_0 = 0.0027、Beta0_0 = 0.69 と BS の 1/100 の係数)の計算曲線で切り分けたところ、
  遅い減衰の時定数(Si 111: 97.92 μs、220: 55.12、422: 30.16)は (TOF/1000) ρ(d) × β₀' の
  4.5〜3.9 倍で d にも t にも乗らず、β₀' × Γ_TCH(143.37 × 0.683 = 97.92 μs)に 0.03% で一致した。
  BS の 18 反射の経験値 c·d も Γ_TCH と 0.3% 以内(経験式 ρ(d) は BS の Γ/(TOF/1000) の近似
  だった)。つまり α', β' は「ピーク幅に対する時定数の比」で、Type0m は幅を変えても形が相似。
  SE では立ち上がりと短い裾の時定数が 0.14 μs、0.39 μs(δ 関数同然)で、非対称は R = 0.17 の
  遅い裾だけが担う。Z-Rietveld は α' を 0.001 で下から止める(SE の Si 111 で確認)。
  試料由来の広がり(σ², γ に足す分)も Γ に含める(Z-Rietveld の SigmaSquare1e などに相当)。
- 実装: `TOFInstrument(profile_kind="zrietveld", zr_*)`。`calculate_pattern_tof` は既存の閉形式
  (両側指数 ⊗ TCH)を β₁ と β₀ で 2 回呼んで (1 − R) : R で混ぜる。`from_zdiffractometer` は
  Type0m の 8 個が揃っていれば自動で zrietveld にする。パラメータ名は `h0:inst:zr_alpha1` など。
  標準手順では α, β, R は動かさない(Z-Rietveld の流儀と同じ。装置ファイルの値をそのまま使う)。
  時定数の単位は `TOFInstrument.zr_time_scale(d, sigma2, gamma)`(= Γ_TCH)で、校正定数は無い
  (v0.1.1 までの zr_scale0, zr_scale1 は廃止。古いプロジェクトファイルの値は読み飛ばす)。
- テスト: `tests/validation/test_zrietveld_type0m.py`(参照 `validation/reference/zrietveld_type0m.json`
  = BS、`zrietveld_type0m_se.json` = SE。各 10 反射の計算曲線 + 窓内の隣接反射)。d 依存と
  τ = Γ は 1e-6、形状は最大 BS 4.0e-4、SE 4.8e-4(経験式のときは BS 4.3e-3)、重心は 0.3 μs 以内。

実データでの効果(7 節と同じ標準手順。Type0m では α, β, R は装置ファイルの値に固定):

| データ | GSAS type 3(α, β を精密化) | Type0m、経験式 ρ(d) | Type0m、τ = Γ(現行) |
|---|---|---|---|
| CeO2 BS 155°: Rwp / a | 12.6% / 5.40933 Å | 8.63% / 5.41228(1) Å | 8.13% / 5.41223(1) Å(Z-Rietveld の解 5.41197 Å) |
| CeO2 SE 90°: Rwp / a | 8.17% / 5.40874 Å | 8.70% / 5.41475(3) Å | 6.23% / 5.41379(2) Å |
| Al2O3 BS 155°: Rwp / a, c | 9.55% / 4.75723, 12.98667 Å | 5.91% / 4.76011, 12.9945 Å | 5.57% / 4.76002, 12.99427 Å |
| MnF2 SuperHRPD: Rwp / a, c | 7.50% / 4.87402, 3.30655 Å | 7.08% / 4.87349, 3.30616 Å | 7.17% / 4.87350, 3.30617 Å |

BS バンクの格子定数の −5×10⁻⁴ のずれは消えた(位置の定義が同じになったため)。τ = Γ にすると
CeO2 は BS で 8.63 → 8.13%(幅の段階のガードも入らなくなった)、SE で 8.70 → 6.23% に下がり、
SE の格子定数の BS とのずれは +2.5×10⁻³ から +1.5×10⁻³ Å に縮んだ。残りは SE バンクの
校正(ゼロ点を −0.35 → −0.69 μs に動かしている)の問題と思われる。MnF2(SuperHRPD)は
0.1% 悪くなるが、幅の段階でガードが 2 回入る(σ² が負に流れる)のは前と同じで、装置ファイルの
σ² の形(S0 = S1 = 0)が合っていない影響が大きい。

**Z-Rietveld の精密化結果との直接比較**(同じ CeO2 BS データを Z-Rietveld 2.9.1 で精密化した
プロジェクトから取り出した。範囲 4500〜68000 μs、背景 Legendre 10 項、α・β・R 固定、ゼロ点固定、
σ² と γ の 6 項・格子・B・スケールを精密化。Rwp の定義は同じ: Z-Rietveld の計算曲線から
本プログラムの式で求め直すと 8.219% で一致):

| 項目 | kaisetsu(標準手順) | kaisetsu(幅 6 項すべて、ゼロ点固定) | Z-Rietveld 2.9.1 |
|---|---|---|---|
| Rwp / Rp | 8.12% / 8.20% | 8.20% / 8.22% | 8.22% / 8.26% |
| a [Å] | 5.412226(9) | 5.412352(4) | 5.412343(3) |
| B(Ce) [Å²] | 0.280(3) | 0.278(4) | 0.279(2) |
| B(O) [Å²] | 0.449(3) | 0.446(3) | 0.448(2) |
| σ² (S0, S1, S2) | 1.53, 30.4, 2.37 | 0.704, 32.08, 1.94 | 0.704, 32.10, 1.96 |
| γ (G0, G1, G2) | 0.17, 2.10, 0.59 | 1.03, 0.61, 1.14 | 1.00, 0.66, 1.11 |

(τ = Γ にする前は 8.62% / 8.37%、σ² = 0.12, 29.4, 4.18 だった。)同じ条件で動かすと Rwp、
格子定数(1×10⁻⁵ Å)、B(esd 以内)、σ² の 3 項(0.1〜1%)まで Z-Rietveld と一致し、
プロファイル関数が同じになったことが確かめられた。Rwp が 0.02% 低いのは背景の基底
(Legendre と Chebyshev)と吸収補正(Z-Rietveld は円筒試料 R = 0.3 cm と充填密度 1.014 g/cm³
から計算、本プログラムは μR を精密化して ≈ 0)の違い。標準手順(ゼロ点も動かす)は
8.12% まで下がり、ガードも入らない。

## 9. 残課題

- 複数バンクの同時精密化は、GUI の「すべてのバンク」でバンクごとにヒストグラムを追加して行う
  (バンク間でセル定数などは相パラメータとして自動的に共有される)。
- ローレンツ成分を含む精密化の GSAS-II との比較は、GSAS-II 側の複素指数積分の近似のため同じ極小に
  ならない(6 節)。
- Z-Rietveld の Type0m は 8 節の通り BS と SE の 2 バンクで確認した(時定数の単位 = 擬 Voigt の
  FWHM)。Type0(旧形式、`alpha`, `beta 1_0`, `beta 0_0`, `beta 0_1`, `r0`)と Type2 は未対応。
  SE バンクの格子定数が BS より 1.5×10⁻³ Å 大きいのは装置側の校正の問題と思われるが未確認。
