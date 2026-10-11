# Phase 2 検証記録: 粉末パターン計算(GSAS-II 突き合わせ)

実施日: 2026-10-11
kaisetsu: 0.1.0(Phase 2 完了時点)
比較対象: GSAS-II 5.7.4(tag 5833)
参照値: `validation/reference/*.json`(`validation/gsas2/run_gsas2.py` で生成)
テスト: `tests/validation/test_gsas2_phase2.py`

## 比較の条件

GSAS-II に 0 サイクルの精密化をさせて得た計算パターン ycalc と、同じ装置・試料・相パラメータで
kaisetsu が計算したパターンを点ごとに比べた。散乱因子(f', f''、中性子 b)は GSAS-II の値を
注入し、テーブル由来の差(Phase 1 で記録済み)を除いてプロファイル・強度補正・位置だけを比べた。
Kα2 の扱いは `kalpha2_mode="gsas"`(GSAS-II と同じ近似)にした。

| ケース | 放射線 | 効かせた要素 |
|---|---|---|
| pbso4_xray | Cu Kα1/Kα2 | TCH pseudo-Voigt、偏光 0.7、既定のサイズ 1 μm・歪み 1000 |
| pbso4_neutron | 中性子 1.909 Å | 大きなガウス幅(U=354, V=-760, W=652)、ゼロ点 |
| diamond_xray_sim | Cu Kα1/Kα2 | シミュレーション、立方晶 |
| pbso4_xray_asym | Cu Kα1/Kα2 | FCJ 非対称 SH/L=0.05、ローレンツ幅 X=1, Y=3、ゼロ点 0.02、偏光 0.5、試料変位 100 μm、March-Dollase r=0.8 (001)、サイズ 0.3 μm、歪み 2000 |
| pbso4_neutron_abs | 中性子 | ローレンツ幅、SH/L=0.03、円筒試料吸収 μR=0.5、Debye-Scherrer 変位 (100, -50) μm |

## 結果

### パターン(Bragg 成分 y_calc - y_bkg)

| ケース | 相対差 中央値 | 相対差 最大(ピーク領域) | 最大\|Δ\|/max | 面積比 |
|---|---|---|---|---|
| pbso4_xray | 6.3e-5 | 5.1e-4 | 1.7e-4 | 0.999987 |
| pbso4_neutron | 1.5e-5 | 2.1e-4 | 2.2e-5 | 0.999996 |
| diamond_xray_sim | 8.1e-5 | 3.8e-4 | 1.7e-4 | 0.999986 |
| pbso4_xray_asym | 2.0e-5 | 9.7e-5 | 3.7e-5 | 0.999982 |
| pbso4_neutron_abs | 1.2e-5 | 7.8e-5 | 1.7e-5 | 0.999993 |

(ピーク領域 = Bragg 成分が最大の 1% 以上の点。)

### 反射ごとの量

| 量 | 一致の程度 |
|---|---|
| ピーク位置(ゼロ点・試料変位込み) | 1e-13 deg(倍精度の丸め) |
| ガウス幅 Γ_G | 相対 1e-15 |
| ローレンツ幅 Γ_L | 相対 1e-15(GSAS-II が下限 0.001 centideg を当てる反射を除く) |
| 積分強度 m\|F\|^2·LP·選択配向·吸収/V | 相対 1e-6 以内 |

## 突き合わせの過程で分かったこと(GSAS-II の慣例)

1. **既定のサイズ・歪み広がり。** GSAS-II は相を追加した時点で結晶子サイズ 1 μm、微小歪み 1000
   (いずれもローレンツ型)を既定値として持ち、これがピーク幅に効く(実験室 X 線では無視できない)。
   kaisetsu では `PhaseContribution.size_um`, `strain_e6` として明示的に持つ(既定は広がりなし)。
2. **ピーク計算範囲。** 低角側 50×(Γ_G + Γ_L + SH/L·|cos 2θ|)、高角側 75×(Γ_G + Γ_L)。
   幅に TCH の全 FWHM を使うと GSAS-II より狭くなり、ローレンツ裾の打ち切りで面積が 0.1% ずれた。
3. **Kα2 成分。** GSAS-II は位置を 2θ1 + (360/π)(Δλ/λ) tanθ の近似で置き、強度補正(LP 等)と
   幅は Kα1 の位置で求めた値を使い回す。LP を Kα2 の位置で評価すると低角で 0.5% 違う。
   kaisetsu の既定(`kalpha2_mode="exact"`)は Kα2 の位置で Bragg 式・LP・幅を評価する
   (物理的にはこちらが正しい)。GSAS-II と比べるときだけ `"gsas"` を使う。
4. **FCJ 非対称。** GSAS-II の PSVFCJO は S/L = H/L = (SH/L)/2 とし、[2φ_min, 2θ] を
   Gauss-Legendre 則の正の半分の節点で積分する。節点数は 3·(2θ - 2φ_min)/Γ_L を超える表の値。
   kaisetsu は同じ手順(倍精度)で実装した。
5. **スケール因子の定義。** GSAS-II の強度は 1/V を陽に含み、プロファイルは centideg あたりで
   規格化されている。kaisetsu は 1/V を含み、プロファイルは deg あたりで規格化するので
   S_here = S_hist·S_phase/100。中性子では F の単位(fm vs 10^-12 cm)でさらに 1/100。
6. **Lorentz 因子** は 1/(2 sin^2θ cosθ)、偏光因子は (1 - P) cos^2 2θ + P(中性子では 1)。
7. **試料変位の単位。** GSAS-II の Shift / DisplaceX / DisplaceY は μm。
   Bragg-Brentano の Δ2θ = -(2 s/R) cosθ、Debye-Scherrer の Δ2θ = -(Δx cos 2θ + Δy sin 2θ)/R。

## 判断

Phase 2 の受け入れ基準(ピーク領域で相対差 1e-3 以内)を全ケースで満たした
(最大でも 5e-4、中央値 1e-4 以下)。残差の主因は GSAS-II の Fortran 部分が単精度(REAL*4)
であることと考えられる。

## 再現方法

```powershell
uv run python validation/gsas2/run_gsas2.py                 # 参照値の再生成
uv run pytest tests/validation/test_gsas2_phase2.py -s      # 差の要約を表示
```
