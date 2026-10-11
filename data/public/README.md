# 公開テストデータ

このディレクトリのデータは検証用。出典とライセンスを以下に記す。

## pbso4/

IUCr Commission on Powder Diffraction の Rietveld ラウンドロビン(R. J. Hill, J. Appl. Cryst. 25, 589 (1992))で配布された PbSO4 のデータ。GSAS-II 5.7.4 の `tests/testinp/` から複製した(GSAS-II: UChicago Argonne, LLC のオープンソースライセンス)。

| ファイル | 内容 |
|---|---|
| `PBSO4.XRA` | 実験室 X 線(Cu Kα1/Kα2、Bragg-Brentano)、10-159° 2θ、0.025° 刻み、GSAS 形式(CONST, STD) |
| `PBSO4.CWN` | ILL D1A 中性子、λ = 1.909 Å、10-155.9° 2θ、0.05° 刻み、GSAS 形式(CONST, ESD) |
| `INST_XRY.PRM` | X 線用装置パラメータ(λ1 = 1.5405, λ2 = 1.5443 Å、Kα2/Kα1 = 0.5、偏光 0.7) |
| `inst_d1a.prm` | D1A 用装置パラメータ(λ = 1.909 Å、ゼロ点 -0.1) |
| `PbSO4-Wyckoff.cif` | PbSO4 の初期構造(Pnma, a = 8.48, b = 5.398, c = 6.958 Å) |

## cif/

| ファイル | 内容 | 出典 |
|---|---|---|
| `diamond.cif` | ダイヤモンド(Fd-3m, a = 3.56679 Å) | Wyckoff, Crystal Structures (1963) 経由 AMCSD 0011242。GSAS-II `tests/testinp/` から複製 |
