# 公開テストデータ

このディレクトリのデータは検証と練習用。出典とライセンスを以下に記す。
GUI の「はじめての精密化」ウィザードでそのまま選べる。
スクリプトでの精密化例は `examples/refine_sample_data.py`(`uv run python examples/refine_sample_data.py fap` など)。

| フォルダ | 試料 | 放射線・装置 | 難しさ | 用途 |
|---|---|---|---|---|
| `fap/` | 蛍石燐灰石 Ca5(PO4)3F | 実験室 Cu Kα、Bragg-Brentano | ★ | いちばん最初に試すもの |
| `lab6/` | LaB6(NIST SRM 660 系) | 実験室 Cu Kα、Bruker D8(.raw) | ★★ | Bruker 形式の読み込み、表面粗さ補正の練習 |
| `pbso4/` | PbSO4 | 実験室 Cu Kα と ILL D1A 中性子 | ★★ | GSAS-II との突き合わせ(X 線+中性子の同時精密化) |
| `garnet/` | Y3(Fe,Al)5O12 ガーネット | ILL D1A 中性子 1.909 Å | ★★ | 中性子、混合サイトの占有率 |
| `nac/` | Na2Ca3Al2F14 + CaF2 | APS 11-BM 放射光 0.4139 Å | ★★★ | 放射光、2 相、異方性変位パラメータ |
| `pdf/` | Ni | X 線・中性子の G(r) | — | Phase 5(PDF)用。まだ使わない |
| `kaolinite/` | カオリナイト | Bruker .raw(2 レンジ) | — | 複数スキャンの .raw 読み込みテスト用(積層不整のため Rietveld には不向き) |

## 出典とライセンス

- **GSAS-II チュートリアル**(<https://github.com/AdvancedPhotonSource/GSAS-II-tutorials>、UChicago Argonne, LLC のオープンソースライセンス): `fap/FAP.XRA`, `fap/INST_XRY.PRM`, `lab6/LaB6_Jan2018.raw`, `garnet/garnet.raw`, `garnet/inst_d1a.prm`, `nac/*`, `pdf/*`, `kaolinite/*`。
- **GSAS-II 5.7.4 同梱テストデータ**: `pbso4/*`, `cif/diamond.cif`。
- **Crystallography Open Database (COD)**(<https://www.crystallography.net/>、パブリックドメイン相当): `fap/fluorapatite_COD9001232.cif`, `lab6/LaB6_COD1000055.cif`。
- 自作: `garnet/Y3Fe2Al3O12_start.cif`(チュートリアルの出発値から作成)。

## fap/ — 蛍石燐灰石(実験室 X 線)

GSAS-II チュートリアル "Fitting laboratory X-ray powder data for fluoroapatite" のデータ。Cu Kα、Bragg-Brentano、15–130° 2θ、0.02° 刻み、GSAS 形式(STD)。

| ファイル | 内容 |
|---|---|
| `FAP.XRA` | 回折データ(GSAS 形式) |
| `INST_XRY.PRM` | 装置パラメータ(λ1 = 1.5405, λ2 = 1.5443 Å、偏光 0.7) |
| `fluorapatite_COD9001232.cif` | 出発構造。COD 9001232(Hughes, Cameron & Crowley, Am. Mineral. 74, 870 (1989))の U_ij を U_eq に置き換えたもの |

参考結果(標準の手順、背景 10 項、試料高さずれ): Rwp = 8.5%、χ² = 2.3、a = 9.3733, c = 6.8870 Å、R_Bragg = 4.2%。
GSAS-II チュートリアルの最終値は wR = 10.4%(背景関数が違う)、a = 9.3719, c = 6.8859 Å。格子定数の差 1.5×10⁻⁴ は波長(1.5405 と 1.5405929)と 2θ ずれモデルの違いによる。

## lab6/ — LaB6(Bruker .raw)

GSAS-II チュートリアル "Determine Starting Profile from a Standard" のデータ。Bruker D8、Cu Kα(Ni フィルタ)、10–70° 2θ、ゴニオメータ半径 250 mm。ファイルは Bruker RAW1.01 形式で、kaisetsu がそのまま読む(ターゲット・波長はファイルから入る)。

| ファイル | 内容 |
|---|---|
| `LaB6_Jan2018.raw` | 回折データ(Bruker RAW1.01) |
| `LaB6_COD1000055.cif` | 出発構造。COD 1000055(P m -3 m, a = 4.1576 Å) |

参考結果: 標準の手順だけでは Rwp = 7.8%、χ² = 21 で U_iso が負になる。これは吸収の強い試料を平板に詰めたときの表面粗さ効果で、「表面粗さを動かす」を付けると Rwp = 3.9%、χ² = 5.3、U_iso が正になる(La 0.016, B 0.021 Å²)。負の U が出たときの典型例として残してある。

## pbso4/ — PbSO4(X 線+中性子)

IUCr Commission on Powder Diffraction の Rietveld ラウンドロビン(R. J. Hill, J. Appl. Cryst. 25, 589 (1992))で配布された PbSO4 のデータ。GSAS-II 5.7.4 の `tests/testinp/` から複製した。

| ファイル | 内容 |
|---|---|
| `PBSO4.XRA` | 実験室 X 線(Cu Kα1/Kα2、Bragg-Brentano)、10-159° 2θ、0.025° 刻み、GSAS 形式(CONST, STD) |
| `PBSO4.CWN` | ILL D1A 中性子、λ = 1.909 Å、10-155.9° 2θ、0.05° 刻み、GSAS 形式(CONST, ESD) |
| `INST_XRY.PRM` | X 線用装置パラメータ(λ1 = 1.5405, λ2 = 1.5443 Å、Kα2/Kα1 = 0.5、偏光 0.7) |
| `inst_d1a.prm` | D1A 用装置パラメータ(λ = 1.909 Å、ゼロ点 -0.1) |
| `PbSO4-Wyckoff.cif` | PbSO4 の初期構造(Pnma, a = 8.48, b = 5.398, c = 6.958 Å) |

GSAS-II との突き合わせ結果は `docs/validation/` を参照。

## garnet/ — Y3(Fe,Al)5O12(中性子)

GSAS-II チュートリアル "CW Neutron Powder Data" のデータ。ILL D1A、λ = 1.909 Å、24–157.9° 2θ、0.05° 刻み。16a と 24d が Fe/Al の混合サイトで、占有率精密化の練習になる。

| ファイル | 内容 |
|---|---|
| `garnet.raw` | 回折データ(GSAS 形式。拡張子は .raw だが Bruker ではない。中身で判定する) |
| `inst_d1a.prm` | D1A の装置パラメータ(U, V, W = 354, -760, 652 centideg²) |
| `Y3Fe2Al3O12_start.cif` | 出発構造(Ia-3d, a = 12.18 Å、混合サイトは 0.5/0.5 から) |

装置設定では「中性子」、波長 1.909 Å、Debye-Scherrer を選ぶ。混合サイトの占有率は和が 1 に保たれる(Al1 は Fe1 に、Fe2 は Al2 に従属)。
参考結果(`examples/refine_sample_data.py garnet`: 標準の手順+占有率、試料変位 x, y、約 2 分):
Rwp = 4.2%、χ² = 2.5、a = 12.1867 Å、16a: Fe 0.576(7) / Al 0.424、24d: Al 0.705(6) / Fe 0.295。
ゼロ点だけを動かした場合は Rwp = 6.4%、a = 12.1775 Å。格子定数が 2θ ずれのモデルで 10⁻³ 動くのは、
データが 24° から始まり低角の反射が無いため。
GSAS-II チュートリアルの最終値は wR = 5.2%、a = 12.1817 Å(ゼロ点+試料変位 2 項)、16a: Fe 0.577、24d: Al 0.701。

## nac/ — Na2Ca3Al2F14 + CaF2(放射光)

GSAS-II チュートリアル "TOF-CW Joint Refinement" の 11-BM データ。λ = 0.413909 Å、0.5–60° 2θ、0.001° 刻み、59,497 点、FXYE 形式。主相 NAC(I2₁3)と少量の CaF2。

| ファイル | 内容 |
|---|---|
| `11BM_NAC.fxye` | 回折データ(GSAS FXYE 形式、2θ は centidegree) |
| `11bm_gsas.prm` | 11-BM の装置パラメータ(U, V, W, X = 1.163, -0.126, 0.063, 0.173、偏光 0.99) |
| `NAC.cif`, `CaF2.cif` | 出発構造(異方性 U_ij 付き) |

点数が多く反射も多いので、まず 2θ の上限を 30° にして試す(スクリプト例は `x_max=30`)。
参考結果(0.5–30°、標準の手順、2 相): Rwp = 8.2%、χ² = 8.3、NAC R_Bragg = 7.7%、CaF2 R_Bragg = 28.5%(少量相)。55 変数で約 50 秒。

## pdf/ — Ni の PDF データ(Phase 5 用)

GSAS-II チュートリアル "PDFfit-I" の Ni の G(r)(X 線は APS 11-ID-B、中性子は NOMAD)と、fullrmc チュートリアルの Ni.cif。実空間フィット(PDFfit2 相当)を実装するときの検証に使う。

## cif/

| ファイル | 内容 | 出典 |
|---|---|---|
| `diamond.cif` | ダイヤモンド(Fd-3m, a = 3.56679 Å) | Wyckoff, Crystal Structures (1963) 経由 AMCSD 0011242。GSAS-II `tests/testinp/` から複製 |
