# kaisetsu(仮称)

初心者向け・日本語対応の粉末回折解析プログラム。Rietveld 解析から始めて、PDF(二体分布関数)解析と MEM(最大エントロピー法)密度解析まで同じ構造モデル・同じ精密化エンジンで扱うことを目標にしている。

計画と設計: [docs/ROADMAP.md](docs/ROADMAP.md)
開発規約: [CLAUDE.md](CLAUDE.md)

## 現在の状態

- Phase 0(基盤)、Phase 1(結晶学コア: 格子・空間群・CIF・散乱因子・反射生成・構造因子)、
  Phase 2(パターン計算: TCH pseudo-Voigt、FCJ 非対称、Kα 二重線、LP・選択配向・吸収・試料変位、
  Chebyshev 背景、GSAS/Bruker/Rigaku/PANalytical リーダー)、Phase 3(精密化エンジン: パラメータと対称拘束、
  最小二乗、R 因子、Le Bail、Pawley、レシピ、プロジェクトファイル、スクリプト API)、
  Phase 4(日本語 GUI)、Phase 5(PDF: I(Q)→G(r) 変換と PDFfit2 相当の実空間フィット)、
  Phase 6(MEM 電子密度・核密度、MPF、VESTA 出力)が完了。
- 外部ソフトとの突き合わせ: GSAS-II 5.7.4(Rietveld)
  [Phase 1](docs/validation/phase1_structure_factors.md)、[Phase 2](docs/validation/phase2_pattern.md)、
  [Phase 3](docs/validation/phase3_refinement.md)、diffpy.pdffit2 1.6.0(PDF)
  [Phase 5](docs/validation/phase5_pdf.md)、MEM の合成・実データ検証 [Phase 6](docs/validation/phase6_mem.md)
- 実行例: `uv run python examples/refine_sample_data.py fap|lab6|garnet|nac|ni_pdf`、
  `uv run python examples/mem_pbso4.py --mpf 2`

## 使い方(開発者向け)

```powershell
uv sync                       # 依存のインストール
uv run pytest                 # テスト(GSAS-II 参照値があれば突き合わせも実行)
uv run ruff check . ; uv run ruff format --check . ; uv run mypy
uv run python validation/gsas2/run_gsas2.py   # GSAS-II で参照値を再生成(gsas2main が必要)
```

Windows のコンソールで日本語が化ける場合は `$env:PYTHONUTF8 = "1"` を設定する。

### スクリプト例(Rietveld 精密化)

```python
from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    read_powder_data,
)
from kaisetsu.refine.recipe import standard_recipe

st = Structure.from_cif("data/public/pbso4/PbSO4-Wyckoff.cif")
data = read_powder_data("data/public/pbso4/PBSO4.XRA")  # GSAS 形式 / xy / xye
inst = CWInstrument(
    Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01, polarization=0.7
)

proj = Project("PbSO4")
proj.add_phase(st, name="PbSO4")
proj.add_histogram(data, inst, SampleGeometry(), n_background=6)
standard_recipe("PbSO4", "h0").run(proj)  # スケール/背景 → 格子 → 幅 → 座標 → U_iso
print(proj.summary())  # Rwp, χ², R_Bragg, パラメータと esd

# 個別に動かす場合
proj.vary("PbSO4:O3:*", "h0:inst:s_l")
proj.refine()
```

パラメータ名は `相:cell:a`、`相:サイト:x|y|z|occ|uiso`、`h0:inst:zero|u|v|w|x|y|s_l|h_l`、
`h0:bkg:0..`、`h0:相:scale|md_r|size|strain` の形で、glob(`*`)で選べる。
特殊位置の座標拘束(例 y = x、y = 1/4 固定)と晶系による格子定数拘束は自動で付く。

### スクリプト例(PDF フィットと MEM)

```python
from kaisetsu.api import Project, Structure, mem_from_project, pdf_recipe, read_pdf_data
from kaisetsu.mem import write_ccp4

# PDF: Ni の G(r) を PDFfit2 と同じ定式化でフィット
proj = Project("Ni PDF")
proj.add_phase(Structure.from_cif("data/public/pdf/Ni.cif"), name="Ni")
proj.add_pdf(read_pdf_data("data/public/pdf/Ni-xray.gr"), "xray", qdamp=0.06, r_min=1.5, r_max=20)
pdf_recipe("Ni", "g0").run(proj)      # スケール → 格子 → U と δ2 → Q_damp → 座標 → 最終
print(proj.summary())                 # Rw ≈ 7.1%、a = 3.5316 Å

# MEM: Rietveld 済みのプロジェクト(上の PbSO4)から電子密度
mem = mem_from_project(proj_rietveld, hist=0, phase="PbSO4", grid=(64, 40, 54))
print(mem.chi2, mem.maxima(5))        # C ≈ 1、極大は Pb の位置
write_ccp4(mem.density, mem.lattice, mem.spacegroup, "pbso4_mem.ccp4")  # VESTA で開く
```

## ライセンス

未定(研究室内利用の段階)。依存ライブラリはすべて商用利用可能なライセンス(BSD / MIT / MPL / LGPL / PSF)のものに限定している。散乱因子テーブルは cctbx(BSD-3)から出典を明記して取り込んだ(`tools/gen_scattering_tables.py`)。

## GUI

```powershell
uv sync --extra gui        # PySide6 / pyqtgraph / matplotlib を入れる(開発環境では uv sync で入る)
uv run kaisetsu-gui        # または uv run python -m kaisetsu.gui
```

「データ → はじめての精密化(ウィザード)」で、回折データ・CIF・装置プリセットを選ぶだけで
標準の手順(スケールと背景 → 格子定数 → ピーク幅 → 原子座標 → 変位パラメータ)を実行できる。
パラメータ表では精密化する量をチェックで選び、F9 で精密化、F5 で再計算、Esc で中止。
PDF データの読み込みと標準手順、I(Q) → G(r) の変換、MEM(断面表示、VESTA への書き出し、MPF)、
相の設定(Le Bail / Pawley、選択配向軸、異方性 U)、構造の 3D 表示、χ² の推移、図の PNG/SVG 保存も GUI から使える。
使い方の詳細は [docs/user-guide/getting-started.md](docs/user-guide/getting-started.md)。
