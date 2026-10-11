# kaisetsu(仮称)

初心者向け・日本語対応の粉末回折解析プログラム。Rietveld 解析から始めて、PDF(二体分布関数)解析と MEM(最大エントロピー法)密度解析まで同じ構造モデル・同じ精密化エンジンで扱うことを目標にしている。

計画と設計: [docs/ROADMAP.md](docs/ROADMAP.md)
開発規約: [CLAUDE.md](CLAUDE.md)

## 現在の状態

- Phase 0(基盤)、Phase 1(結晶学コア: 格子・空間群・CIF・散乱因子・反射生成・構造因子)、
  Phase 2(パターン計算: TCH pseudo-Voigt、FCJ 非対称、Kα 二重線、LP・選択配向・吸収・試料変位、
  Chebyshev 背景、GSAS 形式リーダー)、Phase 3(精密化エンジン: パラメータと対称拘束、
  最小二乗、R 因子、Le Bail、レシピ、プロジェクトファイル、スクリプト API)が完了。
- GSAS-II 5.7.4 との突き合わせ結果:
  [Phase 1](docs/validation/phase1_structure_factors.md)、[Phase 2](docs/validation/phase2_pattern.md)、
  [Phase 3](docs/validation/phase3_refinement.md)
- 実行例: `uv run python examples/refine_pbso4.py`(PbSO4 の X 線データを標準レシピで精密化)

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

## ライセンス

未定(研究室内利用の段階)。依存ライブラリはすべて商用利用可能なライセンス(BSD / MIT / MPL / LGPL / PSF)のものに限定している。散乱因子テーブルは cctbx(BSD-3)から出典を明記して取り込んだ(`tools/gen_scattering_tables.py`)。
