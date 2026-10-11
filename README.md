# kaisetsu(仮称)

初心者向け・日本語対応の粉末回折解析プログラム。Rietveld 解析から始めて、PDF(二体分布関数)解析と MEM(最大エントロピー法)密度解析まで同じ構造モデル・同じ精密化エンジンで扱うことを目標にしている。

計画と設計: [docs/ROADMAP.md](docs/ROADMAP.md)
開発規約: [CLAUDE.md](CLAUDE.md)

## 現在の状態

- Phase 0(基盤)、Phase 1(結晶学コア: 格子・空間群・CIF・散乱因子・反射生成・構造因子)、
  Phase 2(パターン計算: TCH pseudo-Voigt、FCJ 非対称、Kα 二重線、LP・選択配向・吸収・試料変位、
  Chebyshev 背景、GSAS 形式リーダー)が完了。
- GSAS-II 5.7.4 との突き合わせ結果:
  [Phase 1](docs/validation/phase1_structure_factors.md)、[Phase 2](docs/validation/phase2_pattern.md)

## 使い方(開発者向け)

```powershell
uv sync                       # 依存のインストール
uv run pytest                 # テスト(GSAS-II 参照値があれば突き合わせも実行)
uv run ruff check . ; uv run ruff format --check . ; uv run mypy
uv run python validation/gsas2/run_gsas2.py   # GSAS-II で参照値を再生成(gsas2main が必要)
```

Windows のコンソールで日本語が化ける場合は `$env:PYTHONUTF8 = "1"` を設定する。

### スクリプト例

```python
import numpy as np
from kaisetsu.crystal import Structure
from kaisetsu.io import read_powder_data
from kaisetsu.pattern import CWInstrument, PhaseContribution, SampleGeometry, calculate_pattern
from kaisetsu.profile import ChebyshevBackground
from kaisetsu.scattering import Radiation

st = Structure.from_cif("data/public/pbso4/PbSO4-Wyckoff.cif")
data = read_powder_data("data/public/pbso4/PBSO4.XRA")  # GSAS 形式 / xy / xye
inst = CWInstrument(Radiation.characteristic("Cu"), w=0.003, s_l=0.02, h_l=0.02, polarization=0.7)
bkg = ChebyshevBackground([150.0], data.x[0], data.x[-1])
res = calculate_pattern(data.x, [PhaseContribution(st, scale=2.0)], inst, SampleGeometry(), bkg)
print(res.y_calc[:5], res.peaks[0].two_theta[:3])
```

## ライセンス

未定(研究室内利用の段階)。依存ライブラリはすべて商用利用可能なライセンス(BSD / MIT / MPL / LGPL / PSF)のものに限定している。散乱因子テーブルは cctbx(BSD-3)から出典を明記して取り込んだ(`tools/gen_scattering_tables.py`)。
