# kaisetsu(仮称)

初心者向け・日本語対応の粉末回折解析プログラム。Rietveld 解析から始めて、PDF(二体分布関数)解析と MEM(最大エントロピー法)密度解析まで同じ構造モデル・同じ精密化エンジンで扱うことを目標にしている。

計画と設計: [docs/ROADMAP.md](docs/ROADMAP.md)
開発規約: [CLAUDE.md](CLAUDE.md)

## 現在の状態

- Phase 0(基盤)完了、Phase 1(結晶学コア: 格子・空間群・CIF・散乱因子・反射生成・構造因子)完了。
- GSAS-II 5.7.4 との突き合わせ結果: [docs/validation/phase1_structure_factors.md](docs/validation/phase1_structure_factors.md)

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
from kaisetsu.crystal import Structure
from kaisetsu.reflections import generate_reflections, structure_factors
from kaisetsu.scattering import Radiation

st = Structure.from_cif("data/public/pbso4/PbSO4-Wyckoff.cif")
refl = generate_reflections(st.lattice, st.spacegroup, d_min=1.0)
f = structure_factors(st, refl.hkl, Radiation.characteristic("Cu"))
for hkl, d, m, f2 in zip(refl.hkl, refl.d, refl.multiplicity, abs(f) ** 2):
    print(hkl, f"d={d:.4f}", f"m={m}", f"|F|^2={f2:.1f}")
```

## ライセンス

未定(研究室内利用の段階)。依存ライブラリはすべて商用利用可能なライセンス(BSD / MIT / MPL / LGPL / PSF)のものに限定している。散乱因子テーブルは cctbx(BSD-3)から出典を明記して取り込んだ(`tools/gen_scattering_tables.py`)。
