"""Le Bail の分配更新: 範囲外の反射は決めない、フィット範囲の点だけ使う、収束まで回す。

PbSO4 CuKα(データ 160° まで)で、x_max + 2° のマージンに入る 2θ = 161° の反射が裾だけで
分配を受けて 1e3 倍に膨らみ、更新のたびに Rwp が 51 → 2700% と発散していた。標準手順
(le_bail_recipe)の最終 Rwp が 3e14 % になり「Le Bail で強度が合わない」と見えた件。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import (
    CWInstrument,
    Project,
    Radiation,
    SampleGeometry,
    Structure,
    read_powder_data,
)
from kaisetsu.refine.rfactors import partition_intensities, reflections_inside


def _project(data_dir: Path, x_min: float = 20.0, x_max: float = 60.0) -> Project:
    st = Structure.from_cif(str(data_dir / "pbso4" / "PbSO4-Wyckoff.cif"))
    data = read_powder_data(data_dir / "pbso4" / "PBSO4.XRA").cropped(x_min, x_max)
    inst = CWInstrument(
        Radiation.xray(1.5405, 1.5443, 0.5), w=0.003, s_l=0.01, h_l=0.01, polarization=0.7
    )
    proj = Project("lebail-outside")
    proj.add_phase(st, name="PbSO4")
    proj.add_histogram(data, inst, SampleGeometry(), n_background=6)
    proj.build_parameters()
    return proj


def _rwp(proj: Project) -> float:
    proj.calculate(0)
    return proj.rfactors()["total"].rwp


def _inside_outside(proj: Project) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    h = proj.histograms[0]
    tbl = proj.calculate(0).peaks[0]
    inside = reflections_inside(h.data.x, tbl.two_theta, h.mask())
    keys = [(int(a), int(b), int(c)) for a, b, c in tbl.hkl]
    return (
        [k for k, ins in zip(keys, inside, strict=True) if ins],
        [k for k, ins in zip(keys, inside, strict=True) if not ins],
    )


def test_reflections_inside_uses_data_range_and_mask() -> None:
    x = np.linspace(10.0, 20.0, 101)
    mask = np.ones(101, dtype=bool)
    mask[40:61] = False  # 14〜16° を除外
    cen = np.array([9.9, 10.0, 15.0, 17.0, 20.0, 20.1, np.nan])
    assert reflections_inside(x, cen, mask).tolist() == [
        False, True, False, True, True, False, False,
    ]  # fmt: skip
    assert reflections_inside(x, cen).tolist() == [False, True, True, True, True, False, False]
    assert reflections_inside(x, np.zeros(0)).tolist() == []


def test_partition_returns_calc_for_outside_and_ignores_masked_points() -> None:
    x = np.arange(0.0, 10.0, 0.1)
    prof = np.exp(-0.5 * ((x - 5.0) / 0.3) ** 2)
    y_calc = 10.0 * prof + 1.0
    y_bkg = np.ones_like(x)
    y_obs = 20.0 * prof + 1.0  # 観測は計算の 2 倍
    win = [(0, len(x))]
    contrib = [10.0 * prof]
    i_calc = np.array([10.0 * prof.sum()])
    plain = partition_intensities(y_obs, y_calc, y_bkg, win, contrib, i_calc)
    assert plain[0] == pytest.approx(2.0 * i_calc[0])
    # 中心が範囲外なら計算値のまま
    out = partition_intensities(
        y_obs, y_calc, y_bkg, win, contrib, i_calc, inside=np.array([False])
    )
    assert out[0] == pytest.approx(i_calc[0])
    # マスクで低角側半分を外しても、観測/計算の比は同じ(比で決めるので切り詰めに強い)
    mask = x >= 5.0
    half = partition_intensities(y_obs, y_calc, y_bkg, win, contrib, i_calc, mask=mask)
    assert half[0] == pytest.approx(2.0 * i_calc[0])
    # 全点を外すと使える点が無く、計算値のまま
    none = partition_intensities(
        y_obs, y_calc, y_bkg, win, contrib, i_calc, mask=np.zeros_like(mask)
    )
    assert none[0] == pytest.approx(i_calc[0])


def test_update_does_not_run_away_and_outside_reflections_take_median(data_dir: Path) -> None:
    proj = _project(data_dir)
    proj.start_le_bail("PbSO4", update=False)
    rwp = [_rwp(proj)]
    for _ in range(8):
        proj.le_bail_update()
        rwp.append(_rwp(proj))
    assert rwp[0] > 2.0  # 構造の強度 × 固定スケール(1)なので観測と桁が違う
    assert rwp[1] < 1.0  # 1 回の分配で観測と同じ大きさになる(幅はまだ合っていない)
    assert max(rwp[1:]) < rwp[1] * 1.02  # 更新を重ねても悪くならない(発散していた)
    inside, outside = _inside_outside(proj)
    assert outside, "x_max + 2° のマージンに反射があるはず"
    table = proj.phases["PbSO4"].f2_tables[0]
    med = float(np.median([table[k] for k in inside]))
    for k in outside:
        assert table[k] == pytest.approx(med)
    # 範囲内の反射は暴走していない(中央値の 1000 倍以内)
    assert max(table[k] for k in inside) < 1e3 * med


def test_start_le_bail_updates_to_data_and_refine_converges(data_dir: Path) -> None:
    proj = _project(data_dir)
    proj.start_le_bail("PbSO4")
    assert _rwp(proj) < 1.0  # 開始直後から観測と同じ大きさ(幅はまだ合っていない)
    proj.vary(
        "h0:bkg:*", "PbSO4:cell:*", "h0:inst:zero", "h0:inst:w", "h0:inst:u", "h0:inst:v",
        "h0:inst:x", "h0:inst:y",
    )  # fmt: skip
    proj.refine(max_cycles=10)  # le_bail_cycles=0: 改善が止まるまで強度の更新を繰り返す
    reported = proj.rfactors()["total"].rwp
    assert reported < 0.15
    # 終了時の計算パターンは更新後の強度で計算済み(メッセージの Rwp と図の Rwp が一致)
    assert _rwp(proj) == pytest.approx(reported, rel=1e-9)


def test_reflections_in_excluded_region_take_median(data_dir: Path) -> None:
    proj = _project(data_dir)
    h = proj.histograms[0]
    h.excluded = [(29.0, 31.0)]
    proj.start_le_bail("PbSO4")
    inside, outside = _inside_outside(proj)
    tbl = proj.calculate(0).peaks[0]
    in_gap = [
        (int(a), int(b), int(c))
        for (a, b, c), tt in zip(tbl.hkl, tbl.two_theta, strict=True)
        if 29.0 <= tt <= 31.0
    ]
    assert in_gap and set(in_gap) <= set(outside)
    table = proj.phases["PbSO4"].f2_tables[0]
    med = float(np.median([table[k] for k in inside]))
    for k in in_gap:
        assert table[k] == pytest.approx(med)


def test_weighted_partition_matches_least_squares_scale_for_isolated_peak() -> None:
    x = np.arange(0.0, 10.0, 0.05)
    prof = np.exp(-0.5 * ((x - 5.0) / 0.3) ** 2)
    c = 10.0 * prof
    y_bkg = 5.0 + 0.0 * x
    y_calc = c + y_bkg
    rng = np.random.default_rng(1)
    y_obs = 1.7 * c + y_bkg + rng.normal(0.0, 0.5, len(x))
    w = 1.0 / np.maximum(y_obs, 1.0)
    win = [(0, len(x))]
    i_calc = np.array([c.sum()])
    got = partition_intensities(y_obs, y_calc, y_bkg, win, [c], i_calc, weights=w)
    core = c > 0.01 * c.max()
    net = (y_obs - y_bkg)[core]
    lsq = float(np.sum(w[core] * c[core] * net) / np.sum(w[core] * c[core] ** 2))
    assert got[0] == pytest.approx(i_calc[0] * lsq, rel=1e-9)
    assert got[0] == pytest.approx(1.7 * i_calc[0], rel=0.02)
