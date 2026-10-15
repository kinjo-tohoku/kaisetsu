"""pdCIF(粉末データ入りの CIF)の読み込み: COD 2240592 の Ba(ClO4)2。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.io.readers import read_powder_data
from kaisetsu.io.readers.pdcif import read_pdcif


def test_read_pdcif_cod2240592(data_dir: Path) -> None:
    path = data_dir / "baclo4" / "BaClO4_COD2240592.cif"
    d = read_powder_data(path)
    assert d.metadata["format"] == "pdCIF"
    assert d.metadata["intensity_tag"] == "_pd_meas_counts_total"
    # 4423 点のうち重み 0(除外)の 1 点を落とす
    assert len(d.x) == 4422 and d.metadata["n_dropped"] == 1
    assert abs(d.x[0] - 14.9918) < 1e-4
    assert np.allclose(np.diff(d.x), 0.026, atol=1e-6)
    assert d.y.max() > 20000
    assert d.weight is not None and np.all(d.weight > 0)
    # 重みは 1/y(ポアソン)に近い
    assert np.allclose(d.weight * d.y, 1.0, rtol=0.2)
    assert abs(float(d.metadata["wavelength"]) - 1.54059) < 1e-5  # type: ignore[arg-type]


def test_structure_only_cif_is_rejected_as_data(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="粉末データの loop"):
        read_pdcif(data_dir / "lab6" / "LaB6_COD1000055.cif")


def test_pdcif_with_scan_column_and_missing_points(tmp_path: Path) -> None:
    text = """data_test
_diffrn_radiation_wavelength 1.5406
loop_
_pd_meas_2theta_scan
_pd_meas_intensity_total
10.0 100
10.1 .
10.2 120(3)
10.3 130
"""
    p = tmp_path / "t.cif"
    p.write_text(text, encoding="utf-8")
    d = read_pdcif(p)
    assert list(d.x) == [10.0, 10.2, 10.3]
    assert list(d.y) == [100.0, 120.0, 130.0]
    # 重みの列が無ければ Poisson(1/y)
    assert d.weight is not None and np.allclose(d.weight, 1.0 / d.y)
    assert d.metadata["n_dropped"] == 1
