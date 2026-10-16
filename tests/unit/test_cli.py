"""コマンドライン(kaisetsu.cli)。

GUI なしで、プロジェクトの精密化・計算・格子定数・逐次精密化が動く。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaisetsu.api import CWInstrument, Project, Radiation, SampleGeometry, Structure
from kaisetsu.cli import build_parser, main
from kaisetsu.io import read_powder_data
from kaisetsu.io.project import load_project, save_project
from kaisetsu.pattern.presets import PRESETS, instrument_from_preset, resolve_preset_name

ROOT = Path(__file__).resolve().parents[2]
PBSO4_XRA = ROOT / "data/public/pbso4/PBSO4.XRA"
PBSO4_CIF = ROOT / "data/public/pbso4/PbSO4-Wyckoff.cif"


def _project_file(tmp_path: Path) -> Path:
    proj = Project("PbSO4")
    proj.add_phase(Structure.from_cif(str(PBSO4_CIF)), name="PbSO4")
    proj.add_histogram(
        read_powder_data(PBSO4_XRA),
        CWInstrument(
            Radiation.xray(1.5405, 1.5443, 0.5), u=0.002, v=-0.002, w=0.004, x=0.02, y=0.02
        ),
        SampleGeometry(),
        n_background=6,
        x_max=60.0,
    )
    proj.build_parameters()
    proj.vary("h0:bkg:*", "h0:PbSO4:scale")
    return save_project(proj, tmp_path / "pbso4.kaisetsu.json")


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "kaisetsu 0." in capsys.readouterr().out
    parser = build_parser()
    assert {"refine", "calc", "lattice", "sequence", "presets", "selftest"} <= set(
        parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    )


def test_presets_and_selftest(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "1. " in out and "(cu, cu-mono)" in out
    assert resolve_preset_name("2") == list(PRESETS)[1]
    assert resolve_preset_name("CO") == "実験室 X 線 Co Kα (Bragg-Brentano)"
    with pytest.raises(KeyError):
        resolve_preset_name("nope")
    inst, smp = instrument_from_preset("neutron")
    assert inst.radiation.is_neutron and smp.kind.value == "debye-scherrer"
    assert main(["selftest"]) == 0
    assert "FAP pattern" in capsys.readouterr().out


def test_calc_and_refine_current(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pf = _project_file(tmp_path)
    csv = tmp_path / "calc.csv"
    assert main(["calc", str(pf), "--csv", str(csv), "-q"]) == 0
    out = capsys.readouterr().out
    assert "h0 (" in out and "Rwp" in out and csv.exists()
    header = csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == "two_theta,y_obs,y_calc,y_bkg,diff,PbSO4"
    # いまのチェック(背景とスケール)だけで精密化して保存
    out_file = tmp_path / "out.kaisetsu.json"
    refl = tmp_path / "refl.csv"
    cif = tmp_path / "out.cif"
    peaks = tmp_path / "peaks.csv"
    code = main(
        [
            "refine", str(pf), "--recipe", "current", "--cycles", "5", "--out", str(out_file),
            "--reflections", str(refl), "--cif", str(cif), "--unexplained", str(peaks), "-q",
        ]
    )  # fmt: skip
    assert code == 0 and out_file.exists() and refl.exists() and cif.exists() and peaks.exists()
    text = capsys.readouterr().out
    assert f"保存: {out_file}" in text and "説明できないピークの一覧" in text
    assert refl.read_text(encoding="utf-8").startswith("phase,h,k,l,mult,d,two_theta")
    assert peaks.read_text(encoding="utf-8").startswith("hist,kind,position,unit,d,height")
    saved = load_project(out_file)
    assert saved.params["h0:PbSO4:scale"].value > 0
    # 既定の保存先は <名前>_refined.kaisetsu.json(入力は上書きしない)
    before = pf.read_text(encoding="utf-8")
    assert main(["refine", str(pf), "--recipe", "current", "--cycles", "2", "-q"]) == 0
    assert (tmp_path / "pbso4_refined.kaisetsu.json").exists()
    assert pf.read_text(encoding="utf-8") == before


def test_refine_standard_recipe(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pf = _project_file(tmp_path)
    code = main(
        ["refine", str(pf), "--cycles", "3", "--no-coordinates", "--no-adp", "--no-kalpha2-trial",
         "--no-humps", "--out", str(tmp_path / "std.kaisetsu.json")]
    )  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "[格子定数の初期値を振る]" not in out  # 走査は精密化の段階ではないので経過に出ない
    assert "[スケールと背景]" in out and "[最終(全パラメータ)]" in out
    assert "相 PbSO4:" in out and "nma" in out.replace(" ", "")
    rwp = float(out.split("Rwp = ")[-1].split("%")[0])
    assert rwp < 25.0


def test_refine_current_without_vary_fails(tmp_path: Path) -> None:
    proj = load_project(_project_file(tmp_path))
    proj.fix_all()
    pf = save_project(proj, tmp_path / "fixed.kaisetsu.json")
    assert main(["refine", str(pf), "--recipe", "current", "-q"]) == 1


def test_missing_project_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["calc", str(tmp_path / "nope.kaisetsu.json")]) == 1
    assert "エラー" in capsys.readouterr().err


def test_lattice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "lattice.csv"
    code = main(
        ["lattice", str(PBSO4_XRA), "--cif", str(PBSO4_CIF), "--preset", "cu", "--x-max", "50",
         "--csv", str(csv), "-q"]
    )  # fmt: skip
    assert code == 0 and csv.exists()
    out = capsys.readouterr().out
    assert "a = 8.4" in out
    # 装置の指定が無ければエラー
    with pytest.raises(SystemExit):
        main(["lattice", str(PBSO4_XRA), "--cif", str(PBSO4_CIF)])


def test_sequence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pf = _project_file(tmp_path)
    proj = load_project(pf)
    proj.vary("PbSO4:cell:*", "h0:inst:zero")
    save_project(proj, pf)
    # 同じデータを 2 回(1 回目と同じ結果になる)
    d = read_powder_data(PBSO4_XRA)
    p1 = tmp_path / "t1.xy"
    np.savetxt(p1, np.column_stack([d.x, d.y]), fmt="%.4f %.1f")
    csv = tmp_path / "seq.csv"
    code = main(
        ["sequence", str(pf), str(p1), str(p1), "--cycles", "5", "--params", "PbSO4:cell:a",
         "--csv", str(csv), "-q"]
    )  # fmt: skip
    assert code == 0 and csv.exists()
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("name,ok,Rwp,chi2,PbSO4:cell:a") and len(lines) == 3
    a1, a2 = (float(ln.split(",")[4]) for ln in lines[1:])
    assert abs(a1 - a2) < 1e-4 and abs(a1 - 8.48) < 0.02
    assert "t1" in capsys.readouterr().out
