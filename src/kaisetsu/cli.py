"""コマンドライン(``kaisetsu`` / ``python -m kaisetsu`` / 配布物の ``kaisetsu-cli.exe``)。

GUI で作ったプロジェクトファイル(*.kaisetsu.json)を端末から精密化したり、多数のデータを
順に処理したりする。GUI(Qt)には依存しない。

    kaisetsu refine   PROJECT [--out OUT] [--csv ...] [--reflections ...] [--unexplained ...]
                      [--cif ...]
    kaisetsu calc     PROJECT [--csv ...]            精密化せずに計算と R 因子だけ
    kaisetsu lattice  DATA... --cif CIF [--inst TEMPLATE | --preset NAME] [--csv ...]
    kaisetsu sequence PROJECT DATA... [--csv ...]    連続データ(温度・時間系列)の逐次精密化
    kaisetsu presets                                  装置プリセットの一覧
    kaisetsu selftest                                 動作確認
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from kaisetsu import __version__

PROJECT_SUFFIX = ".kaisetsu.json"


def _out_path(project_path: Path, out: str | None) -> Path:
    """--out が無ければ <名前>_refined.kaisetsu.json(入力は上書きしない)。"""
    if out:
        return Path(out)
    name = project_path.name
    stem = name[: -len(PROJECT_SUFFIX)] if name.endswith(PROJECT_SUFFIX) else project_path.stem
    return project_path.with_name(f"{stem}_refined{PROJECT_SUFFIX}")


def _say(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg, flush=True)


def _report(project: object, quiet: bool) -> None:
    """R 因子と診断を出す。"""
    from kaisetsu.refine.project import Project

    assert isinstance(project, Project)
    rf = project.rfactors()
    for i, h in enumerate(project.histograms):
        print(f"h{i} ({h.name}): {rf[h.name]}")
    if len(project.histograms) > 1 and "total" in rf:
        print(f"全体: {rf['total']}")
    for i, h in enumerate(project.histograms):
        for name in h.contributions:
            rep = project.reflection_report(i, name)
            print(f"  {name}: R_B = {100 * rep.r_bragg:.2f}%  R_F = {100 * rep.r_f:.2f}%")
    for name, ph in project.phases.items():
        ph.sync()
        print(f"相 {name}: {ph.structure.spacegroup.symbol}, {ph.structure.lattice}")
    if not quiet:
        for note in project.diagnostics():
            print(f"注意: {note}")


def _save_outputs(project: object, args: argparse.Namespace) -> None:
    from kaisetsu.crystal.cif import write_structure
    from kaisetsu.io.export import (
        write_pattern_csv,
        write_reflections_csv,
        write_unexplained_peaks_csv,
    )
    from kaisetsu.refine.project import Project

    assert isinstance(project, Project)
    if args.csv:
        p = write_pattern_csv(project, args.hist, args.csv)
        print(f"計算パターン: {p}")
    if args.reflections:
        p = write_reflections_csv(project, args.hist, args.reflections)
        print(f"反射リスト: {p}")
    if args.unexplained:
        p = write_unexplained_peaks_csv(project, args.unexplained, args.hist)
        n = max(len(p.read_text(encoding="utf-8").splitlines()) - 1, 0)
        print(f"説明できないピークの一覧({n} 本): {p}")
    if args.cif:
        name = args.phase or next(iter(project.phases))
        ph = project.phases[name]
        ph.sync()
        write_structure(ph.structure, args.cif)
        print(f"CIF({name}): {args.cif}")


# ---------------------------------------------------------------- サブコマンド
def cmd_refine(args: argparse.Namespace) -> int:
    from kaisetsu.io.project import load_project, save_project
    from kaisetsu.refine.recipe import standard_recipe

    path = Path(args.project)
    project = load_project(path)
    t0 = time.perf_counter()
    if args.recipe == "current":
        if not project.params.free():
            print(
                "精密化にチェックの入ったパラメータがありません(--recipe standard を使ってください)"
            )
            return 1
        _say(
            args.quiet, f"現在チェックの入っている {len(project.params.free())} 変数で精密化します"
        )
        project.refine(max_cycles=args.cycles)
    else:
        recipe = standard_recipe(
            args.phase or "*",
            "h*",
            coordinates=not args.no_coordinates,
            adp=not args.no_adp,
            occupancy=args.occupancy,
            sample_displacement=args.displacement,
            preferred_orientation=args.preferred_orientation,
            roughness=args.roughness,
            try_no_kalpha2=not args.no_kalpha2_trial,
            humps=not args.no_humps,
            cell_scan=not args.no_cell_scan,
        )
        for s in recipe.steps:
            s.max_cycles = args.cycles * 2 if s.name == "最終(全パラメータ)" else args.cycles

        def _progress(step: object, res: object) -> None:
            rf = project.rfactors()["total"]
            _say(
                args.quiet,
                f"[{getattr(step, 'name', '')}] 変数 {len(project.params.free())}"
                f"  Rwp = {100 * rf.rwp:.3f}%  χ² = {rf.chi2:.3f}",
            )

        n0 = len(project.log)
        recipe.run(project, progress=_progress)
        if args.verbose:
            for line in project.log[n0:]:
                print("  " + line)
    _say(args.quiet, f"({time.perf_counter() - t0:.1f} s)")
    _report(project, args.quiet)
    out = _out_path(path, args.out)
    save_project(project, out)
    print(f"保存: {out}")
    _save_outputs(project, args)
    return 0


def cmd_calc(args: argparse.Namespace) -> int:
    from kaisetsu.io.project import load_project

    project = load_project(Path(args.project))
    project.calculate_all()
    _report(project, args.quiet)
    _save_outputs(project, args)
    return 0


def _instrument_from_args(args: argparse.Namespace) -> tuple[object, object]:
    from kaisetsu.io.project import load_instrument_template
    from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
    from kaisetsu.pattern.presets import instrument_from_preset

    if args.inst:
        p = Path(args.inst)
        if p.suffix.lower() == ".json":
            return load_instrument_template(p)
        inst = CWInstrument.from_instrument_file(p)
        return inst, SampleGeometry(kind=GeometryKind.DEBYE_SCHERRER)
    if args.preset:
        return instrument_from_preset(args.preset)
    raise SystemExit(
        "装置を --inst(テンプレート json / .instprm / .prm)か --preset で指定してください"
    )


def cmd_lattice(args: argparse.Namespace) -> int:
    from kaisetsu.refine.lattice import refine_lattice_batch

    inst, sample = _instrument_from_args(args)
    batch = refine_lattice_batch(
        args.data,
        list(args.cif),
        inst,  # type: ignore[arg-type]
        sample,  # type: ignore[arg-type]
        progress=None if args.quiet else print,
        position=args.position,
        widths=args.widths,
        n_background=args.background,
        x_min=args.x_min,
        x_max=args.x_max,
    )
    print(batch.summary())
    if args.csv:
        print(f"CSV: {batch.to_csv(args.csv)}")
    return 0 if all(r.ok for r in batch.results) else 1


def cmd_sequence(args: argparse.Namespace) -> int:
    from kaisetsu.io.project import load_project
    from kaisetsu.refine.recipe import standard_recipe
    from kaisetsu.refine.sequential import refine_sequence

    project = load_project(Path(args.project))
    recipe = None
    if args.recipe == "standard":
        recipe = standard_recipe(
            "*",
            "h*",
            coordinates=not args.no_coordinates,
            adp=not args.no_adp,
            try_no_kalpha2=False,
            humps=False,
            cell_scan=not args.no_cell_scan,
        )
        for s in recipe.steps:
            s.max_cycles = args.cycles
    elif not project.params.free():
        print("精密化にチェックの入ったパラメータがありません(--recipe standard を使ってください)")
        return 1
    res = refine_sequence(
        project,
        args.data,
        hist=args.hist,
        recipe=recipe,
        params=args.params,
        max_cycles=args.cycles,
        reset_background=not args.keep_background,
        cell_scan=not args.no_cell_scan,
        progress=None if args.quiet else print,
    )
    print(res.summary())
    if args.csv:
        print(f"CSV: {res.to_csv(args.csv)}")
    return 0 if all(r.ok for r in res.rows) else 1


def cmd_presets(_args: argparse.Namespace) -> int:
    from kaisetsu.pattern.presets import PRESET_ALIASES, PRESETS

    for i, name in enumerate(PRESETS, 1):
        short = [k for k, v in PRESET_ALIASES.items() if v == name]
        print(f"{i}. {name}" + (f"  ({', '.join(short)})" if short else ""))
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    from kaisetsu.selftest import quick_check

    print(quick_check())
    return 0


# ---------------------------------------------------------------- 引数
def _add_output_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", help="計算パターン(観測・計算・背景・差・相ごと)を CSV に書く")
    p.add_argument("--reflections", help="反射リストを CSV に書く")
    p.add_argument(
        "--unexplained",
        help="相の反射で説明できないピークの一覧(位置、d 値、高さ、相対強度)を CSV に書く"
        "(他のソフトで第二相を探す用)",
    )
    p.add_argument("--cif", help="精密化した構造を CIF に書く(--phase で相を選ぶ)")
    p.add_argument("--hist", type=int, default=0, help="CSV に書くヒストグラム番号(既定 0)")
    p.add_argument("--phase", help="対象の相の名前(既定: 全部 / CIF は最初の相)")
    p.add_argument("-q", "--quiet", action="store_true", help="経過を出さない")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaisetsu",
        description=(
            "kaisetsu のコマンドライン。GUI で保存したプロジェクト(*.kaisetsu.json)を端末から扱う。"
        ),
    )
    parser.add_argument("--version", action="version", version=f"kaisetsu {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "refine", help="プロジェクトを標準手順(または現在のチェック)で精密化して保存する"
    )
    p.add_argument("project", help="プロジェクトファイル(*.kaisetsu.json)")
    p.add_argument("--out", help="保存先(既定: <名前>_refined.kaisetsu.json。入力は上書きしない)")
    p.add_argument(
        "--recipe",
        choices=["standard", "current"],
        default="standard",
        help="standard: 標準手順(既定)、current: いまチェックの入っているパラメータだけ",
    )
    p.add_argument("--cycles", type=int, default=15, help="各段階の最大反復回数(既定 15)")
    p.add_argument("--no-coordinates", action="store_true", help="原子座標を動かさない")
    p.add_argument("--no-adp", action="store_true", help="変位パラメータ U を動かさない")
    p.add_argument("--occupancy", action="store_true", help="占有率も動かす(混合・部分占有サイト)")
    p.add_argument("--displacement", action="store_true", help="ゼロ点の代わりに試料変位を動かす")
    p.add_argument(
        "--preferred-orientation", action="store_true", help="選択配向(March-Dollase)も動かす"
    )
    p.add_argument("--roughness", action="store_true", help="表面粗さも動かす(Bragg-Brentano)")
    p.add_argument("--no-cell-scan", action="store_true", help="格子定数の初期値を振らない")
    p.add_argument("--no-kalpha2-trial", action="store_true", help="Kα2 なしを試さない")
    p.add_argument("--no-humps", action="store_true", help="残差のこぶを背景のハンプにして試さない")
    p.add_argument("-v", "--verbose", action="store_true", help="精密化のログを全部出す")
    _add_output_options(p)
    p.set_defaults(func=cmd_refine)

    p = sub.add_parser("calc", help="精密化せずにパターンを計算して R 因子と診断を出す")
    p.add_argument("project", help="プロジェクトファイル(*.kaisetsu.json)")
    _add_output_options(p)
    p.set_defaults(func=cmd_calc)

    p = sub.add_parser(
        "lattice", help="Le Bail で自動解析(格子定数とプロファイルだけ。複数データ可)"
    )
    p.add_argument("data", nargs="+", help="回折データ(複数可)")
    p.add_argument("--cif", action="append", required=True, help="相の CIF(複数回指定可)")
    p.add_argument("--inst", help="装置テンプレート(.json)か GSAS-II の .instprm / GSAS の .prm")
    p.add_argument("--preset", help="装置プリセット(番号か短い名前。`kaisetsu presets` で一覧)")
    p.add_argument(
        "--position",
        choices=["none", "zero", "distance"],
        default="zero",
        help="位置の補正: zero(ゼロ点、既定)、distance(試料位置)、none",
    )
    p.add_argument(
        "--widths",
        choices=["none", "sample", "instrument"],
        default="sample",
        help="ピーク幅: sample(試料の広がりだけ、既定)、instrument(装置の幅も)、none",
    )
    p.add_argument("--x-min", type=float, help="フィット範囲の下限")
    p.add_argument("--x-max", type=float, help="フィット範囲の上限")
    p.add_argument("--background", type=int, default=6, help="背景(Chebyshev)の項数(既定 6)")
    p.add_argument("--csv", help="結果の表を CSV に書く")
    p.add_argument("-q", "--quiet", action="store_true", help="経過を出さない")
    p.set_defaults(func=cmd_lattice)

    p = sub.add_parser("sequence", help="連続データ(温度・時間系列)を順に精密化して表にする")
    p.add_argument("project", help="1 つ目のデータで精密化済みのプロジェクト(*.kaisetsu.json)")
    p.add_argument("data", nargs="+", help="順に差し替える回折データ")
    p.add_argument("--hist", type=int, default=0, help="差し替えるヒストグラム番号(既定 0)")
    p.add_argument(
        "--recipe",
        choices=["current", "standard"],
        default="current",
        help="current: いまチェックの入っているパラメータだけ(既定)、standard: 標準手順",
    )
    p.add_argument("--cycles", type=int, default=20, help="最大反復回数(既定 20)")
    p.add_argument("--params", nargs="+", help="表に出すパラメータ名(既定: 精密化したもの全部)")
    p.add_argument(
        "--keep-background", action="store_true", help="データごとに背景を推定し直さない"
    )
    p.add_argument("--no-cell-scan", action="store_true", help="失敗時に格子定数を振り直さない")
    p.add_argument("--no-coordinates", action="store_true", help="(standard)原子座標を動かさない")
    p.add_argument("--no-adp", action="store_true", help="(standard)変位パラメータを動かさない")
    p.add_argument("--csv", help="結果の表を CSV に書く")
    p.add_argument("-q", "--quiet", action="store_true", help="経過を出さない")
    p.set_defaults(func=cmd_sequence)

    p = sub.add_parser("presets", help="装置プリセットの一覧")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("selftest", help="動作確認(サンプルデータで 1 回計算する)")
    p.set_defaults(func=cmd_selftest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("中断しました", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
