"""精密化レシピ: どのパラメータをどの順番で動かすかを宣言的に書く。

初心者向けの標準手順(Young, *The Rietveld Method*, Ch. 1.6 / McCusker et al.,
J. Appl. Cryst. 32, 36-50 (1999) の推奨順序に準拠):

    1. スケール因子とバックグラウンド
    2. 格子定数とゼロ点(または試料変位)
    3. ピーク幅 W → U, V → ローレンツ成分 X, Y → 非対称
    4. 原子座標
    5. 等方性変位パラメータ
    6. (必要なら)占有率、選択配向、異方性変位

各段階では前の段階で動かしたパラメータも動かし続ける(累積)。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaisetsu.refine.lsq import FitResult
    from kaisetsu.refine.parameters import Parameter
    from kaisetsu.refine.project import Project


@dataclass
class Step:
    """レシピの 1 段階。"""

    name: str
    patterns: tuple[str, ...]
    max_cycles: int = 15
    fix: tuple[str, ...] = ()  # この段階で固定に戻すパラメータ
    description: str = ""
    # patterns に一致したパラメータをさらに絞る条件(None なら全部)
    filter: Callable[[Project, Parameter], bool] | None = None


@dataclass
class Recipe:
    name: str
    steps: list[Step] = field(default_factory=list)

    def add(self, name: str, patterns: Iterable[str], max_cycles: int = 15, **kw: object) -> Recipe:
        self.steps.append(Step(name, tuple(patterns), max_cycles, **kw))  # type: ignore[arg-type]
        return self

    def run(
        self,
        project: Project,
        *,
        reset: bool = True,
        progress: Callable[[Step, FitResult], None] | None = None,
        iteration_progress: Callable[[int, float], None] | None = None,
    ) -> list[FitResult]:
        """レシピを順に実行する。reset=True なら最初に全パラメータを固定に戻す。"""
        if reset:
            project.fix_all()
        results: list[FitResult] = []
        for step in self.steps:
            available: list[str] = []
            for pat in step.patterns:
                for p in project.params.select(pat):
                    if step.filter is None or step.filter(project, p):
                        available.append(p.name)
            if step.fix:
                for pat in step.fix:
                    if project.params.select(pat):
                        project.fix(pat)
            if available:
                project.vary(*available)
            if not project.params.free():
                continue
            res = project.refine(max_cycles=step.max_cycles, progress=iteration_progress)
            project.log.append(
                f"レシピ '{self.name}' 段階 '{step.name}': {project.rfactors()['total']}"
            )
            results.append(res)
            if progress is not None:
                progress(step, res)
        return results


def _occupancy_is_refinable(project: Project, p: Parameter) -> bool:
    """占有率を動かす価値があるサイトか。

    占有率 1 で混合でもないサイトの占有率はスケール因子と完全に相関し、精密化すると
    発散や巨大な esd の原因になる。部分占有(値 < 1)か、混合サイトの主パラメータ
    (他のサイトの占有率がこれに従属している)だけを動かす。
    """
    if p.value < 1.0 - 1e-6:
        return True
    return any(q.tie is not None and p.name in q.tie.masters for q in project.params)


def _is_bragg_brentano(project: Project, p: Parameter) -> bool:
    """ヒストグラムパラメータ 'h{i}:...' の試料が Bragg-Brentano か。"""
    from kaisetsu.pattern.instrument import GeometryKind

    head = p.name.split(":", 1)[0]
    if not (head.startswith("h") and head[1:].isdigit()):
        return False
    i = int(head[1:])
    if i >= len(project.histograms):
        return False
    return project.histograms[i].sample.kind is GeometryKind.BRAGG_BRENTANO


def _shift_matches_geometry(project: Project, p: Parameter) -> bool:
    """試料変位パラメータをヒストグラムの光学系に合うものだけに絞る(それ以外は通す)。"""
    tail = p.name.rsplit(":", 1)[-1]
    if tail not in ("displacement", "disp_x", "disp_y"):
        return True
    bb = _is_bragg_brentano(project, p)
    return bb if tail == "displacement" else not bb


def standard_recipe(
    phase: str = "*",
    hist: str = "h*",
    *,
    coordinates: bool = True,
    adp: bool = True,
    occupancy: bool = False,
    sample_displacement: bool = False,
    asymmetry: bool = False,
    preferred_orientation: bool = False,
    roughness: bool = False,
) -> Recipe:
    """標準の段階的精密化レシピ。

    Args:
        phase: 相の名前(glob 可)
        hist: ヒストグラムの接頭辞(例 'h0', 'h*')
        coordinates, adp, occupancy: 構造パラメータを動かすか
        sample_displacement: ゼロ点の代わりに試料変位を動かす(Bragg-Brentano では高さずれ、
            Debye-Scherrer では x, y ずれ)。平板試料では推奨
        asymmetry: 軸発散 S/L, H/L を動かす
        preferred_orientation: March-Dollase r を動かす
        roughness: 表面粗さ(Suortti)を動かす。Bragg-Brentano のヒストグラムだけに適用。
            吸収の強い試料で変位パラメータが負になるときに有効
    """
    r = Recipe("標準")
    r.add("スケールと背景", [f"{hist}:bkg:*", f"{hist}:{phase}:scale"], 10)
    if sample_displacement:
        # Bragg-Brentano では高さずれ、Debye-Scherrer では x, y ずれ(filter で振り分ける)
        r.add(
            "格子定数とピーク位置",
            [
                f"{phase}:cell:*",
                f"{hist}:sample:displacement",
                f"{hist}:sample:disp_x",
                f"{hist}:sample:disp_y",
            ],
            10,
            filter=_shift_matches_geometry,
        )
    else:
        r.add("格子定数とピーク位置", [f"{phase}:cell:*", f"{hist}:inst:zero"], 10)
    r.add("ピーク幅 W", [f"{hist}:inst:w"], 10)
    r.add(
        "ピーク幅 U, V, X, Y",
        [f"{hist}:inst:u", f"{hist}:inst:v", f"{hist}:inst:x", f"{hist}:inst:y"],
        15,
    )
    if asymmetry:
        r.add("非対称", [f"{hist}:inst:s_l", f"{hist}:inst:h_l"], 10)
    if preferred_orientation:
        r.add("選択配向", [f"{hist}:{phase}:md_r"], 10)
    if coordinates:
        r.add("原子座標", [f"{phase}:*:x", f"{phase}:*:y", f"{phase}:*:z"], 15)
    if adp:
        r.add(
            "変位パラメータ",
            [
                f"{phase}:*:uiso",
                f"{phase}:*:u11",
                f"{phase}:*:u22",
                f"{phase}:*:u33",
                f"{phase}:*:u12",
                f"{phase}:*:u13",
                f"{phase}:*:u23",
            ],
            20,
        )
    if roughness:
        r.add(
            "表面粗さ",
            [f"{hist}:sample:rough_a", f"{hist}:sample:rough_b"],
            15,
            filter=_is_bragg_brentano,
        )
    if occupancy:
        r.add(
            "占有率(混合・部分占有サイト)",
            [f"{phase}:*:occ"],
            15,
            filter=_occupancy_is_refinable,
        )
    r.add("最終(全パラメータ)", [], 30)
    return r
