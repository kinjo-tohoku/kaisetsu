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
    from kaisetsu.refine.project import Project


@dataclass
class Step:
    """レシピの 1 段階。"""

    name: str
    patterns: tuple[str, ...]
    max_cycles: int = 15
    fix: tuple[str, ...] = ()  # この段階で固定に戻すパラメータ
    description: str = ""


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
            available = [p for p in step.patterns if project.params.select(p)]
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
) -> Recipe:
    """標準の段階的精密化レシピ。

    Args:
        phase: 相の名前(glob 可)
        hist: ヒストグラムの接頭辞(例 'h0', 'h*')
        coordinates, adp, occupancy: 構造パラメータを動かすか
        sample_displacement: ゼロ点の代わりに試料変位を動かす(Bragg-Brentano で推奨)
        asymmetry: 軸発散 S/L, H/L を動かす
        preferred_orientation: March-Dollase r を動かす
    """
    r = Recipe("標準")
    r.add("スケールと背景", [f"{hist}:bkg:*", f"{hist}:{phase}:scale"], 10)
    shift = f"{hist}:sample:displacement" if sample_displacement else f"{hist}:inst:zero"
    r.add("格子定数とピーク位置", [f"{phase}:cell:*", shift], 10)
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
    if occupancy:
        r.add("占有率", [f"{phase}:*:occ"], 15)
    r.add("最終(全パラメータ)", [], 30)
    return r
