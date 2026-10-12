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
    #: 真なら、段階の後でガウス幅 Γ_G² が負(境界)になっていたら U を固定してやり直す
    guard_widths: bool = True

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
            snapshot = {q.name: q.value for q in project.params}
            res = project.refine(max_cycles=step.max_cycles, progress=iteration_progress)
            if self.guard_widths:
                res = self._guard_widths(project, step, snapshot, res, iteration_progress)
            project.log.append(
                f"レシピ '{self.name}' 段階 '{step.name}': {project.rfactors()['total']}"
            )
            results.append(res)
            if progress is not None:
                progress(step, res)
        return results

    def _guard_widths(
        self,
        project: Project,
        step: Step,
        snapshot: dict[str, float],
        res: FitResult,
        iteration_progress: Callable[[int, float], None] | None,
    ) -> FitResult:
        """ガウス幅 Γ_G² が負になったら、段階の前の値に戻して U を固定しやり直す。

        負の範囲はガウス成分を 0 として計算される(GSAS-II と同じ)ので精密化は通るが、
        U < 0 に流れていく解は装置の特性として意味を失う。U を初期値に固定しても Rwp は
        ほとんど変わらない(PbSO4 9.88 → 9.86%、LaB6 は同じ)ので、初心者向けに固定する。
        それでも V, W の組み合わせで負が残ることはあり、その場合は diagnostics が知らせる。
        """
        from kaisetsu.refine.width_check import width_violations

        fixed: list[str] = []
        for v in width_violations(project):
            name = f"h{v.hist}:inst:u"
            if v.kind == "gauss" and name in project.params and project.params[name].is_free:
                fixed.append(name)
        if not fixed:
            return res
        for q in project.params:
            if q.name in snapshot:
                q.value = snapshot[q.name]
        project.params.evaluate_ties()
        project.params.apply_all()
        for name in fixed:
            project.fix(name)
        project.log.append(
            f"レシピ '{self.name}' 段階 '{step.name}': ガウス幅 Γ_G² がデータの一部で負になった"
            f"ので {', '.join(fixed)} を段階前の値に固定してやり直します"
        )
        return project.refine(max_cycles=step.max_cycles, progress=iteration_progress)


def _is_phase_param(project: Project, p: Parameter) -> bool:
    """相のパラメータ(名前の先頭が相の名前)だけに絞る。

    phase='*' のとき `*:*:z` が装置の `h0:inst:z`(X, Y, Z)にも一致してしまい、
    「原子座標」の段階で装置のローレンツ幅まで動いていた。
    """
    return p.name.split(":", 1)[0] in project.phases


def _occupancy_is_refinable(project: Project, p: Parameter) -> bool:
    """占有率を動かす価値があるサイトか。

    占有率 1 で混合でもないサイトの占有率はスケール因子と完全に相関し、精密化すると
    発散や巨大な esd の原因になる。部分占有(値 < 1)か、混合サイトの主パラメータ
    (他のサイトの占有率がこれに従属している)だけを動かす。
    """
    if not _is_phase_param(project, p):
        return False
    if p.value < 1.0 - 1e-6:
        return True
    return any(q.tie is not None and p.name in q.tie.masters for q in project.params)


def _is_phase_or_pdf_param(project: Project, p: Parameter) -> bool:
    """相のパラメータか PDF データセット(g*)のパラメータ(粉末ヒストグラムの h* は除く)。"""
    head = p.name.split(":", 1)[0]
    return head in project.phases or (head.startswith("g") and head[1:].isdigit())


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
        asymmetry: 軸発散 SH/L(FCJ 非対称)を動かす
        preferred_orientation: March-Dollase r を動かす
        roughness: 表面粗さ(Suortti の a。b は 1 に固定)を動かす。Bragg-Brentano の
            ヒストグラムだけに適用。
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
        r.add("非対称", [f"{hist}:inst:sh_l"], 10)
    if preferred_orientation:
        r.add("選択配向", [f"{hist}:{phase}:md_r"], 10)
    if coordinates:
        r.add(
            "原子座標",
            [f"{phase}:*:x", f"{phase}:*:y", f"{phase}:*:z"],
            15,
            filter=_is_phase_param,
        )
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
            filter=_is_phase_param,
        )
    if roughness:
        r.add(
            "表面粗さ",
            [f"{hist}:sample:rough_a"],
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


def pdf_recipe(
    phase: str = "*",
    dataset: str = "g*",
    *,
    coordinates: bool = True,
    adp: bool = True,
    delta2: bool = True,
    delta1: bool = False,
    qdamp: bool = True,
    qbroad: bool = False,
    spdiameter: bool = False,
    occupancy: bool = False,
) -> Recipe:
    """PDF(実空間)フィットの標準手順(PDFgui のチュートリアルの順序に準拠)。

    1. スケール因子
    2. 格子定数
    3. 変位パラメータと相関運動補正 δ2(ピーク幅)
    4. 装置分解能 Q_damp(遠距離の減衰)
    5. 原子座標
    6. 最終(全パラメータ)

    Args:
        phase: 相の名前(glob 可)
        dataset: PDF データセットの接頭辞(例 'g0', 'g*')
        delta1, delta2: 相関運動補正を動かすか(通常は室温以下なら δ2 だけ)
        qdamp, qbroad: 装置分解能を動かすか(標準試料で決めた値があれば固定する)
        spdiameter: 球状ナノ粒子の直径を動かすか
    """
    r = Recipe("PDF 標準")
    r.add("スケール", [f"{dataset}:pdf:scale"], 10)
    r.add("格子定数", [f"{phase}:cell:*"], 10)
    width = [
        f"{phase}:*:uiso", f"{phase}:*:u11", f"{phase}:*:u22", f"{phase}:*:u33",
        f"{phase}:*:u12", f"{phase}:*:u13", f"{phase}:*:u23",
    ]  # fmt: skip
    if not adp:
        width = []
    if delta2:
        width.append(f"{dataset}:{phase}:delta2")
    if delta1:
        width.append(f"{dataset}:{phase}:delta1")
    if width:
        r.add("ピーク幅(変位パラメータ、δ)", width, 15, filter=_is_phase_or_pdf_param)
    inst = []
    if qdamp:
        inst.append(f"{dataset}:pdf:qdamp")
    if qbroad:
        inst.append(f"{dataset}:pdf:qbroad")
    if inst:
        r.add("装置分解能", inst, 10)
    if spdiameter:
        r.add("粒子径", [f"{dataset}:{phase}:spdiameter"], 10)
    if coordinates:
        r.add(
            "原子座標",
            [f"{phase}:*:x", f"{phase}:*:y", f"{phase}:*:z"],
            15,
            filter=_is_phase_param,
        )
    if occupancy:
        r.add("占有率", [f"{phase}:*:occ"], 10, filter=_occupancy_is_refinable)
    r.add("最終(全パラメータ)", [], 30)
    return r
