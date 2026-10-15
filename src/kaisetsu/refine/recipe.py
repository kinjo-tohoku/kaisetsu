"""精密化レシピ: どのパラメータをどの順番で動かすかを宣言的に書く。

初心者向けの標準手順(Young, *The Rietveld Method*, Ch. 1.6 / McCusker et al.,
J. Appl. Cryst. 32, 36-50 (1999) の推奨順序に準拠):

    1. スケール因子とバックグラウンド
    2. 格子定数とゼロ点(または試料変位)
    3. ピーク幅 W → U, V → ローレンツ成分 X, Y → 非対称
       (TOF では sig1 と立ち上がり α・減衰 β → sig0, sig2, X, Y)
    4. 原子座標
    5. 等方性変位パラメータ
    6. (必要なら)占有率、選択配向、異方性変位

各段階では前の段階で動かしたパラメータも動かし続ける(累積)。

Le Bail 相があるときは、最初に強度の更新を 5 回回してから始め、各段階の精密化でも
強度の更新を 3 回ずつ回す(`Project.refine(le_bail_cycles=3)`)。段階ごとに 1 回しか
更新しないと、幅が合わないうちに抽出強度が崩れて Rietveld より悪い Le Bail になる
(opXRD の実験室データ 2 本で Rwp 23.6% → 11.1%、23.6% → 11.9%)。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from kaisetsu.errors import PatternSizeError
from kaisetsu.pattern.instrument import CWInstrument, GeometryKind
from kaisetsu.refine.residual import find_residual_features

if TYPE_CHECKING:
    from kaisetsu.refine.lsq import FitResult
    from kaisetsu.refine.parameters import Parameter
    from kaisetsu.refine.project import HistogramEntry, Project


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
    #: 段階の前に走らせる準備(拘束を張るなど)。ログに出す文を返す
    prepare: Callable[[Project, Step], list[str]] | None = None
    #: "refine"(普通の段階)、"kalpha2_trial"(Kα2 なしで合うか試して良い方を採る)、
    #: "hump_trial"(残差の広いこぶを背景のハンプにして合うか試す)
    kind: str = "refine"


@dataclass
class Recipe:
    name: str
    steps: list[Step] = field(default_factory=list)
    #: 真なら、段階の後でガウス幅 Γ_G²(TOF では σ²)が負(境界)になっていたら
    #: U → V → W の順(TOF では sig0, sig2 → sig1)に固定してやり直す(3 回まで)
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
        lb_cycles = 3 if _has_le_bail(project) else 0
        if lb_cycles:
            for _ in range(5):
                project.le_bail_update()
        results: list[FitResult] = []
        for step in self.steps:
            if step.kind in ("kalpha2_trial", "hump_trial"):
                trial_fn = self._kalpha2_trial if step.kind == "kalpha2_trial" else self._hump_trial
                trial = trial_fn(project, step, iteration_progress)
                if trial is not None:
                    results.append(trial)
                    if progress is not None:
                        progress(step, trial)
                continue
            if step.prepare is not None:
                for line in step.prepare(project, step):
                    project.log.append(f"レシピ '{self.name}' 段階 '{step.name}': {line}")
            available: list[str] = []
            for pat in step.patterns:
                for p in project.params.select(pat):
                    if step.filter is None or step.filter(project, p):
                        available.append(p.name)
            if step.fix:
                for pat in step.fix:
                    if project.params.select(pat):
                        project.fix(pat)
            if step.patterns and not available:
                # 対象が無い段階(平板でない試料の表面粗さ、実験室データの放射光向け非対称など)
                project.log.append(
                    f"レシピ '{self.name}' 段階 '{step.name}': 対象のパラメータが無いので飛ばします"
                )
                continue
            if available:
                project.vary(*available)
            if not project.params.free():
                continue
            snapshot = {q.name: q.value for q in project.params}
            try:
                res = project.refine(
                    max_cycles=step.max_cycles,
                    progress=iteration_progress,
                    le_bail_cycles=lb_cycles,
                )
            except PatternSizeError as exc:
                # 格子定数や幅が発散した。発散した値のまま残さず、段階前の値に戻して知らせる
                for q in project.params:
                    if q.name in snapshot:
                        q.value = snapshot[q.name]
                project.params.evaluate_ties()
                project.params.apply_all()
                msg = (
                    f"レシピ '{self.name}' 段階 '{step.name}' でパラメータが発散したので、"
                    f"段階前の値に戻して止めました。{exc}"
                )
                project.log.append(msg)
                raise PatternSizeError(msg) from exc
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
        """ガウス幅 Γ_G²(TOF では σ²)が負になったら、段階の前の値に戻して固定しやり直す。

        負の範囲はガウス成分を 0 として計算される(GSAS-II と同じ)ので精密化は通るが、
        U < 0 に流れていく解は装置の特性として意味を失う。U を初期値に固定しても Rwp は
        ほとんど変わらない(PbSO4 9.88 → 9.86%、LaB6 は同じ)ので、初心者向けに固定する。
        U を固定しても V, W の組み合わせで負が残れば V を段階前の値に、それでも残れば W を
        最小値 G2_REF(FWHM 0.01°)に固定する(3 回まで)。ガウス成分の無いデータで V, W が
        負のまま「収束」すると、全範囲でガウス幅が 0 に切られて V, W の微分が 0 になり、
        共分散が特異になる(Z-Rietveld サンプルの CaF2 実験室データで V = −1.8, W = −0.43)。

        TOF では σ² = sig0 + sig1 d² + sig2 d⁴ が負になったら、動いている sig0, sig2 を固定する。
        sig1 だけが動いている段階(「ピーク幅 W」)で負になるのは、立ち上がり α や減衰 β が
        合っていなくて幅の辻褄を σ² < 0 で合わせようとしたときなので、sig1 を段階前の値に
        固定して次の段階(α, β も動く)に進む(iMATERIA の CeO2 で α の既定値のまま sig1 を
        動かすと σ² = −4000 μs² まで流れた)。
        """
        from kaisetsu.refine.width_check import G2_REF, width_violations

        def is_free(name: str) -> bool:
            return name in project.params and project.params[name].is_free

        for _attempt in range(3):
            fixed: list[str] = []
            floors: dict[str, float] = {}
            for v in width_violations(project):
                if v.kind == "gauss":
                    order = [f"h{v.hist}:inst:{k}" for k in ("u", "v", "w")]
                    cands = [n for n in order if is_free(n)][:1]
                    if cands and cands[0].endswith(":w"):
                        # U, V を固定しても負ならガウス成分が要らないデータ。W は段階前の値
                        # (ガウスだけで幅を合わせた大きめの値)でなく、分解能より細い最小値にする
                        w0 = snapshot.get(cands[0], 0.0)
                        floors[cands[0]] = w0 if 0.0 < w0 < G2_REF else G2_REF
                elif v.kind == "tof_sigma2":
                    cands = [f"h{v.hist}:inst:sig0", f"h{v.hist}:inst:sig2"]
                    if not any(is_free(n) for n in cands):
                        cands = [f"h{v.hist}:inst:sig1"]
                else:
                    continue
                fixed.extend(n for n in cands if is_free(n) and n not in fixed)
            if not fixed:
                return res
            for q in project.params:
                if q.name in snapshot:
                    q.value = snapshot[q.name]
            for name, val in floors.items():
                project.params[name].value = val
            project.params.evaluate_ties()
            project.params.apply_all()
            for name in fixed:
                project.fix(name)
            what = ", ".join(f"{n} = {project.params[n].value:g}" for n in fixed)
            project.log.append(
                f"レシピ '{self.name}' 段階 '{step.name}': ガウス幅(Γ_G² または σ²)がデータの"
                f"一部で負になったので {what} を固定してやり直します"
            )
            res = project.refine(
                max_cycles=step.max_cycles,
                progress=iteration_progress,
                le_bail_cycles=3 if _has_le_bail(project) else 0,
            )
        return res

    def _kalpha2_trial(
        self,
        project: Project,
        step: Step,
        iteration_progress: Callable[[int, float], None] | None,
    ) -> FitResult | None:
        """Kα2 を外した方が合うか試し、良い方を採る。

        データのラベルが Kα1 だけでも単色化されていないことがあり、その逆もある(opXRD の
        実験室データ 13 本中、Kα2 なしが効いたのは 2 本、Kα2 なしにすると悪化が 1 本)。
        今の自由パラメータのまま Kα2 の強度比を 0 にして精密化し、Rwp が KALPHA2_GAIN
        (0.3 ポイント)以上下がれば採用、そうでなければ値も比も元に戻す。
        Kα2 の無い装置(放射光、中性子、単色化済み)なら何もしない。
        """
        targets = [
            h
            for h in project.histograms
            if isinstance(h.instrument, CWInstrument)
            and h.instrument.radiation.is_xray
            and h.instrument.radiation.wavelength2 is not None
            and h.instrument.radiation.ratio2 > 0.0
        ]
        prefix = f"レシピ '{self.name}' 段階 '{step.name}'"
        if not targets or not project.params.free():
            project.log.append(f"{prefix}: Kα2 の無い装置なので飛ばします")
            return None
        values = project.snapshot()
        tables = {
            name: {i: dict(t) for i, t in ph.f2_tables.items()}
            for name, ph in project.phases.items()
        }
        rwp0 = project.rfactors()["total"].rwp
        ratios: list[float] = []
        for h in targets:
            assert isinstance(h.instrument, CWInstrument)
            ratios.append(h.instrument.radiation.ratio2)
            h.instrument.radiation = replace(h.instrument.radiation, ratio2=0.0)
        res: FitResult | None = None
        rwp1 = float("inf")
        try:
            res = project.refine(
                max_cycles=step.max_cycles,
                progress=iteration_progress,
                le_bail_cycles=3 if _has_le_bail(project) else 0,
            )
            rwp1 = project.rfactors()["total"].rwp
        except (PatternSizeError, ValueError, np.linalg.LinAlgError) as exc:
            project.log.append(f"{prefix}: Kα2 なしの試行でエラー({exc})")
        if res is not None and rwp0 - rwp1 >= KALPHA2_GAIN:
            project.log.append(
                f"{prefix}: Kα2 なしの方が合います(Rwp {100 * rwp0:.2f}% → {100 * rwp1:.2f}%)。"
                "単色化されたデータとみなして Kα2 を外しました"
            )
            return res
        for h, r2 in zip(targets, ratios, strict=True):
            assert isinstance(h.instrument, CWInstrument)
            h.instrument.radiation = replace(h.instrument.radiation, ratio2=r2)
        project.restore(values)
        for name, tbl in tables.items():
            project.phases[name].f2_tables = tbl
        for h in project.histograms:
            h.result = None
        project.log.append(
            f"{prefix}: Kα2 なしでは良くなりません(Rwp {100 * rwp0:.2f}% → {100 * rwp1:.2f}%)。"
            "Kα2 ありのまま進めます"
        )
        return None

    def _hump_trial(
        self,
        project: Project,
        step: Step,
        iteration_progress: Callable[[int, float], None] | None,
    ) -> FitResult | None:
        """残差に残った広いこぶ(ガラスキャピラリ・非晶質成分)を背景のハンプにして合うか試す。

        各ヒストグラムの残差を `find_residual_features` で点検し、幅が代表のピーク幅の 4 倍を
        超える山(高さが正味の最大強度の HUMP_MIN_FRACTION 以上、1 ヒストグラムに 3 つまで)を
        ガウスのハンプとして背景に足し、背景・スケールと一緒に精密化する。Rwp が HUMP_GAIN
        (0.5 ポイント)以上下がれば採用して以後の段階でもハンプを動かし、そうでなければ外して
        値を戻す。Chebyshev の項数を増やしても広いこぶは吸えるが、ピークの裾まで吸い始める
        (PbSO4 に幅 12° のこぶを合成した試験: 12 項 Rwp 28% → ハンプ 8%、20 項 11%)。
        """
        prefix = f"レシピ '{self.name}' 段階 '{step.name}'"
        values = project.snapshot()
        tables = {
            name: {i: dict(t) for i, t in ph.f2_tables.items()}
            for name, ph in project.phases.items()
        }
        rwp0 = project.rfactors()["total"].rwp
        added: dict[int, int] = {}
        notes: list[str] = []
        for i in range(len(project.histograms)):
            feats = find_residual_features(project, i, hump_min_fraction=HUMP_MIN_FRACTION)
            cands = feats.humps[:HUMP_MAX_PER_HISTOGRAM]
            if not cands:
                continue
            for hm in cands:
                project.add_hump(i, hm.center, hm.fwhm, hm.height)
                notes.append(
                    f"h{i}: {hm.center:.1f}{feats.unit} を中心に幅 {hm.fwhm:.1f}{feats.unit}"
                    f"(高さは最大ピークの {100 * hm.height / feats.net_max:.0f}%)"
                )
            added[i] = len(cands)
        if not added:
            project.log.append(f"{prefix}: 残差に広いこぶは無いので飛ばします")
            return None
        names = [q.name for q in project.params.select("h*:hump*:*", "h*:bkg:*", "h*:*:scale")]
        project.vary(*names)
        res: FitResult | None = None
        rwp1 = float("inf")
        try:
            res = project.refine(
                max_cycles=step.max_cycles,
                progress=iteration_progress,
                le_bail_cycles=3 if _has_le_bail(project) else 0,
            )
            rwp1 = project.rfactors()["total"].rwp
        except (PatternSizeError, ValueError, np.linalg.LinAlgError) as exc:
            project.log.append(f"{prefix}: こぶをハンプにする試行でエラー({exc})")
        if res is not None and rwp0 - rwp1 >= HUMP_GAIN:
            project.log.append(
                f"{prefix}: 残差の広いこぶを背景のガウスのハンプにしました("
                + "、".join(notes)
                + f"。Rwp {100 * rwp0:.2f}% → {100 * rwp1:.2f}%)。非晶質成分(ガラスキャピラリ"
                "など)の可能性があります。第二相の広いピークなら、その相の CIF を足してください。"
                "パラメータは h*:hump*:center|fwhm|height です"
            )
            return res
        for i, n in added.items():
            del project.histograms[i].humps[-n:]
        project.build_parameters()
        project.restore(values)
        for name, tbl in tables.items():
            project.phases[name].f2_tables = tbl
        for h in project.histograms:
            h.result = None
        project.log.append(
            f"{prefix}: 残差のこぶ(" + "、".join(notes) + f")を背景のハンプにしても良くなりません"
            f"(Rwp {100 * rwp0:.2f}% → {100 * rwp1:.2f}%)。ハンプ無しのまま進めます"
        )
        return None


#: 軽原子とみなす原子番号の上限(H〜Ne)。X 線では重原子が共存すると U_iso が個別に決まらない
LIGHT_ATOM_Z_MAX = 10
#: 軽原子の散乱能(占有率 × 多重度 × Z、中性子では |b|)の割合がこれ未満なら U_iso を共通にする
LIGHT_ATOM_FRACTION = 0.3
#: Kα2 なしを採用する Rwp の改善の下限(0.3 ポイント)
KALPHA2_GAIN = 0.003
#: 残差のこぶを背景のハンプとして採用する Rwp の改善の下限(0.5 ポイント)
HUMP_GAIN = 0.005
#: ハンプの候補にする、こぶの高さの正味の最大強度に対する下限。背景の多項式がこぶを半分吸った残りは
#: 最強ピークの 1〜3% でも Rwp に大きく効く(COD 2240592 の Ba(ClO4)2: 11.5% → 7.0%)
HUMP_MIN_FRACTION = 0.01
#: 1 ヒストグラムあたり試すこぶの数の上限
HUMP_MAX_PER_HISTOGRAM = 3
#: 放射光とみなす波長の上限 [Å](軸発散 SH/L も動かす)
SYNCHROTRON_WAVELENGTH_MAX = 1.2


def _has_le_bail(project: Project) -> bool:
    return any(ph.le_bail and ph.f2_update for ph in project.phases.values())


def _is_synchrotron(project: Project, p: Parameter) -> bool:
    """ヒストグラムパラメータが放射光(CW X 線、λ < 1.2 Å)のものか。"""
    h = _hist_of(project, p)
    if h is None or h.is_tof or not isinstance(h.instrument, CWInstrument):
        return False
    rad = h.instrument.radiation
    return rad.is_xray and rad.wavelength < SYNCHROTRON_WAVELENGTH_MAX


def light_atom_uiso_groups(project: Project) -> list[tuple[str, list[str], float]]:
    """相ごとの軽原子サイト(等方性 U のもの)と、軽原子が散乱能に占める割合。

    散乱能はサイトごとに 占有率 × 多重度 × Z(X 線)。すべてのヒストグラムが中性子なら
    |b| を使う(中性子では軽原子も強く散乱するので、普通は共通化の条件に入らない)。

    Returns:
        [(相の名前, [軽原子サイトのラベル], 軽原子の割合)]。Le Bail / Pawley の相は除く
    """
    from kaisetsu.scattering import NeutronScatteringLengths, XrayScatteringFactors

    neutron = bool(project.histograms) and all(
        h.is_tof or (isinstance(h.instrument, CWInstrument) and h.instrument.radiation.is_neutron)
        for h in project.histograms
    )
    ntab = NeutronScatteringLengths() if neutron else None
    out: list[tuple[str, list[str], float]] = []
    for ph in project.phases.values():
        if ph.le_bail or ph.pawley:
            continue
        st = ph.structure
        total = 0.0
        light_power = 0.0
        light: list[str] = []
        for site, m in zip(st.sites, st.multiplicities(), strict=True):
            try:
                z = XrayScatteringFactors.atomic_number(site.species)
            except ValueError:
                continue
            power = float(z)
            if ntab is not None:
                try:
                    power = abs(ntab.scattering_length(site.species))
                except (KeyError, ValueError):
                    pass
            w = float(site.occupancy) * float(m) * power
            total += w
            if z <= LIGHT_ATOM_Z_MAX:
                light_power += w
                if site.u_aniso is None:
                    light.append(site.label)
        if total > 0 and light:
            out.append((ph.name, light, light_power / total))
    return out


def _tie_light_atom_uiso(project: Project, step: Step) -> list[str]:
    """重原子が散乱を支配する相では、軽原子の U_iso を 1 つの共通パラメータにする。

    Te や Bi と共存する O, C, Li の U_iso は実験室 X 線では個別に決まらず、下限に張り付くか
    負になる(opXRD の 13 本中 5 本)。人がやるように、軽原子の U_iso を共通にして動かす
    (固定よりも、共通の 1 つなら決まることが多い)。すでに拘束されているサイトは触らない。
    """
    msgs: list[str] = []
    for phase, labels, frac in light_atom_uiso_groups(project):
        if frac >= LIGHT_ATOM_FRACTION:
            continue
        names = [f"{phase}:{lab}:uiso" for lab in labels]
        names = [n for n in names if n in project.params and not project.params[n].locked]
        masters_of_others = {m for q in project.params if q.tie is not None for m in q.tie.masters}
        candidates = [n for n in names if project.params[n].tie is None]
        if len(candidates) < 2:
            continue
        # 主パラメータ: 既に他(混合サイトの鏡像)の主になっているものがあればそれ
        master = next((n for n in candidates if n in masters_of_others), candidates[0])
        dependents = [n for n in candidates if n != master and n not in masters_of_others]
        if not dependents:
            continue
        mean = float(np.mean([project.params[n].value for n in [master, *dependents]]))
        project.params[master].value = max(mean, 0.005)  # 負や 0 から始めない
        for n in dependents:
            project.tie(n, master)
        project.params.evaluate_ties()
        project.params.apply_all()
        shown = ", ".join(n.split(":")[1] for n in [master, *dependents])
        msgs.append(
            f"相 {phase}: 重原子が散乱能の {100 * (1 - frac):.0f}% を占め軽原子の U_iso は個別に"
            f"決まらないので、{shown} の U_iso を共通にします(主パラメータ {master})"
        )
    return msgs


def _is_phase_param(project: Project, p: Parameter) -> bool:
    """相のパラメータ(名前の先頭が相の名前)だけに絞る。

    phase='*' のとき `*:*:z` が装置の `h0:inst:z`(X, Y, Z)にも一致してしまい、
    「原子座標」の段階で装置のローレンツ幅まで動いていた。Le Bail / Pawley の相は
    構造パラメータが強度に効かない(特異になる)ので対象外。
    """
    head = p.name.split(":", 1)[0]
    ph = project.phases.get(head)
    return ph is not None and not (ph.le_bail or ph.pawley)


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
    head = p.name.split(":", 1)[0]
    if not (head.startswith("h") and head[1:].isdigit()):
        return False
    i = int(head[1:])
    if i >= len(project.histograms):
        return False
    return project.histograms[i].sample.kind is GeometryKind.BRAGG_BRENTANO


def _hist_of(project: Project, p: Parameter) -> HistogramEntry | None:
    head = p.name.split(":", 1)[0]
    if head.startswith("h") and head[1:].isdigit() and int(head[1:]) < len(project.histograms):
        return project.histograms[int(head[1:])]
    return None


def _shift_matches_geometry(project: Project, p: Parameter) -> bool:
    """ピーク位置のパラメータをヒストグラムの種類に合うものだけに絞る(それ以外は通す)。

    Bragg-Brentano は高さずれ、Debye-Scherrer は x, y ずれ、TOF は zero(試料変位は
    TOF では意味が無く、動かすと esd が発散する)。
    """
    tail = p.name.rsplit(":", 1)[-1]
    if tail not in ("displacement", "disp_x", "disp_y", "zero"):
        return True
    h = _hist_of(project, p)
    if h is None:
        return True
    if h.is_tof:
        return tail == "zero"
    if tail == "zero":
        return False
    bb = h.sample.kind is GeometryKind.BRAGG_BRENTANO
    return bb if tail == "displacement" else not bb


def standard_recipe(
    phase: str = "*",
    hist: str = "h*",
    *,
    coordinates: bool = True,
    adp: bool = True,
    occupancy: bool = False,
    sample_displacement: bool = False,
    asymmetry: bool | None = None,
    preferred_orientation: bool = False,
    roughness: bool = False,
    try_no_kalpha2: bool = True,
    light_atom_uiso: bool = True,
    humps: bool = True,
) -> Recipe:
    """標準の段階的精密化レシピ。

    Args:
        phase: 相の名前(glob 可)
        hist: ヒストグラムの接頭辞(例 'h0', 'h*')
        coordinates, adp, occupancy: 構造パラメータを動かすか
        sample_displacement: ゼロ点の代わりに試料変位を動かす(Bragg-Brentano では高さずれ、
            Debye-Scherrer では x, y ずれ)。平板試料では推奨
        asymmetry: 軸発散 SH/L(FCJ 非対称)を動かす。None(既定)なら放射光(λ < 1.2 Å)の
            ヒストグラムだけ動かす(FWHM 0.02° 級のピークでは形が合わないと最強ピークの残差が
            Rwp を決める)
        preferred_orientation: March-Dollase r を動かす
        roughness: 表面粗さ(Suortti の a。b は 1 に固定)を動かす。Bragg-Brentano の
            ヒストグラムだけに適用。
            吸収の強い試料で変位パラメータが負になるときに有効
        try_no_kalpha2: ピーク幅のあとで Kα2 を外した方が合うか試し、Rwp が 0.3 ポイント以上
            下がれば採用する(ラベルが Kα1 でも単色化されていないデータがあり、逆もある)
        light_atom_uiso: 重原子が散乱能の 70% 以上を占める相では、軽原子(Z ≤ 10)の U_iso を
            共通の 1 つにする(個別には決まらず下限に張り付くため)
        humps: ピーク幅のあとで残差の広いこぶ(ガラスキャピラリ・非晶質成分)を背景のガウスの
            ハンプにして試し、Rwp が 0.5 ポイント以上下がれば採用する。人が足したハンプ
            (h*:hump*:*)は「スケールと背景」で背景と一緒に動く
    """
    r = Recipe("標準")
    r.add("スケールと背景", [f"{hist}:bkg:*", f"{hist}:hump*:*", f"{hist}:{phase}:scale"], 10)
    if sample_displacement:
        # Bragg-Brentano では高さずれ、Debye-Scherrer では x, y ずれ(filter で振り分ける)
        r.add(
            "格子定数とピーク位置",
            [
                f"{phase}:cell:*",
                f"{hist}:sample:displacement",
                f"{hist}:sample:disp_x",
                f"{hist}:sample:disp_y",
                f"{hist}:inst:zero",
            ],
            10,
            filter=_shift_matches_geometry,
        )
    else:
        r.add("格子定数とピーク位置", [f"{phase}:cell:*", f"{hist}:inst:zero"], 10)
    # TOF では sig1(σ² の d² の係数)が W に相当する。立ち上がり α と減衰 β も一緒に動かす:
    # α は装置ごとに桁で違い(POWGEN 0.13、iMATERIA は 5 以上)、α, β が合わないまま sig1 だけ
    # 動かすと幅の辻褄を σ² < 0 で合わせようとして流れる(iMATERIA の CeO2 で σ² = −4000 μs²)。
    # 立ち上がりが分解能より速い装置では α が上限側で決まらなくなる(esd が大きい)が害はない
    r.add(
        "ピーク幅 W",
        [
            f"{hist}:inst:w",
            f"{hist}:inst:sig1",
            f"{hist}:inst:alpha",
            f"{hist}:inst:beta0",
            f"{hist}:inst:beta1",
        ],
        10,
    )
    r.add(
        "ピーク幅 U, V, X, Y",
        [
            f"{hist}:inst:u",
            f"{hist}:inst:v",
            f"{hist}:inst:x",
            f"{hist}:inst:y",
            # TOF: σ² の定数項と d⁴ の係数(x, y は TOF のローレンツ幅にも一致する)
            f"{hist}:inst:sig0",
            f"{hist}:inst:sig2",
        ],
        15,
    )
    if humps:
        r.add("残差のこぶを背景に試す", [], 10, kind="hump_trial")
    if try_no_kalpha2:
        r.add("Kα2 なしを試す", [], 10, kind="kalpha2_trial")
    if asymmetry:
        r.add("非対称", [f"{hist}:inst:sh_l"], 10)
    elif asymmetry is None:
        r.add("非対称(放射光)", [f"{hist}:inst:sh_l"], 10, filter=_is_synchrotron)
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
            prepare=_tie_light_atom_uiso if light_atom_uiso else None,
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
