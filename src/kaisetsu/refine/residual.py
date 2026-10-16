"""残差に残った、相の反射で説明できない特徴(未同定ピーク・非晶質のこぶ)の検出。

opXRD の公開データ 13 本の比較(docs/validation/opxrd_cases.md)で、一発の標準解析が外れる主因は
不純物・試料台のピーク(6 本)と非晶質のこぶ(2 本)だった。ここでは候補を出すだけで、除外や
ハンプの追加は人が選ぶ。自動で除外すると、構造モデルの強度が合わない反射まで異物とみなして
しまう(同じ比較で 2 本がそうなった)。

手順(2 段構え):

1. 残差 (obs − calc) を 10 FWHM の移動平均で平滑化し、半値幅が 4 FWHM より広い山を「こぶ」とする。
2. こぶをガウスで引いた残差から、2 FWHM の窓で局所最大となる鋭いピークを拾う。近くに計算反射が
   あって、そこで計算の正味強度が残差の半分以上なら「強度の不一致」(構造の問題、除外しない)、
   それ以外を「未同定」(除外の候補)とする。

しきい値は統計誤差の 8 倍、かつ正味の最大強度の 4%(雑音の多いデータで小さな山を拾わない)。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from kaisetsu.pattern.instrument import CWInstrument
from kaisetsu.pattern.tof import TOFInstrument

if TYPE_CHECKING:
    from kaisetsu.refine.project import HistogramEntry, Project

FloatArray = npt.NDArray[np.float64]
#: 一覧の 1 マスの値
Cell = float | int | str | None


@dataclass(frozen=True)
class ResidualPeak:
    """残差に残った鋭いピーク。"""

    hist: int
    position: float
    """位置(2θ [°] か TOF [μs])"""
    height: float
    """残差の高さ(計数)"""
    lo: float
    hi: float
    """除外するならこの範囲(±2.5 FWHM)"""
    sigma: float
    """高さ / その点の統計誤差"""
    nearest_reflection: float | None
    """最も近い計算反射の位置"""
    calc_fraction: float
    """その点の計算の正味強度 / 残差"""
    unindexed: bool
    """真: 相の反射で説明できない(不純物・試料台の候補)。偽: 反射はあるが強度が合わない"""
    d: float = float("nan")
    """観測位置の d 値 [Å](ゼロ点・試料位置の補正を外したもの。他のソフトで第二相を探す用)"""


@dataclass(frozen=True)
class ResidualHump:
    """残差に残った広い山(非晶質のこぶ、背景の不足)。"""

    hist: int
    center: float
    fwhm: float
    height: float


@dataclass
class ResidualFeatures:
    """1 つのヒストグラムの残差の点検結果。"""

    hist: int
    fwhm: float
    """代表のピーク幅(反射の FWHM の中央値)"""
    net_max: float
    """正味(背景を引いた)の観測強度の最大値"""
    unit: str = "°"
    peaks: list[ResidualPeak] = field(default_factory=list)
    humps: list[ResidualHump] = field(default_factory=list)

    @property
    def unindexed(self) -> list[ResidualPeak]:
        return [p for p in self.peaks if p.unindexed]

    @property
    def misfit(self) -> list[ResidualPeak]:
        return [p for p in self.peaks if not p.unindexed]

    def unexplained(self, *, include_misfit: bool = True) -> list[ResidualPeak]:
        """書き出す対象のピーク(未同定、include_misfit なら強度の不一致も)。高い順。"""
        return [p for p in self.peaks if p.unindexed or include_misfit]

    def notes(self, max_items: int = 3) -> list[str]:
        """初心者向けの注意文(問題が無ければ空)。"""
        out: list[str] = []
        tag = f"h{self.hist}"
        u = self.unit
        for p in self.unindexed[:max_items]:
            out.append(
                f"{tag}: {p.position:.2f}{u} に相の反射で説明できないピークがあります"
                f"(高さは最大ピークの {100 * p.height / self.net_max:.0f}%)。不純物や試料台なら"
                "「精密化 → 残差の点検」で除外領域にできます(自動では除外しません)。"
                "第二相なら CIF を追加してください"
            )
        rest = len(self.unindexed) - max_items
        if rest > 0:
            out.append(f"{tag}: ほかにも説明できないピークが {rest} 本あります(残差の点検を参照)")
        for p in self.misfit[:2]:
            out.append(
                f"{tag}: {p.position:.2f}{u} の反射は観測が計算より強いです(残差は最大ピークの "
                f"{100 * p.height / self.net_max:.0f}%)。占有率・選択配向・第二相の重なりを"
                "疑ってください"
            )
        for h in self.humps[:max_items]:
            out.append(
                f"{tag}: {h.center:.1f}{u} を中心に幅 {h.fwhm:.1f}{u} のこぶが残っています"
                f"(高さは最大ピークの {100 * h.height / self.net_max:.0f}%)。非晶質成分"
                "(ガラスキャピラリなど)なら「精密化 → 残差の点検」で背景のハンプにできます。"
                "背景の項数を増やすのも手です"
            )
        return out


def observed_d_spacing(h: HistogramEntry, positions: npt.ArrayLike) -> FloatArray:
    """観測ピークの位置(2θ [°] か TOF [μs])を d 値 [Å] に直す。

    精密化したゼロ点と試料位置のずれ(平板の変位・透過、キャピラリの x, y)の分を外してから
    Bragg の式で d にする。試料位置の補正は 2θ に依存するので 3 回の反復で逆算する
    (補正は 0.1° 以下なので 1 回でほぼ収束)。TOF は `TOFInstrument.d_from_tof`。
    他のソフトで第二相を探すときの d と相対強度の一覧に使う。
    """
    pos = np.asarray(positions, dtype=np.float64)
    inst = h.instrument
    if isinstance(inst, TOFInstrument):
        return inst.d_from_tof(pos)
    assert isinstance(inst, CWInstrument)
    tt = pos - inst.zero
    for _ in range(3):
        tt = pos - inst.zero - h.sample.position_shift(tt)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = inst.radiation.wavelength / (2.0 * np.sin(np.radians(tt / 2.0)))
    return np.asarray(np.where(np.isfinite(d) & (d > 0), d, np.nan))


#: 説明できないピーク一覧の列(CSV と表計算ソフト向けの文字列で共通)
UNEXPLAINED_COLUMNS = (
    "hist",
    "kind",
    "position",
    "unit",
    "d",
    "height",
    "rel_height_pct",
    "sigma",
    "nearest_reflection",
    "lo",
    "hi",
)
UNEXPLAINED_LABELS_JA = (
    "データ",
    "種類",
    "位置",
    "単位",
    "d (Å)",
    "高さ",
    "最大比 (%)",
    "高さ/σ",
    "近い反射",
    "範囲(下)",
    "範囲(上)",
)


def unexplained_peak_rows(
    features: Iterable[ResidualFeatures], *, include_misfit: bool = True
) -> list[dict[str, Cell]]:
    """説明できないピーク(と強度の不一致)の一覧を、列名 → 値の辞書の並びで返す。

    kind は "unindexed"(相の反射で説明できない)か "misfit"(反射はあるが観測が計算より強い)。
    ヒストグラムの順、各ヒストグラムの中は高い順。
    """
    rows: list[dict[str, Cell]] = []
    for f in features:
        for p in f.unexplained(include_misfit=include_misfit):
            rows.append(
                {
                    "hist": f.hist,
                    "kind": "unindexed" if p.unindexed else "misfit",
                    "position": p.position,
                    "unit": f.unit.strip(),
                    "d": p.d,
                    "height": p.height,
                    "rel_height_pct": 100.0 * p.height / f.net_max if f.net_max > 0 else 0.0,
                    "sigma": p.sigma,
                    "nearest_reflection": p.nearest_reflection,
                    "lo": p.lo,
                    "hi": p.hi,
                }
            )
    return rows


def _fmt_cell(key: str, value: Cell) -> str:
    if value is None:
        return ""
    if key == "hist":
        return f"h{value}"
    if key in ("kind", "unit") or isinstance(value, str):
        return str(value)
    if key == "d":
        return f"{float(value):.5f}" if math.isfinite(float(value)) else ""
    if key in ("position", "nearest_reflection"):
        return f"{float(value):.4f}"
    if key in ("lo", "hi"):
        return f"{float(value):.3f}"
    if key == "sigma":
        v = float(value)
        return f"{v:.1f}" if math.isfinite(v) else "inf"
    return f"{float(value):.1f}"


def format_unexplained_peaks(
    features: Iterable[ResidualFeatures],
    *,
    sep: str = "\t",
    include_misfit: bool = True,
    japanese: bool = True,
) -> str:
    """一覧を区切り文字で並べた文字列(先頭行は列名)。既定はタブ区切りで表計算ソフトに貼れる。"""
    rows = unexplained_peak_rows(features, include_misfit=include_misfit)
    labels = UNEXPLAINED_LABELS_JA if japanese else UNEXPLAINED_COLUMNS
    lines = [sep.join(labels)]
    for r in rows:
        cells = [_fmt_cell(k, r[k]) for k in UNEXPLAINED_COLUMNS]
        if japanese:
            cells[1] = "未同定" if r["kind"] == "unindexed" else "強度の不一致"
        lines.append(sep.join(cells))
    return "\n".join(lines) + "\n"


def _boxcar(a: np.ndarray, half: int) -> np.ndarray:
    """端を繰り返して埋める移動平均(窓 2 half + 1)。"""
    n = 2 * half + 1
    if n <= 1 or len(a) < n:
        return a.copy()
    padded = np.pad(a, (half, half), mode="edge")
    return np.asarray(np.convolve(padded, np.ones(n) / n, mode="valid"))


def _gauss(x: np.ndarray, center: float, fwhm: float, height: float) -> np.ndarray:
    sigma = max(fwhm, 1e-9) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return np.asarray(height * np.exp(-0.5 * ((x - center) / sigma) ** 2))


def find_residual_features(
    project: Project,
    hist: int = 0,
    *,
    min_sigma: float = 8.0,
    min_fraction: float = 0.04,
    hump_min_fraction: float = 0.01,
    max_peaks: int = 12,
    max_humps: int = 6,
) -> ResidualFeatures:
    """残差の未同定ピークとこぶを探す(候補を返すだけで、プロジェクトは変えない)。

    Args:
        project, hist: 対象(最後の計算結果 `HistogramEntry.result` を使う。無ければ計算する)
        min_sigma: 残差が統計誤差の何倍以上なら拾うか
        min_fraction: 鋭いピークの残差が正味の最大強度の何割以上なら拾うか
        hump_min_fraction: こぶの高さが正味の最大強度の何割以上なら拾うか。こぶは平滑化した残差で
            見る(雑音は min_sigma で既に抑えてある)ので鋭いピークより低くてよい。背景の多項式が
            こぶを半分吸ったあとの残りは最強ピークの 1〜3% で、それでも Rwp には大きく効く
            (COD 2240592 の Ba(ClO4)2: ハンプ無し 11.5%、ハンプ有り 7.0%)
        max_peaks, max_humps: 返す数の上限(高い順)
    """
    h = project.histograms[hist]
    res = h.result
    if res is None or len(res.x) != len(h.data.x):
        res = project.calculate(hist)
    unit = " μs" if h.is_tof else "°"
    m = h.mask()
    # 代表のピーク幅: 強い反射(最大の 10% 以上)の FWHM の中央値。弱い高角反射まで含めた中央値は
    # 幅の大きい反射に引きずられる(opXRD の Li2TeC2 で 1.1°。本当のピーク幅は 0.2°)
    fw_all = np.zeros(0)
    if res.peaks:
        strong = [
            pk.fwhm[pk.intensity_total >= 0.1 * pk.intensity_total.max()]
            for pk in res.peaks
            if len(pk.intensity_total) and pk.intensity_total.max() > 0
        ]
        fw_all = np.concatenate(strong) if strong else np.concatenate([pk.fwhm for pk in res.peaks])
    fw_all = fw_all[np.isfinite(fw_all) & (fw_all > 0)]
    n_in = int(np.count_nonzero(m))
    if n_in < 10:
        return ResidualFeatures(hist, float(np.median(fw_all)) if len(fw_all) else 0.0, 1.0, unit)
    x = res.x[m]
    yo = h.data.y[m]
    yc = res.y_calc[m]
    yb = res.y_background[m]
    step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    step = max(step, 1e-9)
    fw = float(np.median(fw_all)) if len(fw_all) else 10.0 * step
    fw = max(fw, 2.5 * step)
    net = yo - yb
    net_max = float(net.max()) if len(net) else 1.0
    if not np.isfinite(net_max) or net_max <= 0:
        net_max = max(float(yo.max()), 1.0)
    weight = h.data.weight[m] if h.data.weight is not None else None
    if weight is not None and np.all(weight > 0):
        sig = 1.0 / np.sqrt(weight)
    else:
        sig = np.sqrt(np.maximum(yo, 1.0))
    resid = yo - yc
    d_obs = observed_d_spacing(h, x)
    refl = np.concatenate([pk.two_theta for pk in res.peaks]) if res.peaks else np.zeros(0)
    refl = refl[np.isfinite(refl)]
    feats = ResidualFeatures(hist, fw, net_max, unit)

    # 1) こぶ: 10 FWHM で平滑化した残差の、幅 4 FWHM を超える山
    k = max(round(10.0 * fw / step), 3)
    sm = _boxcar(resid, k)
    sig_sm = sig / math.sqrt(2 * k + 1)
    span = float(x[-1] - x[0])
    n = len(x)
    i = k
    while i < n - k:
        v = sm[i]
        if (
            v >= max(min_sigma * sig_sm[i], hump_min_fraction * net_max)
            and v >= sm[i - k : i + k + 1].max()
        ):
            hv = 0.5 * v
            a = i
            while a > 0 and sm[a] > hv:
                a -= 1
            b = i
            while b < n - 1 and sm[b] > hv:
                b += 1
            width = float(x[b] - x[a])
            if 4.0 * fw < width < 0.5 * span and not any(
                abs(hm.center - x[i]) < 0.5 * (hm.fwhm + width) for hm in feats.humps
            ):
                feats.humps.append(ResidualHump(hist, float(x[i]), width, float(v)))
            i = b + 1
            continue
        i += 1
    feats.humps.sort(key=lambda hm: -hm.height)
    del feats.humps[max_humps:]

    # 2) 鋭いピーク: こぶを引いた残差の局所最大(窓 2 FWHM)
    r2 = resid.copy()
    for hm in feats.humps:
        r2 -= _gauss(x, hm.center, hm.fwhm, hm.height)
    half = max(round(fw / step), 2)
    for i in range(half, n - half):
        v = r2[i]
        if v < max(min_sigma * sig[i], min_fraction * net_max):
            continue
        if v < r2[i - half : i + half + 1].max():
            continue
        calc_here = float(yc[i] - yb[i])
        nearest: float | None = None
        near = False
        if len(refl):
            j = int(np.argmin(np.abs(refl - x[i])))
            nearest = float(refl[j])
            near = abs(nearest - x[i]) < 1.5 * fw
        unindexed = not (near and calc_here > 0.5 * v)
        feats.peaks.append(
            ResidualPeak(
                hist,
                float(x[i]),
                float(v),
                float(x[i] - 2.5 * fw),
                float(x[i] + 2.5 * fw),
                float(v / sig[i]) if sig[i] > 0 else float("inf"),
                nearest,
                calc_here / v if v > 0 else 0.0,
                unindexed,
                float(d_obs[i]),
            )
        )
    feats.peaks.sort(key=lambda p: -p.height)
    del feats.peaks[max_peaks:]
    return feats


__all__ = [
    "UNEXPLAINED_COLUMNS",
    "ResidualFeatures",
    "ResidualHump",
    "ResidualPeak",
    "find_residual_features",
    "format_unexplained_peaks",
    "observed_d_spacing",
    "unexplained_peak_rows",
]
