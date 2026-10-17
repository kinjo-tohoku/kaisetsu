"""全散乱データの変換: I(Q) → S(Q) → F(Q) → G(r)。

PDFgetX3 型のアドホック補正(Juhás, Davis, Farrow & Billinge, J. Appl. Cryst. 46, 560 (2013)):

    1. 容器・空セルの差し引き       I_c(Q) = I(Q) - b I_bkg(Q)
    2. 偏光補正(X 線)、円筒試料の吸収補正(任意)
    3. 規格化                     高 Q 側(Q > q_norm)で I_c / <f²> の平均が 1 になるよう α を決める
    4. 生の F                     F_raw(Q) = Q (α I_c(Q) - <f²>(Q)) / <f>²(Q)
    5. 多項式補正                 次数 n = round(Q_max r_poly / π) の多項式を F_raw に当てて差し引く
                                  (r < r_poly の構造は PDF に影響しない)
    6. 正弦フーリエ変換           G(r) = (2/π) ∫_{Q_min}^{Q_max} F(Q) M(Q) sin(Qr) dQ
                                  M(Q) は Lorch 窓 sin(πQ/Q_max)/(πQ/Q_max)(任意、既定は無し)

Q_max はデータの上限 Q(Cu Kα の 2θ = 150° で 7.9 Å⁻¹、Mo Kα で 17 Å⁻¹)で切り、その値を
多項式の次数、Lorch 窓、フーリエ変換、出力のメタデータ(qmax、qmin、lorch。フィットの終端効果に
使う)のすべてに使う。設定の Q_max(既定 25)がデータより大きいままだと、窓と次数と終端効果が
データと食い違い、フィットの U とスケールが大きく偏る(レビュー 2026-10-17 精密化 #2、#3)。

中性子では <f²>, <f>² を <b²>, <b>²(定数)に置き換える。入力はバナジウム規格化などを済ませた
強度を想定し、同じアドホック補正を使う(PDFgetN3 と同じ考え方)。

散乱因子の平均は組成(元素 → 原子数)から: <f>(Q) = Σ c_i f_i(Q), <f²>(Q) = Σ c_i f_i(Q)²。
X 線は f0(s = Q/4π)(異常分散は含めない)。

出典: 上記 Juhás et al. (2013); T. Egami & S. J. L. Billinge, *Underneath the Bragg Peaks*,
2nd ed. (2012) Ch. 2-3; E. Lorch, J. Phys. C 2, 229 (1969)(Lorch 窓)。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from kaisetsu.crystal.atoms import normalize_species, split_species
from kaisetsu.pattern.histogram import PowderData
from kaisetsu.pdf.calculator import lorch_window
from kaisetsu.pdf.data import PDFData
from kaisetsu.profile.corrections import cylinder_absorption, polarization_factor
from kaisetsu.scattering.neutron import NeutronScatteringLengths
from kaisetsu.scattering.xray import XrayScatteringFactors

FloatArray = npt.NDArray[np.float64]

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_composition(formula: str | Mapping[str, float]) -> dict[str, float]:
    """組成式('Ni', 'CaF2', 'Na2Ca3Al2F14', 'La0.5Sr0.5MnO3')を元素 → 原子数に分解する。"""
    if not isinstance(formula, str):
        return {normalize_species(k): float(v) for k, v in formula.items()}
    s = formula.replace(" ", "")
    if not s:
        raise ValueError("組成式が空です")
    comp: dict[str, float] = {}
    pos = 0
    for m in _FORMULA_TOKEN.finditer(s):
        if m.start() != pos:
            raise ValueError(f"組成式を解釈できません: {formula!r}")
        el, num = m.group(1), m.group(2)
        comp[el] = comp.get(el, 0.0) + (float(num) if num else 1.0)
        pos = m.end()
    if pos != len(s):
        raise ValueError(f"組成式を解釈できません: {formula!r}")
    return comp


def two_theta_to_q(two_theta: npt.ArrayLike, wavelength: float) -> FloatArray:
    """Q = 4π sinθ / λ [Å⁻¹]。"""
    th = np.radians(np.asarray(two_theta, dtype=np.float64) / 2.0)
    return np.asarray(4.0 * math.pi * np.sin(th) / wavelength)


def q_to_two_theta(q: npt.ArrayLike, wavelength: float) -> FloatArray:
    s = np.asarray(q, dtype=np.float64) * wavelength / (4.0 * math.pi)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.degrees(2.0 * np.arcsin(np.clip(s, -1.0, 1.0))))


def average_scattering(
    q: FloatArray, composition: Mapping[str, float], radiation: str
) -> tuple[FloatArray, FloatArray]:
    """(<f>²(Q), <f²>(Q))。中性子では Q に依らない定数の配列。"""
    total = float(sum(composition.values()))
    if total <= 0:
        raise ValueError("組成の原子数の和が 0 です")
    mean_f = np.zeros_like(q)
    mean_f2 = np.zeros_like(q)
    if radiation == "xray":
        xt = XrayScatteringFactors()
        s = q / (4.0 * math.pi)
        for el, n in composition.items():
            f = xt.f0(el, s)
            mean_f += (n / total) * f
            mean_f2 += (n / total) * f * f
    elif radiation == "neutron":
        tbl = NeutronScatteringLengths()
        for el, n in composition.items():
            b = tbl.scattering_length(el).real
            mean_f += (n / total) * b
            mean_f2 += (n / total) * b * b
    else:
        raise ValueError("radiation は 'xray' か 'neutron' です")
    return mean_f * mean_f, mean_f2


@dataclass
class TransformSettings:
    """I(Q) → G(r) の変換条件。

    Attributes:
        composition: 組成式(例 'Ni', 'CaF2')または元素 → 原子数
        radiation: 'xray' または 'neutron'
        qmin, qmax: フーリエ変換に使う Q の範囲 [Å⁻¹]。qmax がデータの上限より大きければ
            データの上限に切り詰める(TransformResult.qmax_eff)
        qmaxinst: データとして信頼できる Q の上限(多項式補正はここまで当てる)。0 なら qmax
        rpoly: 多項式補正の r 側の目安 [Å](次数 n = round(qmax rpoly / π)。小さいほど補正は緩やか)
        bgscale: 容器・空セル強度に掛ける係数
        polarization: X 線の偏光係数(None なら補正しない。放射光は 0.9〜1、実験室は 0.5)
        wavelength: 偏光補正と 2θ → Q 変換に使う波長 [Å]
        mu_r: 円筒試料の μR(> 0 なら吸収補正)
        lorch: Lorch 窓を掛けるか
        rmin, rmax, rstep: 出力する r のグリッド [Å]
        q_norm_fraction: 規格化に使う高 Q 側の割合(既定: 上位 20%)
    """

    composition: str | Mapping[str, float] = "Ni"
    radiation: str = "xray"
    qmin: float = 0.5
    qmax: float = 25.0
    qmaxinst: float = 0.0
    rpoly: float = 0.9
    bgscale: float = 1.0
    polarization: float | None = None
    wavelength: float | None = None
    mu_r: float = 0.0
    lorch: bool = False
    rmin: float = 0.0
    rmax: float = 30.0
    rstep: float = 0.01
    q_norm_fraction: float = 0.2


@dataclass
class TransformResult:
    """変換結果(途中の量も持つ。GUI で確認用に表示する)。

    qmax_eff は実際にフーリエ変換・Lorch 窓・多項式の次数に使った Q_max(設定の qmax とデータの
    上限 Q の小さい方)。フィットの終端効果にはこちらを使う(to_pdf_data のメタデータ)。
    """

    q: FloatArray
    i_corrected: FloatArray
    sq: FloatArray
    fq_raw: FloatArray
    fq: FloatArray
    polynomial: FloatArray
    scale: float
    r: FloatArray
    gr: FloatArray
    settings: TransformSettings
    log: list[str] = field(default_factory=list)
    qmax_eff: float = 0.0

    def to_pdf_data(self, name: str = "") -> PDFData:
        """G(r) を PDFData にする。メタデータの qmax / qmin / lorch はフィットの終端効果に使う。"""
        meta: dict[str, object] = {
            "qmax": self.qmax_eff if self.qmax_eff > 0 else self.settings.qmax,
            "qmin": self.settings.qmin,
            "lorch": bool(self.settings.lorch),
            "rpoly": self.settings.rpoly,
            "composition": str(self.settings.composition),
            "format": "kaisetsu-transform",
        }
        return PDFData(self.r, self.gr, None, name=name, metadata=meta)


def powder_to_q(data: PowderData, wavelength: float) -> tuple[FloatArray, FloatArray]:
    """2θ のパターンを (Q, I) にする。"""
    return two_theta_to_q(data.x, wavelength), data.y.copy()


def sine_transform(
    q: FloatArray, fq: FloatArray, r: FloatArray, qmax: float, lorch: bool = False
) -> FloatArray:
    """G(r) = (2/π) ∫ F(Q) M(Q) sin(Qr) dQ(台形則)。M(Q) は Lorch 窓(lorch が真のとき)。"""
    f = fq.copy()
    if lorch and qmax > 0:
        f = f * lorch_window(q, qmax)
    dq = np.gradient(q)
    w = f * dq
    # 台形則の端点補正
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.asarray((2.0 / math.pi) * (np.sin(np.outer(r, q)) @ w))


def transform(
    q: npt.ArrayLike,
    intensity: npt.ArrayLike,
    settings: TransformSettings,
    background: npt.ArrayLike | None = None,
) -> TransformResult:
    """I(Q) から G(r) を得る(手順はモジュールの docstring)。

    Args:
        q: Q [Å⁻¹](昇順)
        intensity: 試料の強度
        settings: 変換条件
        background: 容器・空セルの強度(同じ Q グリッド)。None なら差し引かない
    """
    qa = np.asarray(q, dtype=np.float64).ravel()
    ia = np.asarray(intensity, dtype=np.float64).ravel()
    if qa.shape != ia.shape:
        raise ValueError("Q と強度の長さが一致しません")
    if np.any(np.diff(qa) <= 0):
        raise ValueError("Q は単調増加でなければなりません")
    log: list[str] = []
    ic = ia.copy()
    if background is not None:
        bg = np.asarray(background, dtype=np.float64).ravel()
        if bg.shape != qa.shape:
            raise ValueError("背景(容器)の長さが Q と一致しません")
        ic = ic - settings.bgscale * bg
        log.append(f"容器・空セルを差し引き(係数 {settings.bgscale:g})")
    if settings.radiation == "xray" and settings.polarization is not None:
        if settings.wavelength is None:
            raise ValueError("偏光補正には波長が必要です")
        tt = q_to_two_theta(qa, settings.wavelength)
        pol = polarization_factor(tt, settings.polarization)
        ic = ic / np.maximum(pol, 1e-6)
        log.append(f"偏光補正(P = {settings.polarization:g})")
    if settings.mu_r > 0:
        if settings.wavelength is None:
            raise ValueError("吸収補正には波長が必要です")
        tt = q_to_two_theta(qa, settings.wavelength)
        ic = ic / np.maximum(cylinder_absorption(settings.mu_r, tt), 1e-6)
        log.append(f"円筒試料の吸収補正(μR = {settings.mu_r:g})")

    comp = parse_composition(settings.composition)
    mean_f_sq, mean_f2 = average_scattering(qa, comp, settings.radiation)

    qmaxinst = settings.qmaxinst if settings.qmaxinst > 0 else settings.qmax
    use = (qa >= settings.qmin) & (qa <= qmaxinst) & np.isfinite(ic)
    if np.count_nonzero(use) < 20:
        raise ValueError("Q_min〜Q_max の範囲にデータ点が少なすぎます")
    q_use = qa[use]
    # Q_max はデータの上限で切る(実験室 X 線では設定の既定 25 Å⁻¹ よりずっと低い)。上限は
    # 使う範囲の最後の点ではなくデータそのものの最大 Q(q <= Q_max で切った最後の点を使うと、
    # np.arange の丸めで 24.0 が 23.99 になる)
    q_top = float(np.max(qa[np.isfinite(ic)]))
    qmax_eff = min(settings.qmax, q_top)
    qmaxinst = min(qmaxinst, q_top)
    if qmax_eff < settings.qmax * (1.0 - 1e-9):
        log.append(
            f"Q_max をデータの上限 {qmax_eff:.3f} Å⁻¹ に下げました(設定 {settings.qmax:g} Å⁻¹)"
        )
    # 規格化: 高 Q 側で I_c/<f²> の平均が 1
    q_hi = q_use[-1] - settings.q_norm_fraction * (q_use[-1] - q_use[0])
    hi = use & (qa >= q_hi)
    ratio = ic[hi] / np.maximum(mean_f2[hi], 1e-12)
    alpha = 1.0 / float(np.mean(ratio)) if np.mean(ratio) > 0 else 1.0
    log.append(f"規格化係数 α = {alpha:.6g}(Q > {q_hi:.2f} Å⁻¹ の平均)")
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = (alpha * ic - mean_f2 + mean_f_sq) / mean_f_sq
    fq_raw = qa * (sq - 1.0)
    # 多項式補正(次数はデータで切った Q_max から)
    n = round(qmax_eff * settings.rpoly / math.pi)
    poly = np.zeros_like(qa)
    if n > 0:
        qf = qa[use]
        ff = fq_raw[use]
        ok = np.isfinite(ff)
        # Q = 0 で F = 0 となる多項式 Q × P_{n-1}(Q) を当てる
        vand = np.vander(qf[ok] / qmaxinst, n + 1, increasing=True)[:, 1:]
        coef, *_ = np.linalg.lstsq(vand, ff[ok], rcond=None)
        poly = np.vander(qa / qmaxinst, n + 1, increasing=True)[:, 1:] @ coef
        log.append(f"多項式補正: 次数 {n}(r_poly = {settings.rpoly:g} Å)")
    fq = np.where(np.isfinite(fq_raw), fq_raw - poly, 0.0)

    r = np.arange(settings.rmin, settings.rmax + 0.5 * settings.rstep, settings.rstep)
    ft = (qa >= settings.qmin) & (qa <= qmax_eff) & np.isfinite(fq)
    gr = sine_transform(qa[ft], fq[ft], r, qmax_eff, settings.lorch)
    log.append(
        f"フーリエ変換: Q = {settings.qmin:g}〜{qmax_eff:.3f} Å⁻¹"
        + ("(Lorch 窓あり)" if settings.lorch else "")
    )
    return TransformResult(
        q=qa,
        i_corrected=ic,
        sq=sq,
        fq_raw=fq_raw,
        fq=fq,
        polynomial=poly,
        scale=alpha,
        r=r,
        gr=gr,
        settings=settings,
        log=log,
        qmax_eff=qmax_eff,
    )


def element_symbols(composition: Mapping[str, float]) -> list[str]:
    return [split_species(k)[0] for k in composition]
