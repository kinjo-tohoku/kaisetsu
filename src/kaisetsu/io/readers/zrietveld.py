"""Z-Rietveld の装置ファイル(*.zDiffractometer)。

`[キー] 値` と `[キー] … [End]` の入れ子で書かれたテキスト(行末は CR のことがある)。
ここでは装置の設定に要るものだけを取り出す:

- [Beam type] X-Ray / Neutron、[Measurement method] Time Of Flight / Dispersive …
- [Wave length] [Value] λ(2 つあれば Kα1, Kα2)、[Wave length ratio] [Value](2 つ目が Kα2 の比)
- [Conversion parameters] [Value] × 3: TOF = c0 + c1 d + c2 d²
  (c0 = ゼロ点 [μs]、c1 = DIFC、c2 = DIFA)
- [Diffraction angle] 2θ [deg](TOF のバンク角)
- [Profile function initial value] [Identifier] TypeA1 / TypeA1FCJ の U, V, W, P, X, Y, S/L, H/L
  (定波長の擬 Voigt。単位は deg²、deg と見なす)
- [Profile function] [Identifier] Type0m / Type0(TOF)の SigmaSquare0..2、Gamma0..2、
  Alpha*, Beta*, r0*(プロファイルのブロック内の `[名前] [Value] v` はすべて拾う)。
  [Profile function] は装置ファイルで今使っている値、[Profile function initial value] は
  「初期値に戻す」用の値で、同じ名前のプロファイルが両方にあることがある(iMATERIA の
  配布ファイルでは Type0m の α, β が両者でまったく違う)。前者を優先する
- [Default profile model function identifier] 既定のプロファイル名
- [Global fitting range] の [Exclude range](MIN/MAX は None)

サンプルデータ(z-code-software.com)の実験室 X 線と iMATERIA の TOF で確認した。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_PAIR = re.compile(r"\[([^\]]+)\]\s*([^\[\r\n]*)")


@dataclass
class ZDiffractometer:
    """zDiffractometer から取り出した装置設定。"""

    beam: str = ""  # "X-Ray" / "Neutron"
    method: str = ""  # "Time Of Flight" など
    wavelengths: list[float] = field(default_factory=list)
    wavelength_ratios: list[float] = field(default_factory=list)
    conversion: list[float] = field(default_factory=list)  # c0, c1, c2
    two_theta: float | None = None
    #: [Profile function](今使っている値)。名前 → {パラメータ名: 値}
    profiles: dict[str, dict[str, float]] = field(default_factory=dict)
    #: [Profile function initial value](初期値)
    profiles_initial: dict[str, dict[str, float]] = field(default_factory=dict)
    #: [Default profile model function identifier]
    default_profile: str = ""
    #: プロファイル以外の単独の量: "Z"(ゼロ点シフト [deg])、"Ds"(試料変位)
    misc: dict[str, float] = field(default_factory=dict)
    #: [Exclude range] の (下限, 上限)。MIN / MAX は None
    excluded: list[tuple[float | None, float | None]] = field(default_factory=list)

    @property
    def is_tof(self) -> bool:
        return "flight" in self.method.lower()

    @property
    def is_neutron(self) -> bool:
        return "neutron" in self.beam.lower()

    def profile(self, *names: str) -> dict[str, float]:
        """既定のプロファイル、なければ names の順に最初に見つかったものを返す。

        同じ名前が [Profile function] と [Profile function initial value] の両方にあれば
        前者(今使っている値)を優先し、足りないキーだけ後者で補う。
        """
        for name in (self.default_profile, *names):
            if not name:
                continue
            if name in self.profiles or name in self.profiles_initial:
                return {**self.profiles_initial.get(name, {}), **self.profiles.get(name, {})}
        return {}


def is_zdiffractometer(head: str) -> bool:
    return head.lstrip().startswith("[File format version]") or (
        "[Diffracto meter format version]" in head[:400]
    )


def _pairs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in re.split(r"[\r\n]+", text):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for key, val in _PAIR.findall(line):
            out.append((key.strip(), val.strip()))
    return out


def _float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


#: プロファイルのブロック内で値を持たない(=次に [Value] が来る)見出し
_NOT_PROFILE_KEYS = ("Value", "ID", "End", "Identifier", "Min", "Max")
#: プロファイル以外で単独の値を持つ見出し
_MISC_KEYS = ("Z", "Ds")


def parse_zdiffractometer(path: str | Path) -> ZDiffractometer:
    """zDiffractometer を読む。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    pairs = _pairs(text)
    z = ZDiffractometer()
    block: str | None = None  # 直前の見出し([Wave length] など)
    profile_name: str | None = None
    profile_store = z.profiles
    in_profile = False
    pending_key: str | None = None
    range_lo: float | None = None
    in_exclude = False
    for i, (key, val) in enumerate(pairs):
        if key == "Beam type" and val:
            z.beam = val
        elif key == "Default profile model function identifier" and val:
            z.default_profile = val
        elif key == "Exclude range":
            in_exclude = True
            range_lo = None
        elif key == "Min" and in_exclude:
            range_lo = _float(val)
        elif key == "Max" and in_exclude:
            z.excluded.append((range_lo, _float(val)))
            range_lo = None
        elif key == "Measurement method" and val:
            z.method = val
        elif key == "Diffraction angle" and val:
            v = _float(val)
            if v is not None:
                z.two_theta = v
        elif key in ("Wave length", "Wave length ratio", "Conversion parameters"):
            block = key
            pending_key = None
        elif key in ("Profile function initial value", "Profile function"):
            in_profile = True
            in_exclude = False
            profile_name = None
            pending_key = None
            profile_store = z.profiles_initial if "initial" in key else z.profiles
        elif key == "Identifier" and in_profile:
            profile_name = val or (pairs[i + 1][0] if i + 1 < len(pairs) else None)
            if profile_name:
                profile_store.setdefault(profile_name, {})
        elif in_profile and key not in _NOT_PROFILE_KEYS and not val:
            pending_key = key
        elif not in_profile and key in _MISC_KEYS:
            pending_key = key
        elif key == "Value":
            v = _float(val)
            if v is None:
                continue
            if pending_key is not None and in_profile and profile_name is not None:
                profile_store[profile_name][pending_key] = v
                pending_key = None
            elif pending_key is not None and not in_profile:
                z.misc[pending_key] = v
                pending_key = None
            elif block == "Wave length":
                z.wavelengths.append(v)
                block = None
            elif block == "Wave length ratio":
                z.wavelength_ratios.append(v)
                block = None
            elif block == "Conversion parameters":
                z.conversion.append(v)
                block = None
        elif key == "End":
            pending_key = None
            in_exclude = False
    return z
