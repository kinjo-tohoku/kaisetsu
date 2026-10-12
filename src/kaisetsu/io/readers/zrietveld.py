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
    profiles: dict[str, dict[str, float]] = field(default_factory=dict)
    #: プロファイル以外の単独の量: "Z"(ゼロ点シフト [deg])、"Ds"(試料変位)
    misc: dict[str, float] = field(default_factory=dict)

    @property
    def is_tof(self) -> bool:
        return "flight" in self.method.lower()

    @property
    def is_neutron(self) -> bool:
        return "neutron" in self.beam.lower()


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


_PROFILE_KEYS = ("U", "V", "W", "P", "X", "Y", "S/L", "H/L", "As")


def parse_zdiffractometer(path: str | Path) -> ZDiffractometer:
    """zDiffractometer を読む。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    pairs = _pairs(text)
    z = ZDiffractometer()
    block: str | None = None  # 直前の見出し([Wave length] など)
    profile_name: str | None = None
    in_profile = False
    pending_key: str | None = None
    for i, (key, val) in enumerate(pairs):
        if key == "Beam type" and val:
            z.beam = val
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
            profile_name = None
            pending_key = None
        elif key == "Identifier" and in_profile:
            profile_name = val or (pairs[i + 1][0] if i + 1 < len(pairs) else None)
            if profile_name:
                z.profiles.setdefault(profile_name, {})
        elif in_profile and key in _PROFILE_KEYS:
            pending_key = key
        elif not in_profile and key in ("Z", "Ds"):
            pending_key = key
        elif key == "Value":
            v = _float(val)
            if v is None:
                continue
            if pending_key is not None and in_profile and profile_name is not None:
                z.profiles[profile_name][pending_key] = v
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
    return z
