"""装置のプリセット(GUI の装置設定ダイアログとコマンドラインで共用)。"""

from __future__ import annotations

from typing import Any

from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.profile.corrections import polarization_from_monochromator
from kaisetsu.scattering.radiation import Radiation

PRESETS: dict[str, dict[str, Any]] = {
    "実験室 X 線 Cu Kα (Bragg-Brentano, モノクロメータ)": {
        "target": "Cu", "kalpha2": True, "ratio2": 0.5,
        "polarization": polarization_from_monochromator(26.6),
        "u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02, "sh_l": 0.04,
        "geometry": GeometryKind.BRAGG_BRENTANO,
    },
    "実験室 X 線 Cu Kα (Bragg-Brentano, Ni フィルタ)": {
        "target": "Cu", "kalpha2": True, "ratio2": 0.5, "polarization": 0.5,
        "u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02, "sh_l": 0.04,
        "geometry": GeometryKind.BRAGG_BRENTANO,
    },
    "実験室 X 線 Co Kα (Bragg-Brentano)": {
        "target": "Co", "kalpha2": True, "ratio2": 0.5, "polarization": 0.5,
        "u": 0.005, "v": -0.004, "w": 0.005, "x": 0.02, "y": 0.02, "sh_l": 0.04,
        "geometry": GeometryKind.BRAGG_BRENTANO,
    },
    "放射光 X 線 (キャピラリ, Debye-Scherrer)": {
        "wavelength": 0.5, "polarization": 0.98,
        "u": 0.0005, "v": -0.0003, "w": 0.0005, "x": 0.005, "y": 0.005, "sh_l": 0.01,
        "geometry": GeometryKind.DEBYE_SCHERRER,
    },
    "定波長中性子 (Debye-Scherrer)": {
        "neutron": True, "wavelength": 1.8, "polarization": 0.0,
        "u": 0.3, "v": -0.5, "w": 0.4, "x": 0.0, "y": 0.0, "sh_l": 0.02,
        "geometry": GeometryKind.DEBYE_SCHERRER,
    },
}  # fmt: skip

#: コマンドラインで使う短い名前(番号でも指定できる)
PRESET_ALIASES: dict[str, str] = {
    "cu": "実験室 X 線 Cu Kα (Bragg-Brentano, モノクロメータ)",
    "cu-mono": "実験室 X 線 Cu Kα (Bragg-Brentano, モノクロメータ)",
    "cu-filter": "実験室 X 線 Cu Kα (Bragg-Brentano, Ni フィルタ)",
    "co": "実験室 X 線 Co Kα (Bragg-Brentano)",
    "synchrotron": "放射光 X 線 (キャピラリ, Debye-Scherrer)",
    "neutron": "定波長中性子 (Debye-Scherrer)",
}


def resolve_preset_name(key: str) -> str:
    """プリセットの正式名を返す。

    番号(1 始まり)、短い名前(cu, co, synchrotron, neutron)、正式名のどれでも受け付ける。
    """
    names = list(PRESETS)
    if key in PRESETS:
        return key
    k = key.strip().lower()
    if k in PRESET_ALIASES:
        return PRESET_ALIASES[k]
    if k.isdigit() and 1 <= int(k) <= len(names):
        return names[int(k) - 1]
    raise KeyError(
        f"プリセット '{key}' はありません。番号 1〜{len(names)}、"
        f"短い名前({', '.join(PRESET_ALIASES)})、または正式名で指定してください"
    )


def instrument_from_preset(name: str) -> tuple[CWInstrument, SampleGeometry]:
    """プリセット名から装置と試料形状を作る(ダイアログを開かずに使うため)。"""
    d = PRESETS[resolve_preset_name(name)]
    if d.get("neutron"):
        rad = Radiation.neutron(float(d["wavelength"]))
    elif "target" in d:
        rad = Radiation.characteristic(str(d["target"]))
        if not d.get("kalpha2", True):
            rad = Radiation.xray(rad.wavelength)
        elif rad.wavelength2 is not None:
            rad = Radiation.xray(rad.wavelength, rad.wavelength2, float(d.get("ratio2", 0.5)))
    else:
        rad = Radiation.xray(float(d["wavelength"]))
    sh_l = float(d.get("sh_l", 0.0))
    inst = CWInstrument(
        rad,
        u=float(d["u"]),
        v=float(d["v"]),
        w=float(d["w"]),
        x=float(d["x"]),
        y=float(d["y"]),
        s_l=0.5 * sh_l,
        h_l=0.5 * sh_l,
        polarization=float(d.get("polarization", 0.5)),
    )
    kind = d.get("geometry", GeometryKind.BRAGG_BRENTANO)
    geometry = kind if isinstance(kind, GeometryKind) else GeometryKind(str(kind))
    return inst, SampleGeometry(kind=geometry)
