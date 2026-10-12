"""ダイアログ: 装置・試料の設定、標準レシピの選択。"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.dialog_utils import fit_dialog_to_screen
from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.profile.corrections import polarization_from_monochromator
from kaisetsu.refine.recipe import Recipe, standard_recipe
from kaisetsu.scattering.radiation import CHARACTERISTIC_WAVELENGTHS, KBETA_WAVELENGTHS, Radiation

PRESETS: dict[str, dict[str, object]] = {
    "実験室 X 線 Cu Kα (Bragg-Brentano, モノクロメータ)": {
        "target": "Cu", "kalpha2": True, "ratio2": 0.5, "polarization": polarization_from_monochromator(26.6),
        "u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02, "s_l": 0.02, "h_l": 0.02,
        "geometry": GeometryKind.BRAGG_BRENTANO,
    },
    "実験室 X 線 Cu Kα (Bragg-Brentano, Ni フィルタ)": {
        "target": "Cu", "kalpha2": True, "ratio2": 0.5, "polarization": 0.5,
        "u": 0.004, "v": -0.003, "w": 0.004, "x": 0.02, "y": 0.02, "s_l": 0.02, "h_l": 0.02,
        "geometry": GeometryKind.BRAGG_BRENTANO,
    },
    "実験室 X 線 Co Kα (Bragg-Brentano)": {
        "target": "Co", "kalpha2": True, "ratio2": 0.5, "polarization": 0.5,
        "u": 0.005, "v": -0.004, "w": 0.005, "x": 0.02, "y": 0.02, "s_l": 0.02, "h_l": 0.02,
        "geometry": GeometryKind.BRAGG_BRENTANO,
    },
    "放射光 X 線 (キャピラリ, Debye-Scherrer)": {
        "wavelength": 0.5, "polarization": 0.98,
        "u": 0.0005, "v": -0.0003, "w": 0.0005, "x": 0.005, "y": 0.005, "s_l": 0.005, "h_l": 0.005,
        "geometry": GeometryKind.DEBYE_SCHERRER,
    },
    "定波長中性子 (Debye-Scherrer)": {
        "neutron": True, "wavelength": 1.8, "polarization": 0.0,
        "u": 0.3, "v": -0.5, "w": 0.4, "x": 0.0, "y": 0.0, "s_l": 0.01, "h_l": 0.01,
        "geometry": GeometryKind.DEBYE_SCHERRER,
    },
}  # fmt: skip


def _spin(value: float, lo: float, hi: float, decimals: int, step: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    return s


class InstrumentDialog(QDialog):
    """装置パラメータと試料形状の設定。"""

    def __init__(
        self,
        instrument: CWInstrument | None,
        sample: SampleGeometry | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("装置と試料の設定")
        inst = instrument or CWInstrument(
            Radiation.characteristic("Cu"), w=0.004, s_l=0.02, h_l=0.02, polarization=0.5
        )
        smp = sample or SampleGeometry()

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("プリセット:"))
        self.preset = QComboBox()
        self.preset.addItem("(変更しない)")
        for name in PRESETS:
            self.preset.addItem(name)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        top.addWidget(self.preset, 1)
        layout.addLayout(top)
        # 2 列(左: 放射線と試料、右: ピーク幅)にして、ノート PC でも画面に収める
        cols = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        layout.addLayout(cols)

        # 放射線
        g_rad = QGroupBox("放射線(X 線源・波長)")
        f_rad = QFormLayout(g_rad)
        self.kind = QComboBox()
        self.kind.addItems(["X 線", "中性子"])
        self.kind.setCurrentIndex(1 if inst.radiation.is_neutron else 0)
        f_rad.addRow("種類", self.kind)
        self.target = QComboBox()
        for t in CHARACTERISTIC_WAVELENGTHS:
            self.target.addItem(f"{t} Kα", t)
        self.target.addItem("その他(波長を直接入力: 放射光・中性子など)", "")
        self.target.setToolTip("実験室 X 線の対陰極。選ぶと Kα1・Kα2 の波長が自動で入ります")
        self._set_target_from_wavelength(inst.radiation.wavelength, inst.radiation.is_neutron)
        self.target.currentIndexChanged.connect(self._apply_target)
        f_rad.addRow("ターゲット(対陰極)", self.target)
        self.lam1 = _spin(inst.radiation.wavelength, 0.1, 10.0, 6, 0.0001)
        f_rad.addRow("波長 λ1 [Å]", self.lam1)
        self.use_ka2 = QCheckBox("Kα2 あり")
        self.use_ka2.setChecked(inst.radiation.wavelength2 is not None)
        self.lam2 = _spin(inst.radiation.wavelength2 or 1.5444274, 0.1, 10.0, 6, 0.0001)
        self.ratio2 = _spin(inst.radiation.ratio2 or 0.5, 0.0, 1.0, 3, 0.01)
        row = QHBoxLayout()
        row.addWidget(self.use_ka2)
        row.addWidget(QLabel("λ2 [Å]"))
        row.addWidget(self.lam2)
        row.addWidget(QLabel("強度比"))
        row.addWidget(self.ratio2)
        f_rad.addRow("", row)
        self.use_kb = QCheckBox("Kβ あり(フィルタ漏れ)")
        self.use_kb.setToolTip(
            "Ni フィルタ付き Cu 管などで Kβ が少し残るとき。強度比は 0.005〜0.03 程度"
        )
        self.use_kb.setChecked(inst.radiation.wavelength3 is not None)
        self.lam3 = _spin(inst.radiation.wavelength3 or 1.392225, 0.1, 10.0, 6, 0.0001)
        self.ratio3 = _spin(inst.radiation.ratio3 or 0.01, 0.0, 0.5, 4, 0.001)
        row3 = QHBoxLayout()
        row3.addWidget(self.use_kb)
        row3.addWidget(QLabel("λβ [Å]"))
        row3.addWidget(self.lam3)
        row3.addWidget(QLabel("強度比"))
        row3.addWidget(self.ratio3)
        f_rad.addRow("", row3)
        self.pol = _spin(inst.polarization, 0.0, 1.0, 3, 0.01)
        self.pol.setToolTip(
            "0.5: 無偏光(フィルタ)、1/(1+cos²2θm): 回折側モノクロメータ、1: 放射光(鉛直散乱)"
        )
        f_rad.addRow("偏光係数 P", self.pol)
        left.addWidget(g_rad)

        # プロファイル
        g_prof = QGroupBox(
            "ピーク幅(Caglioti: Γ_G² = U tan²θ + V tanθ + W + P/cos²θ、Γ_L = X tanθ + Y/cosθ + Z)"
        )
        f_prof = QFormLayout(g_prof)
        self.zero = _spin(inst.zero, -5.0, 5.0, 4, 0.001)
        f_prof.addRow("ゼロ点 [deg]", self.zero)
        self.u = _spin(inst.u, -10.0, 10.0, 5, 0.0005)
        self.v = _spin(inst.v, -10.0, 10.0, 5, 0.0005)
        self.w = _spin(inst.w, -1.0, 10.0, 5, 0.0005)
        self.p = _spin(inst.p, -1.0, 10.0, 5, 0.0005)
        self.x = _spin(inst.x, -1.0, 5.0, 4, 0.001)
        self.y = _spin(inst.y, -1.0, 5.0, 4, 0.001)
        self.z = _spin(inst.z, -1.0, 5.0, 4, 0.001)
        for lab, wdg in (
            ("U [deg²]", self.u),
            ("V [deg²]", self.v),
            ("W [deg²]", self.w),
            ("P [deg²]", self.p),
            ("X [deg]", self.x),
            ("Y [deg]", self.y),
            ("Z [deg]", self.z),
        ):
            f_prof.addRow(lab, wdg)
        self.s_l = _spin(inst.s_l, 0.0, 0.2, 4, 0.001)
        self.h_l = _spin(inst.h_l, 0.0, 0.2, 4, 0.001)
        f_prof.addRow("軸発散 S/L", self.s_l)
        f_prof.addRow("軸発散 H/L", self.h_l)
        right.addWidget(g_prof)
        right.addStretch(1)

        # 試料
        g_smp = QGroupBox("試料・光学系")
        f_smp = QFormLayout(g_smp)
        self.geometry = QComboBox()
        self.geometry.addItems(["Bragg-Brentano(反射・平板)", "Debye-Scherrer(透過・キャピラリ)"])
        self.geometry.setCurrentIndex(1 if smp.kind is GeometryKind.DEBYE_SCHERRER else 0)
        f_smp.addRow("光学系", self.geometry)
        self.radius = _spin(smp.radius_mm, 10.0, 2000.0, 1, 1.0)
        f_smp.addRow("ゴニオメータ半径 [mm]", self.radius)
        self.displacement = _spin(smp.displacement_mm, -5.0, 5.0, 4, 0.01)
        f_smp.addRow("試料高さずれ [mm] (BB)", self.displacement)
        self.mu_r = _spin(smp.mu_r, 0.0, 20.0, 3, 0.1)
        f_smp.addRow("円筒試料 μR (DS)", self.mu_r)
        left.addWidget(g_smp)
        left.addStretch(1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        fit_dialog_to_screen(self, self.buttons)

    def apply_preset_by_name(self, name: str) -> None:
        idx = self.preset.findText(name)
        if idx >= 0:
            self.preset.setCurrentIndex(idx)

    def _set_target_from_wavelength(self, wavelength: float, neutron: bool = False) -> None:
        """波長に合うターゲットを選ぶ(合わなければ「その他」)。"""
        chosen = self.target.count() - 1
        if not neutron:
            for i in range(self.target.count()):
                t = self.target.itemData(i)
                if t and abs(CHARACTERISTIC_WAVELENGTHS[t][0] - wavelength) < 2e-4:
                    chosen = i
                    break
        self.target.blockSignals(True)
        self.target.setCurrentIndex(chosen)
        self.target.blockSignals(False)

    def _apply_target(self, idx: int) -> None:
        t = self.target.itemData(idx)
        if not t:
            return
        l1, l2 = CHARACTERISTIC_WAVELENGTHS[str(t)]
        self.kind.setCurrentIndex(0)
        self.lam1.setValue(l1)
        self.lam2.setValue(l2)
        self.use_ka2.setChecked(True)
        if str(t) in KBETA_WAVELENGTHS:
            self.lam3.setValue(KBETA_WAVELENGTHS[str(t)])

    def apply_hints(self, meta: dict[str, object] | None) -> None:
        """データファイルから読めた情報(ターゲット・波長)を反映する。"""
        if not meta:
            return
        target = str(meta.get("target", "") or "").strip().capitalize()
        lam1 = meta.get("wavelength")
        lam2 = meta.get("wavelength2")
        if target in CHARACTERISTIC_WAVELENGTHS:
            i = self.target.findData(target)
            if i >= 0:
                self.target.setCurrentIndex(i)
        if isinstance(lam1, int | float) and lam1 > 0:
            self.kind.setCurrentIndex(0)
            self.lam1.setValue(float(lam1))
            self._set_target_from_wavelength(float(lam1))
        if isinstance(lam2, int | float) and lam2 > 0:
            self.lam2.setValue(float(lam2))
            self.use_ka2.setChecked(True)
            ratio = meta.get("ratio2")
            if isinstance(ratio, int | float) and ratio > 0:
                self.ratio2.setValue(float(ratio))

    def _apply_preset(self, idx: int) -> None:
        if idx <= 0:
            return
        name = self.preset.itemText(idx)
        p = PRESETS[name]
        neutron = bool(p.get("neutron", False))
        self.kind.setCurrentIndex(1 if neutron else 0)
        if "target" in p:
            l1, l2 = CHARACTERISTIC_WAVELENGTHS[str(p["target"])]
            self.lam1.setValue(l1)
            self.lam2.setValue(l2)
            self.use_ka2.setChecked(bool(p.get("kalpha2", True)))
            self.ratio2.setValue(float(p.get("ratio2", 0.5)))
            self.use_kb.setChecked(bool(p.get("kbeta", False)))
            if str(p["target"]) in KBETA_WAVELENGTHS:
                self.lam3.setValue(KBETA_WAVELENGTHS[str(p["target"])])
            self._set_target_from_wavelength(l1)
        else:
            self.lam1.setValue(float(p["wavelength"]))
            self.use_ka2.setChecked(False)
            self.use_kb.setChecked(False)
            self._set_target_from_wavelength(float(p["wavelength"]), neutron)
        self.pol.setValue(float(p["polarization"]))
        for key in ("u", "v", "w", "x", "y", "s_l", "h_l"):
            getattr(self, key).setValue(float(p[key]))
        self.geometry.setCurrentIndex(1 if p["geometry"] is GeometryKind.DEBYE_SCHERRER else 0)

    def result_instrument(self) -> tuple[CWInstrument, SampleGeometry]:
        if self.kind.currentIndex() == 1:
            rad = Radiation.neutron(self.lam1.value())
        else:
            rad = Radiation.xray(
                self.lam1.value(),
                self.lam2.value() if self.use_ka2.isChecked() else None,
                self.ratio2.value(),
                self.lam3.value() if self.use_kb.isChecked() else None,
                self.ratio3.value(),
            )
        inst = CWInstrument(
            radiation=rad, zero=self.zero.value(), u=self.u.value(), v=self.v.value(), w=self.w.value(),
            p=self.p.value(), x=self.x.value(), y=self.y.value(), z=self.z.value(),
            s_l=self.s_l.value(), h_l=self.h_l.value(), polarization=self.pol.value(),
        )  # fmt: skip
        smp = SampleGeometry(
            kind=GeometryKind.DEBYE_SCHERRER
            if self.geometry.currentIndex() == 1
            else GeometryKind.BRAGG_BRENTANO,
            radius_mm=self.radius.value(),
            displacement_mm=self.displacement.value(),
            mu_r=self.mu_r.value(),
        )
        return inst, smp


class RecipeDialog(QDialog):
    """標準レシピのオプションを選ぶ。"""

    def __init__(self, phase_names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("標準の手順で精密化")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "推奨される順序でパラメータを少しずつ増やしながら精密化します:\n"
            "1. スケールと背景 → 2. 格子定数とピーク位置 → 3. ピーク幅 → 4. 原子座標 → 5. 変位パラメータ"
        ))  # fmt: skip
        form = QFormLayout()
        self.phase = QComboBox()
        self.phase.addItem("すべての相", "*")
        for n in phase_names:
            self.phase.addItem(n, n)
        form.addRow("対象の相", self.phase)
        self.coords = QCheckBox("原子座標を動かす")
        self.coords.setChecked(True)
        self.adp = QCheckBox("変位パラメータ(U)を動かす")
        self.adp.setChecked(True)
        self.occ = QCheckBox("占有率を動かす(混合サイト・部分占有のサイトだけ)")
        self.disp = QCheckBox(
            "ゼロ点ではなく試料変位を動かす(平板: 高さずれ、キャピラリ: x, y ずれ)"
        )
        self.asym = QCheckBox("ピーク非対称 (S/L, H/L) を動かす")
        self.po = QCheckBox("選択配向 (March-Dollase) を動かす")
        self.rough = QCheckBox(
            "表面粗さを動かす(吸収の強い試料で U が負になるとき。Bragg-Brentano)"
        )
        for w in (self.coords, self.adp, self.occ, self.disp, self.asym, self.po, self.rough):
            form.addRow(w)
        self.cycles = QSpinBox()
        self.cycles.setRange(1, 200)
        self.cycles.setValue(15)
        form.addRow("各段階の最大反復回数", self.cycles)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def recipe(self) -> Recipe:
        r = standard_recipe(
            str(self.phase.currentData()),
            "h*",
            coordinates=self.coords.isChecked(),
            adp=self.adp.isChecked(),
            occupancy=self.occ.isChecked(),
            sample_displacement=self.disp.isChecked(),
            asymmetry=self.asym.isChecked(),
            preferred_orientation=self.po.isChecked(),
            roughness=self.rough.isChecked(),
        )
        for s in r.steps:
            s.max_cycles = (
                max(self.cycles.value(), 5)
                if s.name != "最終(全パラメータ)"
                else self.cycles.value() * 2
            )
        return r


class PhaseNameDialog(QDialog):
    def __init__(self, default: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("相の名前")
        layout = QFormLayout(self)
        self.name = QLineEdit(default)
        layout.addRow("名前(':' は使えません)", self.name)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def value(self) -> str:
        return self.name.text().strip().replace(":", "_")


__all__ = ["PRESETS", "InstrumentDialog", "PhaseNameDialog", "RecipeDialog", "replace"]
