"""ダイアログ: 装置・試料の設定、標準レシピの選択。"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.gui.dialog_utils import fit_dialog_to_screen
from kaisetsu.gui.paths import last_dir, remember_dir
from kaisetsu.pattern.instrument import CWInstrument, GeometryKind, SampleGeometry
from kaisetsu.pattern.presets import PRESETS
from kaisetsu.refine.recipe import Recipe, standard_recipe
from kaisetsu.scattering.radiation import CHARACTERISTIC_WAVELENGTHS, KBETA_WAVELENGTHS, Radiation


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
        self._inst = inst
        self._smp = smp

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("プリセット:"))
        self.preset = QComboBox()
        self.preset.addItem("(変更しない)")
        for name in PRESETS:
            self.preset.addItem(name)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        top.addWidget(self.preset, 1)
        self.btn_load = QPushButton("装置ファイルを読む…")
        self.btn_load.setToolTip(
            "GSAS-II の .instprm、GSAS の .prm / .inst(APS 11-BM、NIST BT-1 などが配っている装置"
            "パラメータ)、Z-Rietveld の .zDiffractometer から波長・ピーク幅・非対称・偏光を取り込みます"
        )
        self.btn_load.clicked.connect(self.load_instrument_file)
        top.addWidget(self.btn_load)
        self.btn_template_load = QPushButton("テンプレートを読む…")
        self.btn_template_load.setToolTip(
            "「テンプレートに保存」で保存した装置と試料形状(*.kaisetsu-inst.json)を読み込みます"
        )
        self.btn_template_load.clicked.connect(self.load_template)
        top.addWidget(self.btn_template_load)
        self.btn_template_save = QPushButton("テンプレートに保存…")
        self.btn_template_save.setToolTip(
            "いまの装置と試料形状をファイルに保存します。標準試料で校正した値を、"
            "別のプロジェクトや「Le Bail で自動解析」で使い回せます"
        )
        self.btn_template_save.clicked.connect(self.save_template)
        top.addWidget(self.btn_template_save)
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
        g_prof = QGroupBox("ピーク幅(Caglioti: Γ_G² = U tan²θ + V tanθ + W、Γ_L = X tanθ + Y/cosθ)")
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
            ("P [deg²](詳細)", self.p),
            ("X [deg]", self.x),
            ("Y [deg]", self.y),
            ("Z [deg](詳細)", self.z),
        ):
            f_prof.addRow(lab, wdg)
        self.p.setToolTip("ガウス幅の 1/cos²θ 項(FullProf 流)。通常は 0 のまま")
        self.z.setToolTip("ローレンツ幅の定数項。Y と相関 −0.996 で、実験室 X 線では不要(通常 0)")
        self.sh_l = _spin(inst.sh_l, 0.0, 0.4, 4, 0.002)
        self.sh_l.setToolTip(
            "Finger-Cox-Jephcoat の軸発散非対称 SH/L = S/L + H/L(GSAS-II の SH/L と同じ)。\n"
            "試料の長さ S と受光スリットの高さ H を試料-検出器距離 L で割った和。実験室 X 線で 0.02〜0.06 程度"
        )
        f_prof.addRow("軸発散 SH/L", self.sh_l)
        self._advanced_rows: list[tuple[QFormLayout, QWidget]] = [
            (f_prof, self.p),
            (f_prof, self.z),
        ]
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
        self.transparency = _spin(smp.transparency_mm, 0.0, 10.0, 3, 0.01)
        self.transparency.setToolTip(
            "透過補正の有効侵入深さ 1/μ_eff [mm]。0 で補正なし。軽い元素の平板試料で"
        )
        f_smp.addRow("透過 1/μ_eff [mm] (BB)", self.transparency)
        self.rough_a = _spin(smp.roughness_a, 0.0, 1.0, 3, 0.01)
        self.rough_a.setToolTip(
            "Suortti の表面粗さ補正。1 で補正なし、小さいほど低角の強度が落ちる。\n"
            "重い元素など吸収の強い平板試料で U_iso が負になるときに動かす(h0:sample:rough_a)"
        )
        f_smp.addRow("表面粗さ a (BB; 1 = なし)", self.rough_a)
        self.rough_b = _spin(smp.roughness_b, 0.01, 10.0, 2, 0.1)
        self.rough_b.setToolTip("表面粗さの角度依存の鋭さ b。a と相関が強いので通常は 1 に固定")
        f_smp.addRow("表面粗さ b(詳細)", self.rough_b)
        self.mu_r = _spin(smp.mu_r, 0.0, 20.0, 3, 0.1)
        f_smp.addRow("円筒試料 μR (DS)", self.mu_r)
        self._advanced_rows.append((f_smp, self.rough_b))
        left.addWidget(g_smp)
        left.addStretch(1)
        self.show_advanced = QCheckBox("詳細な量も表示(P, Z, 表面粗さ b)")
        self.show_advanced.toggled.connect(self._set_advanced_visible)
        self.show_advanced.setChecked(inst.p != 0.0 or inst.z != 0.0 or smp.roughness_b != 1.0)
        self._set_advanced_visible(self.show_advanced.isChecked())
        layout.addWidget(self.show_advanced)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        fit_dialog_to_screen(self, self.buttons)

    def _set_advanced_visible(self, flag: bool) -> None:
        for form, widget in self._advanced_rows:
            form.setRowVisible(widget, flag)

    def load_instrument_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "GSAS / GSAS-II の装置パラメータ",
            last_dir(),
            "装置パラメータ (*.instprm *.prm *.inst *.iparm *.zDiffractometer);;すべて (*)",
        )
        if not path:
            return
        try:
            inst = CWInstrument.from_instrument_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "装置ファイル", f"読めませんでした:\n{exc}")
            return
        remember_dir(path)
        self.apply_instrument(inst)
        self.preset.setCurrentIndex(0)

    def load_template(self) -> None:
        from kaisetsu.io.project import INSTRUMENT_TEMPLATE_FILTER, load_instrument_template

        path, _ = QFileDialog.getOpenFileName(
            self, "装置テンプレート", last_dir(), INSTRUMENT_TEMPLATE_FILTER
        )
        if not path:
            return
        try:
            inst, smp = load_instrument_template(path)
        except Exception as exc:
            QMessageBox.warning(self, "装置テンプレート", f"読めませんでした:\n{exc}")
            return
        if not isinstance(inst, CWInstrument):
            QMessageBox.warning(
                self, "装置テンプレート", "TOF のテンプレートはこのダイアログでは使えません"
            )
            return
        remember_dir(path)
        self.apply_instrument(inst)
        self.apply_sample(smp)
        self.preset.setCurrentIndex(0)

    def save_template(self) -> None:
        from kaisetsu.io.project import (
            INSTRUMENT_TEMPLATE_FILTER,
            INSTRUMENT_TEMPLATE_SUFFIX,
            save_instrument_template,
        )

        path, _ = QFileDialog.getSaveFileName(
            self, "装置テンプレートに保存", last_dir(), INSTRUMENT_TEMPLATE_FILTER
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += INSTRUMENT_TEMPLATE_SUFFIX
        inst, smp = self.result_instrument()
        save_instrument_template(path, inst, smp)
        remember_dir(path)
        QMessageBox.information(self, "装置テンプレート", f"保存しました:\n{path}")

    @staticmethod
    def _keep(spin: QDoubleSpinBox, original: float) -> float:
        """スピンボックスの値が元の値と表示桁の範囲で同じなら、元の値(丸める前)を返す。

        精密化した値(例: U = 0.00123456)は表示桁(5 桁)に丸まって入るので、開いて OK を
        押しただけで値が変わり esd も消えていた(レビュー 2026-10-17 GUI #17)。
        """
        v = float(spin.value())
        return original if abs(v - original) < 0.5 * 10.0 ** (-spin.decimals()) else v

    def apply_sample(self, smp: SampleGeometry) -> None:
        """SampleGeometry の値をウィジェットに反映する。"""
        self._smp = smp
        self.geometry.setCurrentIndex(1 if smp.kind is GeometryKind.DEBYE_SCHERRER else 0)
        self.radius.setValue(smp.radius_mm)
        self.displacement.setValue(smp.displacement_mm)
        self.transparency.setValue(smp.transparency_mm)
        self.rough_a.setValue(smp.roughness_a)
        self.rough_b.setValue(smp.roughness_b)
        self.mu_r.setValue(smp.mu_r)

    def apply_instrument(self, inst: CWInstrument) -> None:
        """CWInstrument の値をウィジェットに反映する(プリセットは触らない)。"""
        self._inst = inst
        rad = inst.radiation
        self.kind.setCurrentIndex(1 if rad.is_neutron else 0)
        self.lam1.setValue(rad.wavelength)
        self.use_ka2.setChecked(rad.wavelength2 is not None)
        if rad.wavelength2 is not None:
            self.lam2.setValue(rad.wavelength2)
            self.ratio2.setValue(rad.ratio2 or 0.5)
        self.use_kb.setChecked(rad.wavelength3 is not None)
        if rad.wavelength3 is not None:
            self.lam3.setValue(rad.wavelength3)
            self.ratio3.setValue(rad.ratio3 or 0.0)
        self._set_target_from_wavelength(rad.wavelength, rad.is_neutron)
        self.pol.setValue(inst.polarization)
        self.zero.setValue(inst.zero)
        for key in ("u", "v", "w", "p", "x", "y", "z", "sh_l"):
            getattr(self, key).setValue(float(getattr(inst, key)))
        if inst.p != 0.0 or inst.z != 0.0:
            self.show_advanced.setChecked(True)

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
        for key in ("u", "v", "w", "x", "y", "sh_l"):
            getattr(self, key).setValue(float(p[key]))
        self.geometry.setCurrentIndex(1 if p["geometry"] is GeometryKind.DEBYE_SCHERRER else 0)

    def result_instrument(self) -> tuple[CWInstrument, SampleGeometry]:
        """ダイアログの値から装置と試料を作る。変えていない欄は元の値(丸める前)を保つ。"""
        o = self._inst
        orad = o.radiation
        keep = self._keep
        lam1 = keep(self.lam1, orad.wavelength)
        if self.kind.currentIndex() == 1:
            rad = Radiation.neutron(lam1)
        else:
            lam2 = (
                keep(self.lam2, orad.wavelength2 if orad.wavelength2 is not None else -1.0)
                if self.use_ka2.isChecked()
                else None
            )
            lam3 = (
                keep(self.lam3, orad.wavelength3 if orad.wavelength3 is not None else -1.0)
                if self.use_kb.isChecked()
                else None
            )
            rad = Radiation.xray(
                lam1,
                lam2,
                keep(self.ratio2, orad.ratio2) if lam2 is not None else 0.0,
                lam3,
                keep(self.ratio3, orad.ratio3) if lam3 is not None else 0.0,
            )
        if rad == orad:
            rad = orad
        sh_l = keep(self.sh_l, o.sh_l)
        s_l, h_l = (o.s_l, o.h_l) if sh_l == o.sh_l else (0.5 * sh_l, 0.5 * sh_l)
        inst = CWInstrument(
            radiation=rad, zero=keep(self.zero, o.zero), u=keep(self.u, o.u), v=keep(self.v, o.v),
            w=keep(self.w, o.w), p=keep(self.p, o.p), x=keep(self.x, o.x), y=keep(self.y, o.y),
            z=keep(self.z, o.z), s_l=s_l, h_l=h_l, polarization=keep(self.pol, o.polarization),
        )  # fmt: skip
        s = self._smp
        smp = replace(
            s,
            kind=GeometryKind.DEBYE_SCHERRER
            if self.geometry.currentIndex() == 1
            else GeometryKind.BRAGG_BRENTANO,
            radius_mm=keep(self.radius, s.radius_mm),
            displacement_mm=keep(self.displacement, s.displacement_mm),
            transparency_mm=keep(self.transparency, s.transparency_mm),
            roughness_a=keep(self.rough_a, s.roughness_a),
            roughness_b=keep(self.rough_b, s.roughness_b),
            mu_r=keep(self.mu_r, s.mu_r),
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
        self.asym = QCheckBox("ピーク非対称 (SH/L) を動かす")
        self.po = QCheckBox("選択配向 (March-Dollase) を動かす")
        self.rough = QCheckBox(
            "表面粗さを動かす(吸収の強い試料で U が負になるとき。Bragg-Brentano)"
        )
        self.ka2 = QCheckBox("Kα2 なしも試して良い方を採る(単色化されたデータかどうか)")
        self.ka2.setChecked(True)
        self.ka2.setToolTip(
            "ピーク幅のあとで Kα2 の強度比を 0 にして精密化し、Rwp が 0.3 ポイント以上下がれば"
            "採用します。Kα2 の無い装置(放射光・中性子)では何もしません"
        )
        self.light = QCheckBox("軽原子の U_iso を共通にする(重原子が散乱を支配するとき)")
        self.light.setChecked(True)
        self.light.setToolTip(
            "重原子が散乱能の 70% 以上を占める相では、軽原子(Z ≤ 10)の U_iso は個別に決まらず"
            "下限に張り付きます。共通の 1 つにして動かします"
        )
        self.humps = QCheckBox("残差の広いこぶを背景のハンプにして試す(ガラスキャピラリ・非晶質)")
        self.humps.setChecked(True)
        self.humps.setToolTip(
            "ピーク幅のあとで残差に幅の広い山が残っていれば、背景にガウスのハンプを足して精密化し、"
            "Rwp が 0.5 ポイント以上下がれば採用します。「背景の設定」で足したハンプは"
            "「スケールと背景」で背景と一緒に動きます"
        )
        self.cell_scan = QCheckBox("格子定数の初期値を ±1% 振って、ピークが重なる谷から始める")
        self.cell_scan.setChecked(True)
        self.cell_scan.setToolTip(
            "最初に格子定数を ±1% の範囲で細かく振り(刻みは高角端のピーク幅)、計算ピークが"
            "観測ピークに重なる初期値から始めます。初期値が谷の外だと最小二乗は格子定数を"
            "動かせません(放射光級の鋭いピークでは 0.6% のずれで止まり、3% で発散します)。"
            "数秒で済みます"
        )
        for w in (
            self.cell_scan,
            self.coords,
            self.adp,
            self.occ,
            self.disp,
            self.asym,
            self.po,
            self.rough,
            self.ka2,
            self.light,
            self.humps,
        ):
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
            asymmetry=True if self.asym.isChecked() else None,
            preferred_orientation=self.po.isChecked(),
            roughness=self.rough.isChecked(),
            try_no_kalpha2=self.ka2.isChecked(),
            light_atom_uiso=self.light.isChecked(),
            humps=self.humps.isChecked(),
            cell_scan=self.cell_scan.isChecked(),
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
