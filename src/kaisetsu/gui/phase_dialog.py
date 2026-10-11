"""相の設定ダイアログ: 強度の扱い(構造モデル / Le Bail / Pawley)、選択配向の軸、サイズ・歪み、
サイトの異方性変位パラメータへの切り替え、Brindley 微小吸収補正の粒子半径。"""

from __future__ import annotations

import numpy as np
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from kaisetsu.refine.project import Project


def _spin(value: float, lo: float, hi: float, decimals: int, step: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    return s


class PhaseDialog(QDialog):
    """1 つの相の設定を編集する。apply() でプロジェクトに反映する。"""

    def __init__(
        self, project: Project, phase: str, hist: int, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.phase = phase
        self.hist = hist
        ph = project.phases[phase]
        self.setWindowTitle(f"相の設定: {phase}")
        layout = QVBoxLayout(self)

        mode = QGroupBox("反射強度の扱い")
        mf = QFormLayout(mode)
        self.mode = QComboBox()
        self.mode.addItem("構造モデルから計算(Rietveld)", "structure")
        self.mode.addItem("Le Bail 法(構造を使わず強度を抽出)", "le_bail")
        self.mode.addItem("Pawley 法(強度を最小二乗で解く。σ と相関が出る)", "pawley")
        self.mode.setCurrentIndex(
            {"structure": 0, "le_bail": 1, "pawley": 2}[
                "pawley" if ph.pawley else ("le_bail" if ph.le_bail else "structure")
            ]
        )
        mf.addRow("モード", self.mode)
        self.reset_table = QCheckBox("抽出強度を初期化し直す(構造モデルから。構造が無ければ一様)")
        mf.addRow(self.reset_table)
        mf.addRow(QLabel(
            "Le Bail / Pawley は格子定数とプロファイルを先に決めるときや、\n"
            "未知構造の強度抽出(MEM の入力)に使います。"
        ))  # fmt: skip
        layout.addWidget(mode)

        pc = project.histograms[hist].contributions[phase] if project.histograms else None
        po = QGroupBox(f"選択配向とサイズ・歪み(ヒストグラム h{hist})")
        pf = QFormLayout(po)
        axis = pc.march_dollase_axis if pc is not None else (0, 0, 1)
        self.axis = [QSpinBox() for _ in range(3)]
        row = QHBoxLayout()
        for sb, v, lab in zip(self.axis, axis, ("h", "k", "l"), strict=True):
            sb.setRange(-20, 20)
            sb.setValue(int(v))
            row.addWidget(QLabel(lab))
            row.addWidget(sb)
        row.addStretch(1)
        pf.addRow("March-Dollase の配向軸 (hkl)", row)
        pf.addRow(
            QLabel("板状結晶なら劈開面の法線(例 001)、針状なら伸びる方向。r は h0:相:md_r で動かす")
        )
        self.size_lf = _spin(pc.size_lorentz_fraction if pc else 1.0, 0.0, 1.0, 2, 0.1)
        self.strain_lf = _spin(pc.strain_lorentz_fraction if pc else 1.0, 0.0, 1.0, 2, 0.1)
        pf.addRow("サイズ広がりのローレンツ成分の割合", self.size_lf)
        pf.addRow("歪み広がりのローレンツ成分の割合", self.strain_lf)
        po.setEnabled(pc is not None)
        layout.addWidget(po)

        br = QGroupBox("Brindley 微小吸収補正(重量分率の補正)")
        bf = QFormLayout(br)
        self.radius = _spin(ph.brindley_radius_um, 0.0, 1000.0, 2, 1.0)
        self.radius.setToolTip(
            "粒子半径 [μm]。0 で補正なし。重い相と軽い相の混合で分率が系統的にずれるときに使う"
        )
        bf.addRow("粒子半径 [μm]", self.radius)
        layout.addWidget(br)

        adp = QGroupBox("サイトの変位パラメータ")
        af = QVBoxLayout(adp)
        af.addWidget(
            QLabel("チェックしたサイトを異方性(U11〜U23)にします。外すと等方性(U_iso)に戻します。")
        )
        self.site_boxes: dict[str, QCheckBox] = {}
        for site in ph.structure.sites:
            cb = QCheckBox(f"{site.label} ({site.species})")
            cb.setChecked(site.u_aniso is not None)
            self.site_boxes[site.label] = cb
            af.addWidget(cb)
        layout.addWidget(adp)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self) -> list[str]:
        """設定をプロジェクトに反映し、ログに出す行を返す。"""
        proj = self.project
        ph = proj.phases[self.phase]
        notes: list[str] = []
        mode = str(self.mode.currentData())
        was = "pawley" if ph.pawley else ("le_bail" if ph.le_bail else "structure")
        if mode != was or self.reset_table.isChecked():
            if mode == "structure":
                ph.le_bail = False
                ph.pawley = False
                ph.f2_update = True
            elif mode == "le_bail":
                ph.f2_update = True
                if self.reset_table.isChecked() or not ph.f2_tables:
                    proj.start_le_bail(self.phase, from_structure=bool(ph.structure.sites))
                else:
                    ph.le_bail, ph.pawley = True, False
            else:
                if self.reset_table.isChecked() or not ph.f2_tables:
                    proj.start_pawley(self.phase, from_structure=bool(ph.structure.sites))
                else:
                    ph.pawley, ph.le_bail = True, False
            notes.append(f"{self.phase}: 強度の扱いを {self.mode.currentText()} にしました")
        if proj.histograms:
            pc = proj.histograms[self.hist].contributions[self.phase]
            axis = tuple(int(sb.value()) for sb in self.axis)
            if axis == (0, 0, 0):
                axis = (0, 0, 1)
            if axis != pc.march_dollase_axis:
                pc.march_dollase_axis = axis
                notes.append(f"{self.phase}: 選択配向の軸を {axis} にしました")
            pc.size_lorentz_fraction = float(self.size_lf.value())
            pc.strain_lorentz_fraction = float(self.strain_lf.value())
        if abs(ph.brindley_radius_um - float(self.radius.value())) > 1e-9:
            ph.brindley_radius_um = float(self.radius.value())
            notes.append(
                f"{self.phase}: Brindley 補正の粒子半径を {ph.brindley_radius_um:g} μm にしました"
            )
        changed = False
        for site in ph.structure.sites:
            want = self.site_boxes[site.label].isChecked()
            if want and site.u_aniso is None:
                u = site.u_iso if site.u_iso > 0 else 0.01
                site.u_aniso = np.array([u, u, u, 0.0, 0.0, 0.0])
                changed = True
                notes.append(f"{site.label} を異方性にしました(U11 = U22 = U33 = {u:g})")
            elif not want and site.u_aniso is not None:
                site.u_iso = float(np.mean(site.u_aniso[:3]))
                site.u_aniso = None
                changed = True
                notes.append(f"{site.label} を等方性に戻しました(U_iso = {site.u_iso:.4f})")
        if changed or notes:
            proj.build_parameters()
        return notes


__all__ = ["PhaseDialog"]
