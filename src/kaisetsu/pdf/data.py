"""PDF データ G(r) の入れ物と読み込み。

対応形式:
    - PDFgetX2 / PDFgetN の .gr(ヘッダの後に ``#### start data``、``#L r G(r) dr dG(r)``)
    - PDFgetX3 / xPDFsuite の .gr(``[PDFgetX3]`` ヘッダ、``#### start data``、2 列)
    - 2〜4 列のテキスト(r, G[, dr][, dG])。'#' で始まる行は読み飛ばす
ヘッダから qmax, qmin, qdamp, qbroad, composition, wavelength を拾えれば metadata に入れる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_SPLIT = re.compile(r"[\s,;]+")
_KEYS = ("qmax", "qmin", "qdamp", "qbroad", "rpoly", "rstep", "rmin", "rmax", "bgscale")
_KV = re.compile(r"(?i)\b(" + "|".join(_KEYS) + r"|wavelength|composition)\s*=\s*([^\s,;#]+)")


@dataclass
class PDFData:
    """観測 PDF。

    Attributes:
        r: 距離 r [Å](昇順、等間隔が望ましい)
        g: G(r) [Å^-2]
        dg: G(r) の標準偏差(無ければ None。重みは 1 になる)
        name: 表示名
        metadata: 読み込み元の情報(qmax 等)
    """

    r: FloatArray
    g: FloatArray
    dg: FloatArray | None = None
    name: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.r = np.asarray(self.r, dtype=np.float64).ravel()
        self.g = np.asarray(self.g, dtype=np.float64).ravel()
        if self.r.shape != self.g.shape:
            raise ValueError("r と G(r) の長さが一致しません")
        if len(self.r) < 4:
            raise ValueError("PDF のデータ点が少なすぎます")
        if np.any(np.diff(self.r) <= 0):
            raise ValueError("r は単調増加でなければなりません")
        if self.dg is not None:
            self.dg = np.asarray(self.dg, dtype=np.float64).ravel()
            if self.dg.shape != self.r.shape:
                raise ValueError("dG の長さが r と一致しません")
            if not np.any(self.dg > 0):
                self.dg = None

    def __len__(self) -> int:
        return len(self.r)

    @property
    def step(self) -> float:
        return float((self.r[-1] - self.r[0]) / (len(self.r) - 1))

    @property
    def weight(self) -> FloatArray:
        """最小二乗の重み。dG があれば 1/dG²、無ければ 1。"""
        if self.dg is None:
            return np.ones_like(self.r)
        with np.errstate(divide="ignore"):
            return np.asarray(np.where(self.dg > 0, 1.0 / np.maximum(self.dg, 1e-300) ** 2, 0.0))

    def mask_range(self, r_min: float | None = None, r_max: float | None = None) -> BoolArray:
        m = np.ones(len(self.r), dtype=bool)
        if r_min is not None:
            m &= self.r >= r_min
        if r_max is not None:
            m &= self.r <= r_max
        return m

    def cropped(self, r_min: float | None = None, r_max: float | None = None) -> PDFData:
        m = self.mask_range(r_min, r_max)
        return PDFData(
            self.r[m],
            self.g[m],
            None if self.dg is None else self.dg[m],
            self.name,
            dict(self.metadata),
        )


def _parse_header(lines: list[str]) -> dict[str, object]:
    meta: dict[str, object] = {}
    for line in lines:
        for key, val in _KV.findall(line):
            k = key.lower()
            if k in meta:
                continue
            try:
                meta[k] = float(val)
            except ValueError:
                meta[k] = val
    return meta


def read_pdf_data(path: str | Path) -> PDFData:
    """G(r) ファイルを読む。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF データファイルが見つかりません: {p}")
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] = []
    columns: list[str] = []
    blocks: list[list[list[float]]] = [[]]
    after_marker: list[list[float]] = []
    in_data = False
    for raw in text:
        line = raw.strip()
        if not line:
            blocks.append([])
            continue
        if line.startswith("#L"):
            columns = line[2:].split()
            continue
        if line.startswith("#### start data") or line.startswith("##### start data"):
            in_data = True
            continue
        if line[0] in "#!/;'[":
            header.append(line)
            blocks.append([])
            continue
        parts = [t for t in _SPLIT.split(line) if t]
        try:
            vals = [float(t) for t in parts]
        except ValueError:
            header.append(line)
            blocks.append([])
            continue
        if len(vals) < 2:
            blocks.append([])
            continue
        if in_data:
            after_marker.append(vals)
        else:
            blocks[-1].append(vals)
    # ``#### start data`` があればその後だけ、無ければいちばん長い連続した数値ブロックを使う
    rows = after_marker if after_marker else max(blocks, key=len)
    if len(rows) < 4:
        raise ValueError(f"{p.name}: PDF の数値データが読めませんでした")
    ncol = min(len(r) for r in rows)
    arr = np.array([r[:ncol] for r in rows], dtype=np.float64)
    r = arr[:, 0]
    g = arr[:, 1]
    dg: FloatArray | None = None
    lower = [c.lower() for c in columns]
    if ncol >= 4:
        dg = arr[:, 3]
    elif ncol == 3:
        # 3 列目が dr(PDFgetN の古い形式)でなければ dG と見なす
        dg = None if "dr" in lower[2:3] else arr[:, 2]
    if dg is not None and not np.any(dg > 0):
        dg = None
    order = np.argsort(r, kind="stable")
    meta = _parse_header(header)
    meta.update({"format": "gr", "path": str(p)})
    return PDFData(
        r[order], g[order], None if dg is None else dg[order], name=p.stem, metadata=meta
    )
