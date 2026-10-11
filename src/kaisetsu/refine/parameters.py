"""精密化パラメータの集合。

各パラメータは名前(例 'PbSO4:cell:a', 'h0:inst:zero', 'h0:PbSO4:scale')、値、
精密化フラグ、上下限、数値微分の刻み、結果の esd を持ち、`setter` でモデルの
オブジェクトに値を書き込む。

拘束(tie)は「従属パラメータ = Σ c_j × 主パラメータ_j + d」の形の線形関係で、
対称性による座標・U_ij の結び付き(y = x、U22 = U11 など)や、
ユーザー定義の等値拘束(占有率の和など)に使う。従属パラメータは精密化変数にならない。
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass
class Tie:
    """線形拘束: value = Σ coeffs[j] × masters[j] + offset。"""

    masters: tuple[str, ...]
    coeffs: tuple[float, ...]
    offset: float = 0.0

    def __post_init__(self) -> None:
        if len(self.masters) != len(self.coeffs):
            raise ValueError("拘束の主パラメータと係数の数が一致しません")


@dataclass
class Parameter:
    """1 つの精密化パラメータ。

    Attributes:
        name: 一意な名前
        value: 現在値
        vary: 精密化するか
        min, max: 上下限
        step: 数値微分の刻み(絶対値)
        esd: 直近の精密化で得た標準偏差(未精密化なら None)
        setter: 値をモデルに書き込む関数(None なら値を保持するだけ)
        tie: 線形拘束(あれば vary は無視される)
        locked: 真なら対称性等の理由で動かせない(vary を立てても無視)
        unit: 表示用の単位
        description: 表示用の説明(日本語)
    """

    name: str
    value: float
    vary: bool = False
    min: float = -math.inf
    max: float = math.inf
    step: float = 1.0e-4
    esd: float | None = None
    setter: Callable[[float], None] | None = field(default=None, repr=False)
    tie: Tie | None = None
    locked: bool = False
    unit: str = ""
    description: str = ""

    @property
    def is_free(self) -> bool:
        """精密化変数として扱われるか。"""
        return self.vary and self.tie is None and not self.locked

    def apply(self) -> None:
        if self.setter is not None:
            self.setter(self.value)

    def __repr__(self) -> str:
        esd = f" ± {self.esd:.3g}" if self.esd is not None else ""
        flag = "*" if self.is_free else ("=" if self.tie is not None else " ")
        return f"{flag}{self.name} = {self.value:.6g}{esd}"


class ParameterSet:
    """順序付きのパラメータ集合。"""

    def __init__(self, params: Iterable[Parameter] = ()) -> None:
        self._params: dict[str, Parameter] = {}
        for p in params:
            self.add(p)

    # ------------------------------------------------------------------
    # 追加・参照
    # ------------------------------------------------------------------
    def add(self, param: Parameter) -> Parameter:
        if param.name in self._params:
            raise ValueError(f"パラメータ名が重複しています: {param.name}")
        self._params[param.name] = param
        return param

    def __getitem__(self, name: str) -> Parameter:
        try:
            return self._params[name]
        except KeyError:
            raise KeyError(f"パラメータ {name!r} はありません") from None

    def __contains__(self, name: object) -> bool:
        return name in self._params

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    def names(self) -> list[str]:
        return list(self._params)

    def select(self, *patterns: str) -> list[Parameter]:
        """glob パターン(例 'h0:bkg:*', '*:cell:*')に一致するパラメータを返す。"""
        out: list[Parameter] = []
        for p in self._params.values():
            if any(fnmatch.fnmatchcase(p.name, pat) for pat in patterns):
                out.append(p)
        return out

    def remove(self, name: str) -> None:
        del self._params[name]

    # ------------------------------------------------------------------
    # 精密化フラグ
    # ------------------------------------------------------------------
    def vary(self, *patterns: str, value: bool = True) -> list[str]:
        """パターンに一致するパラメータの vary を設定し、対象の名前を返す。"""
        hit = self.select(*patterns)
        if not hit:
            raise KeyError(f"パターン {patterns} に一致するパラメータがありません")
        names = []
        for p in hit:
            if p.tie is None and not p.locked:
                p.vary = value
                names.append(p.name)
        return names

    def fix(self, *patterns: str) -> list[str]:
        return self.vary(*patterns, value=False)

    def fix_all(self) -> None:
        for p in self._params.values():
            p.vary = False

    def free(self) -> list[Parameter]:
        """精密化変数(vary かつ拘束・ロック無し)を順序通りに返す。"""
        return [p for p in self._params.values() if p.is_free]

    def free_names(self) -> list[str]:
        return [p.name for p in self.free()]

    # ------------------------------------------------------------------
    # 拘束
    # ------------------------------------------------------------------
    def tie(
        self,
        name: str,
        masters: str | Iterable[str],
        coeffs: float | Iterable[float] = 1.0,
        offset: float = 0.0,
    ) -> None:
        """name を主パラメータに線形拘束する。"""
        m = (masters,) if isinstance(masters, str) else tuple(masters)
        c = (
            (float(coeffs),) * len(m)
            if isinstance(coeffs, int | float)
            else tuple(float(v) for v in coeffs)
        )
        for mm in m:
            if mm not in self._params:
                raise KeyError(f"主パラメータ {mm!r} はありません")
            if self._params[mm].tie is not None:
                raise ValueError(f"拘束の連鎖はできません: {mm!r} はすでに従属パラメータです")
        p = self[name]
        p.tie = Tie(m, c, offset)
        p.vary = False
        self._evaluate_tie(p)

    def untie(self, name: str) -> None:
        self[name].tie = None

    def _evaluate_tie(self, p: Parameter) -> None:
        assert p.tie is not None
        v = p.tie.offset
        for m, c in zip(p.tie.masters, p.tie.coeffs, strict=True):
            v += c * self._params[m].value
        p.value = v

    def evaluate_ties(self) -> None:
        for p in self._params.values():
            if p.tie is not None:
                self._evaluate_tie(p)

    # ------------------------------------------------------------------
    # ベクトル入出力
    # ------------------------------------------------------------------
    def vector(self) -> FloatArray:
        """精密化変数の値ベクトル。"""
        return np.array([p.value for p in self.free()], dtype=np.float64)

    def bounds(self) -> tuple[FloatArray, FloatArray]:
        fr = self.free()
        return (
            np.array([p.min for p in fr], dtype=np.float64),
            np.array([p.max for p in fr], dtype=np.float64),
        )

    def steps(self) -> FloatArray:
        return np.array([p.step for p in self.free()], dtype=np.float64)

    def set_vector(self, values: npt.ArrayLike, apply: bool = True) -> None:
        """精密化変数に値を入れ、拘束を評価し、モデルに書き込む。"""
        fr = self.free()
        vals = np.asarray(values, dtype=np.float64).ravel()
        if len(vals) != len(fr):
            raise ValueError(f"ベクトル長 {len(vals)} が精密化変数の数 {len(fr)} と一致しません")
        for p, v in zip(fr, vals, strict=True):
            p.value = float(v)
        self.evaluate_ties()
        if apply:
            self.apply_all()

    def apply_all(self) -> None:
        for p in self._params.values():
            p.apply()

    def set_esd(self, esd: npt.ArrayLike) -> None:
        """精密化変数の esd を入れ、拘束されたパラメータへ伝播する。"""
        fr = self.free()
        e = np.asarray(esd, dtype=np.float64).ravel()
        for p in self._params.values():
            p.esd = None
        for p, v in zip(fr, e, strict=True):
            p.esd = float(v)
        for p in self._params.values():
            if p.tie is not None:
                var = 0.0
                for m, c in zip(p.tie.masters, p.tie.coeffs, strict=True):
                    me = self._params[m].esd
                    if me is not None:
                        var += (c * me) ** 2
                p.esd = math.sqrt(var) if var > 0 else None

    # ------------------------------------------------------------------
    def summary(self, only_varied: bool = False) -> str:
        lines = []
        for p in self._params.values():
            if only_varied and not (p.is_free or (p.tie is not None and p.esd)):
                continue
            lines.append(repr(p) + (f"  [{p.unit}]" if p.unit else ""))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ParameterSet({len(self)} params, {len(self.free())} free)"
