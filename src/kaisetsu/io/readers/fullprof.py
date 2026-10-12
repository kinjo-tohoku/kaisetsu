"""FullProf の例題に多い強度データ形式。

FullProf は PCR の INSTRM で形式を指定するが、ここでは中身の構造で判定する:

1. 自由形式(INSTRM = 0 など): 先頭に「開始角 刻み 終了角 [題名]」の行(その前にコメント行や
   空行があってもよい)、以降は強度を 1 行に何個でも。2θ は 開始角 + 刻み × i。
   PSI HRPT や D1A の出力にも同じ構造のものがある(題名 2 行の後に「開始角 刻み 終了角 モニタ」)。
2. 検出器数と強度の組(ILL D1A/D2B 流): 同じ「開始角 刻み 終了角」の行の後に
   "n I n I n I …"(n = その点に寄与した検出器の数、I = 計数の和)。強度は I/n、分散は I/n²。
   古い ILL 形式では題名、「… 刻み …」の行、開始角の行、モニタの行の後に同じ組が続く。

RIETAN-FP の「点数 開始角 刻み」形式は `rietan` モジュール、X Y [σ] の列は `xy` モジュール。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from kaisetsu.pattern.histogram import PowderData

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


def _is_num(tok: str) -> bool:
    return bool(_NUM.match(tok))


def _tokens(line: str) -> list[str]:
    return line.replace(",", " ").split()


def _leading_numbers(line: str) -> list[float]:
    """行頭から連続する数値トークンを返す(題名などの文字列が続いてもよい)。"""
    out: list[float] = []
    for tok in _tokens(line):
        if not _is_num(tok):
            break
        out.append(float(tok.replace("D", "E").replace("d", "e")))
    return out


def _is_comment(line: str) -> bool:
    s = line.strip()
    return not s or s[0] in "#!/;'*"


def _find_header(lines: list[str]) -> int | None:
    """「開始角 刻み 終了角」で始まる行の添字(コメント・空行を除き先頭 6 行以内)。無ければ None。"""
    seen = 0
    for i, line in enumerate(lines[:40]):
        if _is_comment(line):
            continue
        seen += 1
        if seen > 6:
            break
        nums = _leading_numbers(line)
        if len(nums) >= 3 and nums[1] > 0 and nums[2] > nums[0]:
            n = (nums[2] - nums[0]) / nums[1]
            if 5 <= n <= 1_000_000:
                return i
    return None


def _all_numeric_line(line: str) -> bool:
    toks = _tokens(line)
    return bool(toks) and all(_is_num(t) for t in toks)


def _looks_like_pairs(toks: list[str]) -> bool:
    """ "n I n I …" の組か(偶数個で、奇数番目が全部同じ小さな整数)。"""
    if len(toks) < 4 or len(toks) % 2:
        return False
    firsts = toks[0::2]
    if not all(re.fullmatch(r"\d{1,3}", t) for t in firsts):
        return False
    return len(set(firsts)) == 1 and all(_is_num(t) for t in toks[1::2])


def detect_fullprof(head: str) -> str | None:
    """先頭部分から形式を判定する: 'free'、'pairs'、'ill_d1a'、または None。"""
    lines = head.splitlines()
    k = _find_header(lines)
    if k is not None:
        body = [ln for ln in lines[k + 1 : k + 6] if ln.strip()]
        if not body or not all(_all_numeric_line(ln) for ln in body):
            return None
        toks = _tokens(body[0])
        if _looks_like_pairs(toks):
            return "pairs"
        # 2 列(2θ 強度)の X-Y データは対象外(xy モジュールが読む)
        if len(toks) == 2 and abs(float(toks[0]) - _leading_numbers(lines[k])[0]) < 1e-9:
            return None
        return "free"
    # 古い ILL D1A/D2B 形式: 題名、7 個の数(3 番目が刻み)、開始角、モニタ、組
    if len(lines) >= 5:
        nums = _leading_numbers(lines[1])
        if len(nums) >= 6 and 0 < nums[2] < 5 and len(_leading_numbers(lines[2])) == 1:
            if _looks_like_pairs(_tokens(lines[4])):
                return "ill_d1a"
    return None


def _powder(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, name: str, meta: dict[str, object]
) -> PowderData:
    return PowderData(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(w, dtype=float),
        name=name,
        metadata=meta,
    )


def _poisson_weights(y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(y > 0, 1.0 / np.maximum(y, 1e-300), 0.0)


def _collect(lines: list[str]) -> list[str]:
    """数値だけの行からトークンを集める(途中で数値でない行が来たら終わり)。"""
    toks: list[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if not _all_numeric_line(ln):
            break
        toks.extend(_tokens(ln))
    return toks


def read_fullprof(path: str | Path) -> PowderData:
    """FullProf の自由形式 / 組形式 / 古い ILL D1A 形式を読む。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    kind = detect_fullprof(text[:6000])
    if kind is None:
        raise ValueError(f"{p.name}: FullProf の強度データ形式として認識できませんでした")
    meta: dict[str, object] = {"format": f"fullprof-{kind}"}
    if kind == "ill_d1a":
        title = lines[0].strip()
        step = _leading_numbers(lines[1])[2]
        start = _leading_numbers(lines[2])[0]
        toks = _collect(lines[4:])
        meta["comment"] = title
    else:
        k = _find_header(lines)
        assert k is not None
        hdr = _leading_numbers(lines[k])
        start, step, end = hdr[0], hdr[1], hdr[2]
        rest = lines[k].split(None, 3)
        if len(rest) > 3 and not _is_num(rest[3].split()[0] if rest[3].split() else ""):
            meta["comment"] = rest[3].strip()
        elif k > 0:
            meta["comment"] = " ".join(ln.strip() for ln in lines[:k] if not _is_comment(ln))
        toks = _collect(lines[k + 1 :])
        n_expected = round((end - start) / step) + 1
        meta["n_expected"] = n_expected
    if kind in ("pairs", "ill_d1a"):
        vals = np.array([float(t) for t in toks], dtype=float)
        if len(vals) % 2:
            vals = vals[:-1]
        n_det = vals[0::2]
        counts = vals[1::2]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(n_det > 0, counts / np.maximum(n_det, 1), 0.0)
            w = np.where((n_det > 0) & (counts > 0), n_det**2 / np.maximum(counts, 1e-300), 0.0)
    else:
        y = np.array([float(t.replace("D", "E").replace("d", "e")) for t in toks], dtype=float)
        if "n_expected" in meta and len(y) > int(meta["n_expected"]):  # type: ignore[call-overload]
            y = y[: int(meta["n_expected"])]  # type: ignore[call-overload]
        w = _poisson_weights(y)
    if len(y) < 2:
        raise ValueError(f"{p.name}: 強度が読めませんでした")
    x = start + step * np.arange(len(y), dtype=float)
    return _powder(x, y, w, p.stem, meta)
