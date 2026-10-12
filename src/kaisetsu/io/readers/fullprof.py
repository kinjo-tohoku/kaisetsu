"""FullProf の例題に多い強度データ形式。

FullProf は PCR の INSTRM で形式を指定するが、ここでは中身の構造で判定する:

1. 自由形式(INSTRM = 0 など): 先頭に「開始角 刻み 終了角 [題名]」の行(その前にコメント行や
   空行、題名行があってもよい)、以降は強度を 1 行に何個でも。2θ は 開始角 + 刻み × i。
   PSI HRPT や D1A の出力にも同じ構造のものがある(題名 2 行の後に「開始角 刻み 終了角 モニタ」)。
2. 検出器数と強度の組(ILL D1A/D2B 流): 同じ「開始角 刻み 終了角」の行の後に、Fortran の
   (10(I2, I6)) で "n I n I …"(n = その点に寄与した検出器の数、I = 計数の和)。固定桁なので
   n が 2 桁になると空白で区切れない。強度は I のまま、分散は I/n(GSAS の STD と同じ)。
   負の値は終端記号。
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


def _pairs_in_line(line: str) -> list[tuple[int, int]] | None:
    """(I2, I6) の固定桁で "n I" の組を読む。形が合わなければ None。"""
    s = line.rstrip("\r\n")
    if not s.strip() or len(s) % 8 or len(s) > 80:
        return None
    out: list[tuple[int, int]] = []
    for i in range(0, len(s), 8):
        a, b = s[i : i + 2].strip(), s[i + 2 : i + 8].strip()
        if not a and not b:
            break
        try:
            out.append((int(a or 0), int(b or 0)))
        except ValueError:
            return None
    return out or None


def _looks_like_pairs(line: str) -> bool:
    """ "n I n I …" の組か(固定桁で読めて、n が全部同じ小さな非負整数)。"""
    pairs = _pairs_in_line(line)
    if not pairs or len(pairs) < 2:
        return False
    ns = {n for n, _ in pairs}
    return len(ns) == 1 and 0 <= next(iter(ns)) <= 99


def detect_fullprof(head: str) -> str | None:
    """先頭部分から形式を判定する: 'free'、'pairs'、'ill_d1a'、または None。"""
    lines = head.splitlines()
    k = _find_header(lines)
    if k is not None:
        body = [ln for ln in lines[k + 1 : k + 6] if ln.strip()]
        if not body or not all(_all_numeric_line(ln) for ln in body):
            return None
        if _looks_like_pairs(body[0]):
            return "pairs"
        toks = _tokens(body[0])
        # 2 列(2θ 強度)の X-Y データは対象外(xy モジュールが読む)
        if len(toks) == 2 and abs(float(toks[0]) - _leading_numbers(lines[k])[0]) < 1e-9:
            return None
        return "free"
    # 古い ILL D1A/D2B 形式: 題名、7 個の数(3 番目が刻み)、開始角、モニタ、組
    if len(lines) >= 5:
        nums = _leading_numbers(lines[1])
        if len(nums) >= 6 and 0 < nums[2] < 5 and len(_leading_numbers(lines[2])) == 1:
            if _looks_like_pairs(lines[4]):
                return "ill_d1a"
    return None


def _poisson_weights(y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(y > 0, 1.0 / np.maximum(y, 1e-300), 0.0)


def _collect_free(lines: list[str]) -> list[float]:
    """数値だけの行から強度を集める(数値でない行が来たら終わり)。"""
    vals: list[float] = []
    for ln in lines:
        if not ln.strip():
            continue
        if not _all_numeric_line(ln):
            break
        vals.extend(float(t.replace("D", "E").replace("d", "e")) for t in _tokens(ln))
    return vals


def _collect_pairs(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(I2, I6) の組を集める。負の値(終端記号)か形の合わない行で終わり。"""
    n_det: list[int] = []
    counts: list[int] = []
    for ln in lines:
        if not ln.strip():
            continue
        pairs = _pairs_in_line(ln)
        if pairs is None:
            break
        stop = False
        for n, c in pairs:
            if n < 0 or c < 0:
                stop = True
                break
            n_det.append(n)
            counts.append(c)
        if stop:
            break
    return np.array(n_det, dtype=float), np.array(counts, dtype=float)


def read_fullprof(path: str | Path) -> PowderData:
    """FullProf の自由形式 / 組形式 / 古い ILL D1A 形式を読む。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    kind = detect_fullprof(text[:6000])
    if kind is None:
        raise ValueError(f"{p.name}: FullProf の強度データ形式として認識できませんでした")
    meta: dict[str, object] = {"format": f"fullprof-{kind}"}
    n_expected: int | None = None
    if kind == "ill_d1a":
        meta["comment"] = lines[0].strip()
        step = _leading_numbers(lines[1])[2]
        start = _leading_numbers(lines[2])[0]
        body_lines = lines[4:]
    else:
        k = _find_header(lines)
        assert k is not None
        hdr = _leading_numbers(lines[k])
        start, step, end = hdr[0], hdr[1], hdr[2]
        rest = lines[k].split(None, 3)
        tail = rest[3].strip() if len(rest) > 3 else ""
        if tail and not _is_num(tail.split()[0]):
            meta["comment"] = tail
        elif k > 0:
            meta["comment"] = " ".join(ln.strip() for ln in lines[:k] if not _is_comment(ln))
        n_expected = round((end - start) / step) + 1
        meta["n_expected"] = n_expected
        body_lines = lines[k + 1 :]
    if kind in ("pairs", "ill_d1a"):
        n_det, counts = _collect_pairs(body_lines)
        # 強度 I はそのまま(GSAS の STD 形式と同じ: 同じ測定の PBSO4.CWN と一致する)。
        # n は分散の重み付けにだけ使う(分散 = I / n)。I/n に直すと Rwp が 4% → 14% に悪化した
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(n_det > 0, counts, 0.0)
            w = np.where((n_det > 0) & (counts > 0), n_det / np.maximum(counts, 1e-300), 0.0)
    else:
        y = np.array(_collect_free(body_lines), dtype=float)
        w = _poisson_weights(y)
    if n_expected is not None and len(y) > n_expected:
        y, w = y[:n_expected], w[:n_expected]
    if len(y) < 2:
        raise ValueError(f"{p.name}: 強度が読めませんでした")
    x = start + step * np.arange(len(y), dtype=float)
    return PowderData(
        x, np.asarray(y, dtype=float), np.asarray(w, dtype=float), name=p.stem, metadata=meta
    )
