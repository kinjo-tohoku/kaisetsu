"""CIF の読み書き(gemmi を利用)。

読み込みの方針:
    - 対称操作は CIF に明示されていればそれを優先する(設定・原点の取り違えを防ぐ)。
      無ければ H-M 記号、Hall 記号、国際番号の順に解釈する。原点選択が 2 つある群
      (Fd-3m、Pnnn、I41/amd など)で原点が書かれていなければ両方を試し、CIF の多重度
      (`_atom_site_symmetry_multiplicity`)に合う方、無ければ最短原子間距離が妥当な方を選ぶ。
    - ADP は U_iso / U_ij を使う。B_iso しか無い場合は gemmi が U に換算する。
    - 座標の '1/3' のような分数は小数に直して受け付け、'?' は誤りとして止める。
    - 読み込んだ後に CIF の多重度と計算した多重度を照合し、合わなければ警告する。
      対称等価な像どうしが 0.3 Å 以内に残っている(特殊位置から僅かにずれた座標)ときも警告する。

警告は `logging`(logger 名 `kaisetsu.crystal.cif`)に出し、`read_structure(..., warnings=[])`
のようにリストを渡せば利用者向けの日本語メッセージとして受け取れる。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np

from kaisetsu.crystal.atoms import AtomSite, normalize_species
from kaisetsu.crystal.lattice import Lattice
from kaisetsu.crystal.spacegroup import SpaceGroup
from kaisetsu.crystal.structure import Structure

log = logging.getLogger(__name__)

#: 原点選択が 2 つある空間群の国際番号(ITA の origin choice 1 / 2)
ORIGIN_CHOICE_NUMBERS = frozenset(
    {48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134, 137, 138, 141, 142}
    | {201, 203, 222, 224, 227, 228}
)
#: 原点選択が CIF に無いときに使う設定(現代の文献・ICSD の既定は原点選択 2)
DEFAULT_ORIGIN_CHOICE = "2"
#: ICSD 流の記号の末尾(例 'F d -3 m Z' = 原点選択 2、'S' = 原点選択 1、'R -3 c H' = 六方軸)
_ICSD_SETTING_SUFFIX = {"S": "1", "Z": "2", "H": "H", "R": "R"}
#: CIF の多重度と合わないサイトを特殊位置へ乗せ直すときの許容 [Å](2 桁の座標 0.33 でも届く)
SNAP_TOL = 0.1
#: 対称等価な像どうしがこれより近ければ警告する [Å]
OVERLAP_WARN_DISTANCE = 0.3
#: 原点選択を最短原子間距離で決めるとき、この比より差が無ければ既定の設定にする
_DISTANCE_DECISION_RATIO = 1.2

_FRACTION_RE = re.compile(r"^\s*([+-]?\d+)\s*/\s*(\d+)\s*$")


class _Notes:
    """読み込み中の注意を logging と呼び出し側のリストの両方に流す。"""

    def __init__(self, fname: str, out: list[str] | None) -> None:
        self.fname = fname
        self.out = out

    def warn(self, message: str) -> None:
        text = f"CIF {self.fname}: {message}"
        log.warning(text)
        if self.out is not None:
            self.out.append(text)


def read_structure(
    path: str | Path, block: str | None = None, *, warnings: list[str] | None = None
) -> Structure:
    """CIF ファイルから結晶構造を読む。

    Args:
        path: CIF ファイル
        block: 読み込むデータブロック名。None なら最初のブロック。
        warnings: 与えると、読み込み中の注意(原点選択の推定、分数座標の解釈、ラベルの付け直し、
            多重度の不一致、特殊位置のずれ)を日本語の文字列として追加する。logging にも出る。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CIF ファイルが見つかりません: {path}")
    doc = _read_document(path)
    cif_block = doc[0] if block is None else doc.find_block(block)
    if cif_block is None:
        raise ValueError(f"CIF にブロック {block!r} がありません")
    st = gemmi.make_small_structure_from_block(cif_block)
    notes = _Notes(path.name, warnings)

    cell = st.cell
    if cell.a <= 0 or cell.b <= 0 or cell.c <= 0:
        raise ValueError(f"CIF {path.name}: 格子定数が読めません")
    lattice = Lattice(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)

    raw = _RawSites.from_block(cif_block, len(st.sites))
    sites = _read_sites(st, cif_block, raw, notes)
    if not sites:
        raise ValueError(f"CIF {path.name}: 原子サイトがありません")
    _rename_duplicate_labels(sites, notes)

    spacegroup = _resolve_spacegroup(st, lattice, sites, raw.multiplicity, notes)
    _check_multiplicities(spacegroup, lattice, sites, raw.multiplicity, notes)

    name = cif_block.name
    for tag in ("_pd_phase_name", "_chemical_name_mineral", "_chemical_name_common"):
        v = cif_block.find_value(tag)
        if v:
            name = gemmi.cif.as_string(v)
            break
    structure = Structure(lattice, spacegroup, sites, name=name)
    _check_overlapping_images(structure, notes)
    return structure


class _RawSites:
    """原子サイトのループの生の文字列(gemmi が数値にできない '1/3' や '?' を見るため)。

    行の対応は、gemmi のサイト数と行数が一致すれば順番で、そうでなければラベルで取る。
    """

    def __init__(self, rows: list[dict[str, str]], by_index: bool) -> None:
        self.rows = rows
        self.by_index = by_index

    @classmethod
    def from_block(cls, block: gemmi.cif.Block, n_sites: int) -> _RawSites:
        tags = [
            "label",
            "?fract_x",
            "?fract_y",
            "?fract_z",
            "?occupancy",
            "?symmetry_multiplicity",
            "?site_symmetry_multiplicity",
        ]
        table = block.find("_atom_site_", tags)
        rows: list[dict[str, str]] = []
        if table:
            for row in table:
                d: dict[str, str] = {}
                for k, tag in enumerate(tags):
                    if row.has(k):
                        d[tag.lstrip("?")] = row[k]
                rows.append(d)
        return cls(rows, by_index=len(rows) == n_sites)

    def row_for(self, index: int, label: str) -> dict[str, str]:
        if self.by_index:
            return self.rows[index]
        for r in self.rows:
            if r.get("label") == label:
                return r
        return {}

    @property
    def multiplicity(self) -> list[int | None]:
        """CIF に書かれたサイト多重度(無ければ None)。順番は rows と同じ。"""
        out: list[int | None] = []
        for r in self.rows:
            v = r.get("symmetry_multiplicity") or r.get("site_symmetry_multiplicity")
            out.append(_as_int(v) if v is not None else None)
        return out


def _as_int(text: str) -> int | None:
    try:
        v = float(gemmi.cif.as_string(text))
    except (ValueError, RuntimeError):
        return None
    if not math.isfinite(v) or v <= 0 or v != round(v):
        return None
    return round(v)


def _read_sites(
    st: gemmi.SmallStructure, block: gemmi.cif.Block, raw: _RawSites, notes: _Notes
) -> list[AtomSite]:
    # _atom_site_type_symbol が無い CIF(AMCSD 由来の COD エントリに多い)では
    # gemmi がラベルを type_symbol に入れるので、元素記号に落として使う
    has_type_symbol = bool(block.find_values("_atom_site_type_symbol"))
    sites: list[AtomSite] = []
    for i, s in enumerate(st.sites):
        row = raw.row_for(i, s.label)
        species = _species_of(s, has_type_symbol, notes.fname)
        aniso = None
        if s.aniso.nonzero():
            aniso = np.array(
                [s.aniso.u11, s.aniso.u22, s.aniso.u33, s.aniso.u12, s.aniso.u13, s.aniso.u23]
            )
        pos = np.array([s.fract.x, s.fract.y, s.fract.z], dtype=np.float64)
        for k, axis in enumerate("xyz"):
            if not math.isfinite(pos[k]):
                pos[k] = _coordinate_from_text(row.get(f"fract_{axis}"), s.label, axis, notes)
        # gemmi は占有率の列が無いときと ? や . のとき(そして '1/2' のような分数でも)1.0 を
        # 返し、明示された 0 は 0.0、負の値はそのまま返す。0 は「無いサイト」として正当な値なので
        # 保つ(レビュー 2026-10-17 #4)。分数は生の文字列から読む
        occ_frac = _parse_fraction(row.get("occupancy", ""))
        if occ_frac is not None:
            notes.warn(
                f"サイト {s.label} の占有率 {row['occupancy']!r} を {occ_frac:.6f} と解釈しました"
            )
            occ = occ_frac
        else:
            occ = float(s.occ) if math.isfinite(s.occ) else 1.0
        if occ < 0.0:
            raise ValueError(
                f"CIF {notes.fname}: サイト {s.label} の占有率が負です({occ:g})。"
                "0 以上 1 以下に直してください"
            )
        try:
            site = AtomSite(
                label=s.label,
                species=species,
                position=pos,
                occupancy=min(occ, 1.0),
                u_iso=float(s.u_iso) if math.isfinite(s.u_iso) else 0.0,
                u_aniso=aniso,
            )
        except ValueError as exc:
            raise ValueError(f"CIF {notes.fname}: {exc}") from exc
        sites.append(site)
    return sites


def _parse_fraction(text: str) -> float | None:
    """'1/3' のような単純な分数を小数にする。分数でなければ None。"""
    m = _FRACTION_RE.match(text)
    if m is None or int(m.group(2)) == 0:
        return None
    return int(m.group(1)) / int(m.group(2))


def _coordinate_from_text(text: str | None, label: str, axis: str, notes: _Notes) -> float:
    """gemmi が数値にできなかった座標の文字列を解釈する。分数は受け付け、'?' は誤り。"""
    shown = "?" if text is None else text.strip()
    if text is not None:
        frac = _parse_fraction(text)
        if frac is not None:
            notes.warn(f"サイト {label} の座標 {axis} = {shown!r} を {frac:.6f} と解釈しました")
            return frac
    raise ValueError(
        f"CIF {notes.fname}: サイト {label} の座標 {axis} が {shown!r} で数値になっていません。"
        "'1/3' や '?' ではなく 0.333333 のように小数で書いてください"
    )


def _rename_duplicate_labels(sites: list[AtomSite], notes: _Notes) -> None:
    """重複したラベルを 'O_1', 'O_2' のように付け直す(手書きの CIF で 'O O O' のように起きる)。"""
    counts = Counter(s.label for s in sites)
    taken = set(counts)
    for label, n in counts.items():
        if n < 2:
            continue
        renamed: list[str] = []
        k = 0
        for s in sites:
            if s.label != label:
                continue
            k += 1
            new = f"{label}_{k}"
            while new in taken:
                k += 1
                new = f"{label}_{k}"
            taken.add(new)
            s.label = new
            renamed.append(new)
        notes.warn(f"ラベル {label!r} が {n} 回あるので {', '.join(renamed)} に付け直しました")


def _read_document(path: Path) -> gemmi.cif.Document:
    """CIF を gemmi で読む。`data_` 見出しの無いファイル(一部のソフトが書き出す)は見出しを補う。"""
    try:
        return gemmi.cif.read(str(path))
    except (ValueError, RuntimeError) as exc:
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^\s*data_", text, flags=re.MULTILINE):
            raise ValueError(f"CIF {path.name} を解釈できません: {exc}") from exc
        try:
            return gemmi.cif.read_string(f"data_{_safe_name(path.stem)}\n{text}")
        except (ValueError, RuntimeError) as exc2:
            raise ValueError(f"CIF {path.name} を解釈できません: {exc2}") from exc2


def _species_of(site: gemmi.SmallStructure.Site, has_type_symbol: bool, fname: str) -> str:
    """サイトの散乱種を決める。

    _atom_site_type_symbol があればそれを正規化する('Pb2+' など)。
    'La0' のように酸化数 0 を付けた表記や、ラベルしか無い場合は元素記号に落とす。
    """
    sym = site.type_symbol.strip() if has_type_symbol else ""
    if sym and sym not in ("?", ".") and sym.lower() != "x":
        try:
            return normalize_species(sym)
        except ValueError:
            m = re.fullmatch(r"([A-Z][a-z]?)0", sym)
            if m:
                return m.group(1)
    el = site.element.name
    if el in ("", "X"):
        raise ValueError(
            f"CIF {fname}: サイト {site.label!r} の元素が判別できません"
            "(_atom_site_type_symbol を書いてください)"
        )
    return el


def _resolve_spacegroup(
    st: gemmi.SmallStructure,
    lattice: Lattice,
    sites: list[AtomSite],
    cif_mults: list[int | None],
    notes: _Notes,
) -> SpaceGroup:
    """CIF の空間群情報を解釈する。対称操作 → H-M 記号 → Hall 記号 → 国際番号の順。

    原点選択が 2 つある群で原点が書かれていなければ、両方の設定を候補にして
    `_choose_origin` で決める(レビュー 2026-10-17 物理 #3)。
    """
    if st.symops:
        return SpaceGroup.from_symops(st.symops)
    candidates = _spacegroup_candidates(st, notes)
    if len(candidates) == 1:
        return SpaceGroup.from_gemmi(candidates[0])
    return _choose_origin(candidates, lattice, sites, cif_mults, notes)


def _spacegroup_candidates(st: gemmi.SmallStructure, notes: _Notes) -> list[gemmi.SpaceGroup]:
    """CIF の記号・番号から候補の設定を返す(原点が曖昧なときは 2 つ)。"""
    cell = st.cell
    hm = st.spacegroup_hm.strip()
    hall = st.spacegroup_hall.strip()
    number = int(st.spacegroup_number) if st.spacegroup_number else 0
    if hm:
        sg, explicit = _find_by_hm(hm, cell.alpha, cell.gamma)
        if sg is not None:
            if explicit or sg.number not in ORIGIN_CHOICE_NUMBERS:
                return [sg]
            return _origin_settings(sg)
        if not hall and not number:
            raise ValueError(
                f"CIF {notes.fname}: 空間群記号を解釈できません: {hm!r}。"
                "例: 'P n m a', 'Fd-3m', 'P 63/m m c', 'R -3 c:H', 'F d -3 m:2'"
            )
        notes.warn(f"空間群記号 {hm!r} を解釈できないので Hall 記号か番号を使います")
    # gemmi の型スタブは None を返さないことになっているが、見つからなければ None
    found: gemmi.SpaceGroup | None
    if hall:
        try:
            ops = gemmi.symops_from_hall(hall)
        except (ValueError, RuntimeError) as exc:
            if not number:
                raise ValueError(f"CIF {notes.fname}: Hall 記号を解釈できません: {hall!r}") from exc
        else:
            found = gemmi.find_spacegroup_by_ops(ops)
            if found is not None:
                return [found]  # Hall 記号は原点・設定まで決まる
    if number:
        found = gemmi.find_spacegroup_by_number(number)
        if found is None:
            raise ValueError(f"CIF {notes.fname}: 空間群番号は 1〜230 で指定してください: {number}")
        if number in ORIGIN_CHOICE_NUMBERS:
            return _origin_settings(found)
        # 三方晶の R 格子は格子定数から六方軸か菱面体軸かを決める
        by_cell = _find_by_name(found.hm, cell.alpha, cell.gamma)
        return [by_cell if by_cell is not None else found]
    raise ValueError(f"CIF {notes.fname}: 空間群の情報(対称操作・H-M 記号・番号)がありません")


def _find_by_hm(hm: str, alpha: float, gamma: float) -> tuple[gemmi.SpaceGroup | None, bool]:
    """H-M 記号を gemmi で引く。戻り値の 2 つ目は原点・軸の設定が明示されていたか。

    'F d -3 m:2' や 'R -3 c:H' のほか、ICSD 流の 'F d -3 m Z'(原点選択 2)、
    'F d -3 m S'(原点選択 1)、'R -3 c H' / 'R -3 c R' も受け付ける。
    """
    sg = _find_by_name(hm, alpha, gamma)
    if sg is not None:
        return sg, ":" in hm
    tokens = hm.split()
    if len(tokens) >= 2 and tokens[-1] in _ICSD_SETTING_SUFFIX:
        base = " ".join(tokens[:-1])
        ext = _ICSD_SETTING_SUFFIX[tokens[-1]]
        with_ext = _find_by_name(f"{base}:{ext}", alpha, gamma)
        if with_ext is not None:
            return with_ext, True
        without = _find_by_name(base, alpha, gamma)
        if without is not None:
            return without, False
    return None, False


def _find_by_name(name: str, alpha: float, gamma: float) -> gemmi.SpaceGroup | None:
    """gemmi の find_spacegroup_by_name(型スタブは None を返さないことになっているが返る)。"""
    found: gemmi.SpaceGroup | None = gemmi.find_spacegroup_by_name(name, alpha, gamma)
    return found


def _origin_settings(sg: gemmi.SpaceGroup) -> list[gemmi.SpaceGroup]:
    """同じ H-M 記号の原点選択 1 と 2 の設定を、既定の設定を先頭にして返す。"""
    found = [
        s
        for s in gemmi.spacegroup_table()
        if s.number == sg.number and s.hm == sg.hm and s.ext in ("1", "2")
    ]
    if len(found) < 2:
        return [sg]
    found.sort(key=lambda s: s.ext != DEFAULT_ORIGIN_CHOICE)
    return found


def _choose_origin(
    candidates: list[gemmi.SpaceGroup],
    lattice: Lattice,
    sites: list[AtomSite],
    cif_mults: list[int | None],
    notes: _Notes,
) -> SpaceGroup:
    """原点選択が書かれていない CIF で、座標に合う設定を選ぶ。

    1. CIF にサイト多重度があれば、計算した多重度が一致するサイトの数が多い設定。
    2. 無ければ(または同数なら)最短原子間距離が長い設定。間違った原点では原子が
       本来とは別のサイトに乗り、近すぎる原子対ができることが多い。
    3. それでも決まらなければ既定(原点選択 2)。
    """
    groups = [SpaceGroup.from_gemmi(c) for c in candidates]
    names = [c.xhm() for c in candidates]
    known = [m for m in cif_mults if m is not None]
    if known and len(cif_mults) == len(sites):
        scores: list[int] = []
        for sg in groups:
            n = 0
            for s, cm in zip(sites, cif_mults, strict=True):
                if cm is not None and sg.multiplicity(s.position, lattice=lattice) == cm:
                    n += 1
            scores.append(n)
        best = max(scores)
        if scores.count(best) == 1 and best > 0:
            k = scores.index(best)
            notes.warn(
                f"空間群の原点選択が書かれていないので、CIF のサイト多重度に合う設定 "
                f"{names[k]!r} を使います({best}/{len(known)} サイトが一致)"
            )
            return groups[k]
    dists = [_shortest_distance(lattice, sg, sites) for sg in groups]
    order = sorted(range(len(groups)), key=lambda i: -dists[i])
    k0, k1 = order[0], order[1]
    if dists[k1] > 0 and dists[k0] / dists[k1] >= _DISTANCE_DECISION_RATIO:
        k = k0
        reason = (
            f"最短原子間距離が {dists[k0]:.2f} Å({names[k1]} では {dists[k1]:.2f} Å)なので"
            f"設定 {names[k]!r} を使います"
        )
    else:
        k = 0
        reason = (
            f"既定の設定 {names[k]!r} を使います(最短原子間距離は "
            + "、".join(f"{n} で {d:.2f} Å" for n, d in zip(names, dists, strict=True))
            + ")"
        )
    notes.warn(
        "空間群の原点選択が書かれていません。"
        + reason
        + "。違っていれば記号に ':1' / ':2' を付けるか対称操作を書いてください"
    )
    return groups[k]


def _shortest_distance(lattice: Lattice, sg: SpaceGroup, sites: list[AtomSite]) -> float:
    """単位胞内の全原子の最短距離 [Å]。同じ座標に置かれた混合サイトどうしは除く。"""
    g = lattice.metric_tensor
    images = [sg.equivalent_positions(s.position, lattice=lattice)[0] for s in sites]
    best = math.inf
    for i, pi in enumerate(images):
        for j in range(i, len(images)):
            pj = images[j]
            d = pi[:, None, :] - pj[None, :, :]
            d -= np.round(d)
            dist2 = np.einsum("abk,kl,abl->ab", d, g, d)
            if i == j:
                dist2[np.diag_indices(len(pi))] = np.inf
            elif float(dist2.min()) < 1e-6:
                continue  # 混合サイト(同じ座標に別の種)
            best = min(best, float(dist2.min()))
    return math.sqrt(best) if math.isfinite(best) else math.inf


def _check_multiplicities(
    spacegroup: SpaceGroup,
    lattice: Lattice,
    sites: list[AtomSite],
    cif_mults: list[int | None],
    notes: _Notes,
) -> None:
    """CIF の多重度と計算した多重度を照合する。

    合わないとき、SNAP_TOL [Å] 以内の特殊位置に乗せ直して合うなら乗せ直して警告する
    (2 桁に丸めた 0.33 など)。それでも合わなければ原点・設定・座標の確認を促す。
    """
    if len(cif_mults) != len(sites):
        return
    for s, cm in zip(sites, cif_mults, strict=True):
        if cm is None:
            continue
        m = spacegroup.multiplicity(s.position, lattice=lattice)
        if m == cm:
            continue
        xs = spacegroup.symmetrized_position(s.position, SNAP_TOL, lattice=lattice)
        moved = float(lattice.distance(s.position, xs))
        if moved > 1e-9 and spacegroup.multiplicity(xs, lattice=lattice) == cm:
            notes.warn(
                f"サイト {s.label} の座標 {_fmt_xyz(s.position)} は特殊位置から {moved:.3f} Å "
                f"ずれていた(多重度 {m})ので、CIF の多重度 {cm} に合う {_fmt_xyz(xs)} に"
                "乗せ直しました"
            )
            s.position[:] = xs
            continue
        notes.warn(
            f"サイト {s.label} の多重度が CIF の {cm} と合いません(計算では {m})。"
            "空間群の原点選択・設定か座標を確認してください"
        )


def _check_overlapping_images(structure: Structure, notes: _Notes) -> None:
    """対称等価な像どうしが近すぎるサイト(特殊位置からのずれ)を警告する。"""
    dists = structure.min_image_distances()
    mults = structure.multiplicities()
    for s, d, m in zip(structure.sites, dists, mults, strict=True):
        if d < OVERLAP_WARN_DISTANCE:
            notes.warn(
                f"サイト {s.label} {_fmt_xyz(s.position)} の対称等価な像どうしが {d:.3f} Å しか"
                f"離れていません(多重度 {m})。特殊位置から僅かにずれた座標(桁落ち)か、"
                "空間群・原点の取り違えの可能性があります"
            )


def _fmt_xyz(xyz: np.ndarray) -> str:
    return "(" + ", ".join(f"{v:.5f}" for v in xyz) + ")"


def write_structure(structure: Structure, path: str | Path) -> None:
    """結晶構造を簡易な CIF に書き出す(対称操作・原子・ADP を含む)。"""
    lat = structure.lattice
    sg = structure.spacegroup
    lines = [
        f"data_{_safe_name(structure.name) or 'phase'}",
        f"_pd_phase_name '{structure.name}'",
        f"_cell_length_a {lat.a:.6f}",
        f"_cell_length_b {lat.b:.6f}",
        f"_cell_length_c {lat.c:.6f}",
        f"_cell_angle_alpha {lat.alpha:.5f}",
        f"_cell_angle_beta {lat.beta:.5f}",
        f"_cell_angle_gamma {lat.gamma:.5f}",
        f"_cell_volume {lat.volume:.4f}",
        f"_space_group_IT_number {sg.number}",
        f"_space_group_name_H-M_alt '{sg.hm}'",
        "loop_",
        "_space_group_symop_id",
        "_space_group_symop_operation_xyz",
    ]
    for i, op in enumerate(sg.operations, start=1):
        lines.append(f"{i} '{op.triplet}'")
    lines += [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_adp_type",
        "_atom_site_U_iso_or_equiv",
        "_atom_site_symmetry_multiplicity",
    ]
    mults = structure.multiplicities()
    for s, m in zip(structure.sites, mults, strict=True):
        x, y, z = s.position
        adp = "Uani" if s.is_anisotropic else "Uiso"
        u = _u_equiv(s) if s.is_anisotropic else s.u_iso
        lines.append(
            f"{s.label} {s.species} {x:.6f} {y:.6f} {z:.6f} {s.occupancy:.4f} {adp} {u:.5f} {m}"
        )
    aniso_sites = [s for s in structure.sites if s.is_anisotropic]
    if aniso_sites:
        lines += [
            "loop_",
            "_atom_site_aniso_label",
            "_atom_site_aniso_U_11",
            "_atom_site_aniso_U_22",
            "_atom_site_aniso_U_33",
            "_atom_site_aniso_U_12",
            "_atom_site_aniso_U_13",
            "_atom_site_aniso_U_23",
        ]
        for s in aniso_sites:
            assert s.u_aniso is not None
            lines.append(s.label + " " + " ".join(f"{v:.5f}" for v in s.u_aniso))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _u_equiv(site: AtomSite) -> float:
    """U_eq の簡易値(直交格子で正確。一般格子では近似)。"""
    assert site.u_aniso is not None
    return float(np.mean(site.u_aniso[:3]))


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
