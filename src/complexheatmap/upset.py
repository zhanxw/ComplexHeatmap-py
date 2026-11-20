"""UpSet combination matrices (`make_comb_mat`), accessors, and `UpSet` plots."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._options import _ENV
from .annotations import HeatmapAnnotation, anno_barplot, anno_text, rowAnnotation
from .colors import to_hex_rgba
from .genomic import GRanges, IRanges
from .grid import grid_lines, grid_points, grid_rect, grid_segments
from .heatmap import Heatmap
from .units import Unit, as_unit, check_gp, convertWidth, gpar, unit


def list_to_matrix(lt: dict[str, Sequence[Any]] | Sequence[Sequence[Any]], universal_set: Sequence[Any] | None = None) -> pd.DataFrame:
    """Convert a dict/list of sets into a 0/1 membership matrix DataFrame (`list_to_matrix.Rd`)."""
    if isinstance(lt, dict):
        keys = [str(k) for k in lt.keys()]
        sets = [set(v) for v in lt.values()]
    else:
        sets = [set(v) for v in lt]
        keys = [f"set_{i + 1}" for i in range(len(sets))]

    if universal_set is not None:
        all_elems = list(dict.fromkeys(universal_set))
    else:
        seen: dict[Any, None] = {}
        for s_item in lt.values() if isinstance(lt, dict) else lt:
            for elem in s_item:
                if elem not in seen:
                    seen[elem] = None
        all_elems = list(seen.keys())

    mat = np.zeros((len(all_elems), len(sets)), dtype=int)
    for col_i, s in enumerate(sets):
        for row_i, elem in enumerate(all_elems):
            if elem in s:
                mat[row_i, col_i] = 1
    return pd.DataFrame(mat, index=all_elems, columns=keys)


class CombMat:
    """Combination matrix wrapper powering `UpSet` plots (`make_comb_mat.Rd`)."""

    def __init__(
        self,
        mat: np.ndarray,
        set_names: list[str],
        comb_codes: list[str],
        set_sizes: np.ndarray,
        comb_sizes: np.ndarray,
        mode: str = "distinct",
        universal_size: float | None = None,
        elem_groups: dict[str, Any] | None = None,
        sets_on_rows: bool = True,
    ) -> None:
        self.matrix = np.asarray(mat, dtype=int)  # shape (n_sets, n_combs)
        self._set_names = list(set_names)
        self._comb_codes = list(comb_codes)
        self._set_sizes = np.asarray(set_sizes, dtype=float)
        self._comb_sizes = np.asarray(comb_sizes, dtype=float)
        self.mode = mode
        self.universal_size = universal_size
        self._elem_groups = elem_groups or {}
        self._sets_on_rows = bool(sets_on_rows)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape if self._sets_on_rows else self.matrix.T.shape

    @property
    def nrow(self) -> int:
        return self.shape[0]

    @property
    def ncol(self) -> int:
        return self.shape[1]

    @property
    def set_names(self) -> list[str]:
        return list(self._set_names)

    @set_names.setter
    def set_names(self, value: Sequence[str]) -> None:
        new_nms = [str(v) for v in value]
        if len(new_nms) != len(self._set_names):
            raise ValueError("Length of new set names must match number of sets.")
        self._set_names = new_nms

    def set_name(self, new_names: Sequence[str] | None = None) -> list[str]:
        if new_names is not None:
            self.set_names = new_names
        return list(self._set_names)

    def comb_name(self, readable: bool = False) -> list[str]:
        if not readable:
            return list(self._comb_codes)
        out = []
        for code in self._comb_codes:
            parts = [nm for nm, bit in zip(self._set_names, code) if bit == "1"]
            if not parts:
                out.append("complement")
            else:
                sep = " & " if self.mode in ("distinct", "intersect") else " | "
                out.append(sep.join(parts))
        return out

    def set_size(self) -> np.ndarray:
        return self._set_sizes.copy()

    def comb_size(self) -> np.ndarray:
        return self._comb_sizes.copy()

    def comb_degree(self) -> np.ndarray:
        return np.sum(self.matrix, axis=0)

    def complement_size(self) -> float:
        for idx, code in enumerate(self._comb_codes):
            if "1" not in code:
                return float(self._comb_sizes[idx])
        if self.universal_size is not None:
            tot_covered = float(np.sum(self._comb_sizes)) if self.mode == "distinct" else 0.0
            return max(0.0, float(self.universal_size) - tot_covered)
        return 0.0

    def extract_comb(self, comb_name: str) -> Any:
        if comb_name in self._elem_groups:
            return self._elem_groups[comb_name]
        # Allow readable form lookup
        readable = self.comb_name(readable=True)
        if comb_name in readable:
            code = self._comb_codes[readable.index(comb_name)]
            return self._elem_groups.get(code, [])
        raise KeyError(f"Combination {comb_name!r} not found.")

    def t(self) -> "CombMat":
        cp = copy.deepcopy(self)
        cp._sets_on_rows = not self._sets_on_rows
        return cp

    def copy_all(self) -> "CombMat":
        return copy.deepcopy(self)

    def __getitem__(self, key: Any) -> "CombMat":
        if isinstance(key, tuple):
            r_key, c_key = key
        else:
            r_key, c_key = key, slice(None)

        if not self._sets_on_rows:
            c_key, r_key = r_key, c_key

        cp = self.copy_all()
        ns, nc = cp.matrix.shape

        def _res(spec: Any, n: int, nms: list[str]) -> list[int]:
            if isinstance(spec, slice):
                return list(range(*spec.indices(n)))
            if isinstance(spec, (int, np.integer)):
                i = int(spec)
                return [i + n if i < 0 else i]
            if isinstance(spec, str):
                return [nms.index(spec)]
            seq = list(spec)
            if seq and all(isinstance(x, (bool, np.bool_)) for x in seq):
                return [i for i, f in enumerate(seq) if f]
            if seq and all(isinstance(x, str) for x in seq):
                return [nms.index(s) for s in seq]
            return [int(x) for x in seq]

        if not (isinstance(r_key, slice) and r_key == slice(None)):
            r_idx = _res(r_key, ns, cp._set_names)
            cp.matrix = cp.matrix[r_idx, :]
            cp._set_names = [cp._set_names[i] for i in r_idx]
            cp._set_sizes = cp._set_sizes[r_idx]
            # Recompute combination codes after set subsetting
            cp._comb_codes = ["".join(str(int(cp.matrix[r, c])) for r in range(len(r_idx))) for c in range(cp.matrix.shape[1])]

        if not (isinstance(c_key, slice) and c_key == slice(None)):
            c_idx = _res(c_key, cp.matrix.shape[1], cp._comb_codes)
            cp.matrix = cp.matrix[:, c_idx]
            cp._comb_codes = [cp._comb_codes[j] for j in c_idx]
            cp._comb_sizes = cp._comb_sizes[c_idx]

        return cp

    def __len__(self) -> int:
        return self.matrix.size

    def __repr__(self) -> str:
        r_or_c = "rows" if self._sets_on_rows else "columns"
        return (
            f"A combination matrix with {len(self._set_names)} sets and {len(self._comb_codes)} combinations.\n"
            f"  mode: {self.mode}, sets on: {r_or_c}\n"
            f"  sets: {', '.join(self._set_names)}"
        )


def _compute_genomic_comb_mat(
    sets_dict: dict[str, GRanges | IRanges],
    mode: str,
    bin_size: float | None = None,
    complement_size: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute exact interval-partitioned combination sizes for GRanges or IRanges lists."""
    set_names = list(sets_dict.keys())
    n_sets = len(set_names)
    first_val = next(iter(sets_dict.values()))
    is_gr = isinstance(first_val, GRanges)

    # Compute elementary disjoint intervals across union of breakpoints per chromosome (or single chr for IRanges)
    by_chr: dict[str, list[tuple[int, int, int]]] = {}  # chr -> list of (start, end, set_idx)
    for si, (nm, obj) in enumerate(sets_dict.items()):
        if is_gr:
            assert isinstance(obj, GRanges)
            for k in range(len(obj)):
                by_chr.setdefault(obj.seqnames[k], []).append((int(obj.start[k]), int(obj.end[k]), si))
        else:
            assert isinstance(obj, IRanges)
            for k in range(len(obj)):
                by_chr.setdefault("_ir", []).append((int(obj.start[k]), int(obj.end[k]), si))

    # Build membership intervals on disjoint elementary segments
    segments: list[tuple[str, int, int, tuple[int, ...]]] = []
    set_sizes = np.zeros(n_sets, dtype=float)

    for ch, ivs in by_chr.items():
        pts = sorted({p for s, e, _ in ivs for p in (s, e + 1)})
        if len(pts) < 2:
            continue
        # For each elementary interval [pts[k], pts[k+1]-1] check which sets cover it
        n_seg = len(pts) - 1
        cov = np.zeros((n_seg, n_sets), dtype=int)
        for s, e, si in ivs:
            i0 = int(np.searchsorted(pts, s))
            i1 = int(np.searchsorted(pts, e + 1))
            if i1 > i0:
                cov[i0:i1, si] = 1
        seg_lens = np.diff(np.asarray(pts, dtype=np.int64))
        for si in range(n_sets):
            set_sizes[si] += float(np.sum(seg_lens[cov[:, si] == 1]))
        for idx in range(n_seg):
            mask_t = tuple(int(cov[idx, si]) for si in range(n_sets))
            if any(mask_t):
                segments.append((ch, int(pts[idx]), int(pts[idx + 1] - 1), mask_t))

    # Group distinct combinations present
    comb_dict: dict[str, list[tuple[str, int, int]]] = {}
    comb_len_map: dict[str, float] = {}
    for ch, s, e, mask_t in segments:
        code = "".join(str(b) for b in mask_t)
        comb_len_map[code] = comb_len_map.get(code, 0.0) + float(e - s + 1)
        comb_dict.setdefault(code, []).append((ch, s, e))

    if mode == "distinct":
        codes = list(comb_len_map.keys())
        sizes = np.array([comb_len_map[c] for c in codes], dtype=float)
        mat = np.array([[int(b) for b in c] for c in codes], dtype=int).T if codes else np.zeros((n_sets, 0), dtype=int)
        elem_groups = {}
        for c, segs in comb_dict.items():
            if is_gr:
                elem_groups[c] = GRanges([x[0] for x in segs], IRanges([x[1] for x in segs], [x[2] for x in segs]))
            else:
                elem_groups[c] = IRanges([x[1] for x in segs], [x[2] for x in segs])
    else:
        # Intersect or union over all non-empty distinct codes
        all_codes = full_comb_code(n_sets, complement=(complement_size is not None and complement_size > 0))
        codes_kept: list[str] = []
        sizes_list: list[float] = []
        elem_groups = {}
        for target_code in all_codes:
            bits = np.array([int(b) for b in target_code], dtype=int)
            tot = 0.0
            segs_out: list[tuple[str, int, int]] = []
            for seg_code, seg_len in comb_len_map.items():
                s_bits = np.array([int(b) for b in seg_code], dtype=int)
                if mode == "intersect":
                    ok = bool(np.all(s_bits[bits == 1] == 1))
                else:  # union
                    ok = bool(np.any((bits == 1) & (s_bits == 1)))
                if ok:
                    tot += seg_len
                    segs_out.extend(comb_dict[seg_code])
            if tot > 0:
                codes_kept.append(target_code)
                sizes_list.append(tot)
                if is_gr:
                    elem_groups[target_code] = GRanges([x[0] for x in segs_out], IRanges([x[1] for x in segs_out], [x[2] for x in segs_out]))
                else:
                    elem_groups[target_code] = IRanges([x[1] for x in segs_out], [x[2] for x in segs_out])
        codes = codes_kept
        sizes = np.asarray(sizes_list, dtype=float)
        mat = np.array([[int(b) for b in c] for c in codes], dtype=int).T if codes else np.zeros((n_sets, 0), dtype=int)

    if complement_size is not None and complement_size > 0:
        z_code = "0" * n_sets
        codes = [z_code] + codes
        sizes = np.concatenate([[float(complement_size)], sizes])
        z_col = np.zeros((n_sets, 1), dtype=int)
        mat = np.hstack([z_col, mat])

    return mat, codes, set_sizes, sizes, elem_groups


def make_comb_mat(
    *args: Any,
    mode: str = "distinct",
    top_n_sets: float = np.inf,
    min_set_size: float = -np.inf,
    universal_set: Sequence[Any] | None = None,
    complement_size: float | None = None,
    remove_empty_comb_set: bool = True,
    remove_complement_set: bool = False,
) -> CombMat:
    """Create a combination matrix from sets, binary matrix, or genomic ranges (`make_comb_mat.Rd`)."""
    mode = str(mode).lower()
    if mode not in ("distinct", "intersect", "union"):
        raise ValueError("`mode` must be 'distinct', 'intersect', or 'union'.")

    if not args:
        raise ValueError("`make_comb_mat` requires at least one input argument.")

    first = args[0]
    # Support positional multiple GRanges/IRanges/sets or single dict/DataFrame/ndarray
    if len(args) > 1:
        lt: dict[str, Any] = {f"set_{i + 1}": a for i, a in enumerate(args)}
    elif isinstance(first, dict):
        lt = dict(first)
    elif isinstance(first, (list, tuple)) and first and isinstance(first[0], (set, list, tuple, np.ndarray, pd.Series, GRanges, IRanges)):
        lt = {f"set_{i + 1}": a for i, a in enumerate(first)}
    elif isinstance(first, (pd.DataFrame, np.ndarray)):
        df = first if isinstance(first, pd.DataFrame) else pd.DataFrame(first, columns=[f"set_{i + 1}" for i in range(first.shape[1])])
        # Apply set filters before computing combos
        bin_mat = (df.to_numpy(dtype=float) != 0).astype(int)
        set_names = [str(c) for c in df.columns]
        set_sizes = np.sum(bin_mat, axis=0).astype(float)
        keep = set_sizes >= float(min_set_size)
        if np.isfinite(top_n_sets) and np.sum(keep) > int(top_n_sets):
            top_idx = np.argsort(-set_sizes)[: int(top_n_sets)]
            keep2 = np.zeros(len(set_names), dtype=bool)
            keep2[top_idx] = True
            keep &= keep2
        bin_mat = bin_mat[:, keep]
        set_names = [s for s, f in zip(set_names, keep) if f]
        set_sizes = set_sizes[keep]
        elem_names = list(df.index)
        n_elem, n_sets = bin_mat.shape

        # Compute element membership codes
        row_codes = ["".join(str(int(bin_mat[r, c])) for c in range(n_sets)) for r in range(n_elem)]
        code_to_elems: dict[str, list[Any]] = {}
        for r_i, c_str in enumerate(row_codes):
            code_to_elems.setdefault(c_str, []).append(elem_names[r_i])

        return _build_comb_from_membership(
            bin_mat=bin_mat,
            set_names=set_names,
            set_sizes=set_sizes,
            code_to_elems=code_to_elems,
            mode=mode,
            universal_size=float(n_elem),
            complement_size=complement_size,
            remove_empty_comb_set=remove_empty_comb_set,
            remove_complement_set=remove_complement_set,
        )
    else:
        raise TypeError(f"Unsupported input type to `make_comb_mat`: {type(first)}")

    # Check if dict of GRanges / IRanges
    vals = list(lt.values())
    if vals and isinstance(vals[0], (GRanges, IRanges)):
        mat, codes, set_sizes, comb_sizes, elem_groups = _compute_genomic_comb_mat(lt, mode=mode, complement_size=complement_size)
        return CombMat(
            mat=mat,
            set_names=list(lt.keys()),
            comb_codes=codes,
            set_sizes=set_sizes,
            comb_sizes=comb_sizes,
            mode=mode,
            elem_groups=elem_groups,
        )

    # Filter sets by min_set_size & top_n_sets
    set_items = [(str(k), set(v)) for k, v in lt.items()]
    set_items = [(k, s) for k, s in set_items if len(s) >= float(min_set_size)]
    if np.isfinite(top_n_sets) and len(set_items) > int(top_n_sets):
        set_items.sort(key=lambda kv: -len(kv[1]))
        set_items = set_items[: int(top_n_sets)]

    filtered_dict = dict(set_items)
    df = list_to_matrix(filtered_dict, universal_set=universal_set)
    bin_mat = df.to_numpy(dtype=int)
    set_names = [str(c) for c in df.columns]
    set_sizes = np.sum(bin_mat, axis=0).astype(float)
    elem_names = list(df.index)
    row_codes = ["".join(str(int(bin_mat[r, c])) for c in range(len(set_names))) for r in range(len(elem_names))]
    code_to_elems = {}
    for r_i, c_str in enumerate(row_codes):
        code_to_elems.setdefault(c_str, []).append(elem_names[r_i])

    eff_univ_size = float(len(elem_names)) if universal_set is not None else None
    return _build_comb_from_membership(
        bin_mat=bin_mat,
        set_names=set_names,
        set_sizes=set_sizes,
        code_to_elems=code_to_elems,
        mode=mode,
        universal_size=eff_univ_size,
        complement_size=complement_size,
        remove_empty_comb_set=remove_empty_comb_set,
        remove_complement_set=remove_complement_set,
    )


def _build_comb_from_membership(
    bin_mat: np.ndarray,
    set_names: list[str],
    set_sizes: np.ndarray,
    code_to_elems: dict[str, list[Any]],
    mode: str,
    universal_size: float | None,
    complement_size: float | None,
    remove_empty_comb_set: bool,
    remove_complement_set: bool,
) -> CombMat:
    n_sets = len(set_names)
    z_code = "0" * n_sets

    if complement_size is not None and complement_size > 0:
        has_complement = True
    elif z_code in code_to_elems and len(code_to_elems[z_code]) > 0 and not remove_complement_set:
        has_complement = True
    else:
        has_complement = False

    if mode == "distinct":
        codes = list(code_to_elems.keys())
        if not has_complement:
            codes = [c for c in codes if c != z_code]
        # Order nicely: complement last or sorted by bits/degree? Keep deterministic order (decreasing comb_size or lexicographical)
        sizes = np.array([float(complement_size if (c == z_code and complement_size is not None) else len(code_to_elems.get(c, []))) for c in codes], dtype=float)
        elem_groups = {c: list(code_to_elems.get(c, [])) for c in codes}
    else:
        all_codes = full_comb_code(n_sets, complement=has_complement)
        codes_out: list[str] = []
        sizes_out: list[float] = []
        elem_groups = {}
        for target_code in all_codes:
            if target_code == z_code:
                sz = float(complement_size) if complement_size is not None else float(len(code_to_elems.get(z_code, [])))
                if not remove_empty_comb_set or sz > 0:
                    codes_out.append(z_code)
                    sizes_out.append(sz)
                    elem_groups[z_code] = list(code_to_elems.get(z_code, []))
                continue
            bits = np.array([int(b) for b in target_code], dtype=bool)
            if mode == "intersect":
                mask = np.all(bin_mat[:, bits] == 1, axis=1)
            else:  # union
                mask = np.any(bin_mat[:, bits] == 1, axis=1)
            sz = float(np.sum(mask))
            if remove_empty_comb_set and sz == 0:
                continue
            codes_out.append(target_code)
            sizes_out.append(sz)
            # Extract matching elements
            elems = [el for c_str, el_list in code_to_elems.items() if (
                (all(c_str[k] == "1" for k in np.where(bits)[0])) if mode == "intersect"
                else (any(c_str[k] == "1" for k in np.where(bits)[0]))
            ) for el in el_list]
            elem_groups[target_code] = elems
        codes = codes_out
        sizes = np.asarray(sizes_out, dtype=float)

    mat = np.array([[int(b) for b in c] for c in codes], dtype=int).T if codes else np.zeros((n_sets, 0), dtype=int)
    return CombMat(
        mat=mat,
        set_names=set_names,
        comb_codes=codes,
        set_sizes=set_sizes,
        comb_sizes=sizes,
        mode=mode,
        universal_size=universal_size,
        elem_groups=elem_groups,
    )


def full_comb_code(n: int, complement: bool = False) -> list[str]:
    """Generate all binary combination strings of length `n` (`full_comb_code.Rd`)."""
    codes = []
    for deg in range(1, n + 1):
        from itertools import combinations
        for idxs in combinations(range(n), deg):
            bits = ["0"] * n
            for idx in idxs:
                bits[idx] = "1"
            codes.append("".join(bits))
    if complement:
        codes.insert(0, "0" * n)
    return codes


def set_name(m: CombMat, new_names: Sequence[str] | None = None) -> Any:
    if new_names is not None:
        cp = m.copy_all()
        cp.set_names = new_names
        return cp
    return m.set_name()


def comb_name(m: CombMat, readable: bool = False) -> list[str]:
    return m.comb_name(readable=readable)


def set_size(m: CombMat) -> np.ndarray:
    return m.set_size()


def comb_size(m: CombMat) -> np.ndarray:
    return m.comb_size()


def comb_degree(m: CombMat) -> np.ndarray:
    return m.comb_degree()


def complement_size(m: CombMat) -> float:
    return m.complement_size()


def extract_comb(m: CombMat, comb_name: str) -> Any:
    return m.extract_comb(comb_name)


def order_comb_mat(m: CombMat, decreasing: bool = True) -> CombMat:
    """Sort combination matrix columns by combination degree & size (`order_comb_mat.Rd`)."""
    sz = m.comb_size()
    deg = m.comb_degree()
    order = sorted(range(len(sz)), key=lambda j: (-sz[j] if decreasing else sz[j], deg[j], m._comb_codes[j]))
    return m[:, order] if m._sets_on_rows else m[order, :]


def normalize_comb_mat(*args: CombMat, full_comb_sets: bool = False) -> list[CombMat]:
    """Align multiple CombMat objects to share identical set & combination ordering (`normalize_comb_mat.Rd`)."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        mats = list(args[0])
    else:
        mats = list(args)
    if not mats:
        return []
    union_sets = list(dict.fromkeys(nm for m in mats for nm in m._set_names))
    if full_comb_sets:
        union_codes = full_comb_code(len(union_sets), complement=any("0" * len(m._set_names) in m._comb_codes for m in mats))
    else:
        # Map each matrix's codes onto union_sets
        seen_codes: list[str] = []
        seen_c: set[str] = set()
        for m in mats:
            s_idx = [union_sets.index(nm) for nm in m._set_names]
            for c_str in m._comb_codes:
                bits = ["0"] * len(union_sets)
                for local_i, ch in enumerate(c_str):
                    if ch == "1":
                        bits[s_idx[local_i]] = "1"
                mapped = "".join(bits)
                if mapped not in seen_c:
                    seen_c.add(mapped)
                    seen_codes.append(mapped)
        union_codes = seen_codes

    out: list[CombMat] = []
    for m in mats:
        s_lookup = {nm: i for i, nm in enumerate(m._set_names)}
        c_lookup = {}
        for j, c_str in enumerate(m._comb_codes):
            bits = ["0"] * len(union_sets)
            for local_i, ch in enumerate(c_str):
                if ch == "1":
                    bits[union_sets.index(m._set_names[local_i])] = "1"
            c_lookup["".join(bits)] = j

        new_set_sizes = np.array([m._set_sizes[s_lookup[nm]] if nm in s_lookup else 0.0 for nm in union_sets], dtype=float)
        new_comb_sizes = np.array([m._comb_sizes[c_lookup[c_str]] if c_str in c_lookup else 0.0 for c_str in union_codes], dtype=float)
        new_mat = np.array([[int(b) for b in c_str] for c_str in union_codes], dtype=int).T if union_codes else np.zeros((len(union_sets), 0), dtype=int)
        out.append(
            CombMat(
                mat=new_mat,
                set_names=union_sets,
                comb_codes=union_codes,
                set_sizes=new_set_sizes,
                comb_sizes=new_comb_sizes,
                mode=m.mode,
                universal_size=m.universal_size,
                sets_on_rows=m._sets_on_rows,
            )
        )
    return out


# ---- UpSet Annotation Helpers ----

def upset_top_annotation(
    m: CombMat,
    gp: dict[str, Any] | None = None,
    height: Unit = unit(3, "cm"),
    show_annotation_name: bool = True,
    annotation_name_gp: dict[str, Any] | None = None,
    annotation_name_offset: Unit | None = None,
    annotation_name_side: str = "left",
    annotation_name_rot: float = 0.0,
    bar_width: float = 0.6,
    add_numbers: bool = False,
    numbers_gp: dict[str, Any] | None = None,
    numbers_rot: float = 0.0,
    axis_param: dict[str, Any] | None = None,
    which: str | None = None,
    width: Unit | None = None,
    **kwargs: Any,
) -> HeatmapAnnotation:
    """Default top combination-size (or set-size when transposed) barplot annotation (`upset_top_annotation.Rd`)."""
    sets_on_rows = m._sets_on_rows
    eff_which = which or "column"
    vals = m.comb_size() if sets_on_rows else m.set_size()
    lbl = "Intersection\nsize" if sets_on_rows else "Set\nsize"
    if gp is None:
        gp = gpar(fill="black", col=None)
    anno = anno_barplot(
        vals,
        gp=gp,
        bar_width=bar_width,
        border=False,
        which=eff_which,
        height=height if eff_which == "column" else None,
        width=width or (height if eff_which == "row" else None),
        add_numbers=add_numbers,
        numbers_gp=numbers_gp,
        numbers_rot=numbers_rot,
        axis_param=axis_param or ({"side": "left", "labels_rot": 0} if eff_which == "column" else {"side": "top"}),
        **kwargs,
    )
    if eff_which == "column":
        return HeatmapAnnotation(
            **{lbl: anno},
            show_annotation_name=show_annotation_name,
            annotation_name_gp=annotation_name_gp or gpar(fontsize=10),
            annotation_name_side=annotation_name_side,
            annotation_name_rot=annotation_name_rot,
            annotation_name_offset=annotation_name_offset,
        )
    return rowAnnotation(
        **{lbl: anno},
        show_annotation_name=show_annotation_name,
        annotation_name_gp=annotation_name_gp or gpar(fontsize=10),
    )


def upset_right_annotation(
    m: CombMat,
    gp: dict[str, Any] | None = None,
    width: Unit = unit(2, "cm"),
    bar_width: float = 0.6,
    add_numbers: bool = False,
    axis_param: dict[str, Any] | None = None,
    show_annotation_name: bool = True,
    annotation_name_gp: dict[str, Any] | None = None,
    annotation_name_side: str = "bottom",
    **kwargs: Any,
) -> HeatmapAnnotation:
    """Default right set-size (or comb-size when transposed) barplot annotation (`upset_right_annotation.Rd`)."""
    sets_on_rows = m._sets_on_rows
    vals = m.set_size() if sets_on_rows else m.comb_size()
    lbl = "Set size" if sets_on_rows else "Intersection size"
    if gp is None:
        gp = gpar(fill="black", col=None)
    ap = axis_param or {"side": "bottom", "labels_rot": 0}
    user_offset = kwargs.pop("annotation_name_offset", None)
    eff_offset = user_offset if user_offset is not None else (unit(7.0, "mm") if ap.get("side") == annotation_name_side else None)
    anno = anno_barplot(
        vals,
        gp=gp,
        bar_width=bar_width,
        border=False,
        which="row",
        width=width,
        add_numbers=add_numbers,
        axis_param=ap,
        **kwargs,
    )
    return rowAnnotation(
        **{lbl: anno},
        show_annotation_name=show_annotation_name,
        annotation_name_gp=annotation_name_gp or gpar(fontsize=10),
        annotation_name_side=annotation_name_side,
        annotation_name_offset=eff_offset,
    )


def upset_left_annotation(
    m: CombMat,
    gp: dict[str, Any] | None = None,
    width: Unit = unit(2, "cm"),
    bar_width: float = 0.6,
    add_numbers: bool = False,
    axis_param: dict[str, Any] | None = None,
    show_annotation_name: bool = True,
    annotation_name_gp: dict[str, Any] | None = None,
    annotation_name_side: str = "bottom",
    **kwargs: Any,
) -> HeatmapAnnotation:
    """Left-aligned inverted set-size barplot annotation (`upset_left_annotation.Rd`)."""
    sets_on_rows = m._sets_on_rows
    vals = m.set_size() if sets_on_rows else m.comb_size()
    lbl = "Set size" if sets_on_rows else "Intersection size"
    if gp is None:
        gp = gpar(fill="black", col=None)
    ap = axis_param or {"side": "bottom", "direction": "reverse", "labels_rot": 0}
    user_offset = kwargs.pop("annotation_name_offset", None)
    eff_offset = user_offset if user_offset is not None else (unit(7.0, "mm") if ap.get("side") == annotation_name_side else None)
    anno = anno_barplot(
        vals,
        gp=gp,
        bar_width=bar_width,
        border=False,
        which="row",
        width=width,
        add_numbers=add_numbers,
        axis_param=ap,
        **kwargs,
    )
    return rowAnnotation(
        **{lbl: anno},
        show_annotation_name=show_annotation_name,
        annotation_name_gp=annotation_name_gp or gpar(fontsize=10),
        annotation_name_side=annotation_name_side,
        annotation_name_offset=eff_offset,
    )


def UpSet(
    m: CombMat,
    comb_col: Any = "black",
    pt_size: Unit = unit(3, "mm"),
    lwd: float = 2.0,
    bg_col: Any = "#F0F0F0",
    bg_pt_col: Any = "#CCCCCC",
    set_order: Sequence[Any] | None = None,
    comb_order: Sequence[Any] | None = None,
    top_annotation: Any = "default",
    right_annotation: Any = "default",
    left_annotation: Any = None,
    bottom_annotation: Any = None,
    row_names_side: str | None = None,
    column_names_side: str = "bottom",
    name: str | None = None,
    **kwargs: Any,
) -> Heatmap:
    """Construct an UpSet plot Heatmap from a CombMat (`UpSet.Rd`)."""
    if not isinstance(m, CombMat):
        raise TypeError("`m` passed to `UpSet` must be a `CombMat` created by `make_comb_mat`.")

    sets_on_rows = m._sets_on_rows
    mat = m.matrix if sets_on_rows else m.matrix.T
    nr, nc = mat.shape

    if sets_on_rows:
        r_names = m.set_name()
        c_names = m.comb_name(readable=False)
        eff_row_ord = list(range(nr)) if set_order is None else [r_names.index(s) if isinstance(s, str) else int(s) for s in set_order]
        if comb_order is None:
            csz = m.comb_size()
            cdeg = m.comb_degree()
            eff_col_ord = sorted(range(nc), key=lambda j: (cdeg[j], -csz[j]))
        else:
            eff_col_ord = [c_names.index(c) if isinstance(c, str) else int(c) for c in comb_order]
        eff_rn_side = "left" if row_names_side is None else row_names_side
        show_rn = kwargs.pop("show_row_names", True)
        show_cn = kwargs.pop("show_column_names", False)
    else:
        r_names = m.comb_name(readable=False)
        c_names = m.set_name()
        if comb_order is None:
            csz = m.comb_size()
            cdeg = m.comb_degree()
            eff_row_ord = sorted(range(nr), key=lambda i: (cdeg[i], -csz[i]))
        else:
            eff_row_ord = [r_names.index(c) if isinstance(c, str) else int(c) for c in comb_order]
        eff_col_ord = list(range(nc)) if set_order is None else [c_names.index(s) if isinstance(s, str) else int(s) for s in set_order]
        eff_rn_side = "left" if row_names_side is None else row_names_side
        show_rn = kwargs.pop("show_row_names", False)
        show_cn = kwargs.pop("show_column_names", True)

    eff_top = upset_top_annotation(m) if top_annotation == "default" else top_annotation
    eff_right = upset_right_annotation(m) if right_annotation == "default" else right_annotation
    pt_u = as_unit(pt_size, "mm")

    # Background stripes and active combination dots + connecting lines
    bg_cols_seq = [bg_col] if isinstance(bg_col, str) else list(bg_col)
    if len(bg_cols_seq) == 1:
        bg_cols_seq = [bg_cols_seq[0], "#FFFFFF"]
    comb_cols_raw = [comb_col] * (nc if sets_on_rows else nr) if isinstance(comb_col, (str, int, np.integer)) else list(comb_col)

    from .utils import restore_matrix

    def _layer_fun(j: Any, i: Any, x: Any, y: Any, w: Any, h: Any, fill: Any) -> None:
        ind_mat = restore_matrix(j, i, x, y)
        nr_s, nc_s = ind_mat.shape
        x_npc = x.to_mm("x") if False else None
        # Compute cell centers directly in npc for slice
        if sets_on_rows:
            # Horizontal alternating bands per row
            for r in range(nr_s):
                y_r = (nr_s - r - 0.5) / nr_s
                c_bg = bg_cols_seq[r % len(bg_cols_seq)]
                if c_bg and str(c_bg).lower() not in ("white", "#ffffff", "transparent", "na"):
                    grid_rect(x=unit(0.5, "npc"), y=unit(y_r, "npc"), width=unit(1.0, "npc"), height=unit(1.0 / nr_s, "npc"), gp=gpar(fill=c_bg, col=None))
            # Dots & vertical lines per column
            for c in range(nc_s):
                pos_col = ind_mat[:, c]
                orig_j = int(j[pos_col[0]])
                orig_i_list = [int(i[p]) for p in pos_col]
                active_rows = [r for r, oi in enumerate(orig_i_list) if mat[oi, orig_j] == 1]
                x_c = (c + 0.5) / nc_s
                ys_all = [(nr_s - r - 0.5) / nr_s for r in range(nr_s)]
                grid_points(np.full(nr_s, x_c), np.asarray(ys_all), pch=16, size=pt_u, gp=gpar(col=bg_pt_col), default_units="npc")
                c_act = to_hex_rgba(comb_cols_raw[orig_j % len(comb_cols_raw)])
                if len(active_rows) >= 2:
                    y_top = (nr_s - min(active_rows) - 0.5) / nr_s
                    y_bot = (nr_s - max(active_rows) - 0.5) / nr_s
                    grid_segments(unit(x_c, "npc"), unit(y_bot, "npc"), unit(x_c, "npc"), unit(y_top, "npc"), gp=gpar(col=c_act, lwd=lwd))
                if active_rows:
                    ys_act = [(nr_s - r - 0.5) / nr_s for r in active_rows]
                    grid_points(np.full(len(ys_act), x_c), np.asarray(ys_act), pch=16, size=pt_u, gp=gpar(col=c_act), default_units="npc")
        else:
            # Vertical alternating bands per column
            for c in range(nc_s):
                x_c = (c + 0.5) / nc_s
                c_bg = bg_cols_seq[c % len(bg_cols_seq)]
                if c_bg and str(c_bg).lower() not in ("white", "#ffffff", "transparent", "na"):
                    grid_rect(x=unit(x_c, "npc"), y=unit(0.5, "npc"), width=unit(1.0 / nc_s, "npc"), height=unit(1.0, "npc"), gp=gpar(fill=c_bg, col=None))
            # Dots & horizontal lines per row
            for r in range(nr_s):
                pos_row = ind_mat[r, :]
                orig_i = int(i[pos_row[0]])
                orig_j_list = [int(j[p]) for p in pos_row]
                active_cols = [c for c, oj in enumerate(orig_j_list) if mat[orig_i, oj] == 1]
                y_r = (nr_s - r - 0.5) / nr_s
                xs_all = [(c + 0.5) / nc_s for c in range(nc_s)]
                grid_points(np.asarray(xs_all), np.full(nc_s, y_r), pch=16, size=pt_u, gp=gpar(col=bg_pt_col), default_units="npc")
                c_act = to_hex_rgba(comb_cols_raw[orig_i % len(comb_cols_raw)])
                if len(active_cols) >= 2:
                    x_left = (min(active_cols) + 0.5) / nc_s
                    x_right = (max(active_cols) + 0.5) / nc_s
                    grid_segments(unit(x_left, "npc"), unit(y_r, "npc"), unit(x_right, "npc"), unit(y_r, "npc"), gp=gpar(col=c_act, lwd=lwd))
                if active_cols:
                    xs_act = [(c + 0.5) / nc_s for c in active_cols]
                    grid_points(np.asarray(xs_act), np.full(len(xs_act), y_r), pch=16, size=pt_u, gp=gpar(col=c_act), default_units="npc")

    df_mat = pd.DataFrame(mat, index=r_names, columns=c_names)
    ht_name = name or f"upset_{_ENV.next_heatmap_index()}"

    return Heatmap(
        df_mat,
        name=ht_name,
        rect_gp=gpar(type="none"),
        layer_fun=_layer_fun,
        cluster_rows=False,
        cluster_columns=False,
        row_order=eff_row_ord,
        column_order=eff_col_ord,
        show_row_names=show_rn,
        row_names_side=eff_rn_side,
        show_column_names=show_cn,
        column_names_side=column_names_side,
        top_annotation=eff_top,
        right_annotation=eff_right,
        left_annotation=left_annotation,
        bottom_annotation=bottom_annotation,
        show_heatmap_legend=False,
        **kwargs,
    )
