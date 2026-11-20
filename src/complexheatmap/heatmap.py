"""Single Heatmap class, clustering/splitting engine, 9x9 layout, and component drawing."""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._base import AdditiveUnit
from ._options import _ENV, ht_opt
from .annotations import AnnotationFunction, HeatmapAnnotation, SingleAnnotation, anno_text
from .colors import ColorMapping, default_col, to_hex_rgba, to_rgba_tuple
from .dendrogram import (
    Dendrogram,
    Hclust,
    adjust_dend_by_x,
    as_dendrogram,
    cut_dendrogram,
    dend_heights,
    dist as calc_dist,
    dist2,
    grid_dendrogram,
    hclust,
    merge_dendrogram,
    nobs,
    order_dendrogram,
    reorder_dendrogram,
)
from .grid import (
    Viewport,
    grid_raster,
    grid_rect,
    grid_segments,
    grid_text,
    max_text_height,
    max_text_width,
    popViewport,
    pushViewport,
    textGrob,
)
from .legend import Legends
from .text import GridText
from .units import (
    Unit,
    as_unit,
    check_gp,
    convertHeight,
    convertWidth,
    gpar,
    is_abs_unit,
    subset_gp,
    unit,
    unit_c,
    unit_sum,
)


HEATMAP_LAYOUT_COLUMN_COMPONENTS = (
    "column_title_top",
    "column_dend_top",
    "column_names_top",
    "column_anno_top",
    "heatmap_body",
    "column_anno_bottom",
    "column_names_bottom",
    "column_dend_bottom",
    "column_title_bottom",
)

HEATMAP_LAYOUT_ROW_COMPONENTS = (
    "row_title_left",
    "row_dend_left",
    "row_names_left",
    "row_anno_left",
    "heatmap_body",
    "row_anno_right",
    "row_names_right",
    "row_dend_right",
    "row_title_right",
)


def _heatmap_layout_index(nm: str) -> tuple[int, int]:
    if nm in HEATMAP_LAYOUT_COLUMN_COMPONENTS and nm != "heatmap_body":
        return (HEATMAP_LAYOUT_COLUMN_COMPONENTS.index(nm), HEATMAP_LAYOUT_ROW_COMPONENTS.index("heatmap_body"))
    if nm in HEATMAP_LAYOUT_ROW_COMPONENTS and nm != "heatmap_body":
        return (HEATMAP_LAYOUT_COLUMN_COMPONENTS.index("heatmap_body"), HEATMAP_LAYOUT_ROW_COMPONENTS.index(nm))
    if nm == "heatmap_body":
        return (HEATMAP_LAYOUT_COLUMN_COMPONENTS.index("heatmap_body"), HEATMAP_LAYOUT_ROW_COMPONENTS.index("heatmap_body"))
    raise ValueError(f"Unknown heatmap layout component: {nm!r}")


def _is_na(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)) and math.isnan(float(v)):
        return True
    try:
        res = pd.isna(v)
        if isinstance(res, (bool, np.bool_)):
            return bool(res)
    except Exception:
        pass
    return False


def _is_empty_title(title: Any) -> bool:
    if title is None or isinstance(title, (list, tuple)) and len(title) == 0:
        return True
    if isinstance(title, np.ndarray) and title.size == 0:
        return True
    if isinstance(title, str) and title == "":
        return True
    if isinstance(title, float) and math.isnan(title):
        return True
    return False


def _normalize_title_list(title: Any) -> list[Any]:
    if _is_empty_title(title):
        return []
    if isinstance(title, GridText):
        return [title]
    if isinstance(title, (str, bytes)):
        s = title.decode() if isinstance(title, bytes) else title
        return [] if s == "" else [s]
    if isinstance(title, (list, tuple, pd.Series, pd.Index, np.ndarray)):
        out = []
        for x in title:
            if not _is_empty_title(x):
                out.append(x)
        return out
    return [str(title)]


def _normalize_side(val: Any, valid: tuple[str, ...], arg_name: str) -> str:
    if isinstance(val, (list, tuple)):
        val = val[0]
    s = str(val).lower()
    if s not in valid:
        raise ValueError(f"`{arg_name}` must be one of {valid}, got {val!r}.")
    return s


def _get_text_just(rot: float, side: str) -> Any:
    rot = float(rot) % 360
    if side == "left":
        if rot == 0:
            return "right"
        if rot == 90:
            return "center"
        if rot == 270:
            return "center"
        return "right"
    if side == "right":
        if rot == 0:
            return "left"
        if rot == 90:
            return "center"
        if rot == 270:
            return "center"
        return "left"
    if side == "top":
        if rot == 0:
            return "center"
        if rot == 90:
            return "left"
        if rot == 270:
            return "right"
        return "center"
    # bottom
    if rot == 0:
        return "center"
    if rot == 90:
        return "right"
    if rot == 270:
        return "left"
    return "center"


def _dimname_just_and_loc(side: str, rot: float, centered: bool) -> tuple[Any, Unit]:
    rot_m = float(rot) % 360
    if side == "left":
        if centered:
            return "center", unit(0.5, "npc")
        if rot_m == 90:
            return "bottom", unit(1.0, "npc")
        if rot_m == 270:
            return "top", unit(1.0, "npc")
        return "right", unit(1.0, "npc")
    if side == "right":
        if centered:
            return "center", unit(0.5, "npc")
        if rot_m == 90:
            return "top", unit(0.0, "npc")
        if rot_m == 270:
            return "bottom", unit(0.0, "npc")
        return "left", unit(0.0, "npc")
    if side == "top":
        if centered:
            return "center", unit(0.5, "npc")
        if rot_m == 0:
            return "bottom", unit(0.0, "npc")
        if rot_m == 90:
            return "left", unit(0.0, "npc")
        if rot_m == 270:
            return "right", unit(0.0, "npc")
        return "left", unit(0.0, "npc")
    # bottom
    if centered:
        return "center", unit(0.5, "npc")
    if rot_m == 0:
        return "top", unit(1.0, "npc")
    if rot_m == 90:
        return "right", unit(1.0, "npc")
    if rot_m == 270:
        return "left", unit(1.0, "npc")
    return "right", unit(1.0, "npc")


def _title_padding_unit(gp: dict[str, Any]) -> Unit:
    tp = ht_opt.TITLE_PADDING
    if tp is not None:
        return as_unit(tp, "mm")
    fs = float(gp.get("fontsize", 13.2))
    # 5.5pt + approximate descender height (~0.25em)
    pad0_pt = 5.5 + 0.25 * fs
    return unit([pad0_pt * 25.4 / 72.0, 0.0], "mm")


def _parse_matrix(matrix: Any, name: str | None) -> tuple[np.ndarray, bool, list[str] | None, list[str] | None, list[str] | None]:
    """Normalize input into a 2D numpy array + numeric flag + rownames + colnames + factor levels."""
    rn: list[str] | None = None
    cn: list[str] | None = None
    fa_levels: list[str] | None = None

    if isinstance(matrix, pd.DataFrame):
        rn = [str(x) for x in matrix.index]
        cn = [str(x) for x in matrix.columns]
        all_num = all(pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt) for dt in matrix.dtypes)
        if all_num:
            arr = matrix.to_numpy(dtype=float, copy=True)
            is_num = True
        else:
            arr = matrix.to_numpy(dtype=object, copy=True)
            is_num = False
        return arr, is_num, rn, cn, fa_levels

    if isinstance(matrix, pd.Series):
        if isinstance(matrix.dtype, pd.CategoricalDtype):
            fa_levels = [str(c) for c in matrix.cat.categories]
        rn = [str(x) for x in matrix.index]
        col_nm = str(matrix.name) if matrix.name is not None else (name or "V1")
        cn = [col_nm]
        if pd.api.types.is_numeric_dtype(matrix.dtype) and not pd.api.types.is_bool_dtype(matrix.dtype):
            arr = matrix.to_numpy(dtype=float, copy=True).reshape(-1, 1)
            is_num = True
        else:
            arr = matrix.to_numpy(dtype=object, copy=True).reshape(-1, 1)
            is_num = False
        return arr, is_num, rn, cn, fa_levels

    if isinstance(matrix, pd.Categorical):
        fa_levels = [str(c) for c in matrix.categories]
        arr = np.asarray(matrix, dtype=object).reshape(-1, 1)
        cn = [name] if name else None
        return arr, False, None, cn, fa_levels

    arr_raw = np.asarray(matrix)
    if arr_raw.ndim == 0:
        arr_raw = arr_raw.reshape(1, 1)
    elif arr_raw.ndim == 1:
        arr_raw = arr_raw.reshape(-1, 1)
        if name is not None:
            cn = [name]
    elif arr_raw.ndim != 2:
        raise ValueError(f"`matrix` must be 1D or 2D, got shape {arr_raw.shape}.")

    # Distinguish numeric vs bool/str/object
    if np.issubdtype(arr_raw.dtype, np.bool_):
        arr = np.where(arr_raw, "TRUE", "FALSE").astype(object)
        is_num = False
    elif np.issubdtype(arr_raw.dtype, np.number):
        arr = arr_raw.astype(float, copy=True)
        is_num = True
    else:
        # Try checking if object array is purely numeric or None/nan
        flat = arr_raw.ravel()
        non_na = [x for x in flat if not _is_na(x)]
        if non_na and all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) for x in non_na):
            arr = np.empty(arr_raw.shape, dtype=float)
            for idx, v in np.ndenumerate(arr_raw):
                arr[idx] = np.nan if _is_na(v) else float(v)
            is_num = True
        elif not non_na:
            # Empty / all NA numeric matrix by default unless object
            arr = np.full(arr_raw.shape, np.nan, dtype=float)
            is_num = True
        else:
            arr = arr_raw.astype(object, copy=True)
            is_num = False

    return arr, is_num, rn, cn, fa_levels


def _format_labels_input(labels: Any, mat_names: list[str] | None, expected_len: int, arg_name: str) -> Any:
    if labels is None:
        return None if mat_names is None else list(mat_names)
    if isinstance(labels, GridText):
        if len(labels) != expected_len:
            raise ValueError(f"Length of `{arg_name}` ({len(labels)}) must equal {expected_len}.")
        return labels
    if isinstance(labels, (list, tuple, pd.Index, pd.Series, np.ndarray)):
        out = [str(x) for x in labels]
        if len(out) == 0:
            return None
        if len(out) != expected_len:
            raise ValueError(f"Length of `{arg_name}` ({len(out)}) must equal {expected_len}.")
        return out
    return [str(labels)]


def _normalize_split_df(split: Any, n: int, arg_name: str) -> tuple[pd.DataFrame, list[str], list[list[Any]]]:
    """Normalize `split` into a DataFrame of ordered categorical columns, slice labels, and ordered index groups."""
    if isinstance(split, pd.DataFrame):
        df = split.copy()
    elif isinstance(split, (list, tuple)) and len(split) > 0 and isinstance(split[0], (list, tuple, pd.Series, pd.Categorical, np.ndarray)):
        df = pd.DataFrame({f"s{i}": list(col) for i, col in enumerate(split)})
    else:
        if isinstance(split, pd.Categorical):
            df = pd.DataFrame({"s0": pd.Series(split)})
        else:
            df = pd.DataFrame({"s0": pd.Series(list(split) if not isinstance(split, np.ndarray) else split.tolist())})

    if len(df) != n:
        raise ValueError(f"Length of `{arg_name}` ({len(df)}) must match dimension size ({n}).")

    # Convert each column to ordered Categorical mirroring R factor rules
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            present = set(s.dropna().unique())
            cats = [c for c in s.cat.categories if c in present]
            df[col] = pd.Categorical(s, categories=cats, ordered=True)
        elif pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
            uniq_sorted = sorted(s.dropna().unique().tolist())
            cats = [str(u) for u in uniq_sorted]
            df[col] = pd.Categorical([str(v) if not _is_na(v) else np.nan for v in s], categories=cats, ordered=True)
        else:
            vals = [str(v) if not _is_na(v) else np.nan for v in s]
            uniq_sorted = sorted({v for v in vals if not _is_na(v)})
            df[col] = pd.Categorical(vals, categories=uniq_sorted, ordered=True)

    # Lexicographical sort by categorical codes (matching R `do.call("order", split)`)
    codes_cols = [df[c].cat.codes.to_numpy() for c in df.columns]
    row_keys = list(zip(*codes_cols))
    sorted_row_pos = sorted(range(n), key=lambda i: row_keys[i])

    # Build combined slice label per row
    str_mat = np.column_stack([[str(v) for v in df[c]] for c in df.columns])
    comb_names = [",".join(str_mat[i, :]) for i in range(n)]

    seen: dict[str, list[int]] = {}
    ordered_levels: list[str] = []
    for pos in sorted_row_pos:
        lbl = comb_names[pos]
        if lbl not in seen:
            seen[lbl] = []
            ordered_levels.append(lbl)
        seen[lbl].append(pos)

    groups = [seen[lbl] for lbl in ordered_levels]
    return df, ordered_levels, groups


def _compute_distance(submat: np.ndarray, distance: Any) -> Any:
    """Compute DistMatrix on rows of `submat`."""
    if isinstance(distance, str):
        return calc_dist(submat, method=distance)
    if callable(distance):
        sig = inspect.signature(distance)
        params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(params) == 1:
            return distance(submat)
        return dist2(submat, distance)
    raise TypeError(f"Unsupported clustering distance specification: {distance!r}")


def _cluster_submatrix(submat: np.ndarray, cluster: Any, distance: Any, method: str) -> Hclust | Dendrogram:
    """Run hierarchical clustering or user-provided clustering function on rows of `submat`."""
    if callable(cluster):
        res = cluster(submat)
        if isinstance(res, (Hclust, Dendrogram)):
            return res
        # Allow scipy linkage ndarray directly
        if isinstance(res, np.ndarray) and res.ndim == 2 and res.shape[1] == 4:
            return Hclust(res, method=method)
        raise TypeError("Custom clustering function must return an Hclust, Dendrogram, or linkage matrix.")
    d = _compute_distance(submat, distance)
    return hclust(d, method=method)


def _kmeans_single(mat: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 50) -> tuple[np.ndarray, float]:
    """Deterministic/seeded k-means++ with Lloyd updates handling NaNs via column-mean imputation."""
    x = np.asarray(mat, dtype=float)
    n, p = x.shape
    if k >= n:
        return np.arange(n, dtype=int), 0.0
    # Impute any NaN with column mean (or 0 if all NaN)
    if np.any(np.isnan(x)):
        x = x.copy()
        col_means = np.nanmean(x, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        r_na, c_na = np.where(np.isnan(x))
        x[r_na, c_na] = col_means[c_na]

    # k-means++ initialization
    centers = np.empty((k, p), dtype=float)
    idx0 = int(rng.integers(0, n))
    centers[0] = x[idx0]
    sq_dists = np.sum((x - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        tot = float(np.sum(sq_dists))
        if tot <= 0:
            rem = [i for i in range(n) if i not in set(range(c))]
            centers[c] = x[rem[0] if rem else 0]
        else:
            probs = sq_dists / tot
            idx = int(rng.choice(n, p=probs))
            centers[c] = x[idx]
        d_new = np.sum((x - centers[c]) ** 2, axis=1)
        sq_dists = np.minimum(sq_dists, d_new)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign to nearest centroid
        dmat = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dmat, axis=1)
        if np.array_equal(new_labels, labels):
            # Re-verify at least once cleanly
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = np.mean(x[mask], axis=0)
            else:
                # Re-seed empty cluster to farthest point
                far = int(np.argmax(np.min(dmat, axis=1)))
                centers[c] = x[far]

    inertia = 0.0
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            inertia += float(np.sum((x[mask] - centers[c]) ** 2))
    return labels, inertia


def _consensus_kmeans(mat: np.ndarray, centers: int, km_repeats: int = 1) -> np.ndarray:
    """K-means clustering with optional multi-run consensus partitioning."""
    n = mat.shape[0]
    if centers >= n:
        return np.arange(n, dtype=int)

    seed = ht_opt.kmeans_seed if hasattr(ht_opt, "kmeans_seed") and ht_opt.kmeans_seed is not None else None
    rng = np.random.default_rng(seed)

    if km_repeats <= 1:
        best_lbl = None
        best_iner = np.inf
        # Multiple random starts even for single consensus run (like nstart)
        for _ in range(5):
            lbl, iner = _kmeans_single(mat, centers, rng)
            if iner < best_iner:
                best_iner = iner
                best_lbl = lbl
        return best_lbl if best_lbl is not None else np.zeros(n, dtype=int)

    # Co-association matrix consensus across km_repeats runs
    coassoc = np.zeros((n, n), dtype=float)
    runs: list[np.ndarray] = []
    for _ in range(int(km_repeats)):
        lbl, _ = _kmeans_single(mat, centers, rng)
        runs.append(lbl)
        same = (lbl[:, None] == lbl[None, :]).astype(float)
        coassoc += same
    coassoc /= float(km_repeats)
    dist_mat = np.maximum(0.0, 1.0 - coassoc)
    np.fill_diagonal(dist_mat, 0.0)
    # Cluster the consensus distance matrix with average linkage & cut into `centers`
    from .dendrogram import as_dist, cutree
    hc = hclust(as_dist(dist_mat), method="average")
    cl = cutree(hc, k=centers)
    # Remap to consecutive 0..k-1
    uniq = sorted(set(int(v) for v in cl))
    mp = {u: idx for idx, u in enumerate(uniq)}
    return np.array([mp[int(v)] for v in cl], dtype=int)


def _format_slice_titles(title_list: list[Any], slice_names: list[str] | None, n_slices: int) -> list[Any]:
    if not title_list:
        return []
    if len(title_list) == 1:
        t0 = title_list[0]
        if isinstance(t0, GridText) or n_slices <= 1:
            return title_list
        if "%s" in str(t0) and slice_names is not None:
            return [str(t0).replace("%s", nm) for nm in slice_names]
        return [t0]
    if len(title_list) < n_slices:
        return [title_list[i % len(title_list)] for i in range(n_slices)]
    return title_list[:n_slices]


def _normalize_annotation(anno: Any, side: str, n_obs: int, mat_names: list[str] | None) -> tuple[HeatmapAnnotation | None, dict[str, Any]]:
    if anno is None:
        return None, {"side": side}
    if not isinstance(anno, HeatmapAnnotation):
        raise TypeError(f"`{side}_annotation` must be a HeatmapAnnotation (or rowAnnotation for left/right).")
    expected_which = "column" if side in ("top", "bottom") else "row"
    if anno.which != expected_which:
        raise ValueError(f"`{side}_annotation` must have which={expected_which!r}, got {anno.which!r}.")
    a_nobs = anno.nobs()
    if a_nobs is not None and a_nobs != n_obs and n_obs > 0:
        raise ValueError(f"`{side}_annotation` length ({a_nobs}) does not match matrix {'columns' if expected_which == 'column' else 'rows'} ({n_obs}).")
    param: dict[str, Any] = {"side": side}
    if side in ("top", "bottom"):
        param["height"] = anno.height + as_unit(ht_opt.COLUMN_ANNO_PADDING, "mm")
    else:
        param["width"] = anno.width + as_unit(ht_opt.ROW_ANNO_PADDING, "mm")
    return anno, param


def _calc_slice_layout(counts: Sequence[int], gap: Unit, axis: str) -> tuple[Unit, Unit]:
    """Compute slice start positions (`x` left-to-right or `y` top-to-bottom) and slice sizes as `Unit`s."""
    n_slice = len(counts)
    total = sum(counts)
    if total == 0:
        fracs = np.full(n_slice, 1.0 / max(1, n_slice))
    else:
        fracs = np.asarray(counts, dtype=float) / float(total)

    if n_slice == 1:
        sizes = unit(1.0, "npc")
        starts = unit(0.0, "npc") if axis == "x" else unit(1.0, "npc")
        return starts, sizes

    # Recycle gap to length n_slice - 1
    gap_mm = gap.to_mm(axis)
    g_vals = [float(gap_mm[i % len(gap_mm)]) for i in range(n_slice - 1)]
    gaps_u = [unit(g, "mm") for g in g_vals]
    total_gap_u = unit(sum(g_vals), "mm")
    avail = unit(1.0, "npc") - total_gap_u
    sizes = avail * fracs

    pos_list: list[Unit] = []
    if axis == "x":
        cur = unit(0.0, "npc")
        for i in range(n_slice):
            pos_list.append(cur)
            if i < n_slice - 1:
                cur = cur + sizes[i] + gaps_u[i]
    else:
        cur = unit(1.0, "npc")
        for i in range(n_slice):
            pos_list.append(cur)
            if i < n_slice - 1:
                cur = cur - sizes[i] - gaps_u[i]
    return unit_c(*pos_list), sizes


class Heatmap(AdditiveUnit):
    """Single heatmap component with clustering, splitting, annotations, and 9x9 layout."""

    def __init__(
        self,
        matrix: Any,
        col: Any = None,
        name: str | None = None,
        na_col: str = "grey",
        color_space: str = "LAB",
        rect_gp: dict[str, Any] | None = None,
        border: Any = None,
        border_gp: dict[str, Any] | None = None,
        cell_fun: Callable[..., Any] | None = None,
        layer_fun: Callable[..., Any] | None = None,
        jitter: Any = False,

        row_title: Any = None,
        row_title_side: str = "left",
        row_title_gp: dict[str, Any] | None = None,
        row_title_rot: float | None = None,
        column_title: Any = None,
        column_title_side: str = "top",
        column_title_gp: dict[str, Any] | None = None,
        column_title_rot: float = 0.0,

        cluster_rows: Any = True,
        cluster_row_slices: bool = True,
        clustering_distance_rows: Any = "euclidean",
        clustering_method_rows: str = "complete",
        row_dend_side: str = "left",
        row_dend_width: Unit | None = None,
        show_row_dend: bool = True,
        row_dend_reorder: Any = None,
        row_dend_gp: dict[str, Any] | None = None,

        cluster_columns: Any = True,
        cluster_column_slices: bool = True,
        clustering_distance_columns: Any = "euclidean",
        clustering_method_columns: str = "complete",
        column_dend_side: str = "top",
        column_dend_height: Unit | None = None,
        show_column_dend: bool = True,
        column_dend_reorder: Any = None,
        column_dend_gp: dict[str, Any] | None = None,

        row_order: Any = None,
        column_order: Any = None,

        row_labels: Any = None,
        row_names_side: str = "right",
        show_row_names: bool = True,
        row_names_max_width: Unit | None = None,
        row_names_gp: dict[str, Any] | None = None,
        row_names_rot: float = 0.0,
        row_names_centered: bool = False,

        column_labels: Any = None,
        column_names_side: str = "bottom",
        show_column_names: bool = True,
        column_names_max_height: Unit | None = None,
        column_names_gp: dict[str, Any] | None = None,
        column_names_rot: float = 90.0,
        column_names_centered: bool = False,

        top_annotation: HeatmapAnnotation | None = None,
        bottom_annotation: HeatmapAnnotation | None = None,
        left_annotation: HeatmapAnnotation | None = None,
        right_annotation: HeatmapAnnotation | None = None,

        km: int = 1,
        km_repeats: int = 1,
        split: Any = None,
        row_km: int | None = None,
        row_km_repeats: int | None = None,
        row_split: Any = None,
        column_km: int = 1,
        column_km_repeats: int = 1,
        column_split: Any = None,
        gap: Unit | None = None,
        row_gap: Unit | None = None,
        column_gap: Unit | None = None,
        show_parent_dend_line: bool | None = None,

        heatmap_width: Unit | None = None,
        width: Unit | None = None,
        heatmap_height: Unit | None = None,
        height: Unit | None = None,

        show_heatmap_legend: bool = True,
        heatmap_legend_param: dict[str, Any] | None = None,

        use_raster: bool | None = None,
        raster_device: str = "png",
        raster_quality: float = 1.0,
        raster_device_param: dict[str, Any] | None = None,
        raster_resize_mat: Any = False,
        raster_by_magick: bool = False,
        raster_magick_filter: str | None = None,
        post_fun: Callable[..., Any] | None = None,
    ) -> None:
        if name is None:
            name = f"matrix_{_ENV.next_heatmap_index()}"
        else:
            name = str(name)
            if name == "":
                raise ValueError("Heatmap `name` cannot be empty string.")
        self.name = name

        # Apply ht_opt defaults when args omitted
        if row_names_gp is None:
            row_names_gp = ht_opt.heatmap_row_names_gp if ht_opt.heatmap_row_names_gp is not None else gpar(fontsize=12)
        if column_names_gp is None:
            column_names_gp = ht_opt.heatmap_column_names_gp if ht_opt.heatmap_column_names_gp is not None else gpar(fontsize=12)
        if row_title_gp is None:
            row_title_gp = ht_opt.heatmap_row_title_gp if ht_opt.heatmap_row_title_gp is not None else gpar(fontsize=13.2)
        if column_title_gp is None:
            column_title_gp = ht_opt.heatmap_column_title_gp if ht_opt.heatmap_column_title_gp is not None else gpar(fontsize=13.2)

        hlp = dict(heatmap_legend_param) if heatmap_legend_param is not None else {"title": name}
        hlp.setdefault("title", name)
        for opt_key in ("title_gp", "title_position", "labels_gp", "grid_width", "grid_height", "border"):
            if opt_key not in hlp:
                gval = ht_opt.get(f"legend_{opt_key}")
                if gval is not None:
                    hlp[opt_key] = gval

        mat, is_num, mat_rn, mat_cn, fa_levels = _parse_matrix(matrix, name)
        nr, nc = mat.shape
        self.matrix = mat
        self._is_numeric = is_num
        self._rownames = mat_rn
        self._colnames = mat_cn

        eff_rect_gp = check_gp(gpar(col=None) if rect_gp is None else rect_gp)
        if border is None:
            border = ht_opt.heatmap_border if ht_opt.heatmap_border is not None else False
        eff_border_gp = check_gp(gpar(col="black") if border_gp is None else border_gp)

        # Normalize km / split / gap aliases
        eff_row_km = int(km if row_km is None else row_km)
        eff_row_km_repeats = int(km_repeats if row_km_repeats is None else row_km_repeats)
        eff_row_split = split if row_split is None else row_split
        eff_column_km = int(column_km)
        eff_column_km_repeats = int(column_km_repeats)
        eff_column_split = column_split

        eff_row_gap = as_unit(unit(1, "mm") if (gap is None and row_gap is None) else (gap if row_gap is None else row_gap), "mm")
        eff_column_gap = as_unit(unit(1, "mm") if column_gap is None else column_gap, "mm")

        if show_parent_dend_line is None:
            show_parent_dend_line = bool(ht_opt.show_parent_dend_line)

        # Matrix ColorMapping
        self.matrix_color_mapping: ColorMapping | None = None
        if nr > 0 and nc > 0:
            if col is None:
                if eff_rect_gp.get("type") == "none" and np.all([_is_na(v) for v in mat.ravel()]):
                    self.matrix_color_mapping = None
                else:
                    inferred = default_col(mat, main_matrix=True)
                    if callable(inferred):
                        self.matrix_color_mapping = ColorMapping(name=name, col_fun=inferred, na_col=na_col)
                    else:
                        if fa_levels is not None and not is_num:
                            # Preserve categorical level ordering
                            lvl_cols = {str(lv): default_col([str(x) for x in fa_levels], main_matrix=True)[str(lv)] for lv in fa_levels if str(lv) in inferred}
                            self.matrix_color_mapping = ColorMapping(name=name, colors=lvl_cols or inferred, na_col=na_col)
                        else:
                            self.matrix_color_mapping = ColorMapping(name=name, colors=inferred, na_col=na_col)
            elif isinstance(col, ColorMapping):
                self.matrix_color_mapping = copy.copy(col)
            elif callable(col):
                self.matrix_color_mapping = ColorMapping(name=name, col_fun=col, na_col=na_col)
            elif isinstance(col, dict):
                eff_na = col.get("_NA_", na_col)
                clean = {str(k): v for k, v in col.items() if k != "_NA_"}
                self.matrix_color_mapping = ColorMapping(name=name, colors=clean, na_col=eff_na, full_col=clean)
            elif isinstance(col, (list, tuple, np.ndarray)):
                col_seq = [str(c) for c in col]
                if not is_num:
                    uniq = list(dict.fromkeys(str(v) for v in mat.ravel() if not _is_na(v)))
                    if len(col_seq) < len(uniq):
                        raise ValueError(f"Length of `col` ({len(col_seq)}) < unique discrete levels ({len(uniq)}).")
                    mp = dict(zip(uniq, col_seq[: len(uniq)]))
                    self.matrix_color_mapping = ColorMapping(name=name, colors=mp, na_col=na_col)
                else:
                    from .colors import colorRamp2
                    vals = mat.astype(float)
                    finite = vals[np.isfinite(vals)]
                    if len(finite) == 0:
                        vmin, vmax = 0.0, 1.0
                    else:
                        vmin, vmax = float(np.min(finite)), float(np.max(finite))
                        if vmin == vmax:
                            vmax = vmin + 1.0
                    if len(col_seq) == 1:
                        col_seq = [col_seq[0], col_seq[0]]
                    cf = colorRamp2(np.linspace(vmin, vmax, len(col_seq)), col_seq, space=color_space)
                    self.matrix_color_mapping = ColorMapping(name=name, col_fun=cf, na_col=na_col)
            else:
                raise TypeError(f"Unsupported `col` type: {type(col)}")

        self.matrix_legend_param = hlp

        # Width / height validation & normalization
        if width is not None and heatmap_width is not None:
            raise ValueError("Cannot specify both `width` and `heatmap_width`.")
        if height is not None and heatmap_height is not None:
            raise ValueError("Cannot specify both `height` and `heatmap_height`.")

        if width is not None:
            w_u = as_unit(width, "mm")
            hw_u = unit(1.0, "npc")
        elif heatmap_width is not None:
            w_u = unit(1.0, "npc")
            hw_u = as_unit(heatmap_width, "mm")
        else:
            w_u = unit(1.0, "npc")
            hw_u = unit(1.0, "npc")

        if height is not None:
            h_u = as_unit(height, "mm")
            hh_u = unit(1.0, "npc")
        elif heatmap_height is not None:
            h_u = unit(1.0, "npc")
            hh_u = as_unit(heatmap_height, "mm")
        else:
            h_u = unit(1.0, "npc")
            hh_u = unit(1.0, "npc")

        # Validate body dims for clusterability
        self.matrix_param: dict[str, Any] = {
            "row_km": eff_row_km,
            "row_km_repeats": eff_row_km_repeats,
            "row_split": eff_row_split,
            "row_gap": eff_row_gap,
            "column_km": eff_column_km,
            "column_km_repeats": eff_column_km_repeats,
            "column_split": eff_column_split,
            "column_gap": eff_column_gap,
            "border": border,
            "border_gp": eff_border_gp,
            "rect_gp": eff_rect_gp,
            "jitter": jitter,
            "cell_fun": cell_fun,
            "layer_fun": layer_fun,
            "width": w_u,
            "height": h_u,
            "_user_split_row": eff_row_split,
            "_user_split_column": eff_column_split,
        }

        # Titles
        rt_side = _normalize_side(row_title_side, ("left", "right"), "row_title_side")
        ct_side = _normalize_side(column_title_side, ("top", "bottom"), "column_title_side")
        rt_rot = (90.0 if rt_side == "left" else 270.0) if row_title_rot is None else float(row_title_rot)
        ct_rot = float(column_title_rot)

        self.row_title = _normalize_title_list(row_title)
        self.row_title_param: dict[str, Any] = {
            "side": rt_side,
            "gp": check_gp(row_title_gp),
            "rot": rt_rot,
            "just": _get_text_just(rt_rot, rt_side),
        }
        self.column_title = _normalize_title_list(column_title)
        self.column_title_param: dict[str, Any] = {
            "side": ct_side,
            "gp": check_gp(column_title_gp),
            "rot": ct_rot,
            "just": _get_text_just(ct_rot, ct_side),
        }

        # Row clustering parameters
        rd_side = _normalize_side(row_dend_side, ("left", "right"), "row_dend_side")
        cd_side = _normalize_side(column_dend_side, ("top", "bottom"), "column_dend_side")
        rd_w = unit(10, "mm") if row_dend_width is None else as_unit(row_dend_width, "mm")
        cd_h = unit(10, "mm") if column_dend_height is None else as_unit(column_dend_height, "mm")

        def _parse_cluster_arg(cl_arg: Any, n_items: int, manual_order: Any, axis_nm: str, reorder_arg: Any) -> dict[str, Any]:
            is_obj = isinstance(cl_arg, (Hclust, Dendrogram)) or (isinstance(cl_arg, np.ndarray) and cl_arg.ndim == 2 and cl_arg.shape[1] == 4)
            is_fn = callable(cl_arg)
            if isinstance(cl_arg, (bool, np.bool_)):
                do_cl = bool(cl_arg)
                obj = None
                fun = None
            elif is_obj:
                do_cl = True
                obj = as_dendrogram(Hclust(cl_arg) if isinstance(cl_arg, np.ndarray) else cl_arg)
                fun = None
            elif is_fn:
                do_cl = True
                obj = None
                fun = cl_arg
            else:
                raise TypeError(f"`cluster_{axis_nm}` must be bool, Hclust, Dendrogram, linkage array, or callable.")

            if n_items <= 1 or not is_num and not is_obj and not is_fn and not callable(distance_arg_for(axis_nm)):
                pass

            eff_reorder = (isinstance(cl_arg, (bool, np.bool_)) or is_fn) if reorder_arg is None else reorder_arg
            return {
                "cluster": do_cl,
                "obj": obj,
                "fun": fun,
                "reorder": eff_reorder,
            }

        def distance_arg_for(axis_nm: str) -> Any:
            return clustering_distance_rows if axis_nm == "rows" else clustering_distance_columns

        def _normalize_manual_order(ord_val: Any, names_list: list[str] | None, n_items: int, axis_nm: str) -> tuple[list[int], bool]:
            if ord_val is None:
                return list(range(n_items)), False
            seq = list(ord_val)
            if len(seq) != n_items:
                raise ValueError(f"Length of `{axis_nm}_order` ({len(seq)}) must equal {n_items}.")
            if all(isinstance(x, str) for x in seq):
                if names_list is None:
                    raise ValueError(f"Character `{axis_nm}_order` requires matrix {axis_nm} names.")
                lookup = {nm: i for i, nm in enumerate(names_list)}
                return [lookup[s] for s in seq], True
            idx = [int(x) for x in seq]
            # Support 1-based R order passed accidentally? Wait: in Python tests we stick strictly to 0-based!
            # Only validate range 0 .. n_items - 1 cleanly.
            if any(i < 0 or i >= n_items for i in idx):
                raise IndexError(f"`{axis_nm}_order` contains out-of-range 0-based indices for length {n_items}.")
            return idx, True

        r_ord_norm, r_ord_set = _normalize_manual_order(row_order, mat_rn, nr, "row")
        c_ord_norm, c_ord_set = _normalize_manual_order(column_order, mat_cn, nc, "column")

        if r_ord_set:
            cluster_rows_eff: Any = False
        else:
            cluster_rows_eff = cluster_rows
        if c_ord_set:
            cluster_cols_eff: Any = False
        else:
            cluster_cols_eff = cluster_columns

        # Non-numeric matrices disable default clustering unless user provides distance func/cluster obj
        rc_info = _parse_cluster_arg(cluster_rows_eff, nr, row_order, "rows", row_dend_reorder)
        if not is_num and rc_info["obj"] is None and rc_info["fun"] is None and not callable(clustering_distance_rows):
            rc_info["cluster"] = False
        if nr <= 1 or nc == 0:
            if rc_info["obj"] is None:
                rc_info["cluster"] = False

        cc_info = _parse_cluster_arg(cluster_cols_eff, nc, column_order, "columns", column_dend_reorder)
        if not is_num and cc_info["obj"] is None and cc_info["fun"] is None and not callable(clustering_distance_columns):
            cc_info["cluster"] = False
        if nc <= 1 or nr == 0:
            if cc_info["obj"] is None:
                cc_info["cluster"] = False

        # Adjust dend width with DENDROGRAM_PADDING when shown
        dend_pad = convertWidth(as_unit(ht_opt.DENDROGRAM_PADDING, "mm"), "mm", valueOnly=True)
        rd_w_mm = convertWidth(rd_w, "mm", valueOnly=True)
        cd_h_mm = convertHeight(cd_h, "mm", valueOnly=True)
        eff_show_rd = bool(show_row_dend and rc_info["cluster"] and nr > 1 and rd_w_mm > 0)
        eff_show_cd = bool(show_column_dend and cc_info["cluster"] and nc > 1 and cd_h_mm > 0)

        self.row_dend_param: dict[str, Any] = {
            "cluster": rc_info["cluster"],
            "obj": rc_info["obj"],
            "fun": rc_info["fun"],
            "cluster_slices": bool(cluster_row_slices),
            "distance": clustering_distance_rows,
            "method": clustering_method_rows,
            "side": rd_side,
            "width": unit(rd_w_mm + dend_pad, "mm") if eff_show_rd else unit(0.0, "mm"),
            "show": eff_show_rd,
            "gp": check_gp(gpar() if row_dend_gp is None else row_dend_gp),
            "reorder": rc_info["reorder"],
            "split_by_cutree": False,
        }
        self.column_dend_param: dict[str, Any] = {
            "cluster": cc_info["cluster"],
            "obj": cc_info["obj"],
            "fun": cc_info["fun"],
            "cluster_slices": bool(cluster_column_slices),
            "distance": clustering_distance_columns,
            "method": clustering_method_columns,
            "side": cd_side,
            "height": unit(cd_h_mm + dend_pad, "mm") if eff_show_cd else unit(0.0, "mm"),
            "show": eff_show_cd,
            "gp": check_gp(gpar() if column_dend_gp is None else column_dend_gp),
            "reorder": cc_info["reorder"],
            "split_by_cutree": False,
        }

        self.row_dend_list: list[Dendrogram | None] = []
        self.row_dend_slice: Dendrogram | None = None
        self.row_order: list[int] = r_ord_norm
        self.row_order_list: list[list[int]] = [list(r_ord_norm)] if nr > 0 else [[]]

        self.column_dend_list: list[Dendrogram | None] = []
        self.column_dend_slice: Dendrogram | None = None
        self.column_order: list[int] = c_ord_norm
        self.column_order_list: list[list[int]] = [list(c_ord_norm)] if nc > 0 else [[]]

        # Dimension names annotations
        rn_side = _normalize_side(row_names_side, ("right", "left"), "row_names_side")
        cn_side = _normalize_side(column_names_side, ("bottom", "top"), "column_names_side")
        rn_max_w = unit(6, "cm") if row_names_max_width is None else as_unit(row_names_max_width, "mm")
        cn_max_h = unit(6, "cm") if column_names_max_height is None else as_unit(column_names_max_height, "mm")

        r_labels_fmt = _format_labels_input(row_labels, mat_rn, nr, "row_labels")
        c_labels_fmt = _format_labels_input(column_labels, mat_cn, nc, "column_labels")
        rn_anno = None
        eff_show_rn = bool(show_row_names and r_labels_fmt is not None and len(r_labels_fmt) > 0 and nr > 0)
        if eff_show_rn:
            r_just, r_loc = _dimname_just_and_loc(rn_side, row_names_rot, row_names_centered)
            rn_anno = anno_text(
                r_labels_fmt,
                which="row",
                gp=check_gp(row_names_gp),
                rot=float(row_names_rot),
                just=r_just,
                location=r_loc,
                width=max_text_width(r_labels_fmt, gp=check_gp(row_names_gp), rot=float(row_names_rot)),
            )

        cn_anno = None
        eff_show_cn = bool(show_column_names and c_labels_fmt is not None and len(c_labels_fmt) > 0 and nc > 0)
        if eff_show_cn:
            c_just, c_loc = _dimname_just_and_loc(cn_side, column_names_rot, column_names_centered)
            cn_anno = anno_text(
                c_labels_fmt,
                which="column",
                gp=check_gp(column_names_gp),
                rot=float(column_names_rot),
                just=c_just,
                location=c_loc,
                height=max_text_height(c_labels_fmt, gp=check_gp(column_names_gp), rot=float(column_names_rot)),
            )

        self.row_names_param: dict[str, Any] = {
            "labels": r_labels_fmt,
            "side": rn_side,
            "show": eff_show_rn,
            "gp": check_gp(row_names_gp),
            "rot": float(row_names_rot),
            "centered": bool(row_names_centered),
            "max_width": rn_max_w,
            "anno": rn_anno,
        }
        self.column_names_param: dict[str, Any] = {
            "labels": c_labels_fmt,
            "side": cn_side,
            "show": eff_show_cn,
            "gp": check_gp(column_names_gp),
            "rot": float(column_names_rot),
            "centered": bool(column_names_centered),
            "max_height": cn_max_h,
            "anno": cn_anno,
        }

        # Top/Bottom/Left/Right annotations
        self.top_annotation, self.top_annotation_param = _normalize_annotation(top_annotation, "top", nc, mat_cn)
        self.bottom_annotation, self.bottom_annotation_param = _normalize_annotation(bottom_annotation, "bottom", nc, mat_cn)
        self.left_annotation, self.left_annotation_param = _normalize_annotation(left_annotation, "left", nr, mat_rn)
        self.right_annotation, self.right_annotation_param = _normalize_annotation(right_annotation, "right", nr, mat_rn)

        # Raster & global heatmap parameters
        if use_raster is None:
            eff_raster = bool(((nr > 2000 and nc > 10) or (nc > 2000 and nr > 10)) and cell_fun is None and eff_rect_gp.get("type") != "none")
        else:
            eff_raster = bool(use_raster)
            if cell_fun is not None:
                eff_raster = False

        self.heatmap_param: dict[str, Any] = {
            "width": hw_u,
            "height": hh_u,
            "show_heatmap_legend": bool(show_heatmap_legend and self.matrix_color_mapping is not None and nr > 0 and nc > 0),
            "use_raster": eff_raster,
            "raster_device": raster_device,
            "raster_quality": float(raster_quality),
            "raster_device_param": dict(raster_device_param or {}),
            "raster_resize_mat": raster_resize_mat,
            "raster_by_magick": bool(raster_by_magick),
            "raster_magick_filter": raster_magick_filter,
            "show_parent_dend_line": bool(show_parent_dend_line),
            "post_fun": post_fun,
            "direction": "horizontal",
        }

        self.layout: dict[str, Any] = {
            "initialized": False,
            "layout_size": {
                "column_title_top_height": unit(0, "mm"),
                "column_dend_top_height": unit(0, "mm"),
                "column_names_top_height": unit(0, "mm"),
                "column_anno_top_height": unit(0, "mm"),
                "column_anno_bottom_height": unit(0, "mm"),
                "column_names_bottom_height": unit(0, "mm"),
                "column_dend_bottom_height": unit(0, "mm"),
                "column_title_bottom_height": unit(0, "mm"),
                "row_title_left_width": unit(0, "mm"),
                "row_dend_left_width": unit(0, "mm"),
                "row_names_left_width": unit(0, "mm"),
                "row_anno_left_width": unit(0, "mm"),
                "row_anno_right_width": unit(0, "mm"),
                "row_names_right_width": unit(0, "mm"),
                "row_dend_right_width": unit(0, "mm"),
                "row_title_right_width": unit(0, "mm"),
            },
            "layout_index": [],
            "graphic_fun_list": [],
            "slice": {},
        }

    # ---- Dimensions & Properties ----

    _oncoprint_env: dict[str, Any] | None = None

    @property
    def _matrix(self) -> np.ndarray:
        return self.matrix

    @property
    def _matrix_color_mapping(self) -> ColorMapping | None:
        return self.matrix_color_mapping

    @property
    def _row_split_df(self) -> pd.DataFrame | None:
        return self.matrix_param.get("_row_split_df")

    @property
    def _column_split_df(self) -> pd.DataFrame | None:
        return self.matrix_param.get("_column_split_df")

    @property
    def nrow(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def ncol(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrow, self.ncol)

    def dim(self) -> tuple[int, int]:
        return self.shape

    def copy_all(self) -> "Heatmap":
        return copy.deepcopy(self)

    # ---- Annotations Attachment ----

    def attach_annotation(self, ha: HeatmapAnnotation, side: str = "top", gap: Unit | None = None) -> "Heatmap":
        side = _normalize_side(side, ("top", "bottom", "left", "right"), "side")
        cp = copy.copy(self)
        from .annotations import c_HeatmapAnnotation
        slot_name = f"{side}_annotation"
        cur = getattr(cp, slot_name)
        if cur is None:
            combined = ha
        else:
            combined = c_HeatmapAnnotation(cur, ha, gap=gap) if gap is not None else c_HeatmapAnnotation(cur, ha)
        n_obs = cp.ncol if side in ("top", "bottom") else cp.nrow
        nm_list = cp._colnames if side in ("top", "bottom") else cp._rownames
        anno_obj, param = _normalize_annotation(combined, side, n_obs, nm_list)
        setattr(cp, slot_name, anno_obj)
        setattr(cp, f"{side}_annotation_param", param)
        return cp

    # ---- Clustering Engine ----

    def _make_axis_cluster(self, which: str) -> None:
        is_row = which == "row"
        mat_work = self.matrix if is_row else self.matrix.T
        n_items = mat_work.shape[0]
        if n_items == 0:
            if is_row:
                self.row_order = []
                self.row_order_list = [[]]
                self.row_dend_list = [None]
                self.row_dend_slice = None
            else:
                self.column_order = []
                self.column_order_list = [[]]
                self.column_dend_list = [None]
                self.column_dend_slice = None
            return

        # Optional jitter for clustering only
        jitter = self.matrix_param["jitter"]
        if jitter is not False and self._is_numeric:
            jit_val = 1e-10 if jitter is True else float(jitter)
            rng_jit = np.random.default_rng(0)
            mat_work = mat_work.astype(float) + rng_jit.uniform(0.0, jit_val, size=mat_work.shape)

        dend_param = self.row_dend_param if is_row else self.column_dend_param
        order_init = list(self.row_order if is_row else self.column_order)
        km = self.matrix_param["row_km" if is_row else "column_km"]
        km_repeats = self.matrix_param["row_km_repeats" if is_row else "column_km_repeats"]
        split = self.matrix_param["row_split" if is_row else "column_split"]

        dend_param["split_by_cutree"] = False
        dend_list: list[Dendrogram | None] = []
        dend_slice: Dendrogram | None = None
        order_list: list[list[int]] = []
        slice_names: list[str] | None = None

        cluster = dend_param["cluster"]
        cluster_slices = dend_param["cluster_slices"]
        reorder = dend_param["reorder"]
        distance = dend_param["distance"]
        method = dend_param["method"]

        # Helper to compute weights for reordering dendrograms
        def _weight_vec() -> np.ndarray | None:
            if reorder is False or reorder is None:
                return None
            if reorder is True:
                if not self._is_numeric:
                    return None
                with np.errstate(all="ignore"):
                    w = -np.nanmean(mat_work.astype(float), axis=1)
                return np.where(np.isnan(w), 0.0, w)
            arr_w = np.asarray(reorder, dtype=float).ravel()
            if len(arr_w) != n_items:
                raise ValueError(f"Length of `{which}_dend_reorder` ({len(arr_w)}) must equal {n_items}.")
            return arr_w

        weights_full = _weight_vec()

        # Case 1: Path A (cutree on single dendrogram when cluster=True and split is single int >= 1)
        is_scalar_int_split = isinstance(split, (int, np.integer)) and not isinstance(split, (bool, np.bool_))
        if cluster and (dend_param["obj"] is not None or is_scalar_int_split):
            if km > 1 and dend_param["obj"] is not None:
                raise ValueError("Cannot perform k-means when clustering object is supplied.")
            if is_scalar_int_split and km <= 1:
                k_cut = int(split)
                if k_cut < 1:
                    raise ValueError(f"`{which}_split` integer must be >= 1.")
                if dend_param["obj"] is None:
                    hc_full = _cluster_submatrix(mat_work, True if dend_param["fun"] is None else dend_param["fun"], distance, method)
                    dend_full = as_dendrogram(hc_full)
                else:
                    dend_full = as_dendrogram(dend_param["obj"])

                if weights_full is not None and dend_param["obj"] is None:
                    dend_full = reorder_dendrogram(dend_full, weights_full, agglo_fun=np.mean)

                if k_cut == 1:
                    dend_list = [dend_full]
                    order_list = [order_dendrogram(dend_full)]
                    dend_slice = None
                else:
                    dend_param["split_by_cutree"] = True
                    ct = cut_dendrogram(dend_full, k=k_cut)
                    dend_list = list(ct["lower"])
                    dend_slice = ct["upper"]
                    order_list = [order_dendrogram(sub_d) for sub_d in dend_list]
                    # Check order consistency: in R cut_dendrogram produces lower branches ordered by global order
                    if len(order_list) == k_cut:
                        slice_names = [str(i + 1) for i in range(k_cut)]
            else:
                # Clustering object with split=None (or scalar split=1 already handled)
                dend_full = as_dendrogram(dend_param["obj"])
                if split is None:
                    dend_list = [dend_full]
                    order_list = [order_dendrogram(dend_full)]
                else:
                    raise ValueError(f"When clustering object is supplied, `{which}_split` can only be an integer count.")

            # Save into slots & adjust x positions
            final_order = [idx for sl in order_list for idx in sl]
            # Normalize dendrogram x leaf coordinates within each slice (0.5 .. n_s - 0.5)
            adj_dends: list[Dendrogram | None] = []
            for sub_d in dend_list:
                if sub_d is not None and nobs(sub_d) > 0:
                    adj_dends.append(adjust_dend_by_x(sub_d, np.arange(nobs(sub_d), dtype=float) + 0.5))
                else:
                    adj_dends.append(sub_d)
            if is_row:
                self.row_dend_list = adj_dends
                self.row_dend_slice = dend_slice
                self.row_order_list = order_list
                self.row_order = final_order
                if slice_names and len(self.row_title) == 1 and "%s" in str(self.row_title[0]):
                    self.row_title = _format_slice_titles(self.row_title, slice_names, len(order_list))
            else:
                self.column_dend_list = adj_dends
                self.column_dend_slice = dend_slice
                self.column_order_list = order_list
                self.column_order = final_order
                if slice_names and len(self.column_title) == 1 and "%s" in str(self.column_title[0]):
                    self.column_title = _format_slice_titles(self.column_title, slice_names, len(order_list))
            return

        # Case 2: Path B (slice-first: k-means + categorical split DF -> within-slice clustering -> slice dendrogram)
        split_df: pd.DataFrame | None = None
        if split is not None and not is_scalar_int_split:
            if isinstance(split, pd.DataFrame):
                split_df = split.copy()
            elif isinstance(split, pd.Categorical):
                split_df = pd.DataFrame({"split": pd.Series(split)})
            elif isinstance(split, (list, tuple)) and len(split) > 0 and isinstance(split[0], (list, tuple, pd.Series, pd.Categorical, np.ndarray)):
                split_df = pd.DataFrame({f"s{i}": list(col) for i, col in enumerate(split)})
            else:
                split_df = pd.DataFrame({"split": list(split) if not isinstance(split, np.ndarray) else split.tolist()})

        if km > 1 and self._is_numeric and mat_work.shape[1] > 0:
            cl = _consensus_kmeans(mat_work, km, km_repeats=km_repeats)
            uniq_cl = sorted(set(int(v) for v in cl))
            with np.errstate(all="ignore"):
                mean_vecs = [np.nanmean(mat_work[cl == c, :], axis=0) for c in uniq_cl]
            mean_mat = np.vstack(mean_vecs)  # shape (k_eff, p)
            if weights_full is not None:
                slice_wts = np.array([float(np.nanmean(weights_full[cl == c])) for c in uniq_cl])
            else:
                slice_wts = np.nanmean(mean_mat, axis=1)

            if cluster_slices and len(uniq_cl) > 1:
                hc_s = hclust(calc_dist(mean_mat), method="complete")
                dend_s = reorder_dendrogram(as_dendrogram(hc_s), slice_wts, agglo_fun=np.mean)
                cl_ord = order_dendrogram(dend_s)
            else:
                cl_ord = np.argsort(slice_wts, kind="stable").tolist()

            remap = {uniq_cl[old_pos]: str(new_pos + 1) for new_pos, old_pos in enumerate(cl_ord)}
            km_cats = [str(i + 1) for i in range(len(cl_ord))]
            km_col = pd.Categorical([remap[int(c)] for c in cl], categories=km_cats, ordered=True)
            if split_df is None:
                split_df = pd.DataFrame({"km": km_col})
            else:
                split_df.insert(0, "km", km_col)

        if split_df is None:
            order_list = [order_init]
            slice_names = None
        else:
            norm_df, slice_names, groups = _normalize_split_df(split_df, n_items, f"{which}_split")
            # Intersect each group with user's manual order_init to respect row_order/column_order!
            order_set_map = {idx: pos for pos, idx in enumerate(order_init)}
            order_list = [sorted(grp, key=lambda idx: order_set_map[idx]) for grp in groups]
            if is_row:
                self.matrix_param["_row_split_df"] = norm_df
            else:
                self.matrix_param["_column_split_df"] = norm_df

        n_slices = len(order_list)
        dend_list = [None] * n_slices

        # Cluster within each slice
        if cluster:
            for k_s, idx_slice in enumerate(order_list):
                m_s = len(idx_slice)
                if m_s == 0:
                    continue
                if m_s == 1:
                    # Single-leaf dendrogram
                    dend_list[k_s] = Dendrogram(leaf=True, value=int(idx_slice[0]), height=0.0, members=1)
                    continue
                submat = mat_work[idx_slice, :]
                cl_obj = _cluster_submatrix(submat, True if dend_param["fun"] is None else dend_param["fun"], distance, method)
                sub_dend = as_dendrogram(cl_obj)
                # Map local leaf values (0..m_s-1) back to global matrix indices idx_slice
                def _remap_leaf_ids(node: Dendrogram, mapping: list[int]) -> None:
                    if node.is_leaf:
                        node.value = int(mapping[int(node.value)])
                    else:
                        for ch in node.children:
                            _remap_leaf_ids(ch, mapping)

                _remap_leaf_ids(sub_dend, idx_slice)
                if weights_full is not None:
                    sub_dend = reorder_dendrogram(sub_dend, weights_full, agglo_fun=np.mean)
                sub_dend = adjust_dend_by_x(sub_dend, np.arange(m_s, dtype=float) + 0.5)
                dend_list[k_s] = sub_dend
                order_list[k_s] = order_dendrogram(sub_dend)

            # Cluster between slices if n_slices > 1 and cluster_slices
            if n_slices > 1 and cluster_slices:
                mean_rows = []
                for idx_slice in order_list:
                    with np.errstate(all="ignore"):
                        mv = np.nanmean(mat_work[idx_slice, :].astype(float), axis=0)
                    mv = np.where(np.isnan(mv), 0.0, mv)
                    mean_rows.append(mv)
                mean_mat_s = np.vstack(mean_rows)
                try:
                    d_s = _compute_distance(mean_mat_s, distance)
                    hc_slice = hclust(d_s, method=method)
                except Exception:
                    hc_slice = hclust(calc_dist(mean_mat_s, method="euclidean"), method=method)
                dend_slice = as_dendrogram(hc_slice)
                if weights_full is not None:
                    s_wts = np.array([float(np.nanmean(weights_full[idx_slice])) for idx_slice in order_list])
                    dend_slice = reorder_dendrogram(dend_slice, s_wts, agglo_fun=np.mean)
                slice_od = order_dendrogram(dend_slice)
                order_list = [order_list[i] for i in slice_od]
                dend_list = [dend_list[i] for i in slice_od]
                if slice_names is not None:
                    slice_names = [slice_names[i] for i in slice_od]
                # Remap slice leaf values of dend_slice to new sorted order 0..n_slices-1
                inv_od = {old_i: new_i for new_i, old_i in enumerate(slice_od)}
                def _remap_slice_leaves(node: Dendrogram) -> None:
                    if node.is_leaf:
                        node.value = inv_od[int(node.value)]
                    else:
                        for ch in node.children:
                            _remap_slice_leaves(ch)
                _remap_slice_leaves(dend_slice)
                dend_slice = adjust_dend_by_x(dend_slice, np.arange(n_slices, dtype=float) + 0.5)

        final_order = [idx for sl in order_list for idx in sl]
        if is_row:
            self.row_dend_list = dend_list
            self.row_dend_slice = dend_slice
            self.row_order_list = order_list
            self.row_order = final_order
            if slice_names is not None:
                self.row_title = _format_slice_titles(self.row_title, slice_names, n_slices)
            elif len(self.row_title) > n_slices:
                self.row_title = self.row_title[:n_slices]
        else:
            self.column_dend_list = dend_list
            self.column_dend_slice = dend_slice
            self.column_order_list = order_list
            self.column_order = final_order
            if slice_names is not None:
                self.column_title = _format_slice_titles(self.column_title, slice_names, n_slices)
            elif len(self.column_title) > n_slices:
                self.column_title = self.column_title[:n_slices]

    def make_row_cluster(self) -> "Heatmap":
        self._make_axis_cluster("row")
        return self

    def make_column_cluster(self) -> "Heatmap":
        self._make_axis_cluster("column")
        return self

    # ---- Layout Preparation ----

    def prepare(self, process_rows: bool = True, process_columns: bool = True) -> "Heatmap":
        if process_rows:
            self.make_row_cluster()
        if process_columns:
            self.make_column_cluster()
        self.make_layout()
        return self

    def make_layout(self) -> "Heatmap":
        if self.layout["initialized"]:
            return self

        nr_slice = max(1, len(self.row_order_list))
        nc_slice = max(1, len(self.column_order_list))
        snr = [len(s) for s in self.row_order_list] if self.row_order_list else [0]
        snc = [len(s) for s in self.column_order_list] if self.column_order_list else [0]

        slice_y, slice_height = _calc_slice_layout(snr, self.matrix_param["row_gap"], "y")
        slice_x, slice_width = _calc_slice_layout(snc, self.matrix_param["column_gap"], "x")
        self.layout["slice"] = {
            "x": slice_x,
            "y": slice_y,
            "width": slice_width,
            "height": slice_height,
        }

        layout_size = self.layout["layout_size"]
        layout_index: list[tuple[int, int]] = []
        graphic_funs: list[Callable[[], None]] = []

        # 1. Heatmap Body
        if self.nrow > 0 and self.ncol > 0:
            layout_index.append(_heatmap_layout_index("heatmap_body"))

            def _draw_all_bodies() -> None:
                for kr in range(1, nr_slice + 1):
                    for kc in range(1, nc_slice + 1):
                        self.draw_heatmap_body(
                            kr=kr,
                            kc=kc,
                            x=slice_x[kc - 1],
                            y=slice_y[kr - 1],
                            width=slice_width[kc - 1],
                            height=slice_height[kr - 1],
                            just=("left", "top"),
                        )

            graphic_funs.append(_draw_all_bodies)

        # 2. Column Title
        if self.column_title:
            ct_side = self.column_title_param["side"]
            ct_gp = self.column_title_param["gp"]
            ct_rot = self.column_title_param["rot"]
            pad_c = _title_padding_unit(ct_gp)
            ct_h = max_text_height(self.column_title, gp=ct_gp, rot=ct_rot) + unit_sum(pad_c)
            if ct_side == "top":
                layout_size["column_title_top_height"] = ct_h
                layout_index.append(_heatmap_layout_index("column_title_top"))
            else:
                layout_size["column_title_bottom_height"] = ct_h
                layout_index.append(_heatmap_layout_index("column_title_bottom"))

            def _draw_col_titles() -> None:
                if len(self.column_title) == 1 and nc_slice > 1:
                    self.draw_title(which="column", k=1)
                else:
                    for kc in range(1, nc_slice + 1):
                        self.draw_title(which="column", k=kc, x=slice_x[kc - 1], width=slice_width[kc - 1], just="left")

            graphic_funs.append(_draw_col_titles)

        # 3. Row Title
        if self.row_title:
            rt_side = self.row_title_param["side"]
            rt_gp = self.row_title_param["gp"]
            rt_rot = self.row_title_param["rot"]
            pad_r = _title_padding_unit(rt_gp)
            rt_w = max_text_width(self.row_title, gp=rt_gp, rot=rt_rot) + unit_sum(pad_r)
            if rt_side == "left":
                layout_size["row_title_left_width"] = rt_w
                layout_index.append(_heatmap_layout_index("row_title_left"))
            else:
                layout_size["row_title_right_width"] = rt_w
                layout_index.append(_heatmap_layout_index("row_title_right"))

            def _draw_row_titles() -> None:
                if len(self.row_title) == 1 and nr_slice > 1:
                    self.draw_title(which="row", k=1)
                else:
                    for kr in range(1, nr_slice + 1):
                        self.draw_title(which="row", k=kr, y=slice_y[kr - 1], height=slice_height[kr - 1], just="top")

            graphic_funs.append(_draw_row_titles)

        # 4. Row Dendrogram
        if self.row_dend_param["show"] and any(d is not None and nobs(d) > 1 for d in self.row_dend_list):
            rd_side = self.row_dend_param["side"]
            rd_w = self.row_dend_param["width"]
            if rd_side == "left":
                layout_size["row_dend_left_width"] = rd_w
                layout_index.append(_heatmap_layout_index("row_dend_left"))
            else:
                layout_size["row_dend_right_width"] = rd_w
                layout_index.append(_heatmap_layout_index("row_dend_right"))

            child_heights = [dend_heights(d) for d in self.row_dend_list if d is not None]
            max_child_h = max(child_heights) if child_heights else 0.0
            if self.row_dend_param["split_by_cutree"] and self.row_dend_slice is not None:
                row_dend_max_h = dend_heights(self.row_dend_slice)
            elif self.row_dend_slice is not None and self.row_dend_param["cluster_slices"]:
                row_dend_max_h = dend_heights(self.row_dend_slice) + max_child_h
            else:
                row_dend_max_h = max_child_h
            if row_dend_max_h <= 0:
                row_dend_max_h = 1.0

            def _draw_row_dends() -> None:
                pad_u = as_unit(ht_opt.DENDROGRAM_PADDING, "mm")
                if rd_side == "left":
                    pushViewport(Viewport(x=unit(0, "npc"), width=unit(1, "npc") - pad_u, just="left"))
                else:
                    pushViewport(Viewport(x=pad_u, width=unit(1, "npc") - pad_u, just="left"))
                try:
                    for kr in range(1, nr_slice + 1):
                        self.draw_dend(which="row", k=kr, max_height=row_dend_max_h, y=slice_y[kr - 1], height=slice_height[kr - 1], just="top")
                    if nr_slice > 1 and self.row_dend_param["cluster_slices"] and self.row_dend_slice is not None:
                        pushViewport(Viewport(xscale=(0.0, float(row_dend_max_h)), yscale=(0.0, 1.0)))
                        try:
                            leaf_pos_units = []
                            for kr in range(nr_slice):
                                d_k = self.row_dend_list[kr]
                                if d_k is not None and d_k.x is None:
                                    d_k = adjust_dend_by_x(d_k)
                                p_k = float(d_k.x) / max(1, nobs(d_k)) if d_k is not None else 0.5
                                leaf_pos_units.append(slice_y[kr] - slice_height[kr] * p_k)
                            leaf_pos_u = unit_c(*leaf_pos_units)
                            slice_d = self.row_dend_slice
                            if not self.row_dend_param["split_by_cutree"]:
                                slice_d = merge_dendrogram(slice_d, [d for d in self.row_dend_list if d is not None], only_parent=True)
                            slice_d = adjust_dend_by_x(slice_d, leaf_pos_u)
                            grid_dendrogram(slice_d, gp=self.row_dend_param["gp"], facing="right" if rd_side == "left" else "left")
                            if not self.row_dend_param["split_by_cutree"] and self.heatmap_param["show_parent_dend_line"]:
                                x_line = unit(row_dend_max_h - max_child_h, "native") - unit(0.5, "mm") if rd_side == "left" else unit(max_child_h, "native") + unit(0.5, "mm")
                                grid_segments(x_line, unit(0.0, "npc"), x_line, unit(1.0, "npc"), gp=gpar(lty="dotted", col="#666666"))
                        finally:
                            popViewport()
                finally:
                    popViewport()

            graphic_funs.append(_draw_row_dends)

        # 5. Column Dendrogram
        if self.column_dend_param["show"] and any(d is not None and nobs(d) > 1 for d in self.column_dend_list):
            cd_side = self.column_dend_param["side"]
            cd_h = self.column_dend_param["height"]
            if cd_side == "top":
                layout_size["column_dend_top_height"] = cd_h
                layout_index.append(_heatmap_layout_index("column_dend_top"))
            else:
                layout_size["column_dend_bottom_height"] = cd_h
                layout_index.append(_heatmap_layout_index("column_dend_bottom"))

            child_heights_c = [dend_heights(d) for d in self.column_dend_list if d is not None]
            max_child_hc = max(child_heights_c) if child_heights_c else 0.0
            if self.column_dend_param["split_by_cutree"] and self.column_dend_slice is not None:
                col_dend_max_h = dend_heights(self.column_dend_slice)
            elif self.column_dend_slice is not None and self.column_dend_param["cluster_slices"]:
                col_dend_max_h = dend_heights(self.column_dend_slice) + max_child_hc
            else:
                col_dend_max_h = max_child_hc
            if col_dend_max_h <= 0:
                col_dend_max_h = 1.0

            def _draw_col_dends() -> None:
                pad_u = as_unit(ht_opt.DENDROGRAM_PADDING, "mm")
                if cd_side == "top":
                    pushViewport(Viewport(y=pad_u, height=unit(1, "npc") - pad_u, just="bottom"))
                else:
                    pushViewport(Viewport(y=unit(0, "npc"), height=unit(1, "npc") - pad_u, just="bottom"))
                try:
                    for kc in range(1, nc_slice + 1):
                        self.draw_dend(which="column", k=kc, max_height=col_dend_max_h, x=slice_x[kc - 1], width=slice_width[kc - 1], just="left")
                    if nc_slice > 1 and self.column_dend_param["cluster_slices"] and self.column_dend_slice is not None:
                        pushViewport(Viewport(xscale=(0.0, 1.0), yscale=(0.0, float(col_dend_max_h))))
                        try:
                            leaf_pos_units = []
                            for kc in range(nc_slice):
                                d_k = self.column_dend_list[kc]
                                if d_k is not None and d_k.x is None:
                                    d_k = adjust_dend_by_x(d_k)
                                p_k = float(d_k.x) / max(1, nobs(d_k)) if d_k is not None else 0.5
                                leaf_pos_units.append(slice_x[kc] + slice_width[kc] * p_k)
                            leaf_pos_u = unit_c(*leaf_pos_units)
                            slice_d = self.column_dend_slice
                            if not self.column_dend_param["split_by_cutree"]:
                                slice_d = merge_dendrogram(slice_d, [d for d in self.column_dend_list if d is not None], only_parent=True)
                            slice_d = adjust_dend_by_x(slice_d, leaf_pos_u)
                            grid_dendrogram(slice_d, gp=self.column_dend_param["gp"], facing="bottom" if cd_side == "top" else "top")
                            if not self.column_dend_param["split_by_cutree"] and self.heatmap_param["show_parent_dend_line"]:
                                y_line = unit(col_dend_max_h - max_child_hc, "native") - unit(0.5, "mm") if cd_side == "bottom" else unit(max_child_hc, "native") + unit(0.5, "mm")
                                grid_segments(unit(0.0, "npc"), y_line, unit(1.0, "npc"), y_line, gp=gpar(lty="dotted", col="#666666"))
                        finally:
                            popViewport()
                finally:
                    popViewport()

            graphic_funs.append(_draw_col_dends)

        # 6. Row Names
        if self.row_names_param["show"] and self.row_names_param["anno"] is not None:
            dim_pad_mm = convertWidth(as_unit(ht_opt.DIMNAME_PADDING, "mm"), "mm", valueOnly=True)
            rn_w_mm = convertWidth(self.row_names_param["anno"].width, "mm", valueOnly=True) + 2.0 * dim_pad_mm
            rn_max_mm = convertWidth(self.row_names_param["max_width"], "mm", valueOnly=True)
            rn_w_u = unit(min(rn_w_mm, rn_max_mm), "mm")
            if self.row_names_param["side"] == "left":
                layout_size["row_names_left_width"] = rn_w_u
                layout_index.append(_heatmap_layout_index("row_names_left"))
            else:
                layout_size["row_names_right_width"] = rn_w_u
                layout_index.append(_heatmap_layout_index("row_names_right"))

            def _draw_row_names() -> None:
                pad_mm_u = as_unit(ht_opt.DIMNAME_PADDING, "mm")
                for kr in range(1, nr_slice + 1):
                    self.draw_dimnames(which="row", k=kr, y=slice_y[kr - 1], height=slice_height[kr - 1], width=unit(1, "npc") - pad_mm_u * 2.0, just="top")

            graphic_funs.append(_draw_row_names)

        # 7. Column Names
        if self.column_names_param["show"] and self.column_names_param["anno"] is not None:
            dim_pad_mm = convertHeight(as_unit(ht_opt.DIMNAME_PADDING, "mm"), "mm", valueOnly=True)
            cn_h_mm = convertHeight(self.column_names_param["anno"].height, "mm", valueOnly=True) + 2.0 * dim_pad_mm
            cn_max_mm = convertHeight(self.column_names_param["max_height"], "mm", valueOnly=True)
            cn_h_u = unit(min(cn_h_mm, cn_max_mm), "mm")
            if self.column_names_param["side"] == "top":
                layout_size["column_names_top_height"] = cn_h_u
                layout_index.append(_heatmap_layout_index("column_names_top"))
            else:
                layout_size["column_names_bottom_height"] = cn_h_u
                layout_index.append(_heatmap_layout_index("column_names_bottom"))

            def _draw_col_names() -> None:
                pad_mm_u = as_unit(ht_opt.DIMNAME_PADDING, "mm")
                for kc in range(1, nc_slice + 1):
                    self.draw_dimnames(which="column", k=kc, x=slice_x[kc - 1], width=slice_width[kc - 1], height=unit(1, "npc") - pad_mm_u * 2.0, just="left")

            graphic_funs.append(_draw_col_names)

        # 8. Top / Bottom / Left / Right Annotations
        if self.top_annotation is not None and len(self.top_annotation) > 0:
            layout_size["column_anno_top_height"] = self.top_annotation.height + as_unit(ht_opt.COLUMN_ANNO_PADDING, "mm")
            layout_index.append(_heatmap_layout_index("column_anno_top"))

            def _draw_top_anno() -> None:
                pad = as_unit(ht_opt.COLUMN_ANNO_PADDING, "mm")
                for kc in range(1, nc_slice + 1):
                    self.draw_annotation("top", k=kc, x=slice_x[kc - 1], width=slice_width[kc - 1], y=pad, height=unit(1, "npc") - pad, just=("left", "bottom"))

            graphic_funs.append(_draw_top_anno)

        if self.bottom_annotation is not None and len(self.bottom_annotation) > 0:
            layout_size["column_anno_bottom_height"] = self.bottom_annotation.height + as_unit(ht_opt.COLUMN_ANNO_PADDING, "mm")
            layout_index.append(_heatmap_layout_index("column_anno_bottom"))

            def _draw_bottom_anno() -> None:
                pad = as_unit(ht_opt.COLUMN_ANNO_PADDING, "mm")
                for kc in range(1, nc_slice + 1):
                    self.draw_annotation("bottom", k=kc, x=slice_x[kc - 1], width=slice_width[kc - 1], y=unit(0, "npc"), height=unit(1, "npc") - pad, just=("left", "bottom"))

            graphic_funs.append(_draw_bottom_anno)

        if self.left_annotation is not None and len(self.left_annotation) > 0:
            layout_size["row_anno_left_width"] = self.left_annotation.width + as_unit(ht_opt.ROW_ANNO_PADDING, "mm")
            layout_index.append(_heatmap_layout_index("row_anno_left"))

            def _draw_left_anno() -> None:
                pad = as_unit(ht_opt.ROW_ANNO_PADDING, "mm")
                for kr in range(1, nr_slice + 1):
                    self.draw_annotation("left", k=kr, y=slice_y[kr - 1], height=slice_height[kr - 1], x=unit(0, "npc"), width=unit(1, "npc") - pad, just=("left", "top"))

            graphic_funs.append(_draw_left_anno)

        if self.right_annotation is not None and len(self.right_annotation) > 0:
            layout_size["row_anno_right_width"] = self.right_annotation.width + as_unit(ht_opt.ROW_ANNO_PADDING, "mm")
            layout_index.append(_heatmap_layout_index("row_anno_right"))

            def _draw_right_anno() -> None:
                pad = as_unit(ht_opt.ROW_ANNO_PADDING, "mm")
                for kr in range(1, nr_slice + 1):
                    self.draw_annotation("right", k=kr, y=slice_y[kr - 1], height=slice_height[kr - 1], x=pad, width=unit(1, "npc") - pad, just=("left", "top"))

            graphic_funs.append(_draw_right_anno)

        # 9. Resolve heatmap_width vs matrix_param["width"] and heatmap_height vs matrix_param["height"]
        fixed_w_mm = sum(convertWidth(layout_size[f"{c}_width"], "mm", valueOnly=True) for c in HEATMAP_LAYOUT_ROW_COMPONENTS if c != "heatmap_body")
        fixed_h_mm = sum(convertHeight(layout_size[f"{c}_height"], "mm", valueOnly=True) for c in HEATMAP_LAYOUT_COLUMN_COMPONENTS if c != "heatmap_body")

        if is_abs_unit(self.heatmap_param["width"]):
            tot_w_mm = convertWidth(self.heatmap_param["width"], "mm", valueOnly=True)
            body_w_mm = tot_w_mm - fixed_w_mm
            if body_w_mm <= 0:
                raise ValueError("Computed heatmap body width is non-positive; `heatmap_width` is too small for titles/dends/names/annotations.")
            self.matrix_param["width"] = unit(body_w_mm, "mm")
        elif is_abs_unit(self.matrix_param["width"]):
            body_w_mm = convertWidth(self.matrix_param["width"], "mm", valueOnly=True)
            self.heatmap_param["width"] = unit(fixed_w_mm + body_w_mm, "mm")
        else:
            self.heatmap_param["width"] = unit(1.0, "npc")

        if is_abs_unit(self.heatmap_param["height"]):
            tot_h_mm = convertHeight(self.heatmap_param["height"], "mm", valueOnly=True)
            body_h_mm = tot_h_mm - fixed_h_mm
            if body_h_mm <= 0:
                raise ValueError("Computed heatmap body height is non-positive; `heatmap_height` is too small for titles/dends/names/annotations.")
            self.matrix_param["height"] = unit(body_h_mm, "mm")
        elif is_abs_unit(self.matrix_param["height"]):
            body_h_mm = convertHeight(self.matrix_param["height"], "mm", valueOnly=True)
            self.heatmap_param["height"] = unit(fixed_h_mm + body_h_mm, "mm")
        else:
            self.heatmap_param["height"] = unit(1.0, "npc")

        self.layout["layout_index"] = layout_index
        self.layout["graphic_fun_list"] = graphic_funs
        self.layout["initialized"] = True
        return self

    def component_widths_mm(self) -> list[float | None]:
        """Return widths of the 9 row components in mm (`None` means flexible/npc body)."""
        ls = self.layout["layout_size"]
        out: list[float | None] = []
        for c in HEATMAP_LAYOUT_ROW_COMPONENTS:
            if c == "heatmap_body":
                bw = self.matrix_param["width"]
                out.append(convertWidth(bw, "mm", valueOnly=True) if is_abs_unit(bw) else None)
            else:
                out.append(convertWidth(ls[f"{c}_width"], "mm", valueOnly=True))
        return out

    def component_heights_mm(self) -> list[float | None]:
        """Return heights of the 9 column components in mm (`None` means flexible/npc body)."""
        ls = self.layout["layout_size"]
        out: list[float | None] = []
        for c in HEATMAP_LAYOUT_COLUMN_COMPONENTS:
            if c == "heatmap_body":
                bh = self.matrix_param["height"]
                out.append(convertHeight(bh, "mm", valueOnly=True) if is_abs_unit(bh) else None)
            else:
                out.append(convertHeight(ls[f"{c}_height"], "mm", valueOnly=True))
        return out

    # ---- Component Drawing ----

    def draw_heatmap_body(self, kr: int = 1, kc: int = 1, **vp_kwargs: Any) -> None:
        r_idx = self.row_order_list[kr - 1]
        c_idx = self.column_order_list[kc - 1]
        nr_s = len(r_idx)
        nc_s = len(c_idx)
        vp_name = f"{self.name}_heatmap_body_{kr}_{kc}"

        pushViewport(Viewport(name=vp_name, xscale=(0.0, float(max(1, nc_s))), yscale=(0.0, float(max(1, nr_s))), **vp_kwargs))
        try:
            if nr_s > 0 and nc_s > 0:
                submat = self.matrix[np.ix_(r_idx, c_idx)]
                rect_gp = self.matrix_param["rect_gp"]
                rect_type = rect_gp.get("type", None)
                use_raster = bool(self.heatmap_param["use_raster"])
                need_col_mat = (not use_raster) or (self.matrix_param["layer_fun"] is not None) or (self.matrix_param["cell_fun"] is not None)
                if need_col_mat and self.matrix_color_mapping is not None and self._oncoprint_env is None and (
                    rect_type != "none" or self.matrix_param["layer_fun"] is not None or self.matrix_param["cell_fun"] is not None
                ):
                    col_mat = self.matrix_color_mapping.map_to_colors(submat)
                else:
                    col_mat = np.full((nr_s, nc_s), "#FFFFFFFF", dtype=object)

                # Vectorized cell coordinates in npc (row 0 at top -> y = (nr_s - r - 0.5)/nr_s)
                if rect_type != "none":
                    if use_raster:
                        resize_opt = self.heatmap_param["raster_resize_mat"]
                        max_w = int(ht_opt.raster_temp_image_max_width or 30000)
                        max_h = int(ht_opt.raster_temp_image_max_height or 30000)
                        need_downsample = (resize_opt and self._is_numeric and (nr_s > 400 or nc_s > 400)) or nr_s > max_h or nc_s > max_w
                        if need_downsample and self._is_numeric:
                            fn = np.nanmean if (resize_opt is True or not callable(resize_opt)) else resize_opt
                            tgt_r = min(nr_s, 400 if resize_opt else max_h)
                            tgt_c = min(nc_s, 400 if resize_opt else max_w)
                            r_bins = np.array_split(np.arange(nr_s), tgt_r)
                            c_bins = np.array_split(np.arange(nc_s), tgt_c)
                            sub_small = np.empty((tgt_r, tgt_c), dtype=float)
                            for ri, rb in enumerate(r_bins):
                                for ci, cb in enumerate(c_bins):
                                    with np.errstate(all="ignore"):
                                        sub_small[ri, ci] = fn(submat[np.ix_(rb, cb)])
                            if self.matrix_color_mapping is not None:
                                rgba_img = self.matrix_color_mapping.map_to_rgba(sub_small)
                            else:
                                rgba_img = np.ones((tgt_r, tgt_c, 4), dtype=float)
                        else:
                            if self.matrix_color_mapping is not None:
                                rgba_img = self.matrix_color_mapping.map_to_rgba(submat)
                            else:
                                rgba_img = np.ones((nr_s, nc_s, 4), dtype=float)
                        grid_raster(rgba_img, x=0.5, y=0.5, width=1.0, height=1.0, interpolate=False)
                    else:
                        # Column-major flattening to align with layer_fun & restore_matrix
                        x_vec = np.repeat((np.arange(nc_s, dtype=float) + 0.5) / nc_s, nr_s)
                        y_vec = np.tile((nr_s - np.arange(nr_s, dtype=float) - 0.5) / nr_s, nc_s)
                        fill_vec = [col_mat[ri, ci] for ci in range(nc_s) for ri in range(nr_s)]
                        gp_cells = check_gp(rect_gp)
                        gp_cells["fill"] = fill_vec
                        grid_rect(
                            x=unit(x_vec, "npc"),
                            y=unit(y_vec, "npc"),
                            width=unit(1.0 / nc_s, "npc"),
                            height=unit(1.0 / nr_s, "npc"),
                            gp=gp_cells,
                        )

                # layer_fun (vectorized)
                layer_fun = self.matrix_param["layer_fun"]
                if layer_fun is not None:
                    j_vec = np.repeat(np.asarray(c_idx, dtype=int), nr_s)
                    i_vec = np.tile(np.asarray(r_idx, dtype=int), nc_s)
                    x_u = unit(np.repeat((np.arange(nc_s, dtype=float) + 0.5) / nc_s, nr_s), "npc")
                    y_u = unit(np.tile((nr_s - np.arange(nr_s, dtype=float) - 0.5) / nr_s, nc_s), "npc")
                    w_u = unit(np.full(nr_s * nc_s, 1.0 / nc_s), "npc")
                    h_u = unit(np.full(nr_s * nc_s, 1.0 / nr_s), "npc")
                    fill_vec = np.array([col_mat[ri, ci] for ci in range(nc_s) for ri in range(nr_s)], dtype=object)

                    sig = inspect.signature(layer_fun)
                    pos_params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                    has_var = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
                    if len(pos_params) >= 9:
                        layer_fun(j_vec, i_vec, x_u, y_u, w_u, h_u, fill_vec, kr, kc)
                    elif has_var:
                        # Pass 7 or check? In R 9 args vs 7 args: default 7 unless 9 requested
                        layer_fun(j_vec, i_vec, x_u, y_u, w_u, h_u, fill_vec)
                    else:
                        layer_fun(j_vec, i_vec, x_u, y_u, w_u, h_u, fill_vec)

                # cell_fun (per cell)
                cell_fun = self.matrix_param["cell_fun"]
                if cell_fun is not None:
                    w_cell = unit(1.0 / nc_s, "npc")
                    h_cell = unit(1.0 / nr_s, "npc")
                    for ri, orig_i in enumerate(r_idx):
                        y_cell = unit((nr_s - ri - 0.5) / nr_s, "npc")
                        for ci, orig_j in enumerate(c_idx):
                            x_cell = unit((ci + 0.5) / nc_s, "npc")
                            cell_fun(int(orig_j), int(orig_i), x_cell, y_cell, w_cell, h_cell, col_mat[ri, ci])

                # Slice border
                border = self.matrix_param["border"]
                if border is True or (isinstance(border, str) and border.lower() not in ("", "na", "none")):
                    bgp = dict(self.matrix_param["border_gp"])
                    bgp["fill"] = "transparent"
                    if isinstance(border, str):
                        bgp["col"] = border
                    grid_rect(gp=bgp)
        finally:
            popViewport()

    def draw_dend(self, which: str = "row", k: int = 1, max_height: float | None = None, **vp_kwargs: Any) -> None:
        if which == "row":
            side = self.row_dend_param["side"]
            dend = self.row_dend_list[k - 1]
            gp = self.row_dend_param["gp"]
        else:
            side = self.column_dend_param["side"]
            dend = self.column_dend_list[k - 1]
            gp = self.column_dend_param["gp"]

        if dend is None or nobs(dend) <= 1:
            return
        if max_height is None:
            max_height = dend_heights(dend)
        if max_height <= 0:
            max_height = 1.0

        nb = float(nobs(dend))
        if side in ("left", "right"):
            xscale = (0.0, float(max_height))
            yscale = (0.0, nb)
            vp_name = f"{self.name}_dend_row_{k}"
        else:
            xscale = (0.0, nb)
            yscale = (0.0, float(max_height))
            vp_name = f"{self.name}_dend_column_{k}"

        pushViewport(Viewport(**vp_kwargs))
        pushViewport(Viewport(name=vp_name, xscale=xscale, yscale=yscale, width=unit(1, "npc"), height=unit(1, "npc")))
        try:
            facing_map = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
            order_mode = "reverse" if side in ("left", "right") else "normal"
            grid_dendrogram(dend, gp=gp, facing=facing_map[side], order=order_mode)
        finally:
            popViewport(2)

    def draw_dimnames(self, which: str = "row", k: int = 1, **vp_kwargs: Any) -> None:
        if which == "row":
            anno = self.row_names_param["anno"]
            ind = self.row_order_list[k - 1]
        else:
            anno = self.column_names_param["anno"]
            ind = self.column_order_list[k - 1]
        if anno is None or not ind:
            return

        vp_name = f"{self.name}_{which}_names_{k}"
        pushViewport(Viewport(name=vp_name, **vp_kwargs))
        try:
            anno.draw(ind, k=k, n=len(self.row_order_list if which == "row" else self.column_order_list))
        finally:
            popViewport()

    def draw_title(self, which: str = "row", k: int = 1, **vp_kwargs: Any) -> None:
        if which == "row":
            side = self.row_title_param["side"]
            gp = subset_gp(self.row_title_param["gp"], k - 1)
            title = self.row_title[k - 1]
            rot = self.row_title_param["rot"]
            just = self.row_title_param["just"]
            vp_name = f"{self.name}_row_title_{k}"
        else:
            side = self.column_title_param["side"]
            gp = subset_gp(self.column_title_param["gp"], k - 1)
            title = self.column_title[k - 1]
            rot = self.column_title_param["rot"]
            just = self.column_title_param["just"]
            vp_name = f"{self.name}_column_title_{k}"

        pad = _title_padding_unit(gp)
        pushViewport(Viewport(name=vp_name, clip=False, **vp_kwargs))
        try:
            gp2 = dict(gp)
            if "border" in gp2:
                gp2["col"] = gp2["border"]
            if "border" in gp or "fill" in gp:
                grid_rect(gp=gp2)
            if which == "row":
                if side == "left":
                    grid_text(title, x=unit(1, "npc") - pad[0], y=unit(0.5, "npc"), rot=rot, just=just, gp=gp)
                else:
                    grid_text(title, x=pad[0], y=unit(0.5, "npc"), rot=rot, just=just, gp=gp)
            else:
                if side == "top":
                    grid_text(title, x=unit(0.5, "npc"), y=pad[0], rot=rot, just=just, gp=gp)
                else:
                    grid_text(title, x=unit(0.5, "npc"), y=unit(1, "npc") - pad[0], rot=rot, just=just, gp=gp)
        finally:
            popViewport()

    def draw_annotation(self, which: str = "top", k: int = 1, **vp_kwargs: Any) -> None:
        annotation = getattr(self, f"{which}_annotation")
        if annotation is None or len(annotation) == 0:
            return
        if which in ("top", "bottom"):
            index = self.column_order_list[k - 1]
            n = len(self.column_order_list)
        else:
            index = self.row_order_list[k - 1]
            n = len(self.row_order_list)

        # Build anno_mark_param when multi-slice has anno_mark / anno_zoom
        anno_mark_param: dict[str, Any] = {}
        anno_types = [sa.fun.fun_name for sa in annotation.anno_list.values()]
        if any(t in ("anno_mark", "anno_zoom") for t in anno_types):
            if which in ("left", "right"):
                slice_y = self.layout["slice"]["y"]
                slice_h = self.layout["slice"]["height"]
                ro_lt = self.row_order_list
                if len(ro_lt) > 1:
                    pos_units = []
                    for i_s, idx_s in enumerate(ro_lt):
                        m_s = max(1, len(idx_s))
                        rel = (np.arange(m_s, dtype=float) + 0.5) / m_s
                        for r_f in rel:
                            pos_units.append(slice_y[i_s] - slice_h[i_s] * r_f)
                    anno_mark_param[".scale"] = (0.0, 1.0)
                    anno_mark_param[".pos"] = convertHeight(unit_c(*pos_units), "npc", valueOnly=True) if False else None
                    # Convert inside vp once pushed or store Unit list!
                    anno_mark_param["_pos_unit"] = unit_c(*pos_units)
                    anno_mark_param["index"] = [idx for sl in ro_lt for idx in sl]
            else:
                slice_x = self.layout["slice"]["x"]
                slice_w = self.layout["slice"]["width"]
                co_lt = self.column_order_list
                if len(co_lt) > 1:
                    pos_units = []
                    for i_s, idx_s in enumerate(co_lt):
                        m_s = max(1, len(idx_s))
                        rel = (np.arange(m_s, dtype=float) + 0.5) / m_s
                        for c_f in rel:
                            pos_units.append(slice_x[i_s] + slice_w[i_s] * c_f)
                    anno_mark_param[".scale"] = (0.0, 1.0)
                    anno_mark_param["_pos_unit"] = unit_c(*pos_units)
                    anno_mark_param["index"] = [idx for sl in co_lt for idx in sl]

        prev_ht = _ENV.current_heatmap
        _ENV.current_heatmap = self
        try:
            if "_pos_unit" in anno_mark_param:
                u_p = anno_mark_param.pop("_pos_unit")
                from .grid import current_viewport
                full_vp = current_viewport()
                if which in ("left", "right"):
                    anno_mark_param[".pos"] = u_p.to_mm("y", full_vp, pos_mode=True) / max(1e-9, full_vp.height_mm)
                    anno_mark_param["vp_height"] = unit(full_vp.height_mm / 10.0, "cm")
                    anno_mark_param["vp_width"] = unit(1.0, "npc")
                    anno_mark_param["vp_just"] = "top"
                    anno_mark_param["vp_x"] = unit(0.5, "npc")
                    anno_mark_param["vp_y"] = unit(1.0, "npc")
                else:
                    anno_mark_param[".pos"] = u_p.to_mm("x", full_vp, pos_mode=True) / max(1e-9, full_vp.width_mm)
                    anno_mark_param["vp_height"] = unit(1.0, "npc")
                    anno_mark_param["vp_width"] = unit(full_vp.width_mm / 10.0, "cm")
                    anno_mark_param["vp_just"] = "left"
                    anno_mark_param["vp_x"] = unit(0.0, "npc")
                    anno_mark_param["vp_y"] = unit(0.5, "npc")
            pushViewport(Viewport(**vp_kwargs))
            try:
                annotation.draw(index=index, k=k, n=n, anno_mark_param=anno_mark_param)
            finally:
                popViewport()
        finally:
            _ENV.current_heatmap = prev_ht

    def draw_in_viewport(self) -> None:
        """Render the 9x9 layout grid of this prepared Heatmap in the current viewport."""
        if not self.layout["initialized"]:
            self.prepare()

        from .grid import current_viewport

        parent_vp = current_viewport()
        col_widths_mm = self.component_widths_mm()
        row_heights_mm = self.component_heights_mm()

        fixed_w = sum(w for w in col_widths_mm if w is not None)
        fixed_h = sum(h for h in row_heights_mm if h is not None)
        rem_w = max(0.0, parent_vp.width_mm - fixed_w)
        rem_h = max(0.0, parent_vp.height_mm - fixed_h)

        final_w = [rem_w if w is None else w for w in col_widths_mm]
        final_h = [rem_h if h is None else h for h in row_heights_mm]

        # Compute cell boxes inside parent_vp (row 0 is top, col 0 is left)
        x_starts = [0.0] * 9
        cur_x = 0.0
        for c in range(9):
            x_starts[c] = cur_x
            cur_x += final_w[c]

        y_tops = [0.0] * 9
        cur_y = sum(final_h)
        for r in range(9):
            y_tops[r] = cur_y
            cur_y -= final_h[r]

        for (r_idx, c_idx), gfun in zip(self.layout["layout_index"], self.layout["graphic_fun_list"]):
            w_cell = final_w[c_idx]
            h_cell = final_h[r_idx]
            pushViewport(
                Viewport(
                    x=unit(x_starts[c_idx], "mm"),
                    y=unit(y_tops[r_idx], "mm"),
                    width=unit(w_cell, "mm"),
                    height=unit(h_cell, "mm"),
                    just=("left", "top"),
                )
            )
            try:
                gfun()
            finally:
                popViewport()

    # ---- Subsetting ----

    def __getitem__(self, key: Any) -> "Heatmap":
        if self.layout["initialized"]:
            raise RuntimeError("Subsetting is only allowed on a Heatmap before layout initialization.")
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Heatmap expects 1D (rows) or 2D (rows, cols) indexing.")
            i_key, j_key = key
        else:
            i_key, j_key = key, slice(None)

        cp = self.copy_all()
        if not (isinstance(i_key, slice) and i_key == slice(None)):
            cp = _subset_heatmap_rows(cp, i_key)
        if not (isinstance(j_key, slice) and j_key == slice(None)):
            cp = _subset_heatmap_cols(cp, j_key)
        return cp

    # ---- High-level Draw ----

    def draw(self, **kwargs: Any) -> Any:
        from .heatmap_list import HeatmapList
        hl = HeatmapList([self], direction="horizontal")
        return hl.draw(**kwargs)

    def show(self) -> Any:
        return self.draw()

    def to_svg(self, **kwargs: Any) -> str:
        from .heatmap_list import HeatmapList
        return HeatmapList([self], direction="horizontal").to_svg(**kwargs)

    def summary(self) -> str:
        lines = [
            f"A Heatmap object: {self.name}",
            f"  dimension: {self.nrow} x {self.ncol}",
            f"  color mapping: {self.matrix_color_mapping.type if self.matrix_color_mapping else 'none'}",
            f"  cluster rows: {self.row_dend_param['cluster']}",
            f"  cluster columns: {self.column_dend_param['cluster']}",
        ]
        txt = "\n".join(lines)
        print(txt)
        return txt

    def _repr_png_(self) -> bytes:
        import io
        import matplotlib.pyplot as plt
        dev = self.draw(show=False)
        buf = io.BytesIO()
        dev._device.fig.savefig(buf, format="png", dpi=120)
        plt.close(dev._device.fig)
        return buf.getvalue()

    def __repr__(self) -> str:
        return f"Heatmap(name={self.name!r}, shape=({self.nrow}, {self.ncol}))"


def _resolve_index_list(idx_spec: Any, n: int, names_list: list[str] | None) -> list[int]:
    if isinstance(idx_spec, slice):
        return list(range(*idx_spec.indices(n)))
    if isinstance(idx_spec, (int, np.integer)):
        i = int(idx_spec)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"Index {idx_spec} out of range for length {n}.")
        return [i]
    if isinstance(idx_spec, str):
        if names_list is None:
            raise KeyError("Cannot subset by name when names are absent.")
        return [names_list.index(idx_spec)]
    seq = list(idx_spec)
    if seq and all(isinstance(x, (bool, np.bool_)) for x in seq):
        if len(seq) != n:
            raise IndexError(f"Boolean mask length ({len(seq)}) must equal {n}.")
        return [i for i, fl in enumerate(seq) if fl]
    if seq and all(isinstance(x, str) for x in seq):
        if names_list is None:
            raise KeyError("Cannot subset by name when names are absent.")
        lookup = {nm: i for i, nm in enumerate(names_list)}
        return [lookup[s] for s in seq]
    return [int(x) + (n if int(x) < 0 else 0) for x in seq]


def _subset_heatmap_rows(ht: Heatmap, i_spec: Any) -> Heatmap:
    if ht.row_dend_param["obj"] is not None:
        raise ValueError("Cannot subset rows when `cluster_rows` is an explicit Dendrogram/Hclust.")
    ind = _resolve_index_list(i_spec, ht.nrow, ht._rownames)
    ind_set = set(ind)
    ht.matrix = ht.matrix[ind, :]
    if ht._rownames is not None:
        ht._rownames = [ht._rownames[i] for i in ind]
    if ht.row_names_param["labels"] is not None:
        lbl = ht.row_names_param["labels"]
        ht.row_names_param["labels"] = lbl[ind] if isinstance(lbl, GridText) else [lbl[i] for i in ind]
        ht.row_names_param["anno"] = ht.row_names_param["anno"][ind]
    ht.row_names_param["gp"] = subset_gp(ht.row_names_param["gp"], ind)
    # Remap manual row_order preserving relative positions
    pos_map = {old_i: new_i for new_i, old_i in enumerate(ind)}
    kept = [pos_map[old_i] for old_i in ht.row_order if old_i in ind_set]
    ht.row_order = kept if len(kept) == len(ind) else list(range(len(ind)))
    ht.row_order_list = [list(ht.row_order)]
    if ht.matrix_param["row_split"] is not None and not isinstance(ht.matrix_param["row_split"], (int, np.integer)):
        rs = ht.matrix_param["row_split"]
        if isinstance(rs, pd.DataFrame):
            ht.matrix_param["row_split"] = rs.iloc[ind].reset_index(drop=True)
        else:
            ht.matrix_param["row_split"] = [rs[i] for i in ind]
    if isinstance(ht.row_dend_param["reorder"], (list, tuple, np.ndarray, pd.Series)):
        ht.row_dend_param["reorder"] = [ht.row_dend_param["reorder"][i] for i in ind]
    if ht.left_annotation is not None:
        ht.left_annotation = ht.left_annotation[ind]
    if ht.right_annotation is not None:
        ht.right_annotation = ht.right_annotation[ind]
    if ht.nrow <= 1:
        ht.row_dend_param["cluster"] = False
        ht.row_dend_param["show"] = False
    return ht


def _subset_heatmap_cols(ht: Heatmap, j_spec: Any) -> Heatmap:
    if ht.column_dend_param["obj"] is not None:
        raise ValueError("Cannot subset columns when `cluster_columns` is an explicit Dendrogram/Hclust.")
    ind = _resolve_index_list(j_spec, ht.ncol, ht._colnames)
    ind_set = set(ind)
    ht.matrix = ht.matrix[:, ind]
    if ht._colnames is not None:
        ht._colnames = [ht._colnames[j] for j in ind]
    if ht.column_names_param["labels"] is not None:
        lbl = ht.column_names_param["labels"]
        ht.column_names_param["labels"] = lbl[ind] if isinstance(lbl, GridText) else [lbl[j] for j in ind]
        ht.column_names_param["anno"] = ht.column_names_param["anno"][ind]
    ht.column_names_param["gp"] = subset_gp(ht.column_names_param["gp"], ind)
    pos_map = {old_j: new_j for new_j, old_j in enumerate(ind)}
    kept = [pos_map[old_j] for old_j in ht.column_order if old_j in ind_set]
    ht.column_order = kept if len(kept) == len(ind) else list(range(len(ind)))
    ht.column_order_list = [list(ht.column_order)]
    if ht.matrix_param["column_split"] is not None and not isinstance(ht.matrix_param["column_split"], (int, np.integer)):
        cs = ht.matrix_param["column_split"]
        if isinstance(cs, pd.DataFrame):
            ht.matrix_param["column_split"] = cs.iloc[ind].reset_index(drop=True)
        else:
            ht.matrix_param["column_split"] = [cs[j] for j in ind]
    if isinstance(ht.column_dend_param["reorder"], (list, tuple, np.ndarray, pd.Series)):
        ht.column_dend_param["reorder"] = [ht.column_dend_param["reorder"][j] for j in ind]
    if ht.top_annotation is not None:
        ht.top_annotation = ht.top_annotation[ind]
    if ht.bottom_annotation is not None:
        ht.bottom_annotation = ht.bottom_annotation[ind]
    if ht.ncol <= 1:
        ht.column_dend_param["cluster"] = False
        ht.column_dend_param["show"] = False
    return ht
