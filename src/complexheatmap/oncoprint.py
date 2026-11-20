"""OncoPrint plots, alteration parsing, memoSort ordering, and alteration graphics."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._options import _ENV, ht_opt
from .annotations import (
    HeatmapAnnotation,
    anno_oncoprint_barplot,
    anno_text,
    rowAnnotation,
)
from .colors import brewer_pal, to_hex_rgba
from .grid import (
    Viewport,
    grid_newpage,
    grid_points,
    grid_polygon,
    grid_rect,
    grid_segments,
    grid_text,
    max_text_width,
    popViewport,
    pushViewport,
)
from .heatmap import Heatmap
from .legend import Legend
from .units import Unit, as_unit, check_gp, convertWidth, gpar, unit


def default_get_type(x: Any) -> list[str]:
    """Split alteration string on `;`, `:`, `,`, or `|` (`default_get_type.Rd`)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if s == "" or s.upper() == "NA":
        return []
    parts = [p.strip() for p in re.split(r"[;:,|]", s)]
    return [p for p in parts if p != "" and p.upper() != "NA"]


def alter_graphic(
    graphic: str = "rect",
    width: float = 1.0,
    height: float = 1.0,
    horiz_margin: Any = unit(1, "pt"),
    vertical_margin: Any = unit(1, "pt"),
    fill: str = "red",
    col: str | None = None,
    pch: Any = 16,
    size: Any = unit(1, "char"),
    gp: dict[str, Any] | None = None,
    **extra_gp: Any,
) -> Callable[[Any, Any, Any, Any], None]:
    """Factory returning an alteration graphic drawer `fun(x, y, w, h)` (`alter_graphic.Rd`)."""
    graphic = str(graphic).lower()
    if graphic not in ("rect", "point"):
        raise ValueError("`graphic` in `alter_graphic` must be 'rect' or 'point'.")
    hm_u = as_unit(horiz_margin, "mm")
    vm_u = as_unit(vertical_margin, "mm")
    base_gp = check_gp(gp or {})
    if fill is not None:
        base_gp["fill"] = to_hex_rgba(fill)
    if col is not None:
        base_gp["col"] = to_hex_rgba(col)
    elif "col" not in base_gp:
        base_gp["col"] = "#00000000" if graphic == "rect" else to_hex_rgba(fill)
    for k, v in extra_gp.items():
        base_gp[k] = v

    if graphic == "rect":
        def _draw_rect(x: Any, y: Any, w: Any, h: Any) -> None:
            ux = as_unit(x, "npc")
            uy = as_unit(y, "npc")
            uw = as_unit(w, "npc") * float(width) - hm_u * 2.0
            uh = as_unit(h, "npc") * float(height) - vm_u * 2.0
            grid_rect(ux, uy, uw, uh, gp=base_gp)
        return _draw_rect

    sz_u = as_unit(size, "mm")
    def _draw_point(x: Any, y: Any, w: Any, h: Any) -> None:
        grid_points(as_unit(x, "npc"), as_unit(y, "npc"), pch=pch, size=sz_u, gp=base_gp)
    return _draw_point


def unify_mat_list(mat_list: Sequence[Any] | dict[str, Any], default: Any = 0) -> Any:
    """Align a list/dict of named matrices or DataFrames to the union of row/col names (`unify_mat_list.Rd`)."""
    is_dict = isinstance(mat_list, dict)
    keys = list(mat_list.keys()) if is_dict else None
    items = list(mat_list.values()) if is_dict else list(mat_list)
    if not items:
        return {} if is_dict else []

    dfs: list[pd.DataFrame] = []
    for m in items:
        if isinstance(m, pd.DataFrame):
            dfs.append(m.copy())
        else:
            arr = np.asarray(m)
            dfs.append(pd.DataFrame(arr))

    all_rows: list[Any] = []
    all_cols: list[Any] = []
    seen_r: set[Any] = set()
    seen_c: set[Any] = set()
    for df in dfs:
        for r in df.index:
            if r not in seen_r:
                seen_r.add(r)
                all_rows.append(r)
        for c in df.columns:
            if c not in seen_c:
                seen_c.add(c)
                all_cols.append(c)

    out_dfs = []
    for df in dfs:
        re = df.reindex(index=all_rows, columns=all_cols, fill_value=default)
        out_dfs.append(re)

    if is_dict:
        assert keys is not None
        return dict(zip(keys, out_dfs))
    return out_dfs


def _build_default_alter_fun(all_type: list[str], col_map: dict[str, str]) -> dict[str, Callable[..., Any]]:
    funs: dict[str, Callable[..., Any]] = {
        "background": alter_graphic("rect", fill="#CCCCCC", horiz_margin=unit(1, "pt"), vertical_margin=unit(1, "pt"))
    }
    for idx, tp in enumerate(all_type):
        # Match classic shrinking height layers for overlapping events
        h_frac = max(0.3, 1.0 - 0.15 * min(idx, 4))
        funs[tp] = alter_graphic(
            "rect",
            width=1.0,
            height=h_frac,
            horiz_margin=unit(1, "pt"),
            vertical_margin=unit(1, "pt"),
            fill=col_map.get(tp, "#E41A1C"),
        )
    return funs


def test_alter_fun(
    alter_fun: Callable[..., Any] | dict[str, Callable[..., Any]],
    type: Sequence[str] | None = None,
    row_names_gp: dict[str, Any] | None = None,
    width: float = 4.0,
    height: float = 3.0,
) -> Any:
    """Preview all alteration graphics in `alter_fun` as a visual grid (`test_alter_fun.Rd`)."""
    if isinstance(alter_fun, dict):
        keys = [k for k in alter_fun.keys() if k != "background"] if type is None else list(type)
        has_bg = "background" in alter_fun
    else:
        if type is None:
            raise ValueError("`type` must be supplied when `alter_fun` is a single function.")
        keys = list(type)
        has_bg = False

    n = len(keys)
    dev = grid_newpage(width_in=width, height_in=height)
    gp_lbl = check_gp(row_names_gp or gpar(fontsize=10))
    pushViewport(Viewport(x=unit(0.1, "npc"), y=unit(0.5, "npc"), width=unit(0.8, "npc"), height=unit(0.8, "npc")))
    try:
        for idx, tp in enumerate(keys):
            y_pos = unit((n - idx - 0.5) / max(1, n), "npc")
            h_cell = unit(0.8 / max(1, n), "npc")
            w_cell = unit(0.3, "npc")
            x_cell = unit(0.25, "npc")
            if isinstance(alter_fun, dict):
                if has_bg:
                    alter_fun["background"](x_cell, y_pos, w_cell, h_cell)
                if tp in alter_fun:
                    alter_fun[tp](x_cell, y_pos, w_cell, h_cell)
            else:
                alter_fun(x_cell, y_pos, w_cell, h_cell, tp)
            grid_text(tp, x=unit(0.5, "npc"), y=y_pos, just="left", gp=gp_lbl)
    finally:
        popViewport()
    return dev


def _memo_sort_columns(arr: np.ndarray, gene_order: list[int]) -> list[int]:
    """Classic OncoPrint memoSort across columns given ordered genes `gene_order`."""
    ng, ns, nt = arr.shape
    if ns == 0:
        return []
    any_alt = np.any(arr, axis=2)  # (ng, ns)
    # Build lexicographic keys over ordered genes (altered before non-altered; among altered, earliest type)
    keys = []
    for s in range(ns):
        gene_key = []
        for g in gene_order:
            if any_alt[g, s]:
                t_idx = int(np.argmax(arr[g, s, :]))
                gene_key.append((0, t_idx))
            else:
                gene_key.append((1, nt))
        total_cnt = -int(np.sum(any_alt[:, s]))
        keys.append((tuple(gene_key), total_cnt, s))
    keys.sort()
    return [k[-1] for k in keys]


def oncoPrint(
    mat: Any,
    get_type: Callable[[Any], Sequence[str]] = default_get_type,
    alter_fun: Callable[..., Any] | dict[str, Callable[..., Any]] | None = None,
    alter_fun_is_vectorized: bool | None = None,
    col: dict[str, str] | Sequence[str] | None = None,
    top_annotation: Any = "default",
    right_annotation: Any = "default",
    left_annotation: Any = None,
    bottom_annotation: Any = None,
    show_pct: bool = True,
    pct_gp: dict[str, Any] | None = None,
    pct_digits: int = 0,
    pct_side: str = "left",
    pct_include: Sequence[str] | None = None,
    row_labels: Any = None,
    show_row_names: bool = True,
    row_names_side: str = "right",
    row_names_gp: dict[str, Any] | None = None,
    row_split: Any = None,
    column_labels: Any = None,
    column_names_side: str = "bottom",
    column_names_gp: dict[str, Any] | None = None,
    column_split: Any = None,
    show_column_names: bool = False,
    row_order: Any = None,
    column_order: Any = None,
    remove_empty_columns: bool = False,
    remove_empty_rows: bool = False,
    heatmap_legend_param: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> Heatmap:
    """Construct an OncoPrint Heatmap (`oncoPrint.Rd`)."""
    # Parse input: character matrix/DF OR list/dict of binary matrices/DFs
    rownames: list[str] | None = None
    colnames: list[str] | None = None

    if isinstance(mat, (dict, list, tuple)) and not (isinstance(mat, (list, tuple)) and len(mat) > 0 and isinstance(mat[0], (str, bytes))):
        if isinstance(mat, dict):
            type_names = list(mat.keys())
            aligned = unify_mat_list(mat, default=0)
            mat_dfs = [aligned[k] for k in type_names]
        else:
            aligned = unify_mat_list(mat, default=0)
            type_names = [f"type_{i+1}" for i in range(len(aligned))]
            mat_dfs = aligned
        if not mat_dfs:
            raise ValueError("Empty list passed to `oncoPrint`.")
        rownames = [str(r) for r in mat_dfs[0].index]
        colnames = [str(c) for c in mat_dfs[0].columns]
        ng, ns = mat_dfs[0].shape
        all_type = list(type_names)
        arr = np.zeros((ng, ns, len(all_type)), dtype=bool)
        for ti, df_t in enumerate(mat_dfs):
            arr[:, :, ti] = df_t.to_numpy(dtype=float) != 0
        str_mat = np.empty((ng, ns), dtype=object)
        for r in range(ng):
            for c in range(ns):
                str_mat[r, c] = ";".join(all_type[ti] for ti in range(len(all_type)) if arr[r, c, ti])
    else:
        if isinstance(mat, pd.DataFrame):
            rownames = [str(r) for r in mat.index]
            colnames = [str(c) for c in mat.columns]
            raw = mat.to_numpy(dtype=object)
        else:
            raw = np.asarray(mat, dtype=object)
        if raw.ndim != 2:
            raise ValueError("`mat` must be 2D matrix/DataFrame or dict/list of binary matrices.")
        ng, ns = raw.shape
        parsed: list[list[list[str]]] = [[list(get_type(raw[r, c])) for c in range(ns)] for r in range(ng)]
        seen_types: list[str] = []
        seen_set: set[str] = set()
        for r in range(ng):
            for c in range(ns):
                for tp in parsed[r][c]:
                    if tp not in seen_set:
                        seen_set.add(tp)
                        seen_types.append(tp)
        all_type = seen_types
        if col is not None and isinstance(col, dict):
            # Preserve col dict ordering if keys overlap
            extra = [t for t in all_type if t not in col]
            all_type = [t for t in col.keys() if t in seen_set] + extra
        type_to_idx = {tp: i for i, tp in enumerate(all_type)}
        arr = np.zeros((ng, ns, max(1, len(all_type))), dtype=bool)
        for r in range(ng):
            for c in range(ns):
                for tp in parsed[r][c]:
                    arr[r, c, type_to_idx[tp]] = True
        if len(all_type) == 0:
            arr = np.zeros((ng, ns, 0), dtype=bool)
        str_mat = np.array([[";".join(parsed[r][c]) for c in range(ns)] for r in range(ng)], dtype=object)

    # Remove empty rows/columns if requested
    if remove_empty_rows and ng > 0:
        keep_r = np.any(arr, axis=(1, 2))
        arr = arr[keep_r, :, :]
        str_mat = str_mat[keep_r, :]
        if rownames is not None:
            rownames = [r for r, f in zip(rownames, keep_r) if f]
        if row_labels is not None:
            row_labels = [r for r, f in zip(row_labels, keep_r) if f]
        ng = arr.shape[0]
    if remove_empty_columns and ns > 0:
        keep_c = np.any(arr, axis=(0, 2))
        arr = arr[:, keep_c, :]
        str_mat = str_mat[:, keep_c]
        if colnames is not None:
            colnames = [c for c, f in zip(colnames, keep_c) if f]
        if column_labels is not None:
            column_labels = [c for c, f in zip(column_labels, keep_c) if f]
        ns = arr.shape[1]

    # Colors
    nt = len(all_type)
    if col is None:
        pal = brewer_pal(max(nt, 3), "Set1")
        if nt > len(pal):
            from .colors import rand_color
            pal = list(pal) + list(rand_color(nt - len(pal)))
        col_map = {tp: to_hex_rgba(pal[i]) for i, tp in enumerate(all_type)}
    elif isinstance(col, dict):
        col_map = {tp: to_hex_rgba(col.get(tp, "#E41A1C")) for tp in all_type}
    else:
        col_seq = [to_hex_rgba(c) for c in col]
        col_map = {tp: col_seq[i % len(col_seq)] for i, tp in enumerate(all_type)}

    if alter_fun is None:
        eff_alter_fun = _build_default_alter_fun(all_type, col_map)
        is_vec = True
    elif callable(alter_fun) and not isinstance(alter_fun, dict):
        eff_alter_fun = alter_fun
        is_vec = bool(alter_fun_is_vectorized) if alter_fun_is_vectorized is not None else False
    elif isinstance(alter_fun, dict):
        eff_alter_fun = dict(alter_fun)
        is_vec = bool(alter_fun_is_vectorized) if alter_fun_is_vectorized is not None else False
    else:
        raise TypeError("`alter_fun` must be a dict of callables or a callable.")

    # Default gene & sample orders
    if row_order is None:
        gene_freq = np.sum(np.any(arr, axis=2), axis=1) if nt > 0 else np.zeros(ng, dtype=int)
        eff_row_order = np.argsort(-gene_freq, kind="stable").tolist()
    else:
        eff_row_order = list(row_order)

    if column_order is None:
        eff_col_order = _memo_sort_columns(arr, eff_row_order) if nt > 0 else list(range(ns))
    else:
        eff_col_order = list(column_order)

    # Layer function rendering OncoPrint alteration graphics
    def _layer_fun(j: Any, i: Any, x: Any, y: Any, w: Any, h: Any, fill: Any) -> None:
        n_cells = len(j)
        if n_cells == 0:
            return
        if isinstance(eff_alter_fun, dict):
            if "background" in eff_alter_fun:
                bg_fn = eff_alter_fun["background"]
                if is_vec:
                    bg_fn(x, y, w, h)
                else:
                    for k in range(n_cells):
                        bg_fn(x[k], y[k], w[k], h[k])
            for ti, tp in enumerate(all_type):
                if tp not in eff_alter_fun:
                    continue
                mask = arr[np.asarray(i, dtype=int), np.asarray(j, dtype=int), ti]
                idx_m = np.where(mask)[0]
                if len(idx_m) == 0:
                    continue
                fn = eff_alter_fun[tp]
                if is_vec:
                    fn(x[idx_m], y[idx_m], w[idx_m], h[idx_m])
                else:
                    for k in idx_m:
                        fn(x[k], y[k], w[k], h[k])
        else:
            # Single callable alter_fun(x, y, w, h, v)
            sig = inspect.signature(eff_alter_fun)
            for k in range(n_cells):
                v_types = [all_type[ti] for ti in range(nt) if arr[int(i[k]), int(j[k]), ti]]
                if not v_types:
                    v_arg: Any = "background"
                elif len(v_types) == 1:
                    v_arg = v_types[0]
                else:
                    v_arg = v_types
                if len(sig.parameters) >= 5:
                    eff_alter_fun(x[k], y[k], w[k], h[k], v_arg)
                else:
                    eff_alter_fun(x[k], y[k], w[k], h[k])

    # Default top and right barplot annotations
    eff_top = HeatmapAnnotation(cbar=anno_oncoprint_barplot(), name="cbar_anno") if top_annotation == "default" else top_annotation
    eff_right = rowAnnotation(rbar=anno_oncoprint_barplot(), name="rbar_anno") if right_annotation == "default" else right_annotation
    eff_left = left_annotation
    eff_bottom = bottom_annotation

    # Add alteration percentage text annotation
    if show_pct and ng > 0:
        if pct_include is not None:
            inc_idx = [ti for ti, tp in enumerate(all_type) if tp in set(pct_include)]
            alt_any = np.any(arr[:, :, inc_idx], axis=2) if inc_idx else np.zeros((ng, ns), dtype=bool)
        else:
            alt_any = np.any(arr, axis=2) if nt > 0 else np.zeros((ng, ns), dtype=bool)
        pct_vals = np.mean(alt_any, axis=1) * 100.0
        pct_strs = [f"{v:.{int(pct_digits)}f}%" for v in pct_vals]
        p_gp = check_gp(pct_gp or gpar(fontsize=10))
        p_just = "right" if pct_side == "left" else "left"
        p_loc = unit(1.0, "npc") if pct_side == "left" else unit(0.0, "npc")
        pct_anno = anno_text(
            pct_strs,
            which="row",
            gp=p_gp,
            just=p_just,
            location=p_loc,
            width=max_text_width(pct_strs, gp=p_gp) + unit(2.0, "mm"),
        )
        pct_ha = rowAnnotation(pct=pct_anno, show_annotation_name=False)
        if pct_side == "left":
            eff_left = pct_ha if eff_left is None else (pct_ha + eff_left if isinstance(pct_ha, HeatmapAnnotation) else pct_ha)
            if eff_left is not pct_ha:
                from .annotations import c_HeatmapAnnotation
                eff_left = c_HeatmapAnnotation(left_annotation, pct_ha)
        else:
            if eff_right is None:
                eff_right = pct_ha
            else:
                from .annotations import c_HeatmapAnnotation
                eff_right = c_HeatmapAnnotation(pct_ha, eff_right)

    hlp = dict(heatmap_legend_param) if heatmap_legend_param else {"title": "Alterations", "at": all_type, "labels": all_type}
    hlp.setdefault("title", "Alterations")

    df_mat = pd.DataFrame(str_mat, index=rownames, columns=colnames)
    ht_name = name or f"oncoprint_{_ENV.next_heatmap_index()}"

    # ColorMapping representing alterations legend
    from .colors import ColorMapping
    cm_op = ColorMapping(name=ht_name, colors=col_map) if nt > 0 else None

    ht = Heatmap(
        df_mat,
        col=cm_op,
        name=ht_name,
        rect_gp=gpar(type="none"),
        layer_fun=_layer_fun,
        cluster_rows=False,
        cluster_columns=False,
        row_order=eff_row_order,
        column_order=eff_col_order,
        row_labels=row_labels if row_labels is not None else rownames,
        show_row_names=show_row_names,
        row_names_side=row_names_side,
        row_names_gp=row_names_gp,
        row_split=row_split,
        column_labels=column_labels if column_labels is not None else colnames,
        show_column_names=show_column_names,
        column_names_side=column_names_side,
        column_names_gp=column_names_gp,
        column_split=column_split,
        top_annotation=eff_top,
        right_annotation=eff_right,
        left_annotation=eff_left,
        bottom_annotation=eff_bottom,
        heatmap_legend_param=hlp,
        **kwargs,
    )
    ht._oncoprint_env = {
        "arr": arr,
        "all_type": all_type,
        "col": col_map,
    }
    return ht
