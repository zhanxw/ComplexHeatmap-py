"""Built-in annotation functions (`anno_*` and `row_anno_*` aliases)."""

from __future__ import annotations

import copy
import os
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import make_interp_spline
from scipy.stats import gaussian_kde

from ._options import _ENV, ht_opt
from .annotations import (
    AnnotationFunction,
    anno_width_and_height,
    construct_axis_grob,
    default_axis_param,
    subset_by_intersect,
    subset_matrix_by_row,
    subset_no,
    subset_vector,
    update_anno_extend,
    validate_axis_param,
)
from .colors import ColorMapping, add_transparency, brewer_pal, colorRamp2, default_col, to_hex_rgba
from .grid import (
    Viewport,
    current_viewport,
    grid_boxplot,
    grid_draw,
    grid_lines,
    grid_points,
    grid_polygon,
    grid_raster,
    grid_rect,
    grid_roundrect,
    grid_segments,
    grid_text,
    popViewport,
    pushViewport,
    textGrob,
)
from .text import smartAlign, smartAlign2, textbox_grob
from .units import (
    Unit,
    as_unit,
    check_gp,
    convertHeight,
    convertWidth,
    gpar,
    recycle_gp,
    subset_gp,
    unit,
    unit_c,
)
from .grid import max_text_height, max_text_width


def _rle_runs(seq: Sequence[Any]) -> list[tuple[Any, int, int]]:
    """Return (value, start_idx, length) runs for contiguous identical values."""
    if len(seq) == 0:
        return []
    runs: list[tuple[Any, int, int]] = []
    cur = seq[0]
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != cur:
            runs.append((cur, start, i - start))
            cur = seq[i]
            start = i
    runs.append((cur, start, len(seq) - start))
    return runs


def _reverse_vals(vals: np.ndarray, data_scale: tuple[float, float], direction: str) -> np.ndarray:
    if direction == "reverse":
        return data_scale[1] - vals + data_scale[0]
    return vals


def _draw_background(border: Any, background: bool) -> None:
    if background:
        grid_rect(gp=gpar(fill="#DDDDDD", col=None))
    if border is True:
        grid_rect(gp=gpar(fill="transparent", col="black"))
    elif isinstance(border, str):
        grid_rect(gp=gpar(fill="transparent", col=border))


def _recycle_seq(val: Any, n: int) -> list[Any]:
    if isinstance(val, Unit):
        if len(val) == 1:
            return [val] * n
        return [val[i % len(val)] for i in range(n)]
    if isinstance(val, (str, int, float, np.number, bool, np.bool_)) or val is None:
        return [val] * n
    seq = list(val)
    if len(seq) == 0:
        return [None] * n
    return [seq[i % len(seq)] for i in range(n)]


def _to_obs_series(x: Any, which: str) -> list[np.ndarray]:
    """Convert matrix/DataFrame/list-of-vectors into a list of 1D finite float arrays."""
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=float)
        if which == "column":
            return [arr[:, c] for c in range(arr.shape[1])]
        return [arr[r, :] for r in range(arr.shape[0])]
    if isinstance(x, np.ndarray) and x.ndim == 2:
        arr = x.astype(float)
        if which == "column":
            return [arr[:, c] for c in range(arr.shape[1])]
        return [arr[r, :] for r in range(arr.shape[0])]
    if isinstance(x, dict):
        return [np.asarray(v, dtype=float).ravel() for v in x.values()]
    return [np.asarray(v, dtype=float).ravel() for v in x]


# ---- 1. anno_simple ----

def anno_simple(
    x: Any,
    col: Any = None,
    na_col: str = "grey",
    which: str | None = None,
    border: bool = False,
    gp: dict[str, Any] | None = None,
    pch: Any = None,
    pt_size: Any = None,
    pt_gp: dict[str, Any] | None = None,
    simple_anno_size: Unit | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    if simple_anno_size is None:
        simple_anno_size = ht_opt.simple_anno_size
    sas_mm = convertWidth(as_unit(simple_anno_size, "mm"), "mm", valueOnly=True)

    arr = np.asarray(x, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    is_mat = arr.ndim == 2
    n = arr.shape[0]
    nc = arr.shape[1] if is_mat else 1

    if isinstance(col, ColorMapping):
        cm = col
    elif col is None:
        inferred = default_col(arr)
        if callable(inferred):
            cm = ColorMapping(col_fun=inferred, na_col=na_col)
        else:
            cm = ColorMapping(colors=inferred, na_col=na_col)
    elif callable(col):
        cm = ColorMapping(col_fun=col, na_col=na_col)
    elif isinstance(col, dict):
        eff_na = col.get("_NA_", na_col)
        clean = {k: v for k, v in col.items() if k != "_NA_"}
        cm = ColorMapping(colors=clean, na_col=eff_na, full_col=clean)
    else:
        col_seq = list(col)
        uniq = list(dict.fromkeys(v for v in arr.ravel() if v is not None and not (isinstance(v, float) and np.isnan(v))))
        if len(col_seq) == len(uniq):
            cm = ColorMapping(colors=dict(zip([str(u) for u in sorted(uniq)], col_seq)), na_col=na_col)
        else:
            vals = arr.astype(float)
            cf = colorRamp2(np.linspace(float(np.nanmin(vals)), float(np.nanmax(vals)), len(col_seq)), col_seq)
            cm = ColorMapping(col_fun=cf, na_col=na_col)

    sz_default = unit(sas_mm * nc, "mm")
    gp_d = check_gp(gp)
    pt_gp_d = check_gp(pt_gp)
    if pt_size is None:
        pt_sz_u = unit(0.8 / max(1, nc), "snpc")
    else:
        pt_sz_u = as_unit(pt_size, "mm")

    pch_arr = None if pch is None else np.asarray(pch, dtype=object)

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, sz_default)

        def _draw(index, k=1, n_slices=1, env=None):
            val = env["value"]
            sub = val[index] if val.ndim == 1 else val[index, :]
            fill_mat = cm.map_to_colors(sub)
            m = len(index)
            if m == 0:
                return
            p_sub = None
            if env["pch"] is not None:
                pa = env["pch"]
                p_sub = pa[index] if pa.ndim == 1 else pa[index, :]
            if target_which == "column":
                pushViewport(Viewport(xscale=(0.0, float(m)), yscale=(0.0, float(nc)), clip=False))
                try:
                    if not is_mat:
                        fills = list(fill_mat)
                        if cm.type == "discrete" and gp_d.get("col", "#00000000") == "#00000000":
                            for col_val, st, ln in _rle_runs(fills):
                                grid_rect(x=st + ln * 0.5, y=0.5, width=ln, height=1.0, gp=gpar(fill=col_val, col=None), default_units="native")
                        else:
                            rec_gp = recycle_gp(gp_d, m)
                            rec_gp["fill"] = fills
                            grid_rect(x=np.arange(m) + 0.5, y=np.full(m, 0.5), width=np.ones(m), height=np.ones(m), gp=rec_gp, default_units="native")
                        if p_sub is not None:
                            valid = [i for i, pv in enumerate(p_sub) if pv is not None and not (isinstance(pv, float) and np.isnan(pv))]
                            if valid:
                                grid_points(
                                    np.array(valid, dtype=float) + 0.5,
                                    np.full(len(valid), 0.5),
                                    pch=[p_sub[i] for i in valid],
                                    size=pt_sz_u if len(pt_sz_u) == 1 else pt_sz_u[index][valid],
                                    gp=subset_gp(recycle_gp(pt_gp_d, n), [index[i] for i in valid]),
                                    default_units="native",
                                )
                    else:
                        for c_i in range(nc):
                            y_c = nc - c_i - 0.5
                            fills = list(fill_mat[:, c_i])
                            rec_gp = recycle_gp(gp_d, m)
                            rec_gp["fill"] = fills
                            grid_rect(x=np.arange(m) + 0.5, y=np.full(m, y_c), width=np.ones(m), height=np.ones(m), gp=rec_gp, default_units="native")
                            if p_sub is not None:
                                p_col = p_sub[:, c_i] if p_sub.ndim == 2 else p_sub
                                valid = [i for i, pv in enumerate(p_col) if pv is not None and not (isinstance(pv, float) and np.isnan(pv))]
                                if valid:
                                    grid_points(
                                        np.array(valid, dtype=float) + 0.5,
                                        np.full(len(valid), y_c),
                                        pch=[p_col[i] for i in valid],
                                        size=pt_sz_u,
                                        gp=pt_gp_d,
                                        default_units="native",
                                    )
                    if border:
                        grid_rect(gp=gpar(fill="transparent", col="black" if border is True else border))
                finally:
                    popViewport()
            else:
                pushViewport(Viewport(xscale=(0.0, float(nc)), yscale=(0.0, float(m)), clip=False))
                try:
                    if not is_mat:
                        fills = list(fill_mat)
                        if cm.type == "discrete" and gp_d.get("col", "#00000000") == "#00000000":
                            for col_val, st, ln in _rle_runs(fills):
                                y_mid = m - (st + ln * 0.5)
                                grid_rect(x=0.5, y=y_mid, width=1.0, height=ln, gp=gpar(fill=col_val, col=None), default_units="native")
                        else:
                            rec_gp = recycle_gp(gp_d, m)
                            rec_gp["fill"] = fills
                            grid_rect(x=np.full(m, 0.5), y=m - np.arange(m) - 0.5, width=np.ones(m), height=np.ones(m), gp=rec_gp, default_units="native")
                        if p_sub is not None:
                            valid = [i for i, pv in enumerate(p_sub) if pv is not None and not (isinstance(pv, float) and np.isnan(pv))]
                            if valid:
                                grid_points(
                                    np.full(len(valid), 0.5),
                                    m - np.array(valid, dtype=float) - 0.5,
                                    pch=[p_sub[i] for i in valid],
                                    size=pt_sz_u if len(pt_sz_u) == 1 else pt_sz_u[index][valid],
                                    gp=subset_gp(recycle_gp(pt_gp_d, n), [index[i] for i in valid]),
                                    default_units="native",
                                )
                    else:
                        for c_i in range(nc):
                            x_c = c_i + 0.5
                            fills = list(fill_mat[:, c_i])
                            rec_gp = recycle_gp(gp_d, m)
                            rec_gp["fill"] = fills
                            grid_rect(x=np.full(m, x_c), y=m - np.arange(m) - 0.5, width=np.ones(m), height=np.ones(m), gp=rec_gp, default_units="native")
                            if p_sub is not None:
                                p_col = p_sub[:, c_i] if p_sub.ndim == 2 else p_sub
                                valid = [i for i, pv in enumerate(p_col) if pv is not None and not (isinstance(pv, float) and np.isnan(pv))]
                                if valid:
                                    grid_points(
                                        np.full(len(valid), x_c),
                                        m - np.array(valid, dtype=float) - 0.5,
                                        pch=[p_col[i] for i in valid],
                                        size=pt_sz_u,
                                        gp=pt_gp_d,
                                        default_units="native",
                                    )
                    if border:
                        grid_rect(gp=gpar(fill="transparent", col="black" if border is True else border))
                finally:
                    popViewport()

        rules = {"value": subset_matrix_by_row if is_mat else subset_vector}
        if pch_arr is not None:
            rules["pch"] = subset_matrix_by_row if pch_arr.ndim == 2 else subset_vector
        return AnnotationFunction(
            fun=_draw,
            fun_name="anno_simple",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            var_import={"value": arr, "pch": pch_arr},
            subset_rule=rules,
            subsettable=True,
            show_name=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )

    return _build(eff_which)


# ---- 2. anno_empty ----

def anno_empty(
    which: str | None = None,
    border: bool = True,
    zoom: bool = False,
    width: Unit | None = None,
    height: Unit | None = None,
    show_name: bool = False,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(1.0, "cm"))

        def _draw(index, k=1, n_slices=1, env=None):
            if border:
                grid_rect(gp=gpar(fill="transparent", col="black"))

        return AnnotationFunction(
            fun=_draw,
            fun_name="anno_empty",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=None,
            var_import={"zoom": bool(zoom), "border": border},
            subset_rule={},
            subsettable=True,
            show_name=show_name,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )

    return _build(eff_which)


# ---- 3. anno_points ----

def anno_points(
    x: Any,
    which: str | None = None,
    border: bool = True,
    gp: dict[str, Any] | None = None,
    pch: Any = 16,
    size: Any = unit(2, "mm"),
    ylim: Sequence[float] | None = None,
    extend: float = 0.05,
    axis: bool = True,
    background: bool = False,
    axis_param: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
    pch_as_image: bool = False,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    is_mat = arr.ndim == 2
    n = arr.shape[0]
    nc = arr.shape[1] if is_mat else 1

    if ylim is None:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    else:
        lo, hi = float(ylim[0]), float(ylim[1])
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    span = hi - lo
    data_scale = (lo - extend * span, hi + extend * span)

    gp_d = check_gp(gp)
    if not is_mat:
        gp_rec = recycle_gp(gp_d, n)
        pch_rec = _recycle_seq(pch, n)
        sz_rec = as_unit(size, "mm")
        if len(sz_rec) == 1:
            sz_rec = unit(np.full(n, convertWidth(sz_rec, "mm", valueOnly=True)), "mm")
    else:
        gp_rec = recycle_gp(gp_d, nc)
        pch_rec = _recycle_seq(pch, nc)
        sz_rec = as_unit(size, "mm")
        if len(sz_rec) == 1:
            sz_rec = unit(np.full(nc, convertWidth(sz_rec, "mm", valueOnly=True)), "mm")

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(1.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)
        ax_grob = construct_axis_grob(ap, target_which, data_scale) if axis else None

        def _draw(index, k=1, n_slices=1, env=None):
            val = env["value"]
            m = len(index)
            if m == 0:
                return
            if target_which == "column":
                pushViewport(Viewport(xscale=(0.5, m + 0.5), yscale=data_scale, clip=False))
                try:
                    _draw_background(border, background)
                    if not is_mat:
                        sub_v = _reverse_vals(val[index], data_scale, ap["direction"])
                        sub_gp = subset_gp(env["gp"], index)
                        sub_pch = subset_vector(env["pch"], index)
                        sub_sz = env["size"][index]
                        if pch_as_image:
                            for idx_i, (yv, pth) in enumerate(zip(sub_v, sub_pch)):
                                if pth and isinstance(pth, str) and os.path.exists(pth):
                                    img = np.asarray(Image.open(pth).convert("RGBA"))
                                    grid_raster(img, x=idx_i + 1, y=yv, width=sub_sz[idx_i], height=sub_sz[idx_i], default_units="native")
                        else:
                            grid_points(np.arange(1, m + 1), sub_v, pch=sub_pch, size=sub_sz, gp=sub_gp, default_units="native")
                    else:
                        sub_v = _reverse_vals(val[index, :], data_scale, ap["direction"])
                        for c_i in range(nc):
                            grid_points(np.arange(1, m + 1), sub_v[:, c_i], pch=pch_rec[c_i], size=sz_rec[c_i], gp=subset_gp(gp_rec, c_i), default_units="native")
                    show_ax = axis and ((ap["side"] == "left" and k == 1) or (ap["side"] == "right" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()
            else:
                pushViewport(Viewport(xscale=data_scale, yscale=(0.5, m + 0.5), clip=False))
                try:
                    _draw_background(border, background)
                    ys = m - np.arange(m, dtype=float)
                    if not is_mat:
                        sub_v = _reverse_vals(val[index], data_scale, ap["direction"])
                        sub_gp = subset_gp(env["gp"], index)
                        sub_pch = subset_vector(env["pch"], index)
                        sub_sz = env["size"][index]
                        grid_points(sub_v, ys, pch=sub_pch, size=sub_sz, gp=sub_gp, default_units="native")
                    else:
                        sub_v = _reverse_vals(val[index, :], data_scale, ap["direction"])
                        for c_i in range(nc):
                            grid_points(sub_v[:, c_i], ys, pch=pch_rec[c_i], size=sz_rec[c_i], gp=subset_gp(gp_rec, c_i), default_units="native")
                    show_ax = axis and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()

        rules = {"value": subset_matrix_by_row if is_mat else subset_vector}
        if not is_mat:
            rules.update({"gp": subset_gp, "pch": subset_vector, "size": subset_vector})
        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_points",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            data_scale=data_scale,
            var_import={"value": arr, "gp": gp_rec, "pch": pch_rec, "size": sz_rec},
            subset_rule=rules,
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- 4. anno_lines ----

def anno_lines(
    x: Any,
    which: str | None = None,
    border: bool = True,
    gp: dict[str, Any] | None = None,
    add_points: bool | Sequence[bool] | None = None,
    smooth: bool = False,
    pch: Any = 16,
    size: Any = unit(2, "mm"),
    pt_gp: dict[str, Any] | None = None,
    ylim: Sequence[float] | None = None,
    extend: float = 0.05,
    axis: bool = True,
    axis_param: dict[str, Any] | None = None,
    background: bool = False,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    if add_points is None:
        add_points = smooth

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    is_mat = arr.ndim == 2
    n = arr.shape[0]
    nc = arr.shape[1] if is_mat else 1

    if ylim is None:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    else:
        lo, hi = float(ylim[0]), float(ylim[1])
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    span = hi - lo
    data_scale = (lo - extend * span, hi + extend * span)

    gp_d = check_gp(gp)
    pt_gp_d = check_gp(pt_gp)
    if not is_mat:
        gp_line = recycle_gp(gp_d, 1)
        pt_gp_rec = recycle_gp(pt_gp_d, n)
        pch_rec = _recycle_seq(pch, n)
        sz_rec = as_unit(size, "mm")
        if len(sz_rec) == 1:
            sz_rec = unit(np.full(n, convertWidth(sz_rec, "mm", valueOnly=True)), "mm")
        add_pts = [bool(add_points)]
    else:
        gp_line = recycle_gp(gp_d, nc)
        pt_gp_rec = recycle_gp(pt_gp_d, nc)
        pch_rec = _recycle_seq(pch, nc)
        sz_rec = as_unit(size, "mm")
        if len(sz_rec) == 1:
            sz_rec = unit(np.full(nc, convertWidth(sz_rec, "mm", valueOnly=True)), "mm")
        add_pts = [bool(add_points)] * nc if isinstance(add_points, (bool, np.bool_)) else [bool(b) for b in add_points]

    def _smooth_series(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(y)
        tv, yv = t[mask], y[mask]
        if len(tv) < 4:
            return tv, yv
        # Local quadratic regression (LOESS-like) on 100 points
        t_grid = np.linspace(tv[0], tv[-1], 100)
        y_grid = np.empty_like(t_grid)
        bw = 0.75 * (tv[-1] - tv[0])
        for gi, tg in enumerate(t_grid):
            d = np.abs(tv - tg)
            w_rad = max(np.partition(d, min(len(d) - 1, max(2, int(0.75 * len(d)))))[min(len(d) - 1, max(2, int(0.75 * len(d))))], 1e-9)
            u = np.clip(d / w_rad, 0.0, 1.0)
            w = (1.0 - u ** 3) ** 3
            if np.sum(w > 0) >= 3:
                poly = np.polyfit(tv, yv, deg=2, w=np.sqrt(w))
                y_grid[gi] = float(np.polyval(poly, tg))
            else:
                y_grid[gi] = float(np.interp(tg, tv, yv))
        return t_grid, y_grid

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(1.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)
        ax_grob = construct_axis_grob(ap, target_which, data_scale) if axis else None

        def _draw(index, k=1, n_slices=1, env=None):
            val = env["value"]
            m = len(index)
            if m == 0:
                return
            if target_which == "column":
                pushViewport(Viewport(xscale=(0.5, m + 0.5), yscale=data_scale, clip=False))
                try:
                    _draw_background(border, background)
                    xs = np.arange(1, m + 1, dtype=float)
                    if not is_mat:
                        yv = _reverse_vals(val[index], data_scale, ap["direction"])
                        lx, ly = _smooth_series(xs, yv) if smooth else (xs, yv)
                        grid_lines(lx, ly, gp=subset_gp(gp_line, 0), default_units="native")
                        if add_pts[0]:
                            grid_points(xs, yv, pch=subset_vector(env["pch"], index), size=env["size"][index], gp=subset_gp(env["pt_gp"], index), default_units="native")
                    else:
                        sub_v = _reverse_vals(val[index, :], data_scale, ap["direction"])
                        for c_i in range(nc):
                            yv = sub_v[:, c_i]
                            lx, ly = _smooth_series(xs, yv) if smooth else (xs, yv)
                            grid_lines(lx, ly, gp=subset_gp(gp_line, c_i), default_units="native")
                            if add_pts[c_i]:
                                grid_points(xs, yv, pch=pch_rec[c_i], size=sz_rec[c_i], gp=subset_gp(pt_gp_rec, c_i), default_units="native")
                    show_ax = axis and ((ap["side"] == "left" and k == 1) or (ap["side"] == "right" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()
            else:
                pushViewport(Viewport(xscale=data_scale, yscale=(0.5, m + 0.5), clip=False))
                try:
                    _draw_background(border, background)
                    ys = m - np.arange(m, dtype=float)
                    if not is_mat:
                        xv = _reverse_vals(val[index], data_scale, ap["direction"])
                        ly, lx = _smooth_series(ys[::-1], xv[::-1]) if smooth else (ys, xv)
                        grid_lines(lx, ly, gp=subset_gp(gp_line, 0), default_units="native")
                        if add_pts[0]:
                            grid_points(xv, ys, pch=subset_vector(env["pch"], index), size=env["size"][index], gp=subset_gp(env["pt_gp"], index), default_units="native")
                    else:
                        sub_v = _reverse_vals(val[index, :], data_scale, ap["direction"])
                        for c_i in range(nc):
                            xv = sub_v[:, c_i]
                            ly, lx = _smooth_series(ys[::-1], xv[::-1]) if smooth else (ys, xv)
                            grid_lines(lx, ly, gp=subset_gp(gp_line, c_i), default_units="native")
                            if add_pts[c_i]:
                                grid_points(xv, ys, pch=pch_rec[c_i], size=sz_rec[c_i], gp=subset_gp(pt_gp_rec, c_i), default_units="native")
                    show_ax = axis and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()

        rules = {"value": subset_matrix_by_row if is_mat else subset_vector}
        if not is_mat:
            rules.update({"pt_gp": subset_gp, "pch": subset_vector, "size": subset_vector})
        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_lines",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            data_scale=data_scale,
            var_import={"value": arr, "pt_gp": pt_gp_rec, "pch": pch_rec, "size": sz_rec},
            subset_rule=rules,
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- 5. anno_barplot ----

def anno_barplot(
    x: Any,
    baseline: float | str = 0.0,
    which: str | None = None,
    border: bool = True,
    bar_width: float = 0.6,
    beside: bool = False,
    attach: bool = False,
    gp: dict[str, Any] | None = None,
    ylim: Sequence[float] | None = None,
    extend: float = 0.05,
    axis: bool = True,
    background: bool = False,
    axis_param: dict[str, Any] | None = None,
    add_numbers: bool = False,
    numbers_gp: dict[str, Any] | None = None,
    numbers_rot: float | None = None,
    numbers_offset: Any = unit(2, "mm"),
    labels_format: Callable[[Any], Sequence[str]] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple, np.ndarray)):
        arr = np.column_stack([np.asarray(col, dtype=float) for col in x])
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    is_mat = arr.ndim == 2
    n = arr.shape[0]
    nc = arr.shape[1] if is_mat else 1

    if gp is None:
        if is_mat:
            shades = ["#" + f"{int(round(v)):02X}" * 3 for v in np.linspace(230, 80, nc)]
            gp_d = gpar(fill=shades)
        else:
            gp_d = gpar(fill="#CCCCCC")
    else:
        gp_d = check_gp(gp)

    if not is_mat:
        gp_rec = recycle_gp(gp_d, n)
        if baseline == "min":
            base_val = float(np.nanmin(arr))
        elif baseline == "max":
            base_val = float(np.nanmax(arr))
        else:
            base_val = float(baseline)
        if ylim is None:
            lo = min(base_val, float(np.nanmin(arr)))
            hi = max(base_val, float(np.nanmax(arr)))
        else:
            lo, hi = float(ylim[0]), float(ylim[1])
    else:
        gp_rec = recycle_gp(gp_d, nc)
        if baseline == "min":
            base_val = float(np.nanmin(arr))
        elif baseline == "max":
            base_val = float(np.nanmax(arr))
        else:
            base_val = float(baseline)
        if ylim is None:
            if beside:
                lo = min(base_val, float(np.nanmin(arr)))
                hi = max(base_val, float(np.nanmax(arr)))
            else:
                rsums = np.nansum(arr, axis=1)
                lo = min(base_val, float(np.nanmin(rsums)), 0.0 if base_val == 0 else float(np.nanmin(rsums)))
                hi = max(base_val, float(np.nanmax(rsums)))
        else:
            lo, hi = float(ylim[0]), float(ylim[1])

    if lo == hi:
        hi = lo + 1.0
    span = hi - lo

    # Proportion matrix clamp [0, 1]
    is_prop = is_mat and not beside and base_val == 0.0 and np.allclose(np.nansum(arr, axis=1), 1.0)
    if ylim is not None:
        data_scale = (lo, hi)
    elif is_prop:
        data_scale = (0.0, 1.0)
    else:
        lo_e = 0.0 if base_val == 0.0 and lo == 0.0 else lo - extend * span
        hi_e = 0.0 if base_val == 0.0 and hi == 0.0 else hi + extend * span
        if add_numbers and not is_mat:
            hi_e += 0.15 * span
        data_scale = (lo_e, hi_e)

    fmt_fn = labels_format or getattr(x, "labels_format", None)
    num_gp_d = check_gp(numbers_gp or gpar(fontsize=8))
    off_u = as_unit(numbers_offset, "mm")

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(1.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)
        ax_grob = construct_axis_grob(ap, target_which, data_scale) if axis else None
        rot_num = (45.0 if target_which == "column" else 0.0) if numbers_rot is None else float(numbers_rot)

        def _draw(index, k=1, n_slices=1, env=None):
            val = env["value"]
            m = len(index)
            if m == 0:
                return
            b_rev = float(_reverse_vals(np.array([base_val]), data_scale, ap["direction"])[0])
            if target_which == "column":
                pushViewport(Viewport(xscale=(0.5, m + 0.5), yscale=data_scale, clip=False))
                try:
                    _draw_background(border, background)
                    xs = np.arange(1, m + 1, dtype=float)
                    if not is_mat:
                        sub_v = val[index]
                        v_rev = _reverse_vals(sub_v, data_scale, ap["direction"])
                        sub_gp = subset_gp(env["gp"], index)
                        y_pos = np.minimum(v_rev, b_rev)
                        h_bar = np.abs(v_rev - b_rev)
                        grid_rect(x=xs, y=y_pos, width=np.full(m, bar_width), height=h_bar, just="bottom", gp=sub_gp, default_units="native")
                        if add_numbers:
                            lbls = list(fmt_fn(sub_v)) if fmt_fn else [f"{v:g}" for v in sub_v]
                            off_mm = convertHeight(off_u, "mm", valueOnly=True)
                            for xi, yi, lb in zip(xs, v_rev, lbls):
                                grid_text(lb, x=unit(xi, "native"), y=unit(yi, "native") + unit(off_mm, "mm"), just="bottom" if rot_num == 0 else "left", rot=rot_num, gp=num_gp_d)
                    elif not beside:
                        sub_v = val[index, :]
                        cum = np.zeros(m, dtype=float)
                        for c_i in range(nc):
                            col_vals = sub_v[:, c_i]
                            y0_raw = base_val + cum
                            y1_raw = y0_raw + col_vals
                            y0_r = _reverse_vals(y0_raw, data_scale, ap["direction"])
                            y1_r = _reverse_vals(y1_raw, data_scale, ap["direction"])
                            grid_rect(
                                x=xs,
                                y=np.minimum(y0_r, y1_r),
                                width=np.full(m, bar_width),
                                height=np.abs(y1_r - y0_r),
                                just="bottom",
                                gp=subset_gp(gp_rec, c_i),
                                default_units="native",
                            )
                            cum = cum + col_vals
                    else:
                        sub_v = val[index, :]
                        sub_w = bar_width / nc if attach else (bar_width * 0.85) / nc
                        step_w = bar_width / nc
                        for c_i in range(nc):
                            x_off = (c_i - 0.5 * (nc - 1)) * step_w
                            v_rev = _reverse_vals(sub_v[:, c_i], data_scale, ap["direction"])
                            grid_rect(
                                x=xs + x_off,
                                y=np.minimum(v_rev, b_rev),
                                width=np.full(m, sub_w),
                                height=np.abs(v_rev - b_rev),
                                just="bottom",
                                gp=subset_gp(gp_rec, c_i),
                                default_units="native",
                            )
                    show_ax = axis and ((ap["side"] == "left" and k == 1) or (ap["side"] == "right" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()
            else:
                pushViewport(Viewport(xscale=data_scale, yscale=(0.5, m + 0.5), clip=False))
                try:
                    _draw_background(border, background)
                    ys = m - np.arange(m, dtype=float)
                    if not is_mat:
                        sub_v = val[index]
                        v_rev = _reverse_vals(sub_v, data_scale, ap["direction"])
                        sub_gp = subset_gp(env["gp"], index)
                        x_pos = np.minimum(v_rev, b_rev)
                        w_bar = np.abs(v_rev - b_rev)
                        grid_rect(x=x_pos, y=ys, width=w_bar, height=np.full(m, bar_width), just="left", gp=sub_gp, default_units="native")
                        if add_numbers:
                            lbls = list(fmt_fn(sub_v)) if fmt_fn else [f"{v:g}" for v in sub_v]
                            off_mm = convertWidth(off_u, "mm", valueOnly=True)
                            for xi, yi, lb in zip(v_rev, ys, lbls):
                                grid_text(lb, x=unit(xi, "native") + unit(off_mm, "mm"), y=unit(yi, "native"), just="left", rot=rot_num, gp=num_gp_d)
                    elif not beside:
                        sub_v = val[index, :]
                        cum = np.zeros(m, dtype=float)
                        for c_i in range(nc):
                            col_vals = sub_v[:, c_i]
                            x0_raw = base_val + cum
                            x1_raw = x0_raw + col_vals
                            x0_r = _reverse_vals(x0_raw, data_scale, ap["direction"])
                            x1_r = _reverse_vals(x1_raw, data_scale, ap["direction"])
                            grid_rect(
                                x=np.minimum(x0_r, x1_r),
                                y=ys,
                                width=np.abs(x1_r - x0_r),
                                height=np.full(m, bar_width),
                                just="left",
                                gp=subset_gp(gp_rec, c_i),
                                default_units="native",
                            )
                            cum = cum + col_vals
                    else:
                        sub_v = val[index, :]
                        sub_h = bar_width / nc if attach else (bar_width * 0.85) / nc
                        step_h = bar_width / nc
                        for c_i in range(nc):
                            y_off = (0.5 * (nc - 1) - c_i) * step_h
                            v_rev = _reverse_vals(sub_v[:, c_i], data_scale, ap["direction"])
                            grid_rect(
                                x=np.minimum(v_rev, b_rev),
                                y=ys + y_off,
                                width=np.abs(v_rev - b_rev),
                                height=np.full(m, sub_h),
                                just="left",
                                gp=subset_gp(gp_rec, c_i),
                                default_units="native",
                            )
                    show_ax = axis and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()

        rules = {"value": subset_matrix_by_row if is_mat else subset_vector}
        if not is_mat:
            rules["gp"] = subset_gp
        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_barplot",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            data_scale=data_scale,
            var_import={"value": arr, "gp": gp_rec},
            subset_rule=rules,
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- 6. anno_boxplot ----

def anno_boxplot(
    x: Any,
    which: str | None = None,
    border: bool = True,
    gp: dict[str, Any] | None = None,
    ylim: Sequence[float] | None = None,
    extend: float = 0.05,
    outline: bool = True,
    box_width: float = 0.6,
    add_points: bool = False,
    pch: Any = 16,
    size: Any = unit(4, "pt"),
    pt_gp: dict[str, Any] | None = None,
    axis: bool = True,
    background: bool = False,
    axis_param: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    def _build(target_which: str) -> AnnotationFunction:
        series = _to_obs_series(x, target_which)
        n = len(series)
        if ylim is None:
            if not outline:
                lo_vals, hi_vals = [], []
                for s in series:
                    sf = s[np.isfinite(s)]
                    if len(sf):
                        q1, q3 = np.quantile(sf, [0.25, 0.75])
                        iqr = q3 - q1
                        in_w = sf[(sf >= q1 - 1.5 * iqr) & (sf <= q3 + 1.5 * iqr)]
                        lo_vals.append(float(np.min(in_w)) if len(in_w) else float(q1))
                        hi_vals.append(float(np.max(in_w)) if len(in_w) else float(q3))
                lo, hi = min(lo_vals), max(hi_vals)
            else:
                all_v = np.concatenate(series) if n else np.array([0.0, 1.0])
                lo, hi = float(np.nanmin(all_v)), float(np.nanmax(all_v))
        else:
            lo, hi = float(ylim[0]), float(ylim[1])
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        span = hi - lo
        data_scale = (lo - extend * span, hi + extend * span)

        sz = anno_width_and_height(target_which, width, height, unit(2.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)
        ax_grob = construct_axis_grob(ap, target_which, data_scale) if axis else None

        gp_rec = recycle_gp(gp or gpar(fill="#CCCCCC"), n)
        pt_gp_rec = recycle_gp(pt_gp or gpar(), n)
        pch_rec = _recycle_seq(pch, n)
        sz_u = as_unit(size, "pt")
        sz_rec = unit(np.full(n, convertWidth(sz_u, "mm", valueOnly=True)), "mm") if len(sz_u) == 1 else sz_u

        def _draw(index, k=1, n_slices=1, env=None):
            val = env["value"]
            m = len(index)
            if m == 0:
                return
            rng = np.random.default_rng(42)
            if target_which == "column":
                pushViewport(Viewport(xscale=(0.5, m + 0.5), yscale=data_scale, clip=False))
                try:
                    _draw_background(border, background)
                    for pos_i, orig_i in enumerate(index):
                        v = _reverse_vals(val[orig_i], data_scale, ap["direction"])
                        grid_boxplot(
                            v,
                            pos=float(pos_i + 1),
                            outline=outline and not add_points,
                            box_width=box_width,
                            pch=env["pch"][orig_i],
                            size=env["size"][orig_i],
                            gp=subset_gp(env["gp"], orig_i),
                            direction="vertical",
                        )
                        if add_points:
                            vf = v[np.isfinite(v)]
                            jitter_x = (pos_i + 1) + rng.uniform(-0.25 * box_width, 0.25 * box_width, size=len(vf))
                            grid_points(jitter_x, vf, pch=env["pch"][orig_i], size=env["size"][orig_i], gp=subset_gp(env["pt_gp"], orig_i), default_units="native")
                    show_ax = axis and ((ap["side"] == "left" and k == 1) or (ap["side"] == "right" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()
            else:
                pushViewport(Viewport(xscale=data_scale, yscale=(0.5, m + 0.5), clip=False))
                try:
                    _draw_background(border, background)
                    for pos_i, orig_i in enumerate(index):
                        v = _reverse_vals(val[orig_i], data_scale, ap["direction"])
                        grid_boxplot(
                            v,
                            pos=float(m - pos_i),
                            outline=outline and not add_points,
                            box_width=box_width,
                            pch=env["pch"][orig_i],
                            size=env["size"][orig_i],
                            gp=subset_gp(env["gp"], orig_i),
                            direction="horizontal",
                        )
                        if add_points:
                            vf = v[np.isfinite(v)]
                            jitter_y = (m - pos_i) + rng.uniform(-0.25 * box_width, 0.25 * box_width, size=len(vf))
                            grid_points(vf, jitter_y, pch=env["pch"][orig_i], size=env["size"][orig_i], gp=subset_gp(env["pt_gp"], orig_i), default_units="native")
                    show_ax = axis and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices))
                    if show_ax and ax_grob is not None:
                        grid_draw(ax_grob)
                finally:
                    popViewport()

        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_boxplot",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            data_scale=data_scale,
            var_import={"value": series, "gp": gp_rec, "pt_gp": pt_gp_rec, "pch": pch_rec, "size": sz_rec},
            subset_rule={"value": subset_vector, "gp": subset_gp, "pt_gp": subset_gp, "pch": subset_vector, "size": subset_vector},
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- 7. anno_histogram ----

def anno_histogram(
    x: Any,
    which: str | None = None,
    n_breaks: int = 11,
    border: bool = False,
    gp: dict[str, Any] | None = None,
    axis: bool = True,
    axis_param: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    def _build(target_which: str) -> AnnotationFunction:
        series = _to_obs_series(x, target_which)
        n = len(series)
        all_v = np.concatenate(series) if n else np.array([0.0, 1.0])
        lo, hi = float(np.nanmin(all_v)), float(np.nanmax(all_v))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        edges = np.linspace(lo, hi, int(n_breaks))
        data_scale = (lo, hi)

        hists = [np.histogram(s[np.isfinite(s)], bins=edges)[0] for s in series]
        max_cnt = max((int(np.max(h)) for h in hists), default=1)
        if max_cnt <= 0:
            max_cnt = 1

        sz = anno_width_and_height(target_which, width, height, unit(4.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)
        ap["direction"] = "normal"
        ax_grob = construct_axis_grob(ap, target_which, data_scale) if axis else None
        gp_rec = recycle_gp(gp or gpar(fill="#CCCCCC"), n)

        def _draw(index, k=1, n_slices=1, env=None):
            m = len(index)
            if m == 0:
                return
            if target_which == "row":
                for pos_i, orig_i in enumerate(index):
                    pushViewport(Viewport(x=unit(0.0, "npc"), y=unit((m - pos_i - 1) / m, "npc"), height=unit(1.0 / m, "npc"), just=("left", "bottom"), xscale=data_scale, yscale=(0.0, float(max_cnt) * 1.05)))
                    try:
                        cnts = hists[orig_i]
                        sub_gp = subset_gp(env["gp"], orig_i)
                        grid_rect(x=edges[:-1], y=np.zeros(len(cnts)), width=np.diff(edges), height=cnts, just=("left", "bottom"), gp=sub_gp, default_units="native")
                        if border:
                            grid_rect(gp=gpar(fill="transparent", col="black"))
                    finally:
                        popViewport()
                pushViewport(Viewport(xscale=data_scale))
                try:
                    if axis and ax_grob is not None and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices)):
                        grid_draw(ax_grob)
                finally:
                    popViewport()
            else:
                for pos_i, orig_i in enumerate(index):
                    pushViewport(Viewport(x=unit(pos_i / m, "npc"), y=unit(0.0, "npc"), width=unit(1.0 / m, "npc"), just=("left", "bottom"), xscale=(0.0, float(max_cnt) * 1.05), yscale=data_scale))
                    try:
                        cnts = hists[orig_i]
                        sub_gp = subset_gp(env["gp"], orig_i)
                        grid_rect(x=np.zeros(len(cnts)), y=edges[:-1], width=cnts, height=np.diff(edges), just=("left", "bottom"), gp=sub_gp, default_units="native")
                        if border:
                            grid_rect(gp=gpar(fill="transparent", col="black"))
                    finally:
                        popViewport()
                pushViewport(Viewport(yscale=data_scale))
                try:
                    if axis and ax_grob is not None and ((ap["side"] == "left" and k == 1) or (ap["side"] == "right" and k == n_slices)):
                        grid_draw(ax_grob)
                finally:
                    popViewport()

        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_histogram",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            data_scale=data_scale,
            var_import={"value": hists, "gp": gp_rec},
            subset_rule={"value": subset_vector, "gp": subset_gp},
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- 8. anno_density ----

def anno_density(
    x: Any,
    which: str | None = None,
    type: str = "lines",
    xlim: Sequence[float] | None = None,
    max_density: float | None = None,
    heatmap_colors: Sequence[Any] | None = None,
    joyplot_scale: float = 1.0,
    border: bool = True,
    gp: dict[str, Any] | None = None,
    axis: bool = True,
    axis_param: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()
    if heatmap_colors is None:
        heatmap_colors = list(reversed(brewer_pal(11, "RdYlBu")))

    def _build(target_which: str) -> AnnotationFunction:
        series = _to_obs_series(x, target_which)
        n = len(series)
        all_v = np.concatenate(series) if n else np.array([0.0, 1.0])
        if xlim is None:
            lo, hi = float(np.nanmin(all_v)), float(np.nanmax(all_v))
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            data_scale = (lo - pad, hi + pad)
        else:
            data_scale = (float(xlim[0]), float(xlim[1]))

        grid_pts = np.linspace(data_scale[0], data_scale[1], 256)
        dens_list: list[np.ndarray] = []
        for s in series:
            sf = s[np.isfinite(s)]
            if len(sf) >= 2 and np.std(sf) > 1e-12:
                kde = gaussian_kde(sf)
                dens_list.append(kde(grid_pts))
            else:
                dens_list.append(np.zeros_like(grid_pts))

        max_d = float(max_density) if max_density is not None else max((float(np.max(d)) for d in dens_list), default=1.0)
        if max_d <= 0:
            max_d = 1.0
        col_fun = colorRamp2(np.linspace(0.0, max_d, len(heatmap_colors)), heatmap_colors)

        sz = anno_width_and_height(target_which, width, height, unit(4.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)
        ap["direction"] = "normal"
        ax_grob = construct_axis_grob(ap, target_which, data_scale) if axis else None
        gp_rec = recycle_gp(gp or gpar(fill="#CCCCCC"), n)

        def _draw(index, k=1, n_slices=1, env=None):
            m = len(index)
            if m == 0:
                return
            if target_which == "row":
                for pos_i, orig_i in enumerate(index):
                    y_scale = (0.0, max_d) if type == "lines" else (0.0, 1.0)
                    pushViewport(Viewport(x=unit(0.0, "npc"), y=unit((m - pos_i - 1) / m, "npc"), height=unit(1.0 / m, "npc"), just=("left", "bottom"), xscale=data_scale, yscale=y_scale))
                    try:
                        dv = dens_list[orig_i]
                        sub_gp = subset_gp(env["gp"], orig_i)
                        if type == "lines":
                            px = np.concatenate([[grid_pts[0]], grid_pts, [grid_pts[-1]]])
                            py = np.concatenate([[0.0], dv * joyplot_scale, [0.0]])
                            grid_polygon(px, py, gp=sub_gp, default_units="native")
                        elif type == "violin":
                            d_norm = 0.45 * dv / max(float(np.max(dv)), 1e-12)
                            px = np.concatenate([grid_pts, grid_pts[::-1]])
                            py = np.concatenate([0.5 + d_norm, 0.5 - d_norm[::-1]])
                            grid_polygon(px, py, gp=sub_gp, default_units="native")
                            sf = series[orig_i][np.isfinite(series[orig_i])]
                            if len(sf):
                                q1, med, q3 = np.quantile(sf, [0.25, 0.5, 0.75])
                                grid_segments([q1], [0.5], [q3], [0.5], gp=gpar(col="black", lwd=2), default_units="native")
                                grid_points([med], [0.5], pch=16, size=unit(2, "mm"), gp=gpar(col="white"), default_units="native")
                        elif type == "heatmap":
                            cols_h = list(col_fun(dv))
                            step = (data_scale[1] - data_scale[0]) / len(dv)
                            grid_rect(x=grid_pts, y=np.full(len(dv), 0.5), width=np.full(len(dv), step * 1.05), height=np.ones(len(dv)), gp=gpar(fill=cols_h, col=None), default_units="native")
                        if border:
                            grid_rect(gp=gpar(fill="transparent", col="black"))
                    finally:
                        popViewport()
                pushViewport(Viewport(xscale=data_scale))
                try:
                    if axis and ax_grob is not None and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices)):
                        grid_draw(ax_grob)
                finally:
                    popViewport()
            else:
                for pos_i, orig_i in enumerate(index):
                    x_scale = (0.0, max_d) if type == "lines" else (0.0, 1.0)
                    pushViewport(Viewport(x=unit(pos_i / m, "npc"), y=unit(0.0, "npc"), width=unit(1.0 / m, "npc"), just=("left", "bottom"), xscale=x_scale, yscale=data_scale))
                    try:
                        dv = dens_list[orig_i]
                        sub_gp = subset_gp(env["gp"], orig_i)
                        if type == "lines":
                            py = np.concatenate([[grid_pts[0]], grid_pts, [grid_pts[-1]]])
                            px = np.concatenate([[0.0], dv * joyplot_scale, [0.0]])
                            grid_polygon(px, py, gp=sub_gp, default_units="native")
                        elif type == "violin":
                            d_norm = 0.45 * dv / max(float(np.max(dv)), 1e-12)
                            py = np.concatenate([grid_pts, grid_pts[::-1]])
                            px = np.concatenate([0.5 + d_norm, 0.5 - d_norm[::-1]])
                            grid_polygon(px, py, gp=sub_gp, default_units="native")
                            sf = series[orig_i][np.isfinite(series[orig_i])]
                            if len(sf):
                                q1, med, q3 = np.quantile(sf, [0.25, 0.5, 0.75])
                                grid_segments([0.5], [q1], [0.5], [q3], gp=gpar(col="black", lwd=2), default_units="native")
                                grid_points([0.5], [med], pch=16, size=unit(2, "mm"), gp=gpar(col="white"), default_units="native")
                        elif type == "heatmap":
                            cols_h = list(col_fun(dv))
                            step = (data_scale[1] - data_scale[0]) / len(dv)
                            grid_rect(x=np.full(len(dv), 0.5), y=grid_pts, width=np.ones(len(dv)), height=np.full(len(dv), step * 1.05), gp=gpar(fill=cols_h, col=None), default_units="native")
                        if border:
                            grid_rect(gp=gpar(fill="transparent", col="black"))
                    finally:
                        popViewport()
                pushViewport(Viewport(yscale=data_scale))
                try:
                    if axis and ax_grob is not None and ((ap["side"] == "left" and k == 1) or (ap["side"] == "right" and k == n_slices)):
                        grid_draw(ax_grob)
                finally:
                    popViewport()

        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_density",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            data_scale=data_scale,
            var_import={"value": series, "gp": gp_rec},
            subset_rule={"value": subset_vector, "gp": subset_gp},
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- 9. anno_joyplot ----

def _normalize_xy_series(x: Any) -> list[np.ndarray]:
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=float)
        return [np.column_stack([np.arange(1, arr.shape[0] + 1, dtype=float), arr[:, c]]) for c in range(arr.shape[1])]
    if isinstance(x, np.ndarray) and x.ndim == 2:
        arr = x.astype(float)
        return [np.column_stack([np.arange(1, arr.shape[0] + 1, dtype=float), arr[:, c]]) for c in range(arr.shape[1])]
    items = list(x.values()) if isinstance(x, dict) else list(x)
    out: list[np.ndarray] = []
    for it in items:
        if isinstance(it, dict) and "x" in it and "y" in it:
            out.append(np.column_stack([np.asarray(it["x"], dtype=float), np.asarray(it["y"], dtype=float)]))
        elif isinstance(it, pd.DataFrame) and it.shape[1] >= 2:
            out.append(it.iloc[:, :2].to_numpy(dtype=float))
        else:
            a = np.asarray(it, dtype=float)
            if a.ndim == 2 and a.shape[1] >= 2:
                out.append(a[:, :2])
            else:
                out.append(np.column_stack([np.arange(1, len(a) + 1, dtype=float), a.ravel()]))
    return out


def anno_joyplot(
    x: Any,
    which: str | None = None,
    gp: dict[str, Any] | None = None,
    scale: float = 2.0,
    transparency: float = 0.6,
    axis: bool = True,
    axis_param: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    value = _normalize_xy_series(x)
    n = len(value)
    all_x = np.concatenate([v[:, 0] for v in value]) if n else np.array([0.0, 1.0])
    all_y = np.concatenate([v[:, 1] for v in value]) if n else np.array([0.0, 1.0])
    x0, x1 = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    xscale = (x0 - 0.025 * (x1 - x0), x1 + 0.025 * (x1 - x0))
    yscale = (0.0, max(1e-12, float(np.nanmax(all_y)) * 1.05))

    gp_d = check_gp(gp)
    if "fill" not in gp_d:
        gp_d["fill"] = "#000000"
    gp_rec = recycle_gp(gp_d, n)
    gp_rec["fill"] = add_transparency(gp_rec["fill"], transparency)

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(4.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)
        ap["direction"] = "normal"
        ax_grob = construct_axis_grob(ap, target_which, xscale) if axis else None

        def _draw(index, k=1, n_slices=1, env=None):
            val = env["value"]
            m = len(index)
            if m == 0:
                return
            if target_which == "row":
                for pos_i, orig_i in enumerate(index):
                    pushViewport(Viewport(x=unit(0.0, "npc"), y=unit((m - pos_i - 1) / m, "npc"), just=("left", "bottom"), height=unit(1.0 / m, "npc"), xscale=xscale, yscale=yscale))
                    try:
                        arr_xy = val[orig_i]
                        px = np.concatenate([[arr_xy[0, 0]], arr_xy[:, 0], [arr_xy[-1, 0]]])
                        py = np.concatenate([[0.0], arr_xy[:, 1] * scale, [0.0]])
                        sgp = subset_gp(env["gp"], orig_i)
                        gppp = sgp.copy()
                        gppp["col"] = "#00000000"
                        grid_polygon(px, py, gp=gppp, default_units="native")
                        grid_lines(px, py, gp=sgp, default_units="native")
                    finally:
                        popViewport()
                pushViewport(Viewport(xscale=xscale))
                try:
                    if axis and ax_grob is not None and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices)):
                        grid_draw(ax_grob)
                finally:
                    popViewport()
            else:
                for pos_i, orig_i in enumerate(index):
                    pushViewport(Viewport(y=unit(0.0, "npc"), x=unit((pos_i + 1) / m, "npc"), width=unit(1.0 / m, "npc"), just=("right", "bottom"), xscale=yscale, yscale=xscale))
                    try:
                        arr_xy = val[orig_i]
                        py = np.concatenate([[arr_xy[0, 0]], arr_xy[:, 0], [arr_xy[-1, 0]]])
                        px = np.concatenate([[0.0], arr_xy[:, 1] * scale, [0.0]])
                        sgp = subset_gp(env["gp"], orig_i)
                        gppp = sgp.copy()
                        gppp["col"] = "#00000000"
                        grid_polygon(px, py, gp=gppp, default_units="native")
                        grid_lines(px, py, gp=sgp, default_units="native")
                    finally:
                        popViewport()
                pushViewport(Viewport(yscale=xscale))
                try:
                    if axis and ax_grob is not None and ((ap["side"] == "left" and k == 1) or (ap["side"] == "right" and k == n_slices)):
                        grid_draw(ax_grob)
                finally:
                    popViewport()

        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_joyplot",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            data_scale=xscale,
            var_import={"value": value, "gp": gp_rec},
            subset_rule={"value": subset_vector, "gp": subset_gp},
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- 10. anno_horizon ----

def anno_horizon(
    x: Any,
    which: str = "row",
    gp: dict[str, Any] | None = None,
    n_slice: int = 4,
    slice_size: float | None = None,
    negative_from_top: bool = False,
    normalize: bool = True,
    gap: Any = unit(0, "mm"),
    axis: bool = True,
    axis_param: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which = which.lower()
    if which == "column":
        raise ValueError("`anno_horizon()` does not support column annotation.")

    value = _normalize_xy_series(x)
    n = len(value)
    gp_d = gpar(pos_fill="#D73027", neg_fill="#313695") if gp is None else check_gp(gp)
    if "pos_fill" not in gp_d:
        gp_d["pos_fill"] = "#D73027"
    if "neg_fill" not in gp_d:
        gp_d["neg_fill"] = "#313695"
    gp_rec = recycle_gp(gp_d, n)

    if normalize:
        norm_val = []
        for arr_xy in value:
            mx = float(np.nanmax(np.abs(arr_xy[:, 1])))
            out_arr = arr_xy.copy()
            if mx > 0:
                out_arr[:, 1] = out_arr[:, 1] / mx
            norm_val.append(out_arr)
        value = norm_val

    all_x = np.concatenate([v[:, 0] for v in value]) if n else np.array([0.0, 1.0])
    all_y = np.concatenate([np.abs(v[:, 1]) for v in value]) if n else np.array([0.0, 1.0])
    xscale = (float(np.nanmin(all_x)), float(np.nanmax(all_x)))
    ymax = max(1e-12, float(np.nanmax(all_y)))

    sz = anno_width_and_height("row", width, height, unit(4.0, "cm"))
    ap = validate_axis_param(axis_param, "row")
    ap["direction"] = "normal"
    ax_grob = construct_axis_grob(ap, "row", xscale) if axis else None
    gap_u = as_unit(gap, "mm")

    def _draw_horizon_chart(xv: np.ndarray, yv: np.ndarray, sl_size: float, n_sl: int, pos_fill: Any, neg_fill: Any) -> None:
        pos_cf = colorRamp2([0.0, float(n_sl)], ["white", pos_fill])
        neg_cf = colorRamp2([0.0, float(n_sl)], ["white", neg_fill])
        pushViewport(Viewport(xscale=(float(np.min(xv)), float(np.max(xv))), yscale=(0.0, sl_size)))
        try:
            for sign_v, cf, from_top in ((1.0, pos_cf, False), (-1.0, neg_cf, negative_from_top)):
                ys = sign_v * yv
                for i in range(1, n_sl + 1):
                    lo_b = (i - 1) * sl_size
                    hi_b = i * sl_size
                    active = ys >= lo_b
                    if not np.any(active):
                        continue
                    y_band = np.where(ys >= hi_b, sl_size, np.where(ys >= lo_b, ys - lo_b, np.nan))
                    # Split into contiguous non-NaN segments
                    valid = np.isfinite(y_band)
                    idx = 0
                    while idx < len(valid):
                        while idx < len(valid) and not valid[idx]:
                            idx += 1
                        j = idx
                        while j < len(valid) and valid[j]:
                            j += 1
                        if j > idx:
                            sx = xv[idx:j]
                            sy = y_band[idx:j]
                            px = np.concatenate([[sx[0]], sx, [sx[-1]]])
                            if from_top:
                                py = np.concatenate([[sl_size], sl_size - sy, [sl_size]])
                            else:
                                py = np.concatenate([[0.0], sy, [0.0]])
                            grid_polygon(px, py, gp=gpar(fill=cf(i), col=None), default_units="native")
                        idx = j
        finally:
            popViewport()

    def _draw(index, k=1, n_slices=1, env=None):
        val = env["value"]
        m = len(index)
        if m == 0:
            return
        sl_sz = float(slice_size) if slice_size is not None else ymax / n_slice
        n_sl = max(1, int(np.ceil(ymax / sl_sz)))
        for pos_i, orig_i in enumerate(index):
            pushViewport(Viewport(x=unit(0.0, "npc"), y=unit((m - pos_i - 1) / m, "npc"), just=("left", "bottom"), height=unit(1.0 / m, "npc") - gap_u))
            try:
                sgp = subset_gp(env["gp"], orig_i)
                _draw_horizon_chart(val[orig_i][:, 0], val[orig_i][:, 1], sl_sz, n_sl, sgp["pos_fill"], sgp["neg_fill"])
                grid_rect(gp=gpar(fill="transparent", col="black"))
            finally:
                popViewport()
        pushViewport(Viewport(xscale=xscale))
        try:
            if axis and ax_grob is not None and ((ap["side"] == "top" and k == 1) or (ap["side"] == "bottom" and k == n_slices)):
                grid_draw(ax_grob)
        finally:
            popViewport()

    anno = AnnotationFunction(
        fun=_draw,
        fun_name="anno_horizon",
        which="row",
        width=sz["width"],
        height=sz["height"],
        n=n,
        data_scale=xscale,
        var_import={"value": value, "gp": gp_rec},
        subset_rule={"value": subset_vector, "gp": subset_gp},
        subsettable=True,
        _which_explicit=True,
    )
    anno.extended = update_anno_extend(anno, ax_grob, ap)
    return anno


# ---- 11. anno_text ----

def anno_text(
    x: Any,
    which: str | None = None,
    gp: dict[str, Any] | None = None,
    rot: float | None = None,
    just: Any = None,
    offset: Any = None,
    location: Any = None,
    width: Unit | None = None,
    height: Unit | None = None,
    show_name: bool = False,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    from .text import GridText
    is_gt = isinstance(x, GridText)
    n = len(x)
    gp_rec = recycle_gp(gp or gpar(), n)

    def _build(target_which: str) -> AnnotationFunction:
        eff_rot = (90.0 if target_which == "column" else 0.0) if rot is None else float(rot)
        eff_just = ("right" if target_which == "column" else "left") if just is None else just
        loc_raw = offset if location is None and offset is not None else location
        if loc_raw is None:
            eff_loc = unit(1.0, "npc") if target_which == "column" else unit(0.0, "npc")
        else:
            eff_loc = as_unit(loc_raw, "npc")

        if target_which == "column":
            h_u = height if height is not None else max_text_height(x, gp=gp_rec, rot=eff_rot) + unit(2.0, "mm")
            w_u = width
        else:
            w_u = width if width is not None else max_text_width(x, gp=gp_rec, rot=eff_rot) + unit(2.0, "mm")
            h_u = height
        sz = anno_width_and_height(target_which, w_u, h_u, unit(1.0, "cm"))

        def _draw(index, k=1, n_slices=1, env=None):
            val = env["value"]
            sub_val = val[index] if isinstance(val, GridText) else [val[i] for i in index]
            sub_gp = subset_gp(env["gp"], index)
            m = len(index)
            if m == 0:
                return
            has_bg = "fill" in sub_gp or "border" in sub_gp
            if target_which == "column":
                xs = unit((np.arange(m) + 0.5) / m, "npc")
                ys = eff_loc
                if has_bg:
                    grid_rect(x=xs, y=unit(np.full(m, 0.5), "npc"), width=unit(np.full(m, 1.0 / m), "npc"), height=unit(np.ones(m), "npc"), gp=gpar(fill=sub_gp.get("fill", "transparent"), col=sub_gp.get("border", None)))
                grid_text(sub_val, x=xs, y=ys, just=eff_just, rot=eff_rot, gp=sub_gp)
            else:
                ys = unit((m - np.arange(m) - 0.5) / m, "npc")
                xs = eff_loc
                if has_bg:
                    grid_rect(x=unit(np.full(m, 0.5), "npc"), y=ys, width=unit(np.ones(m), "npc"), height=unit(np.full(m, 1.0 / m), "npc"), gp=gpar(fill=sub_gp.get("fill", "transparent"), col=sub_gp.get("border", None)))
                grid_text(sub_val, x=xs, y=ys, just=eff_just, rot=eff_rot, gp=sub_gp)

        val_store = x if is_gt else [str(v) for v in x]
        return AnnotationFunction(
            fun=_draw,
            fun_name="anno_text",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            var_import={"value": val_store, "gp": gp_rec},
            subset_rule={"value": (lambda v, i: v[i]) if is_gt else subset_vector, "gp": subset_gp},
            subsettable=True,
            show_name=show_name,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )

    return _build(eff_which)


# ---- 12. anno_numeric ----

def anno_numeric(
    x: Sequence[float],
    rg: Sequence[float] | None = None,
    labels_gp: dict[str, Any] | None = None,
    x_convert: Callable[[np.ndarray], np.ndarray] | None = None,
    labels_format: Callable[[Sequence[float]], Sequence[str]] | None = None,
    labels_offset: Any = unit(4, "pt"),
    bg_gp: dict[str, Any] | None = None,
    bar_width: Any = None,
    round_corners: bool = True,
    r: Any = unit(0.05, "snpc"),
    which: str = "row",
    align_to: str | float | None = None,
    width: Unit | None = None,
) -> AnnotationFunction:
    which = which.lower()
    if which == "column":
        raise ValueError("`anno_numeric()` can only be used as row annotation.")
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    lbls = [str(v) for v in (labels_format(arr) if labels_format is not None else np.round(arr, 2))]

    if rg is None:
        r_lo, r_hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    else:
        r_lo, r_hi = float(rg[0]), float(rg[1])
    if x_convert is not None:
        arr = np.asarray(x_convert(arr), dtype=float)
        conv_rg = np.asarray(x_convert(np.array([r_lo, r_hi], dtype=float)), dtype=float)
        r_lo, r_hi = float(np.min(conv_rg)), float(np.max(conv_rg))
    if r_lo == r_hi:
        r_hi = r_lo + 1e-6
    arr = np.clip(arr, r_lo, r_hi)

    if align_to is None:
        align_to = 0 if (np.any(arr > 0) and np.any(arr < 0)) else "left"

    lgp_rec = recycle_gp(labels_gp or gpar(fontsize=10), n)
    bgp_rec = recycle_gp(bg_gp or gpar(fill="#8080FF", col="#8080FF"), n)
    bw_u = (unit(1.0, "npc") - unit(4.0, "pt")) if bar_width is None else as_unit(bar_width, "npc")
    off_u = as_unit(labels_offset, "pt")
    r_u = as_unit(r, "snpc")

    if width is None:
        if align_to == 0:
            pos_lbls = [lb for v, lb in zip(arr, lbls) if v >= 0] or ["0"]
            neg_lbls = [lb for v, lb in zip(arr, lbls) if v < 0] or ["0"]
            w_mm = max(20.0, max_text_width(pos_lbls, gp=lgp_rec).to_mm_scalar() + max_text_width(neg_lbls, gp=lgp_rec).to_mm_scalar() + 6.0)
        else:
            w_mm = max(20.0, max_text_width(lbls, gp=lgp_rec).to_mm_scalar() + 4.0)
        w_u = unit(w_mm, "mm")
    else:
        w_u = as_unit(width, "mm")

    def _draw(index, k=1, n_slices=1, env=None):
        vals = env["value"]
        lb_list = env["labels"]
        m = len(index)
        for pos_i, orig_i in enumerate(index):
            pushViewport(Viewport(y=unit(1.0 - (pos_i + 1) / m, "npc"), height=unit(1.0 / m, "npc"), just="bottom", xscale=(r_lo, r_hi)))
            try:
                xv = float(vals[orig_i])
                lb = lb_list[orig_i]
                bgp_i = subset_gp(env["bg_gp"], orig_i)
                lgp_i = subset_gp(env["labels_gp"], orig_i)
                if align_to == "right":
                    w_bar = unit(xv - r_lo, "native")
                    if round_corners:
                        grid_roundrect(x=unit(1, "npc"), width=w_bar, height=bw_u, r=r_u, just="right", gp=bgp_i)
                    else:
                        grid_rect(x=unit(1, "npc"), width=w_bar, height=bw_u, just="right", gp=bgp_i)
                    grid_text(lb, x=unit(1, "npc") - off_u, just="right", gp=lgp_i)
                elif align_to == "left":
                    w_bar = unit(xv - r_lo, "native")
                    if round_corners:
                        grid_roundrect(x=unit(0, "npc"), width=w_bar, height=bw_u, r=r_u, just="left", gp=bgp_i)
                    else:
                        grid_rect(x=unit(0, "npc"), width=w_bar, height=bw_u, just="left", gp=bgp_i)
                    grid_text(lb, x=unit(0, "npc") + off_u, just="left", gp=lgp_i)
                else:  # diverging around 0
                    if xv >= 0:
                        if round_corners:
                            grid_roundrect(x=unit(0.0, "native"), width=unit(xv, "native"), height=bw_u, r=r_u, just="left", gp=bgp_i)
                        else:
                            grid_rect(x=unit(0.0, "native"), width=unit(xv, "native"), height=bw_u, just="left", gp=bgp_i)
                        grid_text(lb, x=unit(0.0, "native") + off_u, just="left", gp=lgp_i)
                    else:
                        if round_corners:
                            grid_roundrect(x=unit(0.0, "native"), width=unit(abs(xv), "native"), height=bw_u, r=r_u, just="right", gp=bgp_i)
                        else:
                            grid_rect(x=unit(0.0, "native"), width=unit(abs(xv), "native"), height=bw_u, just="right", gp=bgp_i)
                        grid_text(lb, x=unit(0.0, "native") - off_u, just="right", gp=lgp_i)
            finally:
                popViewport()

    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_numeric",
        which="row",
        width=w_u,
        height=unit(1.0, "npc"),
        n=n,
        var_import={"value": arr, "labels": lbls, "bg_gp": bgp_rec, "labels_gp": lgp_rec},
        subset_rule={"value": subset_vector, "labels": subset_vector, "bg_gp": subset_gp, "labels_gp": subset_gp},
        subsettable=True,
        show_name=False,
        _which_explicit=True,
    )


# ---- 13. anno_mark ----

def anno_mark(
    at: Sequence[int] | Sequence[bool],
    labels: Sequence[str],
    which: str | None = None,
    side: str | None = None,
    lines_gp: dict[str, Any] | None = None,
    labels_gp: dict[str, Any] | None = None,
    labels_rot: float | None = None,
    padding: Any = unit(1, "mm"),
    link_width: Any = unit(5, "mm"),
    link_height: Any = None,
    link_gp: dict[str, Any] | None = None,
    extend: Any = unit(0, "mm"),
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    at_arr = np.asarray(at)
    if at_arr.dtype == bool:
        at_idx = np.where(at_arr)[0].tolist()
    else:
        at_idx = [int(v) for v in at_arr]

    n = len(at_idx)
    if n < 1:
        return anno_empty(which=eff_which, border=False)

    lbl_list = [str(v) for v in labels]
    lgp_base = check_gp(link_gp if link_gp is not None else (lines_gp or gpar()))
    lbl_gp_base = check_gp(labels_gp or gpar())
    lgp_rec = recycle_gp(lgp_base, n)
    lbl_gp_rec = recycle_gp(lbl_gp_base, n)

    od = np.argsort(at_idx)
    at_sorted = [at_idx[i] for i in od]
    lbl_sorted = [lbl_list[i] for i in od]
    lgp_sorted = subset_gp(lgp_rec, od)
    lbl_gp_sorted = subset_gp(lbl_gp_rec, od)
    at2index = {a: i for i, a in enumerate(at_sorted)}
    at2label = dict(zip(at_sorted, lbl_sorted))

    lw_u = as_unit(link_width, "mm")
    lh_u = lw_u if link_height is None else as_unit(link_height, "mm")
    ext_u = as_unit(extend, "npc")
    if len(ext_u) == 1:
        ext_u = unit_c(ext_u, ext_u)
    else:
        ext_u = ext_u[:2]
    pad_u = as_unit(padding, "mm") if isinstance(padding, Unit) else unit(float(padding) * textGrob("a", gp=subset_gp(lbl_gp_sorted, 0)).height_mm(), "mm")

    def _build(target_which: str) -> AnnotationFunction:
        eff_side = ("top" if target_which == "column" else "right") if side is None else side
        eff_rot = (90.0 if target_which == "column" else 0.0) if labels_rot is None else float(labels_rot) % 360.0

        if target_which == "row":
            w_u = lw_u + max_text_width(lbl_sorted, gp=lbl_gp_sorted, rot=eff_rot)
            h_u = unit(1.0, "npc")
        else:
            w_u = unit(1.0, "npc")
            h_u = lh_u + max_text_height(lbl_sorted, gp=lbl_gp_sorted, rot=eff_rot)

        def _draw(index, k=1, n_slices=1, env=None):
            at_cur = env["at"]
            idx_list = [int(v) for v in index]
            m = len(idx_list)
            present = [a for a in idx_list if a in at2index]
            if not present:
                return
            pos_map_in_index = {a: i for i, a in enumerate(idx_list)}
            dot_pos = env.get(".pos")
            dot_scale = env.get(".scale")

            if target_which == "row":
                # Reverse order so smartAlign runs bottom-to-top
                sel = sorted(present, key=lambda a: pos_map_in_index[a], reverse=True)
                sel_lbls = [at2label[a] for a in sel]
                sel_order = [at2index[a] for a in sel]
                sel_lbl_gp = subset_gp(lbl_gp_sorted, sel_order)
                sel_link_gp = subset_gp(lgp_sorted, sel_order)
                sc = dot_scale if dot_scale is not None else (0.5, m + 0.5)
                pushViewport(Viewport(xscale=(0.0, 1.0), yscale=sc))
                try:
                    ext_n = ext_u.to_mm("y", current_viewport()) / max(1e-9, current_viewport().h_mm) * (sc[1] - sc[0]) if ext_u.is_abs() else ext_u._values * (sc[1] - sc[0])
                    if dot_pos is not None:
                        pos = np.array([dot_pos[pos_map_in_index[a]] for a in sel], dtype=float)
                    else:
                        pos = np.array([m - pos_map_in_index[a] for a in sel], dtype=float)
                    pad_mm = convertHeight(pad_u, "mm", valueOnly=True)
                    nat_per_mm = (sc[1] - sc[0]) / max(1e-9, current_viewport().h_mm)
                    th = np.array([
                        (textGrob(lb, gp=subset_gp(sel_lbl_gp, ii)).width_mm() if eff_rot in (90.0, 270.0) else textGrob(lb, gp=subset_gp(sel_lbl_gp, ii)).height_mm() + pad_mm) * nat_per_mm
                        for ii, lb in enumerate(sel_lbls)
                    ])
                    aligned = smartAlign(pos - 0.5 * th, pos + 0.5 * th, (sc[0] - ext_n[0], sc[1] + ext_n[1]))
                    h_adj = 0.5 * (aligned[:, 0] + aligned[:, 1])

                    if eff_side == "right":
                        just = ("center", "top") if eff_rot == 90 else (("center", "bottom") if eff_rot == 270 else ("left", "center"))
                        for ii in range(len(sel)):
                            grid_text(sel_lbls[ii], x=lw_u, y=unit(h_adj[ii], "native"), just=just, rot=eff_rot, gp=subset_gp(sel_lbl_gp, ii))
                        lw_eff = lw_u - unit(1.0, "mm")
                        grid_segments(unit(np.zeros(len(sel)), "npc"), unit(pos, "native"), lw_eff * (1 / 3), unit(pos, "native"), gp=sel_link_gp)
                        grid_segments(lw_eff * (1 / 3), unit(pos, "native"), lw_eff * (2 / 3), unit(h_adj, "native"), gp=sel_link_gp)
                        grid_segments(lw_eff * (2 / 3), unit(h_adj, "native"), lw_eff, unit(h_adj, "native"), gp=sel_link_gp)
                    else:
                        just = ("center", "bottom") if eff_rot == 90 else (("center", "top") if eff_rot == 270 else ("right", "center"))
                        for ii in range(len(sel)):
                            grid_text(sel_lbls[ii], x=unit(1.0, "npc") - lw_u, y=unit(h_adj[ii], "native"), just=just, rot=eff_rot, gp=subset_gp(sel_lbl_gp, ii))
                        lw_eff = lw_u - unit(1.0, "mm")
                        grid_segments(unit(np.ones(len(sel)), "npc"), unit(pos, "native"), unit(1.0, "npc") - lw_eff * (1 / 3), unit(pos, "native"), gp=sel_link_gp)
                        grid_segments(unit(1.0, "npc") - lw_eff * (1 / 3), unit(pos, "native"), unit(1.0, "npc") - lw_eff * (2 / 3), unit(h_adj, "native"), gp=sel_link_gp)
                        grid_segments(unit(1.0, "npc") - lw_eff * (2 / 3), unit(h_adj, "native"), unit(1.0, "npc") - lw_eff, unit(h_adj, "native"), gp=sel_link_gp)
                finally:
                    popViewport()
            else:
                sel = sorted(present, key=lambda a: pos_map_in_index[a])
                sel_lbls = [at2label[a] for a in sel]
                sel_order = [at2index[a] for a in sel]
                sel_lbl_gp = subset_gp(lbl_gp_sorted, sel_order)
                sel_link_gp = subset_gp(lgp_sorted, sel_order)
                sc = dot_scale if dot_scale is not None else (0.5, m + 0.5)
                pushViewport(Viewport(xscale=sc, yscale=(0.0, 1.0)))
                try:
                    ext_n = ext_u.to_mm("x", current_viewport()) / max(1e-9, current_viewport().w_mm) * (sc[1] - sc[0]) if ext_u.is_abs() else ext_u._values * (sc[1] - sc[0])
                    if dot_pos is not None:
                        pos = np.array([dot_pos[pos_map_in_index[a]] for a in sel], dtype=float)
                    else:
                        pos = np.array([pos_map_in_index[a] + 1 for a in sel], dtype=float)
                    pad_mm = convertWidth(pad_u, "mm", valueOnly=True)
                    nat_per_mm = (sc[1] - sc[0]) / max(1e-9, current_viewport().w_mm)
                    tw = np.array([
                        (textGrob(lb, gp=subset_gp(sel_lbl_gp, ii)).width_mm() if eff_rot in (0.0, 180.0) else textGrob(lb, gp=subset_gp(sel_lbl_gp, ii)).height_mm() + pad_mm) * nat_per_mm
                        for ii, lb in enumerate(sel_lbls)
                    ])
                    aligned = smartAlign(pos - 0.5 * tw, pos + 0.5 * tw, (sc[0] - ext_n[0], sc[1] + ext_n[1]))
                    h_adj = 0.5 * (aligned[:, 0] + aligned[:, 1])

                    if eff_side == "top":
                        just = ("center", "bottom") if eff_rot == 0 else ("left", "center")
                        for ii in range(len(sel)):
                            grid_text(sel_lbls[ii], x=unit(h_adj[ii], "native"), y=lh_u, just=just, rot=eff_rot, gp=subset_gp(sel_lbl_gp, ii))
                        lh_eff = lh_u - unit(1.0, "mm")
                        grid_segments(unit(pos, "native"), unit(np.zeros(len(sel)), "npc"), unit(pos, "native"), lh_eff * (1 / 3), gp=sel_link_gp)
                        grid_segments(unit(pos, "native"), lh_eff * (1 / 3), unit(h_adj, "native"), lh_eff * (2 / 3), gp=sel_link_gp)
                        grid_segments(unit(h_adj, "native"), lh_eff * (2 / 3), unit(h_adj, "native"), lh_eff, gp=sel_link_gp)
                    else:
                        just = ("center", "top") if eff_rot == 0 else ("right", "center")
                        for ii in range(len(sel)):
                            grid_text(sel_lbls[ii], x=unit(h_adj[ii], "native"), y=unit(1.0, "npc") - lh_u, just=just, rot=eff_rot, gp=subset_gp(sel_lbl_gp, ii))
                        lh_eff = lh_u - unit(1.0, "mm")
                        grid_segments(unit(pos, "native"), unit(np.ones(len(sel)), "npc"), unit(pos, "native"), unit(1.0, "npc") - lh_eff * (1 / 3), gp=sel_link_gp)
                        grid_segments(unit(pos, "native"), unit(1.0, "npc") - lh_eff * (1 / 3), unit(h_adj, "native"), unit(1.0, "npc") - lh_eff * (2 / 3), gp=sel_link_gp)
                        grid_segments(unit(h_adj, "native"), unit(1.0, "npc") - lh_eff * (2 / 3), unit(h_adj, "native"), unit(1.0, "npc") - lh_eff, gp=sel_link_gp)
                finally:
                    popViewport()

        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_mark",
            which=target_which,
            width=w_u,
            height=h_u,
            n=-1,
            var_import={"at": at_sorted, ".pos": None, ".scale": None},
            subset_rule={"at": subset_by_intersect},
            subsettable=True,
            show_name=False,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno._called_args = {
            "at": at_sorted, "labels": lbl_sorted, "which": target_which, "side": eff_side,
            "labels_gp": lbl_gp_sorted, "labels_rot": eff_rot, "padding": pad_u,
            "link_width": lw_u, "link_height": lh_u, "link_gp": lgp_sorted, "extend": ext_u,
        }
        return anno

    return _build(eff_which)


# ---- 14 & 15. anno_zoom & anno_link ----

def _contiguous_runs(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    s = positions[0]
    prev = positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
        else:
            runs.append((s, prev))
            s = prev = p
    runs.append((s, prev))
    return runs


def anno_zoom(
    align_to: Any,
    panel_fun: Callable | None = None,
    which: str | None = None,
    side: str | None = None,
    size: Any = None,
    gap: Any = unit(1, "mm"),
    link_width: Any = unit(5, "mm"),
    link_height: Any = None,
    link_gp: dict[str, Any] | None = None,
    extend: Any = unit(0, "mm"),
    width: Unit | None = None,
    height: Unit | None = None,
    internal_line: bool = True,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    if panel_fun is None:
        def panel_fun(index, nm=None):
            grid_rect()

    import inspect
    pf_params = len(inspect.signature(panel_fun).parameters)

    lw_u = as_unit(link_width, "mm")
    lh_u = lw_u if link_height is None else as_unit(link_height, "mm")
    gap_u = as_unit(gap, "mm")
    ext_u = as_unit(extend, "npc")
    if len(ext_u) == 1:
        ext_u = unit_c(ext_u, ext_u)
    else:
        ext_u = ext_u[:2]

    def _build(target_which: str) -> AnnotationFunction:
        eff_side = ("top" if target_which == "column" else "right") if side is None else side
        default_sz = unit(2.0, "cm") + (lw_u if target_which == "row" else lh_u)
        sz = anno_width_and_height(target_which, width, height, default_sz)

        def _draw(index, k=1, n_slices=1, env=None):
            idx_list = [int(v) for v in index]
            m = len(idx_list)
            if m == 0:
                return
            aln_raw = env["align_to"]
            if isinstance(aln_raw, dict):
                aln_dict = {str(k_nm): [int(v) for v in vs] for k_nm, vs in aln_raw.items()}
            else:
                seq = list(aln_raw)
                if seq and isinstance(seq[0], (list, tuple, np.ndarray, set)):
                    aln_dict = {str(i): [int(v) for v in vs] for i, vs in enumerate(seq)}
                elif all(isinstance(v, (int, np.integer)) for v in seq) and len(seq) != m:
                    aln_dict = {"": [int(v) for v in seq]}
                else:
                    # Categorical vector matching m observations
                    if len(seq) != m:
                        raise ValueError("Categorical `align_to` length must equal number of heatmap observations.")
                    # Order groups in appearance order along `index`
                    groups: dict[str, list[int]] = {}
                    for orig_i in idx_list:
                        lbl = str(seq[orig_i])
                        groups.setdefault(lbl, []).append(orig_i)
                    aln_dict = groups

            idx_set = set(idx_list)
            aln_dict = {nm: [v for v in vs if v in idx_set] for nm, vs in aln_dict.items()}
            aln_dict = {nm: vs for nm, vs in aln_dict.items() if len(vs) > 0}
            if not aln_dict:
                return

            box_names = list(aln_dict.keys())
            box_lists = list(aln_dict.values())
            n_boxes = len(box_lists)
            pos_in_idx = {orig: p for p, orig in enumerate(idx_list)}
            runs_per_box = [_contiguous_runs(sorted(pos_in_idx[v] for v in vs)) for vs in box_lists]

            dot_pos = env.get(".pos")
            if target_which == "row":
                pos = np.asarray(dot_pos, dtype=float) if dot_pos is not None else (m - np.arange(m, dtype=float) - 0.5) / m
                half_cell = 0.5 / m if dot_pos is None else max(1e-4, 0.5 * abs(pos[0] - pos[-1]) / max(1, m - 1))
                sc = (0.0, 1.0)
                pushViewport(Viewport(xscale=(0.0, 1.0), yscale=sc))
                try:
                    ext_n = ext_u.to_mm("y", current_viewport()) / max(1e-9, current_viewport().h_mm) if ext_u.is_abs() else ext_u._values
                    gap_n = convertHeight(gap_u, "mm", valueOnly=True) / max(1e-9, current_viewport().h_mm)
                    if size is None:
                        sz_arg = [len(vs) for vs in box_lists]
                        is_sz_unit = False
                    elif isinstance(size, Unit):
                        sz_arg = size
                        is_sz_unit = True
                    else:
                        sz_arg = [float(size)] * n_boxes if np.ndim(size) == 0 else [float(v) for v in size]
                        is_sz_unit = False

                    if not is_sz_unit:
                        if n_boxes == 1:
                            tops = np.array([1.0 + ext_n[1]])
                            bots = np.array([0.0 - ext_n[0]])
                        else:
                            sz_arr = np.asarray(sz_arg, dtype=float)
                            bh = sz_arr / np.sum(sz_arr) * (1.0 + np.sum(ext_n) - gap_n * (n_boxes - 1))
                            tops_rev = np.cumsum(bh) + np.arange(n_boxes) * gap_n - ext_n[0]
                            bots_rev = tops_rev - bh
                            tops = 1.0 - bots_rev
                            bots = 1.0 - tops_rev
                    else:
                        bh_mm = sz_arg.to_mm("y", current_viewport())
                        if len(bh_mm) == 1:
                            bh_mm = np.repeat(bh_mm[0], n_boxes)
                        bh_n = bh_mm / max(1e-9, current_viewport().h_mm)
                        bh2 = bh_n + gap_n
                        if n_boxes > 1:
                            bh2[0] -= 0.5 * gap_n
                            bh2[-1] -= 0.5 * gap_n
                        else:
                            bh2[0] -= gap_n
                        means = np.array([np.mean([0.5 * (pos[s] + pos[e]) for s, e in runs]) for runs in runs_per_box])
                        aligned = smartAlign2((means - 0.5 * bh2)[::-1], (means + 0.5 * bh2)[::-1], (0.0 - ext_n[0], 1.0 + ext_n[1]))
                        aligned = aligned[::-1]
                        bots = aligned[:, 0].copy()
                        tops = aligned[:, 1].copy()
                        if n_boxes > 1:
                            bots[0] += 0.5 * gap_n
                            tops[-1] -= 0.5 * gap_n
                            for bi in range(1, n_boxes - 1):
                                bots[bi] += 0.5 * gap_n
                                tops[bi] -= 0.5 * gap_n
                finally:
                    popViewport()

                # Draw boxes
                box_x = lw_u if eff_side == "right" else unit(0.0, "npc")
                pushViewport(Viewport(x=box_x, just="left", width=sz["width"] - lw_u))
                try:
                    for bi in range(n_boxes):
                        pushViewport(Viewport(y=unit(0.5 * (tops[bi] + bots[bi]), "native"), height=unit(max(1e-6, tops[bi] - bots[bi]), "native")))
                        try:
                            if pf_params <= 1:
                                panel_fun(box_lists[bi])
                            else:
                                panel_fun(box_lists[bi], box_names[bi])
                        finally:
                            popViewport()
                finally:
                    popViewport()

                # Draw ribbons
                lgp = check_gp(link_gp or gpar())
                if "fill" not in lgp:
                    lgp["fill"] = "transparent"
                lgp_rec = recycle_gp(lgp, n_boxes)
                lx = unit(0.0, "npc") if eff_side == "right" else unit(1.0, "npc")
                ljust = "left" if eff_side == "right" else "right"
                pushViewport(Viewport(x=lx, just=ljust, width=lw_u))
                try:
                    for bi in range(n_boxes):
                        gp_i = subset_gp(lgp_rec, bi)
                        for s_pos, e_pos in runs_per_box[bi]:
                            y_hi = pos[s_pos] + half_cell
                            y_lo = pos[e_pos] - half_cell
                            xs_poly = unit_c(unit([0, 0], "npc"), lw_u, lw_u) if eff_side == "right" else unit_c(lw_u, lw_u, unit([0, 0], "npc"))
                            ys_poly = unit([y_lo, y_hi, tops[bi], bots[bi]], "native")
                            if not internal_line:
                                gp_poly = gp_i.copy()
                                gp_poly["col"] = gp_i.get("fill", "transparent")
                                gp_poly.pop("lty", None)
                                grid_polygon(xs_poly, ys_poly, gp=gp_poly)
                                gp_line = gp_i.copy()
                                gp_line["fill"] = "transparent"
                                xs_ln = unit_c(lw_u, unit([0, 0], "npc"), lw_u) if eff_side == "right" else unit_c(unit(0, "npc"), lw_u, lw_u, unit(0, "npc"))
                                grid_lines(xs_ln, unit([bots[bi], y_lo, y_hi, tops[bi]], "native"), gp=gp_line)
                            else:
                                grid_polygon(xs_poly, ys_poly, gp=gp_i)
                finally:
                    popViewport()
            else:
                pos = np.asarray(dot_pos, dtype=float) if dot_pos is not None else (np.arange(m, dtype=float) + 0.5) / m
                half_cell = 0.5 / m if dot_pos is None else max(1e-4, 0.5 * abs(pos[-1] - pos[0]) / max(1, m - 1))
                sc = (0.0, 1.0)
                pushViewport(Viewport(xscale=sc, yscale=(0.0, 1.0)))
                try:
                    ext_n = ext_u.to_mm("x", current_viewport()) / max(1e-9, current_viewport().w_mm) if ext_u.is_abs() else ext_u._values
                    gap_n = convertWidth(gap_u, "mm", valueOnly=True) / max(1e-9, current_viewport().w_mm)
                    if size is None:
                        sz_arg = [len(vs) for vs in box_lists]
                        is_sz_unit = False
                    elif isinstance(size, Unit):
                        sz_arg = size
                        is_sz_unit = True
                    else:
                        sz_arg = [float(size)] * n_boxes if np.ndim(size) == 0 else [float(v) for v in size]
                        is_sz_unit = False

                    if not is_sz_unit:
                        if n_boxes == 1:
                            lefts = np.array([0.0 - ext_n[0]])
                            rights = np.array([1.0 + ext_n[1]])
                        else:
                            sz_arr = np.asarray(sz_arg, dtype=float)
                            bw = sz_arr / np.sum(sz_arr) * (1.0 + np.sum(ext_n) - gap_n * (n_boxes - 1))
                            rights = np.cumsum(bw) + np.arange(n_boxes) * gap_n - ext_n[0]
                            lefts = rights - bw
                    else:
                        bw_mm = sz_arg.to_mm("x", current_viewport())
                        if len(bw_mm) == 1:
                            bw_mm = np.repeat(bw_mm[0], n_boxes)
                        bw_n = bw_mm / max(1e-9, current_viewport().w_mm)
                        bw2 = bw_n + gap_n
                        if n_boxes > 1:
                            bw2[0] -= 0.5 * gap_n
                            bw2[-1] -= 0.5 * gap_n
                        else:
                            bw2[0] -= gap_n
                        means = np.array([np.mean([0.5 * (pos[s] + pos[e]) for s, e in runs]) for runs in runs_per_box])
                        aligned = smartAlign2(means - 0.5 * bw2, means + 0.5 * bw2, (0.0 - ext_n[0], 1.0 + ext_n[1]))
                        lefts = aligned[:, 0].copy()
                        rights = aligned[:, 1].copy()
                        if n_boxes > 1:
                            lefts[0] += 0.5 * gap_n
                            rights[-1] -= 0.5 * gap_n
                            for bi in range(1, n_boxes - 1):
                                lefts[bi] += 0.5 * gap_n
                                rights[bi] -= 0.5 * gap_n
                finally:
                    popViewport()

                box_y = lh_u if eff_side == "top" else unit(0.0, "npc")
                pushViewport(Viewport(y=box_y, just="bottom", height=sz["height"] - lh_u))
                try:
                    for bi in range(n_boxes):
                        pushViewport(Viewport(x=unit(0.5 * (lefts[bi] + rights[bi]), "native"), width=unit(max(1e-6, rights[bi] - lefts[bi]), "native")))
                        try:
                            if pf_params <= 1:
                                panel_fun(box_lists[bi])
                            else:
                                panel_fun(box_lists[bi], box_names[bi])
                        finally:
                            popViewport()
                finally:
                    popViewport()

                lgp = check_gp(link_gp or gpar())
                if "fill" not in lgp:
                    lgp["fill"] = "transparent"
                lgp_rec = recycle_gp(lgp, n_boxes)
                ly = unit(0.0, "npc") if eff_side == "top" else unit(1.0, "npc")
                ljust = "bottom" if eff_side == "top" else "top"
                pushViewport(Viewport(y=ly, just=ljust, height=lh_u))
                try:
                    for bi in range(n_boxes):
                        gp_i = subset_gp(lgp_rec, bi)
                        for s_pos, e_pos in runs_per_box[bi]:
                            x_lo = pos[s_pos] - half_cell
                            x_hi = pos[e_pos] + half_cell
                            xs_poly = unit([x_hi, x_lo, lefts[bi], rights[bi]], "native")
                            ys_poly = unit_c(unit([0, 0], "npc"), lh_u, lh_u) if eff_side == "top" else unit_c(lh_u, lh_u, unit([0, 0], "npc"))
                            grid_polygon(xs_poly, ys_poly, gp=gp_i)
                finally:
                    popViewport()

        def _subset_align_to(x_val: Any, idx: Sequence[int]) -> Any:
            s_idx = set(int(v) for v in idx)
            if isinstance(x_val, dict):
                return {k: [v for v in vs if v in s_idx] for k, vs in x_val.items() if any(v in s_idx for v in vs)}
            return [x_val[i] for i in idx]

        return AnnotationFunction(
            fun=_draw,
            fun_name="anno_zoom",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=-1,
            var_import={"align_to": align_to, ".pos": None},
            subset_rule={"align_to": _subset_align_to},
            subsettable=True,
            show_name=False,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )

    return _build(eff_which)


def anno_link(*args: Any, **kwargs: Any) -> AnnotationFunction:
    """Alias for `anno_zoom`."""
    return anno_zoom(*args, **kwargs)


# ---- 16. anno_block ----

def anno_block(
    align_to: Any = None,
    gp: dict[str, Any] | None = None,
    labels: Sequence[str] | None = None,
    labels_gp: dict[str, Any] | None = None,
    labels_rot: float | None = None,
    labels_offset: Any = unit(0.5, "npc"),
    labels_just: Any = "center",
    which: str | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
    show_name: bool = False,
    panel_fun: Callable | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    if align_to is not None:
        if isinstance(align_to, (list, tuple, np.ndarray)) and align_to and isinstance(align_to[0], (int, np.integer)):
            align_to = {"v": [int(v) for v in align_to]}
        elif not isinstance(align_to, dict):
            raise ValueError("`align_to` must be a dict mapping names to index lists, or a list of indices.")

    gp_d = check_gp(gp)
    lbl_gp_d = check_gp(labels_gp)
    off_u = as_unit(labels_offset, "npc")

    def _build(target_which: str) -> AnnotationFunction:
        eff_rot = (90.0 if target_which == "row" else 0.0) if labels_rot is None else float(labels_rot)
        w_u, h_u = width, height
        if labels is not None:
            if target_which == "column" and h_u is None:
                h_u = max_text_height(labels, gp=lbl_gp_d, rot=eff_rot) + unit(5.0, "mm")
            elif target_which == "row" and w_u is None:
                w_u = max_text_width(labels, gp=lbl_gp_d, rot=eff_rot) + unit(5.0, "mm")
        sz = anno_width_and_height(target_which, w_u, h_u, unit(5.0, "mm"))

        def _draw(index, k=1, n_slices=1, env=None):
            idx_list = [int(v) for v in index]
            ni = len(idx_list)
            if align_to is not None:
                for nm, grp in align_to.items():
                    positions = [pos for pos, orig in enumerate(idx_list) if orig in grp]
                    if not positions:
                        continue
                    if any(positions[i + 1] - positions[i] > 1 for i in range(len(positions) - 1)):
                        raise ValueError("Indices in `align_to` must be continuously adjacent in the heatmap.")
                    if target_which == "row":
                        pushViewport(Viewport(y=unit((ni - positions[-1] - 1) / ni, "npc"), height=unit(len(positions) / ni, "npc"), just="bottom"))
                    else:
                        pushViewport(Viewport(x=unit((positions[-1] + 1) / ni, "npc"), width=unit(len(positions) / ni, "npc"), just="right"))
                    try:
                        if panel_fun is not None:
                            panel_fun([idx_list[p] for p in positions], nm)
                    finally:
                        popViewport()
            elif panel_fun is None:
                grid_rect(gp=subset_gp(recycle_gp(gp_d, n_slices), k - 1))
                if labels is not None:
                    if len(labels) != n_slices:
                        raise ValueError(f"Length of `labels` ({len(labels)}) must equal number of slices ({n_slices}).")
                    lbl = labels[k - 1]
                    lgp = subset_gp(recycle_gp(lbl_gp_d, n_slices), k - 1)
                    tx = off_u if target_which == "row" else unit(0.5, "npc")
                    ty = off_u if target_which == "column" else unit(0.5, "npc")
                    grid_text(lbl, x=tx, y=ty, gp=lgp, rot=eff_rot, just=labels_just)
            else:
                ht = _ENV.current_heatmap
                levels = None
                if ht is not None:
                    sp = ht._row_split_df if target_which == "row" else ht._column_split_df
                    ol = ht.row_order_list if target_which == "row" else ht.column_order_list
                    if sp is not None and k - 1 < len(ol):
                        row_i = ol[k - 1][0]
                        levels = [str(v) for v in sp.iloc[row_i].tolist()]
                panel_fun(idx_list, levels)

        return AnnotationFunction(
            fun=_draw,
            fun_name="anno_block",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=None,
            var_import={},
            subset_rule={},
            subsettable=True,
            show_name=show_name,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )

    return _build(eff_which)


# ---- 17. anno_summary ----

def anno_summary(
    which: str | None = None,
    border: bool = True,
    bar_width: float = 0.8,
    axis: bool = True,
    axis_param: dict[str, Any] | None = None,
    ylim: Sequence[float] | None = None,
    extend: float = 0.05,
    outline: bool = True,
    box_width: float = 0.6,
    pch: Any = 1,
    size: Any = unit(2, "mm"),
    gp: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()
    gp_d = check_gp(gp)
    sz_u = as_unit(size, "mm")

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(2.0, "cm"))
        ap = validate_axis_param(axis_param, target_which)

        def _draw(index, k=1, n_slices=1, env=None):
            ht = _ENV.current_heatmap
            if ht is None:
                return
            mat = ht._matrix
            cm = ht._matrix_color_mapping
            order_list = ht.row_order_list if target_which == "column" else ht.column_order_list
            ng = len(order_list)
            if cm.type == "discrete":
                if ylim is not None:
                    raise ValueError("`ylim` cannot be set for discrete matrix in `anno_summary`.")
                ax_grob = construct_axis_grob(ap, target_which, (0.0, 1.0)) if axis else None
                if target_which == "column":
                    pushViewport(Viewport(xscale=(0.5, ng + 0.5), yscale=(0.0, 1.0)))
                else:
                    pushViewport(Viewport(xscale=(0.0, 1.0), yscale=(0.5, ng + 0.5)))
                try:
                    for gi, od in enumerate(order_list):
                        vals = mat[od, 0] if target_which == "column" else mat[0, od]
                        uniq, cnts = np.unique([str(v) for v in vals], return_counts=True)
                        props = cnts / np.sum(cnts)
                        cum = np.cumsum(props)
                        cols = [cm.map_to_colors(u) for u in uniq]
                        if target_which == "column":
                            grid_rect(x=np.full(len(uniq), gi + 1.0), y=cum, width=np.full(len(uniq), bar_width), height=props, just="top", gp=gpar(fill=cols), default_units="native")
                        else:
                            grid_rect(x=cum, y=np.full(len(uniq), ng - gi), width=props, height=np.full(len(uniq), bar_width), just="right", gp=gpar(fill=cols), default_units="native")
                    if axis and ax_grob is not None:
                        grid_draw(ax_grob)
                    if border:
                        grid_rect(gp=gpar(fill="transparent", col="black"))
                finally:
                    popViewport()
            else:
                vl = [mat[od, 0].astype(float) if target_which == "column" else mat[0, od].astype(float) for od in order_list]
                if ylim is None:
                    all_v = np.concatenate(vl)
                    lo, hi = float(np.nanmin(all_v)), float(np.nanmax(all_v))
                else:
                    lo, hi = float(ylim[0]), float(ylim[1])
                span = max(1e-9, hi - lo)
                ds = (lo - extend * span, hi + extend * span)
                ax_grob = construct_axis_grob(ap, target_which, ds) if axis else None
                gp_rec = recycle_gp(gp_d, ng)
                if target_which == "column":
                    pushViewport(Viewport(xscale=(0.5, ng + 0.5), yscale=ds))
                else:
                    pushViewport(Viewport(xscale=ds, yscale=(0.5, ng + 0.5)))
                try:
                    for gi, v in enumerate(vl):
                        grid_boxplot(
                            v,
                            pos=float(gi + 1) if target_which == "column" else float(ng - gi),
                            box_width=box_width,
                            gp=subset_gp(gp_rec, gi),
                            pch=pch,
                            size=sz_u,
                            outline=outline,
                            direction="vertical" if target_which == "column" else "horizontal",
                        )
                    if axis and ax_grob is not None:
                        grid_draw(ax_grob)
                    if border:
                        grid_rect(gp=gpar(fill="transparent", col="black"))
                finally:
                    popViewport()

        ax0 = construct_axis_grob(ap, target_which, ylim or (0.0, 1.0)) if axis else None
        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_summary",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=1,
            var_import={},
            subset_rule={},
            subsettable=False,
            show_name=False,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax0, ap)
        return anno

    return _build(eff_which)


# ---- 18. anno_image ----

def anno_image(
    image: Sequence[str | None],
    which: str | None = None,
    border: bool | str = True,
    gp: dict[str, Any] | None = None,
    space: Any = unit(1, "mm"),
    width: Unit | None = None,
    height: Unit | None = None,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    paths = [None if p is None or str(p).strip() == "" else str(p) for p in image]
    n = len(paths)
    gp_rec = recycle_gp(gp or gpar(fill=None, col=None), n)
    sp_u = as_unit(space, "mm")

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(1.0, "cm"))

        def _draw(index, k=1, n_slices=1, env=None):
            m = len(index)
            if m == 0:
                return
            p_list = env["image"]
            for pos_i, orig_i in enumerate(index):
                if target_which == "column":
                    pushViewport(Viewport(x=unit((pos_i + 0.5) / m, "npc"), width=unit(1.0 / m, "npc"), height=unit(1.0, "npc")))
                else:
                    pushViewport(Viewport(y=unit((m - pos_i - 0.5) / m, "npc"), height=unit(1.0 / m, "npc"), width=unit(1.0, "npc")))
                try:
                    grid_rect(gp=subset_gp(env["gp"], orig_i))
                    pth = p_list[orig_i]
                    if pth and os.path.exists(pth):
                        im = Image.open(pth).convert("RGBA")
                        arr = np.asarray(im)
                        ih, iw = arr.shape[:2]
                        from .grid import current_viewport
                        cvp = current_viewport()
                        sp_mm = convertWidth(sp_u, "mm", valueOnly=True)
                        avail_w = max(0.5, cvp.w_mm - 2.0 * sp_mm)
                        avail_h = max(0.5, cvp.h_mm - 2.0 * sp_mm)
                        scale = min(avail_w / max(1, iw), avail_h / max(1, ih))
                        grid_raster(arr, width=unit(iw * scale, "mm"), height=unit(ih * scale, "mm"))
                    if border:
                        grid_rect(gp=gpar(fill="transparent", col="black" if border is True else border))
                finally:
                    popViewport()

        return AnnotationFunction(
            fun=_draw,
            fun_name="anno_image",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            var_import={"image": paths, "gp": gp_rec},
            subset_rule={"image": subset_vector, "gp": subset_gp},
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )

    return _build(eff_which)


# ---- 19. anno_customize ----

def anno_customize(
    x: Sequence[Any],
    graphics: dict[str, Callable] | None = None,
    which: str | None = None,
    border: bool = True,
    width: Unit | None = None,
    height: Unit | None = None,
    verbose: bool = True,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    vals = [str(v) for v in x]
    n = len(vals)
    g_map = dict(graphics or {})

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(5.0, "mm"))

        def _draw(index, k=1, n_slices=1, env=None):
            vs = env["value"]
            m = len(index)
            if m == 0:
                return
            if target_which == "row":
                pushViewport(Viewport(yscale=(0.5, m + 0.5)))
                try:
                    for pos_i, orig_i in enumerate(index):
                        fn = g_map.get(vs[orig_i])
                        if fn is not None:
                            pushViewport(Viewport(y=unit(m - pos_i, "native"), height=unit(1.0, "native")))
                            try:
                                fn(unit(0.5, "npc"), unit(0.5, "npc"), unit(1.0, "npc"), unit(1.0, "npc"))
                            finally:
                                popViewport()
                    if border:
                        grid_rect(gp=gpar(fill="transparent", col="black"))
                finally:
                    popViewport()
            else:
                pushViewport(Viewport(xscale=(0.5, m + 0.5)))
                try:
                    for pos_i, orig_i in enumerate(index):
                        fn = g_map.get(vs[orig_i])
                        if fn is not None:
                            pushViewport(Viewport(x=unit(pos_i + 1, "native"), width=unit(1.0, "native")))
                            try:
                                fn(unit(0.5, "npc"), unit(0.5, "npc"), unit(1.0, "npc"), unit(1.0, "npc"))
                            finally:
                                popViewport()
                    if border:
                        grid_rect(gp=gpar(fill="transparent", col="black"))
                finally:
                    popViewport()

        return AnnotationFunction(
            fun=_draw,
            fun_name="anno_customize",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=n,
            var_import={"value": vals},
            subset_rule={"value": subset_vector},
            subsettable=True,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )

    return _build(eff_which)


# ---- 20. anno_textbox ----

def anno_textbox(
    align_to: Any,
    text: Any,
    background_gp: dict[str, Any] | None = None,
    which: str = "row",
    by: str = "anno_link",
    side: str = "right",
    **textbox_kwargs: Any,
) -> AnnotationFunction:
    which = which.lower()
    if which == "column":
        raise ValueError("`anno_textbox()` can only be used as row annotation.")

    bg_gp = gpar(fill="#DDDDDD", col="#AAAAAA", lty=1, lwd=1)
    if background_gp:
        bg_gp.update(check_gp(background_gp))

    if isinstance(align_to, (list, tuple, np.ndarray)) and isinstance(text, (list, tuple, pd.DataFrame)) and not isinstance(text, dict):
        if all(isinstance(v, (int, np.integer)) for v in align_to):
            aln_dict = {"v": [int(v) for v in align_to]}
            txt_dict = {"v": text}
        else:
            raise ValueError("Invalid `align_to` and `text` combination.")
    elif isinstance(align_to, dict) and isinstance(text, dict):
        aln_dict = {str(k): [int(v) for v in vs] for k, vs in align_to.items()}
        txt_dict = {str(k): text[k] for k in align_to.keys()}
    elif isinstance(text, dict):
        # align_to is categorical vector
        groups: dict[str, list[int]] = {}
        for idx, cat in enumerate(align_to):
            groups.setdefault(str(cat), []).append(idx)
        common = [k for k in groups.keys() if k in text]
        if not common:
            raise ValueError("Names of `text` must overlap with levels in `align_to`.")
        aln_dict = {k: groups[k] for k in common}
        txt_dict = {k: text[k] for k in common}
    else:
        raise ValueError("Format of `align_to` or `text` is invalid.")

    gbl: dict[str, Any] = {}
    for k, val in txt_dict.items():
        if isinstance(val, pd.DataFrame):
            words = val.iloc[:, 0].astype(str).tolist()
            gp_args: dict[str, Any] = {}
            for col_nm in ("col", "fontsize", "fontfamily", "fontface"):
                if col_nm in val.columns:
                    gp_args[col_nm] = val[col_nm].tolist()
            gbl[k] = textbox_grob(words, gp=gpar(**gp_args), background_gp=gpar(col=None, fill="transparent"), **textbox_kwargs)
        else:
            gbl[k] = textbox_grob(list(val), background_gp=gpar(col=None, fill="transparent"), **textbox_kwargs)

    gbl_h = unit([gb.height_mm() for gb in gbl.values()], "mm")
    gbl_w_max = max((gb.width_mm() for gb in gbl.values()), default=10.0)

    if by in ("anno_link", "anno_zoom"):
        def _panel(index, nm):
            pushViewport(Viewport())
            try:
                grid_rect(gp=gpar(fill=bg_gp.get("fill", "#DDDDDD"), col=bg_gp.get("fill", "#DDDDDD"), lty=bg_gp.get("lty", 1), lwd=bg_gp.get("lwd", 1)))
                if side == "right":
                    grid_lines([0, 1, 1, 0], [0, 0, 1, 1], gp=gpar(col=bg_gp.get("col", "#AAAAAA"), lty=bg_gp.get("lty", 1), lwd=bg_gp.get("lwd", 1)), default_units="npc")
                else:
                    grid_lines([1, 0, 0, 1], [0, 0, 1, 1], gp=gpar(col=bg_gp.get("col", "#AAAAAA"), lty=bg_gp.get("lty", 1), lwd=bg_gp.get("lwd", 1)), default_units="npc")
                gb = gbl[nm]
                gb.vp.x = unit(0.5 * gb.width_mm(), "mm")
                gb.vp.y = unit(0.5 * gb.height_mm(), "mm")
                gb.draw()
            finally:
                popViewport()

        return anno_link(
            align_to=aln_dict,
            which="row",
            panel_fun=_panel,
            size=gbl_h,
            gap=unit(2.0, "mm"),
            width=unit(gbl_w_max + 5.0, "mm"),
            link_gp=bg_gp,
            internal_line=False,
            side=side,
        )
    else:
        def _panel_blk(index, nm):
            pushViewport(Viewport())
            try:
                grid_rect(gp=bg_gp)
                gb = gbl[nm]
                gb.vp.x = unit(0.0, "npc")
                gb.vp.just = (0.0, 0.5)
                gb.draw()
            finally:
                popViewport()

        return anno_block(align_to=aln_dict, which="row", panel_fun=_panel_blk, width=unit(gbl_w_max + 5.0, "mm"))


# ---- 21. anno_oncoprint_barplot ----

def anno_oncoprint_barplot(
    type: Sequence[str] | None = None,
    which: str | None = None,
    bar_width: float = 0.6,
    beside: bool = False,
    ylim: Sequence[float] | None = None,
    show_fraction: bool = False,
    axis: bool = True,
    axis_param: dict[str, Any] | None = None,
    width: Unit | None = None,
    height: Unit | None = None,
    border: bool = False,
) -> AnnotationFunction:
    which_explicit = which is not None
    eff_which = (which or _ENV.current_annotation_which or "column").lower()

    def _build(target_which: str) -> AnnotationFunction:
        sz = anno_width_and_height(target_which, width, height, unit(2.0, "cm"))
        ap_default = default_axis_param("column") if target_which == "column" else validate_axis_param({"side": "top", "labels_rot": 0}, "row")
        ap = validate_axis_param(axis_param if axis_param is not None else ap_default, target_which)

        def _draw(index, k=1, n_slices=1, env=None):
            ht = _ENV.current_heatmap
            if ht is None or not hasattr(ht, "_oncoprint_env") or ht._oncoprint_env is None:
                return
            op = ht._oncoprint_env
            arr = op["arr"]          # shape (n_genes, n_samples, n_types)
            all_type = list(op["all_type"])
            col_map = op["col"]

            sel_types = list(col_map.keys()) if type is None else list(type)
            use_types = [t for t in all_type if t in sel_types and t in col_map and col_map[t] is not None]
            if not use_types:
                return
            type_indices = [all_type.index(t) for t in use_types]
            sub_arr = arr[:, :, type_indices]
            cols = [col_map[t] for t in use_types]

            if target_which == "column":
                v = np.sum(sub_arr, axis=0).astype(float)   # (n_samples, n_use_types)
                if show_fraction:
                    v = v / max(1, arr.shape[0])
            else:
                v = np.sum(sub_arr, axis=1).astype(float)   # (n_genes, n_use_types)
                if show_fraction:
                    v = v / max(1, arr.shape[1])

            bp = anno_barplot(
                v,
                gp=gpar(fill=cols, col=None),
                which=target_which,
                ylim=ylim,
                baseline=0.0,
                width=sz["width"] if target_which == "row" else None,
                height=sz["height"] if target_which == "column" else None,
                border=border,
                bar_width=bar_width,
                beside=beside,
                axis=axis,
                axis_param=ap,
            )
            bp.fun(index, k, n_slices, env=bp.var_env)

        data_scale = (0.0, 1.0 if show_fraction else 10.0)
        ax_grob = construct_axis_grob(ap, target_which, ylim or data_scale) if axis else None
        anno = AnnotationFunction(
            fun=_draw,
            fun_name="anno_oncoprint_barplot",
            which=target_which,
            width=sz["width"],
            height=sz["height"],
            n=None,
            var_import={"border": border, "type": type, "bar_width": bar_width, "beside": beside, "axis": axis, "axis_param": ap, "ylim": ylim, "show_fraction": show_fraction},
            subset_rule={},
            subsettable=True,
            show_name=False,
            _which_explicit=which_explicit,
            _reorient_fn=_build,
        )
        anno.extended = update_anno_extend(anno, ax_grob, ap)
        return anno

    return _build(eff_which)


# ---- Row Aliases ----

def row_anno_points(*args: Any, **kwargs: Any) -> AnnotationFunction:
    kwargs["which"] = "row"
    return anno_points(*args, **kwargs)


def row_anno_barplot(*args: Any, **kwargs: Any) -> AnnotationFunction:
    kwargs["which"] = "row"
    return anno_barplot(*args, **kwargs)


def row_anno_boxplot(*args: Any, **kwargs: Any) -> AnnotationFunction:
    kwargs["which"] = "row"
    return anno_boxplot(*args, **kwargs)


def row_anno_histogram(*args: Any, **kwargs: Any) -> AnnotationFunction:
    kwargs["which"] = "row"
    return anno_histogram(*args, **kwargs)


def row_anno_density(*args: Any, **kwargs: Any) -> AnnotationFunction:
    kwargs["which"] = "row"
    return anno_density(*args, **kwargs)


def row_anno_text(*args: Any, **kwargs: Any) -> AnnotationFunction:
    kwargs["which"] = "row"
    return anno_text(*args, **kwargs)
