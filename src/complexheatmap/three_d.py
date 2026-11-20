"""3D isometric bar primitive (`bar3D`) and 3D heatmap (`Heatmap3D`)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ._options import _ENV
from .colors import add_luminance, to_hex_rgba
from .grid import GTree, current_viewport, grid_rect, polygonGrob
from .heatmap import Heatmap
from .units import Unit, as_unit, convertWidth, gpar, unit, unit_c


def bar3D(
    x: Any = unit(0.5, "npc"),
    y: Any = unit(0.5, "npc"),
    w: Any = unit(0.2, "npc"),
    h: Any = unit(0.2, "npc"),
    l: Any = unit(1, "cm"),
    theta: float = 60.0,
    default_units: str = "npc",
    fill: Any = "white",
    col: Any = "black",
    lwd: float = 1.0,
    draw: bool = True,
    **kwargs: Any,
) -> GTree:
    """Draw 3D isometric rectangular bar(s) (`bar3D.Rd`). Supports vectorized coordinates."""
    default_units = kwargs.pop("default.units", default_units)
    if float(theta) < 0 or float(theta) > 90:
        raise ValueError("`theta` must be between 0 and 90 degrees.")

    ux = as_unit(x, default_units)
    uy = as_unit(y, default_units)
    uw = as_unit(w, default_units)
    uh = as_unit(h, default_units)
    ul = as_unit(l, default_units)

    n = max(len(ux), len(uy), len(uw), len(uh), len(ul))
    if isinstance(fill, (str, int, np.integer, float, np.floating)) or fill is None:
        fill_seq = [fill] * n
    else:
        fl = list(fill)
        fill_seq = [fl[i % len(fl)] for i in range(n)]

    th_rad = math.radians(float(theta))
    cos_t = math.cos(th_rad)
    sin_t = math.sin(th_rad)

    lum_cache: dict[Any, list[str]] = {}
    pfills: list[str] = []
    for i in range(n):
        f_key = fill_seq[i]
        fill2 = lum_cache.get(f_key)
        if fill2 is None:
            fill2 = add_luminance(f_key)
            lum_cache[f_key] = fill2
        pfills.append(fill2[min(1, len(fill2) - 1)])
        pfills.append(fill2[min(2, len(fill2) - 1)])
        pfills.append(fill2[0])

    if n > 0 and all(u._op is None and u._unit == "mm" for u in (ux, uy, uw, uh, ul)):
        x0 = np.broadcast_to(ux._values, (n,))
        y0 = np.broadcast_to(uy._values, (n,))
        w0 = np.broadcast_to(uw._values, (n,))
        h0 = np.broadcast_to(uh._values, (n,))
        l0 = np.broadcast_to(ul._values, (n,))
        dx = l0 * cos_t
        dy = l0 * sin_t
        x1 = x0 - 0.5 * w0
        x2 = x0 + 0.5 * w0
        y1 = y0 - 0.5 * h0
        y2 = y0 + 0.5 * h0
        a1 = x1 + dx
        a2 = x2 + dx
        b1 = y1 + dy
        b2 = y2 + dy
        px_u = unit(np.column_stack([x1, a1, a2, x2, x1, x1, a1, a1, a1, a1, a2, a2]).ravel(), "mm")
        py_u = unit(np.column_stack([y1, b1, b1, y1, y1, y2, b2, b1, b1, b2, b2, b1]).ravel(), "mm")
        pids: Any = np.repeat(np.arange(1, 3 * n + 1), 4)
    else:
        if len(ux) == 1 and n > 1:
            ux = unit_c(*([ux] * n))
        if len(uy) == 1 and n > 1:
            uy = unit_c(*([uy] * n))
        if len(uw) == 1 and n > 1:
            uw = unit_c(*([uw] * n))
        if len(uh) == 1 and n > 1:
            uh = unit_c(*([uh] * n))
        if len(ul) == 1 and n > 1:
            ul = unit_c(*([ul] * n))

        dx = ul * cos_t
        dy = ul * sin_t

        x1 = ux - uw * 0.5
        x2 = ux + uw * 0.5
        y1 = uy - uh * 0.5
        y2 = uy + uh * 0.5

        xu = ux + dx
        yu = uy + dy
        a1 = xu - uw * 0.5
        a2 = xu + uw * 0.5
        b1 = yu - uh * 0.5
        b2 = yu + uh * 0.5

        px_parts: list[Unit] = []
        py_parts: list[Unit] = []
        pids_list: list[int] = []
        pid = 1
        for i in range(n):
            px_parts.extend([x1[i], a1[i], a2[i], x2[i]])
            py_parts.extend([y1[i], b1[i], b1[i], y1[i]])
            pids_list.extend([pid] * 4)
            pid += 1
            px_parts.extend([x1[i], x1[i], a1[i], a1[i]])
            py_parts.extend([y1[i], y2[i], b2[i], b1[i]])
            pids_list.extend([pid] * 4)
            pid += 1
            px_parts.extend([a1[i], a1[i], a2[i], a2[i]])
            py_parts.extend([b1[i], b2[i], b2[i], b1[i]])
            pids_list.extend([pid] * 4)
            pid += 1
        px_u = unit_c(*px_parts)
        py_u = unit_c(*py_parts)
        pids = pids_list

    col_val = to_hex_rgba(col) if col is not None else "transparent"
    poly = polygonGrob(
        x=px_u,
        y=py_u,
        id=pids,
        gp=gpar(fill=pfills, col=col_val, lwd=float(lwd)),
    )
    gt = GTree(children=[poly], cl="bar3D")
    if draw:
        gt.draw()
    return gt


def Heatmap3D(
    matrix: Any,
    col: Any = None,
    name: str | None = None,
    bar_rel_width: float = 0.6,
    bar_rel_height: float = 0.6,
    bar_max_length: Unit = unit(1, "cm"),
    bar_angle: float = 60.0,
    row_names_side: str = "left",
    show_row_dend: bool = False,
    show_column_dend: bool = False,
    rect_gp: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Heatmap:
    """Create a 3D bar heatmap (`Heatmap3D.Rd`). Matrix must be numeric and non-negative."""
    arr = np.asarray(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) > 0 and float(np.min(finite)) < 0:
        raise ValueError("`Heatmap3D` requires a non-negative numeric matrix.")
    vmax = float(np.max(finite)) if len(finite) > 0 and float(np.max(finite)) > 0 else 1.0

    max_len_mm = convertWidth(as_unit(bar_max_length, "mm"), "mm", valueOnly=True)
    ht_name = name or f"heatmap3d_{_ENV.next_heatmap_index()}"

    from .utils import restore_matrix

    user_cell_col = (rect_gp or {}).get("col", "#444444")
    user_lwd = float((rect_gp or {}).get("lwd", 0.35 if arr.size > 500 else 0.8))

    def _layer_fun(j: Any, i: Any, x: Any, y: Any, w: Any, h: Any, fill: Any) -> None:
        vp = current_viewport()
        grid_rect(x=x, y=y, width=w, height=h, gp=gpar(col="white", fill="#EEEEEE"))
        ind_mat = restore_matrix(j, i, x, y)
        nr_s, nc_s = ind_mat.shape
        x_mm = x.to_mm("x", vp, pos_mode=True) if isinstance(x, Unit) else np.asarray(x, dtype=float)
        y_mm = y.to_mm("y", vp, pos_mode=True) if isinstance(y, Unit) else np.asarray(y, dtype=float)
        w_mm = w.to_mm("x", vp, pos_mode=False) if isinstance(w, Unit) else np.asarray(w, dtype=float)
        h_mm = h.to_mm("y", vp, pos_mode=False) if isinstance(h, Unit) else np.asarray(h, dtype=float)

        cos_t = math.cos(math.radians(float(bar_angle)))
        col_order = range(nc_s - 1, -1, -1) if cos_t >= 0 else range(nc_s)

        xs_l: list[float] = []
        ys_l: list[float] = []
        ws_l: list[float] = []
        hs_l: list[float] = []
        ls_l: list[float] = []
        fills_l: list[Any] = []
        for ri in range(nr_s):
            for ci in col_order:
                pos = int(ind_mat[ri, ci])
                if pos < 0:
                    continue
                orig_i = int(i[pos])
                orig_j = int(j[pos])
                val = float(arr[orig_i, orig_j])
                if np.isnan(val):
                    continue
                xs_l.append(float(x_mm[pos]))
                ys_l.append(float(y_mm[pos]))
                ws_l.append(float(w_mm[pos]) * float(bar_rel_width))
                hs_l.append(float(h_mm[pos]) * float(bar_rel_height))
                ls_l.append((val / vmax) * max_len_mm)
                fills_l.append(fill[pos])

        if xs_l:
            bar3D(
                x=unit(xs_l, "mm"),
                y=unit(ys_l, "mm"),
                w=unit(ws_l, "mm"),
                h=unit(hs_l, "mm"),
                l=unit(ls_l, "mm"),
                theta=bar_angle,
                fill=fills_l,
                col=user_cell_col,
                lwd=user_lwd,
                draw=True,
            )

    return Heatmap(
        matrix,
        col=col,
        name=ht_name,
        rect_gp=gpar(type="none"),
        layer_fun=_layer_fun,
        row_names_side=row_names_side,
        show_row_dend=show_row_dend,
        show_column_dend=show_column_dend,
        **kwargs,
    )
