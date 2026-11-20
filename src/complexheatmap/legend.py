"""Legend, Legends, and packLegend implementation."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Sequence

import numpy as np

from .colors import grid_pretty, pretty_breaks, to_hex_rgba
from .grid import (
    GTree,
    Grob,
    Viewport,
    grid_newpage,
    grid_rect,
    grid_segments,
    grid_text,
    pointsGrob,
    popViewport,
    pushViewport,
    rectGrob,
    segmentsGrob,
    textGrob,
    grid_grabExpr,
)
from .text import smartAlign
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


def _is_diff_equal(x: Sequence[float]) -> bool:
    arr = np.asarray(x, dtype=float)
    if len(arr) <= 2:
        return True
    diffs = np.diff(arr)
    mean_d = float(np.mean(diffs))
    if abs(mean_d) < 1e-12:
        return True
    return bool(np.all(np.abs(np.diff(diffs)) / abs(mean_d) < 1e-4))


def _valid_just(just: Any) -> tuple[float, float]:
    from .grid import parse_just
    return parse_just(just)


def _edit_vp_in_legend_grob(gtree: GTree, x: Any = None, y: Any = None, valid_just: Any = None) -> GTree:
    gt = copy.copy(gtree)
    if gt.vp is None:
        gt.vp = Viewport(x=x, y=y, just=valid_just or "centre")
    else:
        vp = copy.copy(gt.vp)
        if x is not None:
            vp.x = as_unit(x, "npc")
        if y is not None:
            vp.y = as_unit(y, "npc")
        if valid_just is not None:
            vp.just = valid_just
        gt.vp = vp
    return gt


class Legends:
    """Container for one or more packed Legend grobs."""

    def __init__(
        self,
        grob: GTree,
        type: str = "single_legend",
        name: str | None = None,
        n: int = 1,
        multiple: int = 1,
        direction: str = "vertical",
    ) -> None:
        self.grob = grob
        self.type = type
        self.name = name
        self.n = int(n)
        self.multiple = int(multiple)
        self.direction = direction

    @property
    def width(self) -> Unit:
        return unit(self.grob.width_mm(), "mm")

    @property
    def height(self) -> Unit:
        return unit(self.grob.height_mm(), "mm")

    def draw(
        self,
        x: Any = None,
        y: Any = None,
        just: Any = "centre",
        test: bool | str = False,
    ) -> None:
        ux = unit(0.5, "npc") if x is None else as_unit(x, "npc")
        uy = unit(0.5, "npc") if y is None else as_unit(y, "npc")
        lgd_grob = _edit_vp_in_legend_grob(self.grob, x=ux, y=uy, valid_just=_valid_just(just))
        if isinstance(test, str):
            is_test = True
            test_label = test
        else:
            is_test = bool(test)
            test_label = ""
        if is_test:
            grid_newpage()
        lgd_grob.draw()
        if is_test and test_label:
            grid_text(test_label, x=0.5, y=unit(1, "npc") - unit(1, "cm"))

    def _repr_png_(self) -> bytes:
        import io
        import matplotlib.pyplot as plt
        w_mm = max(20.0, self.grob.width_mm() + 10.0)
        h_mm = max(20.0, self.grob.height_mm() + 10.0)
        dev = grid_newpage(width_in=w_mm / 25.4, height_in=h_mm / 25.4)
        self.draw()
        buf = io.BytesIO()
        dev.fig.savefig(buf, format="png", dpi=120)
        plt.close(dev.fig)
        return buf.getvalue()

    def __repr__(self) -> str:
        if self.type == "single_legend":
            return "A single legend"
        if self.type == "single_legend_no_title":
            return "A single legend without title"
        return f"A pack of {self.n} legends"


def _points_grobs_for_legend(x: Unit, y: Unit, pch: Any, gp: Any, size: Unit, width: Unit, height: Unit) -> list[Grob]:
    n = len(x)
    pch_list = [pch] * n if isinstance(pch, (int, np.integer, str)) or pch is None else list(pch)
    if any(isinstance(p, (int, np.integer)) and int(p) in (26, 27, 28, 29, 30, 31) for p in pch_list):
        out: list[Grob] = []
        for i in range(n):
            p = int(pch_list[i]) if isinstance(pch_list[i], (int, np.integer)) else pch_list[i]
            gpi = subset_gp(gp, i)
            if p == 26:
                out.append(segmentsGrob(x[i] - width * 0.4, y[i] - height[i] * 0.4, x[i] + width * 0.4, y[i] + height[i] * 0.4, gp=gpi))
            elif p == 27:
                out.append(segmentsGrob(x[i] + width * 0.4, y[i] - height[i] * 0.4, x[i] - width * 0.4, y[i] + height[i] * 0.4, gp=gpi))
            elif p == 28:
                out.append(
                    segmentsGrob(
                        unit_c(x[i] - width * 0.4, x[i] + width * 0.4),
                        unit_c(y[i] - height[i] * 0.4, y[i] - height[i] * 0.4),
                        unit_c(x[i] + width * 0.4, x[i] - width * 0.4),
                        unit_c(y[i] + height[i] * 0.4, y[i] + height[i] * 0.4),
                        gp=gpi,
                    )
                )
            elif p in (29, 30, 31):
                raise ValueError("pch in 29..31 is not supported in Legend().")
            else:
                out.append(pointsGrob(x[i], y[i], pch=p, size=size[i], gp=gpi, default_units="mm"))
        return out
    return [pointsGrob(x, y, pch=pch_list, size=size, gp=gp, default_units="mm")]


def _discrete_legend_body(
    at: Sequence[Any],
    labels: Sequence[Any],
    nrow: int | None = None,
    ncol: int = 1,
    by_row: bool = False,
    grid_height: Unit = unit(4, "mm"),
    grid_width: Unit = unit(4, "mm"),
    column_gap: Unit = unit(2, "mm"),
    row_gap: Unit = unit(0, "mm"),
    labels_gp: Any = None,
    border: Any = "white",
    background: str = "#EEEEEE",
    type: str | Sequence[str] = "grid",
    graphics: Sequence[Callable] | dict[Any, Callable] | None = None,
    legend_gp: Any = None,
    pch: Any = 16,
    size: Any = unit(2, "mm"),
) -> GTree:
    lbl_list = [str(v) for v in labels]
    n_labels = len(lbl_list)
    if n_labels == 0:
        return GTree(children=[], cl="legend_body", width=unit(0.0, "mm"), height=unit(0.0, "mm"))

    if nrow is None:
        nrow = int(math.ceil(n_labels / max(1, ncol)))
    else:
        ncol = int(math.ceil(n_labels / max(1, nrow)))
    if len(at) == 1:
        nrow = 1
        ncol = 1
    ncol = min(ncol, n_labels)

    # Build grid matrix of indices (None where empty)
    idx_grid: list[list[int | None]] = [[None] * ncol for _ in range(nrow)]
    if by_row:
        for idx in range(n_labels):
            r, c = divmod(idx, ncol)
            idx_grid[r][c] = idx
    else:
        for idx in range(n_labels):
            c, r = divmod(idx, nrow)
            idx_grid[r][c] = idx

    # Trim completely empty columns or rows
    non_empty_cols = [c for c in range(ncol) if any(idx_grid[r][c] is not None for r in range(nrow))]
    ncol = len(non_empty_cols)
    idx_grid = [[idx_grid[r][c] for c in non_empty_cols] for r in range(nrow)]
    non_empty_rows = [r for r in range(nrow) if any(idx_grid[r][c] is not None for c in range(ncol))]
    nrow = len(non_empty_rows)
    idx_grid = [idx_grid[r] for r in non_empty_rows]

    gh_mm_arr = np.broadcast_to(as_unit(grid_height, "mm").to_mm("y"), (nrow,)).copy()
    gw_mm_arr = np.broadcast_to(as_unit(grid_width, "mm").to_mm("x"), (ncol,)).copy()
    col_gap_mm = convertWidth(column_gap, "mm", valueOnly=True)
    row_gap_mm = convertHeight(row_gap, "mm", valueOnly=True)
    lbl_pad_left_mm = 1.0

    lgp_labels = check_gp(labels_gp or gpar(fontsize=10))
    col_max_w = np.zeros(ncol, dtype=float)
    for c in range(ncol):
        for r in range(nrow):
            idx = idx_grid[r][c]
            if idx is not None:
                col_max_w[c] = max(col_max_w[c], textGrob(lbl_list[idx], gp=lgp_labels).width_mm())

    row_h_no_gap = np.zeros(nrow, dtype=float)
    pt4_mm = convertHeight(unit(4, "points"), "mm", valueOnly=True)
    for r in range(nrow):
        th = 0.0
        for c in range(ncol):
            idx = idx_grid[r][c]
            if idx is not None:
                th = max(th, textGrob(lbl_list[idx], gp=lgp_labels).height_mm())
        row_h_no_gap[r] = max(th + pt4_mm, gh_mm_arr[r])

    row_h = row_h_no_gap.copy()
    if nrow > 1:
        row_h[:-1] += row_gap_mm

    tot_w = float(np.sum(gw_mm_arr) + lbl_pad_left_mm * ncol + np.sum(col_max_w) + col_gap_mm * max(0, ncol - 1))
    tot_h = float(np.sum(row_h))

    lgp_rec = recycle_gp(legend_gp or gpar(), n_labels)
    types = [type] if isinstance(type, str) else list(type)

    if graphics is not None:
        if isinstance(graphics, dict):
            g_list = [graphics[k] for k in at]
        else:
            g_list = list(graphics)
        if len(g_list) != n_labels:
            raise ValueError("Length of `graphics` must match number of labels.")
    else:
        g_list = None

    children: list[Grob] = []
    prev_x = 0.0
    for c in range(ncol):
        col_indices = [idx_grid[r][c] for r in range(nrow) if idx_grid[r][c] is not None]
        ni = len(col_indices)
        # y centers from top
        y_top = np.array([0.0] + list(np.cumsum(row_h[: ni - 1])), dtype=float)
        y_bot_from_top = tot_h - y_top
        y_center = y_bot_from_top - 0.5 * row_h_no_gap[:ni]

        lx = prev_x + gw_mm_arr[c] + lbl_pad_left_mm
        children.append(
            textGrob(
                [lbl_list[i] for i in col_indices],
                x=np.full(ni, lx),
                y=y_center,
                just="left",
                gp=lgp_labels,
                default_units="mm",
            )
        )

        sgd = subset_gp(lgp_rec, col_indices)
        sgd2 = gpar()
        if "grid" in types:
            sgd2["fill"] = sgd.get("fill", [to_hex_rgba(background)] * ni)
        else:
            sgd2["fill"] = background
        eff_border = "asis" if (legend_gp and legend_gp.get("col") == "asis") else border
        if eff_border == "asis":
            sgd2["col"] = sgd2["fill"]
        else:
            sgd2["col"] = eff_border

        gx = np.full(ni, prev_x)
        if g_list is None:
            children.append(
                rectGrob(
                    x=gx,
                    y=y_center,
                    width=np.full(ni, gw_mm_arr[c]),
                    height=row_h_no_gap[:ni],
                    just="left",
                    gp=sgd2,
                    default_units="mm",
                )
            )
            if any(t in ("points", "p") for t in types):
                pch_col = [pch] * ni if isinstance(pch, (int, np.integer, str)) or pch is None else [list(pch)[i] for i in col_indices]
                sz_u = as_unit(size, "mm")
                sz_col = unit_c(*([sz_u] * ni)) if len(sz_u) == 1 else sz_u[col_indices]
                children.extend(
                    _points_grobs_for_legend(
                        x=unit(gx + gw_mm_arr[c] * 0.5, "mm"),
                        y=unit(y_center, "mm"),
                        pch=pch_col,
                        gp=sgd,
                        size=sz_col,
                        width=unit(gw_mm_arr[c], "mm"),
                        height=unit(row_h_no_gap[:ni], "mm"),
                    )
                )
            if any(t in ("lines", "l") for t in types):
                children.append(
                    segmentsGrob(
                        x0=gx, y0=y_center,
                        x1=gx + gw_mm_arr[c], y1=y_center,
                        gp=sgd,
                        default_units="mm",
                    )
                )
            if any(t in ("boxplot", "box") for t in types):
                children.append(
                    segmentsGrob(
                        x0=gx + gw_mm_arr[c] * 0.5, y0=y_center - row_h_no_gap[:ni] * 0.45,
                        x1=gx + gw_mm_arr[c] * 0.5, y1=y_center + row_h_no_gap[:ni] * 0.45,
                        gp=sgd, default_units="mm",
                    )
                )
                children.append(
                    rectGrob(
                        x=gx + gw_mm_arr[c] * 0.5, y=y_center,
                        width=np.full(ni, gw_mm_arr[c] * 0.9), height=row_h_no_gap[:ni] * 0.5,
                        gp=sgd, default_units="mm",
                    )
                )
                children.append(
                    segmentsGrob(
                        x0=gx + gw_mm_arr[c] * 0.05, y0=y_center,
                        x1=gx + gw_mm_arr[c] * 0.95, y1=y_center,
                        gp=sgd, default_units="mm",
                    )
                )
        else:
            for r_k, idx in enumerate(col_indices):
                fn = g_list[idx]
                cell_w_mm = gw_mm_arr[c]
                cell_h_mm = row_h_no_gap[r_k]
                # Draw directly into cell viewport inside legend
                def _make_cell_grob(fn_local, cx, cy, cw, ch):
                    vp_c = Viewport(x=unit(cx, "mm"), y=unit(cy, "mm"), width=unit(cw, "mm"), height=unit(ch, "mm"))
                    cap = grid_grabExpr(lambda: fn_local(unit(0.5, "npc"), unit(0.5, "npc"), unit(1.0, "npc"), unit(1.0, "npc")), width_in=cw / 25.4, height_in=ch / 25.4)
                    return GTree(children=[cap], vp=vp_c, width=unit(cw, "mm"), height=unit(ch, "mm"))
                children.append(_make_cell_grob(fn, prev_x + cell_w_mm * 0.5, y_center[r_k], cell_w_mm, cell_h_mm))

        prev_x += gw_mm_arr[c] + col_max_w[c] + lbl_pad_left_mm + col_gap_mm

    return GTree(
        children=children,
        cl="legend_body",
        vp=Viewport(width=unit(tot_w, "mm"), height=unit(tot_h, "mm")),
        width=unit(tot_w, "mm"),
        height=unit(tot_h, "mm"),
    )


def _apply_break_dist(at: np.ndarray, col_fun: Callable, break_dist: Sequence[float] | float) -> tuple[np.ndarray, Callable]:
    n = len(at)
    bd = [float(break_dist)] * (n - 1) if np.ndim(break_dist) == 0 else [float(v) for v in break_dist]
    if len(bd) != n - 1:
        raise ValueError(f"Length of `break_dist` must be {n - 1}.")
    y = np.concatenate([[0.0], np.cumsum(bd) / np.sum(bd)])
    sorted_at = np.sort(at)

    def new_col_fun(vals: Any) -> Any:
        mapped = np.interp(np.asarray(vals, dtype=float), y, sorted_at)
        return col_fun(mapped)

    new_col_fun.breaks = y  # type: ignore[attr-defined]
    new_at = y if at[0] <= at[-1] else y[::-1].copy()
    return new_at, new_col_fun


def _vertical_continuous_legend_body(
    at: Sequence[float],
    labels: Sequence[Any],
    col_fun: Callable,
    break_dist: Any = None,
    grid_width: Unit = unit(4, "mm"),
    legend_height: Unit | None = None,
    tick_length: Unit = unit(0.8, "mm"),
    labels_gp: Any = None,
    border: Any = None,
    legend_gp: Any = None,
) -> GTree:
    at_arr = np.asarray(at, dtype=float)
    lbl_arr = [str(v) for v in labels]
    n = len(at_arr)
    if np.array_equal(np.argsort(at_arr), np.arange(n - 1, -1, -1)):
        if break_dist is not None and np.ndim(break_dist) > 0:
            break_dist = list(reversed(break_dist))
    else:
        od = np.argsort(at_arr)
        if break_dist is not None and not np.array_equal(od, np.arange(n)):
            raise ValueError("`at` must be sorted when `break_dist` is set.")
        at_arr = at_arr[od]
        lbl_arr = [lbl_arr[i] for i in od]

    if break_dist is not None:
        at_arr, col_fun = _apply_break_dist(at_arr, col_fun, break_dist)

    lgp_labels = check_gp(labels_gp or gpar(fontsize=10))
    lgp_rec = recycle_gp(lgp_labels, n)
    lbl_max_w = max((textGrob(lbl_arr[i], gp=subset_gp(lgp_rec, i)).width_mm() for i in range(n)), default=0.0)
    min_h = sum(textGrob(lbl_arr[i], gp=subset_gp(lgp_rec, i)).height_mm() for i in range(n)) + convertHeight(unit(6 * max(0, n - 1), "points"), "mm", valueOnly=True)

    if legend_height is not None:
        lh_mm = convertHeight(legend_height, "mm", valueOnly=True)
        if lh_mm < min_h:
            lh_mm = min_h
    else:
        lh_mm = max(min_h, 21.1)

    gw_mm = convertWidth(grid_width, "mm", valueOnly=True)
    tick_mm = convertWidth(tick_length, "mm", valueOnly=True)
    at_diff_equal = _is_diff_equal(at_arr)
    lbl_pad_left = 1.0 if at_diff_equal else 4.0

    offset_mm = convertHeight(unit(2, "points"), "mm", valueOnly=True)
    ymin, ymax = offset_mm, lh_mm - offset_mm
    span = at_arr[-1] - at_arr[0]
    y_mm = ((at_arr - at_arr[0]) / span) * (ymax - ymin) + ymin if abs(span) > 1e-12 else np.linspace(ymin, ymax, n)
    labels_y = y_mm.copy()

    if not at_diff_equal:
        pt4 = convertHeight(unit(4, "points"), "mm", valueOnly=True)
        lh_items = np.array([textGrob(lbl_arr[i], gp=subset_gp(lgp_rec, i)).height_mm() + pt4 for i in range(n)])
        y_bot = labels_y - 0.5 * lh_items
        y_top = labels_y + 0.5 * lh_items
        yrng = (offset_mm - 0.5 * lh_items[0], lh_mm - offset_mm + 0.5 * lh_items[-1])
        aligned = smartAlign(y_bot, y_top, yrng)
        labels_y = 0.5 * (aligned[:, 0] + aligned[:, 1])

    adjust_text_pos = not bool(np.all(np.abs(labels_y - y_mm) < 1e-4))
    if not adjust_text_pos:
        lbl_pad_left = 1.0

    tot_w = gw_mm + lbl_pad_left + lbl_max_w
    children: list[Grob] = [
        textGrob(lbl_arr, x=np.full(n, gw_mm + lbl_pad_left), y=labels_y, just="left", gp=lgp_labels, default_units="mm")
    ]

    # Sample smooth gradient segments
    at_grad = at_arr if at_diff_equal else np.linspace(at_arr[0], at_arr[-1], n)
    pts_list = []
    for i in range(len(at_grad) - 1):
        m = max(10, int(round(abs(at_grad[i + 1] - at_grad[i]) / max(1e-12, abs(at_arr[-1] - at_arr[0])) * 100)))
        pts_list.append(np.linspace(at_grad[i], at_grad[i + 1], m, endpoint=False))
    pts_list.append(np.array([at_grad[-1]]))
    at2 = np.concatenate(pts_list)
    colors = [to_hex_rgba(c) for c in np.atleast_1d(col_fun(at2))]
    n_c = len(colors)
    step_h = (lh_mm - 2.0 * offset_mm) / n_c
    y2 = np.linspace(offset_mm, lh_mm - offset_mm, n_c, endpoint=False) + 0.5 * step_h
    if at_arr[0] > at_arr[-1]:
        y2 = y2[::-1]
        y2_all = np.concatenate([[lh_mm - 0.5 * offset_mm], y2, [0.5 * offset_mm]])
    else:
        y2_all = np.concatenate([[0.5 * offset_mm], y2, [lh_mm - 0.5 * offset_mm]])
    hh_all = np.concatenate([[offset_mm], np.full(n_c, step_h), [offset_mm]])
    cols_all = [colors[0]] + colors + [colors[-1]]

    children.append(
        rectGrob(
            x=np.zeros(len(cols_all)),
            y=y2_all,
            width=np.full(len(cols_all), gw_mm),
            height=hh_all,
            just="left",
            gp=gpar(col=cols_all, fill=cols_all),
            default_units="mm",
        )
    )

    lgp_tick = check_gp(legend_gp)
    if border is None and "col" in lgp_tick:
        border = lgp_tick["col"]
    if "col" not in lgp_tick:
        lgp_tick["col"] = "white" if border is None else border

    children.append(segmentsGrob(np.zeros(n), y_mm, np.full(n, tick_mm), y_mm, gp=lgp_tick, default_units="mm"))
    children.append(segmentsGrob(np.full(n, gw_mm), y_mm, np.full(n, gw_mm - tick_mm), y_mm, gp=lgp_tick, default_units="mm"))

    if adjust_text_pos:
        x_b0 = np.full(n, gw_mm)
        x_b1 = np.full(n, gw_mm + lbl_pad_left / 3.0)
        x_b2 = np.full(n, gw_mm + lbl_pad_left * 2.0 / 3.0)
        x_b3 = np.full(n, gw_mm + lbl_pad_left - 0.5)
        children.append(segmentsGrob(x_b0, y_mm, x_b1, y_mm, default_units="mm"))
        children.append(segmentsGrob(x_b3, labels_y, x_b2, labels_y, default_units="mm"))
        children.append(segmentsGrob(x_b1, y_mm, x_b2, labels_y, default_units="mm"))

    if border is not None:
        bg_gp = check_gp(lgp_tick)
        bg_gp["fill"] = "transparent"
        children.append(rectGrob(x=0.0, y=0.5 * lh_mm, width=gw_mm, height=lh_mm, just="left", gp=bg_gp, default_units="mm"))

    return GTree(
        children=children,
        cl="legend_body",
        vp=Viewport(width=unit(tot_w, "mm"), height=unit(lh_mm, "mm")),
        width=unit(tot_w, "mm"),
        height=unit(lh_mm, "mm"),
    )


def _horizontal_continuous_legend_body(
    at: Sequence[float],
    labels: Sequence[Any],
    col_fun: Callable,
    break_dist: Any = None,
    grid_height: Unit = unit(4, "mm"),
    legend_width: Unit | None = None,
    tick_length: Unit = unit(0.8, "mm"),
    labels_gp: Any = None,
    labels_rot: float = 0.0,
    border: Any = None,
    legend_gp: Any = None,
    legend_extension: Unit = unit(0, "mm"),
) -> GTree:
    at_arr = np.asarray(at, dtype=float)
    lbl_arr = [str(v) for v in labels]
    n = len(at_arr)
    if np.array_equal(np.argsort(at_arr), np.arange(n - 1, -1, -1)):
        if break_dist is not None and np.ndim(break_dist) > 0:
            break_dist = list(reversed(break_dist))
    else:
        od = np.argsort(at_arr)
        if break_dist is not None and not np.array_equal(od, np.arange(n)):
            raise ValueError("`at` must be sorted when `break_dist` is set.")
        at_arr = at_arr[od]
        lbl_arr = [lbl_arr[i] for i in od]

    if break_dist is not None:
        at_arr, col_fun = _apply_break_dist(at_arr, col_fun, break_dist)

    rot = float(labels_rot) % 360.0
    lgp_labels = check_gp(labels_gp or gpar(fontsize=10))
    lbl_widths = np.array([textGrob(lbl_arr[i], gp=lgp_labels, rot=rot).width_mm() for i in range(n)])
    lbl_max_h = max((textGrob(lbl_arr[i], gp=lgp_labels, rot=rot).height_mm() for i in range(n)), default=0.0)

    min_lw = float(np.sum(lbl_widths)) + 2.1 * n
    lw_mm = min_lw if legend_width is None else convertWidth(legend_width, "mm", valueOnly=True)
    gh_mm = convertHeight(grid_height, "mm", valueOnly=True)
    tick_mm = convertHeight(tick_length, "mm", valueOnly=True)
    at_diff_equal = _is_diff_equal(at_arr)
    lbl_pad_top = 1.0

    offset_mm = convertWidth(unit(2, "points"), "mm", valueOnly=True)
    xmin, xmax = offset_mm, lw_mm - offset_mm
    span = at_arr[-1] - at_arr[0]
    x_mm = ((at_arr - at_arr[0]) / span) * (xmax - xmin) + xmin if abs(span) > 1e-12 else np.linspace(xmin, xmax, n)

    if rot == 0:
        lbl_just = "top"
    elif 0 < rot < 180:
        lbl_just = "right"
    else:
        lbl_just = "left"

    w_pad = lbl_widths + 1.0
    ext_mm = max(convertWidth(legend_extension, "mm", valueOnly=True), 0.5 * w_pad[0] - offset_mm)
    xrng = (-ext_mm, lw_mm - offset_mm + 0.5 * w_pad[-1])
    aligned = smartAlign(x_mm - 0.5 * w_pad, x_mm + 0.5 * w_pad, xrng)
    labels_x = 0.5 * (aligned[:, 0] + aligned[:, 1])

    adjust_text_pos = not bool(np.all(np.abs(labels_x - x_mm) < 1e-4))
    if adjust_text_pos:
        lbl_pad_top = 4.0

    tot_h = gh_mm + lbl_pad_top + lbl_max_h
    labels_y = tot_h - gh_mm - lbl_pad_top
    children: list[Grob] = [
        textGrob(lbl_arr, x=labels_x, y=np.full(n, labels_y), just=lbl_just, gp=lgp_labels, rot=rot, default_units="mm")
    ]

    at_grad = at_arr if at_diff_equal else np.linspace(at_arr[0], at_arr[-1], n)
    pts_list = []
    for i in range(len(at_grad) - 1):
        m = max(10, int(round(abs(at_grad[i + 1] - at_grad[i]) / max(1e-12, abs(at_arr[-1] - at_arr[0])) * 100)))
        pts_list.append(np.linspace(at_grad[i], at_grad[i + 1], m, endpoint=False))
    pts_list.append(np.array([at_grad[-1]]))
    at2 = np.concatenate(pts_list)
    colors = [to_hex_rgba(c) for c in np.atleast_1d(col_fun(at2))]
    n_c = len(colors)
    step_w = (lw_mm - 2.0 * offset_mm) / n_c
    x2 = np.linspace(offset_mm, lw_mm - offset_mm, n_c, endpoint=False) + 0.5 * step_w
    x2_all = np.concatenate([[0.5 * offset_mm], x2, [lw_mm - 0.5 * offset_mm]])
    ww_all = np.concatenate([[offset_mm], np.full(n_c, step_w), [offset_mm]])
    cols_all = [colors[0]] + colors + [colors[-1]]

    children.append(
        rectGrob(
            x=x2_all,
            y=np.full(len(cols_all), tot_h),
            width=ww_all,
            height=np.full(len(cols_all), gh_mm),
            just="top",
            gp=gpar(col=cols_all, fill=cols_all),
            default_units="mm",
        )
    )

    lgp_tick = check_gp(legend_gp)
    if "col" not in lgp_tick:
        lgp_tick["col"] = "white" if border is None else border
    children.append(segmentsGrob(x_mm, np.full(n, tot_h - gh_mm), x_mm, np.full(n, tot_h - gh_mm + tick_mm), gp=lgp_tick, default_units="mm"))
    children.append(segmentsGrob(x_mm, np.full(n, tot_h - tick_mm), x_mm, np.full(n, tot_h), gp=lgp_tick, default_units="mm"))

    if adjust_text_pos:
        y_b0 = np.full(n, tot_h - gh_mm)
        y_b1 = np.full(n, tot_h - gh_mm - lbl_pad_top / 3.0)
        y_b2 = np.full(n, tot_h - gh_mm - lbl_pad_top * 2.0 / 3.0)
        y_b3 = np.full(n, tot_h - gh_mm - lbl_pad_top + 0.5)
        children.append(segmentsGrob(x_mm, y_b0, x_mm, y_b1, default_units="mm"))
        children.append(segmentsGrob(labels_x, y_b3, labels_x, y_b2, default_units="mm"))
        children.append(segmentsGrob(x_mm, y_b1, labels_x, y_b2, default_units="mm"))

    if border is not None:
        bg_gp = check_gp(lgp_tick)
        bg_gp["fill"] = "transparent"
        children.append(rectGrob(x=0.5 * lw_mm, y=tot_h, width=lw_mm, height=gh_mm, just="top", gp=bg_gp, default_units="mm"))

    return GTree(
        children=children,
        cl="legend_body",
        vp=Viewport(width=unit(lw_mm, "mm"), height=unit(tot_h, "mm")),
        width=unit(lw_mm, "mm"),
        height=unit(tot_h, "mm"),
    )


def Legend(
    at: Sequence[Any] | None = None,
    labels: Sequence[Any] | None = None,
    col_fun: Callable | None = None,
    name: str | None = None,
    grob: Grob | None = None,
    break_dist: Sequence[float] | float | None = None,
    nrow: int | None = None,
    ncol: int = 1,
    by_row: bool = False,
    grid_height: Any = unit(4, "mm"),
    grid_width: Any = unit(4, "mm"),
    tick_length: Any = unit(0.8, "mm"),
    gap: Any = unit(2, "mm"),
    column_gap: Any = None,
    row_gap: Any = unit(0, "mm"),
    labels_gp: Any = None,
    labels_rot: float = 0.0,
    border: Any = None,
    background: str = "#EEEEEE",
    type: str | Sequence[str] = "grid",
    graphics: Any = None,
    legend_gp: Any = None,
    pch: Any = 16,
    size: Any = unit(2, "mm"),
    legend_height: Any = None,
    legend_width: Any = None,
    direction: str = "vertical",
    title: str | None = "",
    title_gp: Any = None,
    title_position: str = "topleft",
    title_gap: Any = unit(2, "mm"),
) -> Legends:
    """Construct a single discrete, continuous, or custom legend (`Legends` object)."""
    if at is None and labels is not None:
        at = list(range(len(labels)))
    if labels is None and at is not None:
        labels = list(at)

    lgp_labels = check_gp(labels_gp or gpar(fontsize=10))
    if "fontsize" not in lgp_labels:
        lgp_labels["fontsize"] = 10.0
    lgp_title = check_gp(title_gp or gpar(fontsize=10, fontface="bold"))
    if "fontsize" not in lgp_title:
        lgp_title["fontsize"] = 10.0
    if "fontface" not in lgp_title:
        lgp_title["fontface"] = "bold"

    direction = direction.lower()
    if direction not in ("vertical", "horizontal"):
        raise ValueError(f"Invalid legend `direction`: {direction!r}")
    title_position = title_position.lower()
    valid_tp = ("topleft", "topcenter", "leftcenter", "lefttop", "leftcenter-rot", "lefttop-rot")
    if title_position not in valid_tp:
        raise ValueError(f"Invalid `title_position`: {title_position!r}")

    col_gap_u = as_unit(gap if column_gap is None else column_gap, "mm")
    row_gap_u = as_unit(row_gap, "mm")
    gh_u = as_unit(grid_height, "mm")
    gw_u = as_unit(grid_width, "mm")
    tick_u = as_unit(tick_length, "mm")
    title_pad_mm = convertHeight(as_unit(title_gap, "mm"), "mm", valueOnly=True)

    if grob is not None:
        bw = grob.width_mm()
        bh = grob.height_mm()
        legend_body = GTree(children=[grob], cl="legend_body", vp=Viewport(width=unit(bw, "mm"), height=unit(bh, "mm")), width=unit(bw, "mm"), height=unit(bh, "mm"))
    elif col_fun is None:
        if at is None or labels is None:
            raise ValueError("Provide `at` and/or `labels`, or `col_fun`, or `grob`.")
        eff_border = "white" if border is None else border
        legend_body = _discrete_legend_body(
            at=list(at),
            labels=list(labels),
            nrow=nrow,
            ncol=ncol,
            by_row=by_row,
            grid_height=gh_u,
            grid_width=gw_u,
            column_gap=col_gap_u,
            row_gap=row_gap_u,
            labels_gp=lgp_labels,
            border=eff_border,
            background=background,
            type=type,
            graphics=graphics,
            legend_gp=legend_gp,
            pch=pch,
            size=as_unit(size, "mm"),
        )
    else:
        if at is None:
            brk = getattr(col_fun, "breaks", None)
            if brk is None:
                raise ValueError("Provide `at` when `col_fun` has no `breaks` attribute.")
            if break_dist is None:
                le1 = grid_pretty((float(np.min(brk)), float(np.max(brk))))
                le2 = pretty_breaks((float(np.min(brk)), float(np.max(brk))), n=3)
                at_vals = le1 if abs(len(le1) - 5) < abs(len(le2) - 5) else le2
            else:
                at_vals = np.asarray(brk, dtype=float)
            labels = [f"{v:g}" for v in at_vals]
        else:
            at_vals = list(at)
            if labels is None:
                labels = list(at_vals)

        if direction == "vertical":
            if title_position in ("leftcenter", "lefttop"):
                raise ValueError("'topleft', 'topcenter', 'leftcenter-rot' and 'lefttop-rot' are only allowed for vertical continuous legend.")
            legend_body = _vertical_continuous_legend_body(
                at=at_vals,
                labels=list(labels),
                col_fun=col_fun,
                break_dist=break_dist,
                grid_width=gw_u,
                legend_height=None if legend_height is None else as_unit(legend_height, "mm"),
                tick_length=tick_u,
                labels_gp=lgp_labels,
                border=border,
                legend_gp=legend_gp,
            )
        else:
            if title_position in ("leftcenter-rot", "lefttop-rot"):
                raise ValueError("'topleft', 'topcenter', 'lefttop' and 'leftcenter' are only allowed for horizontal continuous legend.")
            ext_u = unit(0.0, "mm")
            if title_position == "lefttop" and title:
                tg = textGrob(title, gp=lgp_title)
                if tg.height_mm() <= convertHeight(gh_u, "mm", valueOnly=True):
                    ext_u = unit(tg.width_mm() + title_pad_mm, "mm")
            legend_body = _horizontal_continuous_legend_body(
                at=at_vals,
                labels=list(labels),
                col_fun=col_fun,
                break_dist=break_dist,
                grid_height=gh_u,
                legend_width=None if legend_width is None else as_unit(legend_width, "mm"),
                tick_length=tick_u,
                labels_gp=lgp_labels,
                labels_rot=labels_rot,
                border=border,
                legend_gp=legend_gp,
                legend_extension=ext_u,
            )

    from .text import GridText

    has_title = False
    if isinstance(title, GridText):
        has_title = len(title) > 0 and bool(title.as_list()[0].strip())
    elif title is not None and str(title).strip() != "":
        has_title = True

    if not has_title:
        return Legends(grob=legend_body, type="single_legend_no_title", name=name, n=1, multiple=1, direction="vertical")

    tg = textGrob(title, gp=lgp_title)
    t_w = tg.width_mm()
    t_h = tg.height_mm()
    b_w = legend_body.width_mm()
    b_h = legend_body.height_mm()

    if title_position in ("topleft", "topcenter"):
        tot_w = max(t_w, b_w) if title_position == "topleft" else max(t_w, b_w)
        tot_h = t_h + title_pad_mm + b_h
        tx = unit(0.0, "npc") if title_position == "topleft" else unit(0.5, "npc")
        tjust = ("left", "top") if title_position == "topleft" else "top"
        gf = GTree(
            children=[
                textGrob(title, x=tx, y=unit(1.0, "npc"), just=tjust, gp=lgp_title),
                _edit_vp_in_legend_grob(legend_body, x=unit(0.0, "npc"), y=unit(0.0, "npc"), valid_just=(0.0, 0.0)),
            ],
            vp=Viewport(width=unit(tot_w, "mm"), height=unit(tot_h, "mm")),
            cl="legend",
            width=unit(tot_w, "mm"),
            height=unit(tot_h, "mm"),
        )
    elif title_position in ("leftcenter", "lefttop"):
        tot_w = t_w + title_pad_mm + b_w
        tot_h = max(t_h, b_h)
        ty = unit(1.0, "npc") if title_position == "lefttop" else unit(0.5, "npc")
        tjust = ("left", "top") if title_position == "lefttop" else "left"
        gf = GTree(
            children=[
                textGrob(title, x=unit(0.0, "npc"), y=ty, just=tjust, gp=lgp_title),
                _edit_vp_in_legend_grob(legend_body, x=unit(1.0, "npc"), y=unit(1.0, "npc"), valid_just=(1.0, 1.0)),
            ],
            vp=Viewport(width=unit(tot_w, "mm"), height=unit(tot_h, "mm")),
            cl="legend",
            width=unit(tot_w, "mm"),
            height=unit(tot_h, "mm"),
        )
    else:  # leftcenter-rot, lefttop-rot
        tot_w = t_h + title_pad_mm + b_w
        tot_h = max(t_w, b_h)
        if title_position == "lefttop-rot":
            ty = unit(1.0, "npc")
            tjust = ("right", "top")
        else:
            ty = unit(0.5, "npc")
            tjust = "top"
        gf = GTree(
            children=[
                textGrob(title, x=unit(0.0, "npc"), y=ty, just=tjust, gp=lgp_title, rot=90.0),
                _edit_vp_in_legend_grob(legend_body, x=unit(1.0, "npc"), y=unit(1.0, "npc"), valid_just=(1.0, 1.0)),
            ],
            vp=Viewport(width=unit(tot_w, "mm"), height=unit(tot_h, "mm")),
            cl="legend",
            width=unit(tot_w, "mm"),
            height=unit(tot_h, "mm"),
        )

    return Legends(grob=gf, type="single_legend", name=name, n=1, multiple=1, direction="vertical")


def _split_by_max(vals: Sequence[float], max_val: float, gap: float = 0.0) -> list[list[int]]:
    n = len(vals)
    chunks: list[list[int]] = []
    cur: list[int] = []
    acc = 0.0
    for i, v in enumerate(vals):
        step = v + (gap if cur else 0.0)
        if cur and acc + step >= max_val:
            chunks.append(cur)
            cur = [i]
            acc = v
        else:
            cur.append(i)
            acc += step
    if cur:
        chunks.append(cur)
    return chunks


def packLegend(
    *legends: Legends,
    gap: Any = unit(4, "mm"),
    row_gap: Any = None,
    column_gap: Any = None,
    direction: str = "vertical",
    max_width: Any = None,
    max_height: Any = None,
    list: Sequence[Legends] | None = None,
) -> Legends:
    """Pack multiple `Legends` objects vertically or horizontally with optional wrapping."""
    lgd_list = [lg for lg in (list if list is not None else legends) if lg is not None]
    if not lgd_list:
        empty = GTree(children=[], cl="packed_legends", width=unit(0.0, "mm"), height=unit(0.0, "mm"))
        return Legends(grob=empty, type="packed_legends", n=0, multiple=1, direction=direction)
    if len(lgd_list) == 1:
        return lgd_list[0]

    direction = direction.lower()
    rg_mm = convertHeight(as_unit(gap if row_gap is None else row_gap, "mm"), "mm", valueOnly=True)
    cg_mm = convertWidth(as_unit(gap if column_gap is None else column_gap, "mm"), "mm", valueOnly=True)
    grobs = [lg.grob for lg in lgd_list]
    n_lgd = len(grobs)

    children: list[Grob] = []
    if direction == "vertical":
        heights = [g.height_mm() for g in grobs]
        widths = [g.width_mm() for g in grobs]
        if max_height is None:
            chunks = [[i for i in range(n_lgd)]]
        else:
            mh = convertHeight(as_unit(max_height, "mm"), "mm", valueOnly=True)
            chunks = _split_by_max(heights, mh, rg_mm)
        nc = len(chunks)
        col_widths = [max(widths[i] for i in ch) for ch in chunks]
        col_heights = [sum(heights[i] for i in ch) + rg_mm * max(0, len(ch) - 1) for ch in chunks]
        tot_w = sum(col_widths) + cg_mm * max(0, nc - 1)
        tot_h = max(col_heights)

        cur_x = 0.0
        for ci, ch in enumerate(chunks):
            cur_top = tot_h
            for i in ch:
                g = grobs[i]
                h_i = heights[i]
                cur_bot = cur_top - h_i
                children.append(
                    _edit_vp_in_legend_grob(
                        g,
                        x=unit(cur_x, "mm"),
                        y=unit(cur_bot, "mm"),
                        valid_just=(0.0, 0.0),
                    )
                )
                cur_top = cur_bot - rg_mm
            cur_x += col_widths[ci] + cg_mm
        mult = nc
    else:
        heights = [g.height_mm() for g in grobs]
        widths = [g.width_mm() for g in grobs]
        if max_width is None:
            chunks = [[i for i in range(n_lgd)]]
        else:
            mw = convertWidth(as_unit(max_width, "mm"), "mm", valueOnly=True)
            chunks = _split_by_max(widths, mw, cg_mm)
        nr = len(chunks)
        row_heights = [max(heights[i] for i in ch) for ch in chunks]
        row_widths = [sum(widths[i] for i in ch) + cg_mm * max(0, len(ch) - 1) for ch in chunks]
        tot_w = max(row_widths)
        tot_h = sum(row_heights) + rg_mm * max(0, nr - 1)

        cur_top = tot_h
        for ri, ch in enumerate(chunks):
            cur_x = 0.0
            for i in ch:
                g = grobs[i]
                w_i = widths[i]
                children.append(
                    _edit_vp_in_legend_grob(
                        g,
                        x=unit(cur_x + w_i, "mm"),
                        y=unit(cur_top, "mm"),
                        valid_just=(1.0, 1.0),
                    )
                )
                cur_x += w_i + cg_mm
            cur_top -= row_heights[ri] + rg_mm
        mult = nr

    gt = GTree(
        children=children,
        cl="packed_legends",
        vp=Viewport(width=unit(tot_w, "mm"), height=unit(tot_h, "mm")),
        width=unit(tot_w, "mm"),
        height=unit(tot_h, "mm"),
    )
    return Legends(grob=gt, type="packed_legends", n=n_lgd, multiple=mult, direction=direction)
