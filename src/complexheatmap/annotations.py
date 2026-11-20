"""AnnotationFunction, SingleAnnotation, HeatmapAnnotation, columnAnnotation, rowAnnotation, and Axis."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._base import AdditiveUnit
from ._options import _ENV, ht_opt
from .colors import ColorMapping, default_col, pretty_breaks, to_hex_rgba
from .grid import (
    GTree,
    Grob,
    Viewport,
    current_viewport,
    grid_draw,
    grid_newpage,
    grid_rect,
    grid_text,
    linesGrob,
    popViewport,
    pushViewport,
    segmentsGrob,
    textGrob,
)
from .units import (
    Gpar,
    Unit,
    as_unit,
    check_gp,
    convertHeight,
    convertWidth,
    gpar,
    is_abs_unit,
    recycle_gp,
    subset_gp,
    unit,
    unit_c,
    unit_sum,
)


def subset_vector(x: Any, i: Any) -> Any:
    if isinstance(x, Unit):
        return x[i]
    if isinstance(x, np.ndarray):
        return x[i]
    if isinstance(x, (list, tuple)):
        if isinstance(i, slice):
            return list(x)[i]
        if isinstance(i, (int, np.integer)):
            return [x[int(i)]]
        arr_i = np.asarray(i)
        if arr_i.dtype == bool:
            return [v for v, f in zip(x, arr_i) if f]
        return [x[int(k)] for k in arr_i]
    if hasattr(x, "iloc"):
        return x.iloc[i]
    return x


def subset_matrix_by_row(x: Any, i: Any) -> Any:
    arr = np.asarray(x)
    if isinstance(i, (int, np.integer)):
        return arr[[int(i)], :]
    return arr[i, :]


def subset_no(x: Any, i: Any) -> Any:
    return x


def subset_by_intersect(x: Any, i: Any) -> list[Any]:
    s = set(int(v) for v in i)
    return [v for v in x if int(v) in s]


def anno_width_and_height(
    which: str,
    width: Unit | None = None,
    height: Unit | None = None,
    default: Unit = unit(10, "mm"),
) -> dict[str, Unit]:
    if which == "column":
        if height is None:
            height = default
        else:
            height = as_unit(height, "mm")
            if not is_abs_unit(height):
                raise ValueError("Height of a column annotation must be an absolute unit.")
        if width is None:
            width = unit(1.0, "npc")
        else:
            width = as_unit(width, "npc")
    elif which == "row":
        if width is None:
            width = default
        else:
            width = as_unit(width, "mm")
            if not is_abs_unit(width):
                raise ValueError("Width of a row annotation must be an absolute unit.")
        if height is None:
            height = unit(1.0, "npc")
        else:
            height = as_unit(height, "npc")
    else:
        raise ValueError(f"`which` must be 'column' or 'row', got {which!r}")
    return {"width": width, "height": height}


# ---- Axis System ----

def default_axis_param(which: str = "column") -> dict[str, Any]:
    which = which.lower()
    return {
        "at": None,
        "labels": None,
        "labels_rot": 0 if which == "column" else 90,
        "gp": gpar(fontsize=8),
        "side": "left" if which == "column" else "bottom",
        "facing": "outside",
        "direction": "normal",
    }


def validate_axis_param(axis_param: dict[str, Any] | None, which: str) -> dict[str, Any]:
    d = default_axis_param(which)
    if axis_param:
        for k, v in axis_param.items():
            d[k] = v
    d["gp"] = check_gp(d.get("gp"))
    if "fontsize" not in d["gp"]:
        d["gp"]["fontsize"] = 8.0
    if which == "column" and d["side"] not in ("left", "right"):
        raise ValueError("For column annotation, `axis_param['side']` must be 'left' or 'right'.")
    if which == "row" and d["side"] not in ("top", "bottom"):
        raise ValueError("For row annotation, `axis_param['side']` must be 'top' or 'bottom'.")
    if d["facing"] not in ("outside", "inside"):
        raise ValueError("`axis_param['facing']` must be 'outside' or 'inside'.")
    if d["direction"] not in ("normal", "reverse"):
        raise ValueError("`axis_param['direction']` must be 'normal' or 'reverse'.")
    return d


def annotation_axis_grob(
    at: Sequence[float] | None = None,
    labels: Sequence[Any] | None = None,
    labels_rot: float = 0.0,
    gp: dict[str, Any] | None = None,
    side: str = "left",
    facing: str = "outside",
    direction: str = "normal",
    scale: Sequence[float] | None = None,
) -> GTree:
    """Build an `annotation_axis` grob with ticks and labels."""
    if side not in ("left", "right", "top", "bottom"):
        raise ValueError("`side` must be 'left', 'right', 'top', or 'bottom'.")
    if facing not in ("inside", "outside"):
        raise ValueError("`facing` must be 'inside' or 'outside'.")
    if direction not in ("normal", "reverse"):
        raise ValueError("`direction` must be 'normal' or 'reverse'.")

    gp_d = check_gp(gp)
    if scale is None:
        vp = current_viewport()
        scale = vp.xscale if side in ("top", "bottom") else vp.yscale
    sc = (float(scale[0]), float(scale[1]))

    if at is None:
        at_vals = pretty_breaks(sc, n=3)
        lo, hi = min(sc), max(sc)
        at_vals = at_vals[(at_vals >= lo - 1e-12) & (at_vals <= hi + 1e-12)]
        if len(at_vals) <= 2:
            at_vals = pretty_breaks(sc, n=5)
            at_vals = at_vals[(at_vals >= lo - 1e-12) & (at_vals <= hi + 1e-12)]
    else:
        at_vals = np.atleast_1d(np.asarray(at, dtype=float))

    lbls = [f"{v:g}" for v in at_vals] if labels is None else [str(v) for v in labels]

    if direction == "reverse":
        at_pos = sc[1] - at_vals + sc[0]
    else:
        at_pos = at_vals

    rot = float(labels_rot) % 360.0
    children: list[Grob] = []

    if side == "left" and facing == "inside":
        txt_g = textGrob(lbls, unit(2, "mm"), at_pos, default_units="native", rot=0, just="left", gp=gp_d)
        # Note x is 2mm, y is native
        txt_g = textGrob(lbls, unit(2, "mm"), unit(at_pos, "native"), rot=0, just="left", gp=gp_d)
        children = [
            linesGrob(unit([0, 0], "npc"), unit([0, 1], "npc"), gp=gp_d),
            segmentsGrob(unit(np.ones(len(at_pos)), "mm"), unit(at_pos, "native"), unit(np.zeros(len(at_pos)), "npc"), unit(at_pos, "native"), gp=gp_d),
            txt_g,
        ]
        w = unit(max((textGrob(l, gp=gp_d).width_mm() for l in lbls), default=0.0) + 2.0, "mm")
        h = unit(1.0, "npc")
    elif side == "left" and facing == "outside":
        txt_g = textGrob(lbls, unit(-2, "mm"), unit(at_pos, "native"), rot=0, just="right", gp=gp_d)
        children = [
            linesGrob(unit([0, 0], "npc"), unit([0, 1], "npc"), gp=gp_d),
            segmentsGrob(unit(-np.ones(len(at_pos)), "mm"), unit(at_pos, "native"), unit(np.zeros(len(at_pos)), "npc"), unit(at_pos, "native"), gp=gp_d),
            txt_g,
        ]
        w = unit(max((textGrob(l, gp=gp_d).width_mm() for l in lbls), default=0.0) + 2.0, "mm")
        h = unit(1.0, "npc")
    elif side == "right" and facing == "inside":
        txt_g = textGrob(lbls, unit(1, "npc") - unit(2, "mm"), unit(at_pos, "native"), rot=0, just="right", gp=gp_d)
        children = [
            linesGrob(unit([1, 1], "npc"), unit([0, 1], "npc"), gp=gp_d),
            segmentsGrob(unit(1, "npc") - unit(1, "mm"), unit(at_pos, "native"), unit(1, "npc"), unit(at_pos, "native"), gp=gp_d),
            txt_g,
        ]
        w = unit(max((textGrob(l, gp=gp_d).width_mm() for l in lbls), default=0.0) + 2.0, "mm")
        h = unit(1.0, "npc")
    elif side == "right" and facing == "outside":
        txt_g = textGrob(lbls, unit(1, "npc") + unit(2, "mm"), unit(at_pos, "native"), rot=0, just="left", gp=gp_d)
        children = [
            linesGrob(unit([1, 1], "npc"), unit([0, 1], "npc"), gp=gp_d),
            segmentsGrob(unit(1, "npc") + unit(1, "mm"), unit(at_pos, "native"), unit(1, "npc"), unit(at_pos, "native"), gp=gp_d),
            txt_g,
        ]
        w = unit(max((textGrob(l, gp=gp_d).width_mm() for l in lbls), default=0.0) + 2.0, "mm")
        h = unit(1.0, "npc")
    elif side == "top" and facing == "inside":
        j = "top" if rot == 0 else "right"
        txt_g = textGrob(lbls, unit(at_pos, "native"), unit(1, "npc") - unit(2, "mm"), rot=rot, just=j, gp=gp_d)
        children = [
            linesGrob(unit([0, 1], "npc"), unit([1, 1], "npc"), gp=gp_d),
            segmentsGrob(unit(at_pos, "native"), unit(1, "npc") - unit(1, "mm"), unit(at_pos, "native"), unit(1, "npc"), gp=gp_d),
            txt_g,
        ]
        w = unit(1.0, "npc")
        h = unit(max((textGrob(l, gp=gp_d, rot=rot).height_mm() for l in lbls), default=0.0) + 2.0, "mm")
    elif side == "top" and facing == "outside":
        j = "bottom" if rot == 0 else "left"
        txt_g = textGrob(lbls, unit(at_pos, "native"), unit(1, "npc") + unit(2, "mm"), rot=rot, just=j, gp=gp_d)
        children = [
            linesGrob(unit([0, 1], "npc"), unit([1, 1], "npc"), gp=gp_d),
            segmentsGrob(unit(at_pos, "native"), unit(1, "npc") + unit(1, "mm"), unit(at_pos, "native"), unit(1, "npc"), gp=gp_d),
            txt_g,
        ]
        w = unit(1.0, "npc")
        h = unit(max((textGrob(l, gp=gp_d, rot=rot).height_mm() for l in lbls), default=0.0) + 2.0, "mm")
    elif side == "bottom" and facing == "inside":
        j = "bottom" if rot == 0 else "left"
        txt_g = textGrob(lbls, unit(at_pos, "native"), unit(2, "mm"), rot=rot, just=j, gp=gp_d)
        children = [
            linesGrob(unit([0, 1], "npc"), unit([0, 0], "npc"), gp=gp_d),
            segmentsGrob(unit(at_pos, "native"), unit(1, "mm"), unit(at_pos, "native"), unit(0, "npc"), gp=gp_d),
            txt_g,
        ]
        w = unit(1.0, "npc")
        h = unit(max((textGrob(l, gp=gp_d, rot=rot).height_mm() for l in lbls), default=0.0) + 2.0, "mm")
    else:  # bottom & outside
        j = "top" if rot == 0 else "right"
        txt_g = textGrob(lbls, unit(at_pos, "native"), unit(-2, "mm"), rot=rot, just=j, gp=gp_d)
        children = [
            linesGrob(unit([0, 1], "npc"), unit([0, 0], "npc"), gp=gp_d),
            segmentsGrob(unit(at_pos, "native"), unit(-1, "mm"), unit(at_pos, "native"), unit(0, "npc"), gp=gp_d),
            txt_g,
        ]
        w = unit(1.0, "npc")
        h = unit(max((textGrob(l, gp=gp_d, rot=rot).height_mm() for l in lbls), default=0.0) + 2.0, "mm")

    return GTree(children=children, cl="annotation_axis", width=w, height=h)


def grid_annotation_axis(
    at: Sequence[float] | None = None,
    labels: Sequence[Any] | None = None,
    labels_rot: float = 0.0,
    gp: dict[str, Any] | None = None,
    side: str = "left",
    facing: str = "outside",
    direction: str = "normal",
    scale: Sequence[float] | None = None,
) -> GTree:
    gb = annotation_axis_grob(
        at=at, labels=labels, labels_rot=labels_rot, gp=gp,
        side=side, facing=facing, direction=direction, scale=scale,
    )
    gb.draw()
    return gb


def construct_axis_grob(axis_param: dict[str, Any], which: str, data_scale: Sequence[float]) -> GTree:
    return annotation_axis_grob(
        at=axis_param.get("at"),
        labels=axis_param.get("labels"),
        labels_rot=axis_param.get("labels_rot", 0 if which == "column" else 90),
        gp=axis_param.get("gp"),
        side=axis_param["side"],
        facing=axis_param["facing"],
        direction=axis_param["direction"],
        scale=(float(data_scale[0]), float(data_scale[1])),
    )


def update_anno_extend(anno: "AnnotationFunction", axis_grob: GTree | None, axis_param: dict[str, Any]) -> Unit:
    # extended: [bottom, left, top, right] in mm
    ext = [0.0, 0.0, 0.0, 0.0]
    if axis_grob is None or axis_param.get("facing") != "outside":
        return unit(ext, "mm")
    side = axis_param["side"]
    if side == "bottom":
        ext[0] = axis_grob.height_mm()
    elif side == "left":
        ext[1] = axis_grob.width_mm()
    elif side == "top":
        ext[2] = axis_grob.height_mm()
    elif side == "right":
        ext[3] = axis_grob.width_mm()
    return unit(ext, "mm")


# ---- Core Annotation Classes ----

class AnnotationFunction:
    """Subsettable graphic function for row or column heatmap annotations."""

    def __init__(
        self,
        fun: Callable | None = None,
        fun_name: str = "",
        which: str = "column",
        cell_fun: Callable[[int], None] | None = None,
        var_import: dict[str, Any] | None = None,
        n: int | None = None,
        data_scale: Sequence[float] = (0.0, 1.0),
        subset_rule: dict[str, Callable[[Any, Any], Any]] | None = None,
        subsettable: bool | None = None,
        show_name: bool = True,
        width: Unit | None = None,
        height: Unit | None = None,
        *,
        _which_explicit: bool = True,
        _reorient_fn: Callable[[str], "AnnotationFunction"] | None = None,
    ) -> None:
        self.which = which.lower()
        self.fun_name = fun_name
        sz = anno_width_and_height(self.which, width, height, unit(1.0, "cm"))
        self.width: Unit = sz["width"]
        self.height: Unit = sz["height"]
        self.n: int | None = None if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
        self.data_scale: tuple[float, float] = (float(data_scale[0]), float(data_scale[1]))
        self.extended: Unit = unit([0.0, 0.0, 0.0, 0.0], "mm")
        self.show_name: bool = bool(show_name)
        self.var_env: dict[str, Any] = dict(var_import) if var_import else {}
        self._which_explicit = _which_explicit
        self._reorient_fn = _reorient_fn
        self._called_args: dict[str, Any] | None = None

        if cell_fun is not None:
            def _wrapped_cell(index, k=1, n_slices=1, env=None):
                m = len(index)
                for pos_i, orig_i in enumerate(index):
                    if self.which == "column":
                        pushViewport(Viewport(x=unit((pos_i + 1) / m, "npc"), width=unit(1.0 / m, "npc"), just="right"))
                    else:
                        pushViewport(Viewport(y=unit(1.0 - (pos_i + 1) / m, "npc"), height=unit(1.0 / m, "npc"), just="bottom"))
                    try:
                        cell_fun(int(orig_i))
                    finally:
                        popViewport()
            self.fun = _wrapped_cell
            self._takes_env = True
            subsettable_eff = True
        else:
            if fun is None:
                def fun(index, k=1, n=1):
                    return None
            self.fun = fun
            sig = inspect.signature(fun)
            params = list(sig.parameters.keys())
            self._takes_env = "env" in params
            self._n_pos_params = sum(
                1 for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.name != "env"
            )
            subsettable_eff = bool(subset_rule) if subsettable is None else bool(subsettable)

        if subset_rule is None:
            self.subset_rule: dict[str, Callable[[Any, Any], Any]] = {}
            for k_v, v_val in self.var_env.items():
                if isinstance(v_val, np.ndarray) and v_val.ndim == 2:
                    self.subset_rule[k_v] = subset_matrix_by_row
                elif isinstance(v_val, Gpar):
                    self.subset_rule[k_v] = subset_gp
                elif isinstance(v_val, (list, tuple, np.ndarray, Unit)) and len(v_val) > 1:
                    self.subset_rule[k_v] = subset_vector
            if self.subset_rule and subsettable is None:
                subsettable_eff = True
        else:
            self.subset_rule = dict(subset_rule)

        self.subsettable: bool = subsettable_eff

    def reorient(self, new_which: str) -> "AnnotationFunction":
        if self.which == new_which:
            return self
        if not self._which_explicit and self._reorient_fn is not None:
            return self._reorient_fn(new_which)
        raise ValueError(f"`which` in AnnotationFunction ({self.which!r}) differs from container ({new_which!r}).")

    def copy_all(self) -> "AnnotationFunction":
        cp = copy.copy(self)
        cp.var_env = copy.deepcopy(self.var_env)
        cp.subset_rule = dict(self.subset_rule)
        if self._called_args is not None:
            cp._called_args = copy.deepcopy(self._called_args)
        return cp

    def __getitem__(self, i: Any) -> "AnnotationFunction":
        if not self.subsettable:
            raise ValueError("This AnnotationFunction is not subsettable.")
        if isinstance(i, slice):
            if self.n is None or self.n < 0:
                raise ValueError("Cannot slice AnnotationFunction with unknown observation count.")
            idx = list(range(*i.indices(self.n)))
        elif isinstance(i, (int, np.integer)):
            idx = [int(i)]
        else:
            arr_i = np.asarray(i)
            if arr_i.dtype == bool:
                idx = np.where(arr_i)[0].tolist()
            else:
                idx = [int(k) for k in arr_i]

        if self.fun_name == "anno_mark" and self._called_args is not None:
            from .anno_funcs import anno_mark
            ca = copy.deepcopy(self._called_args)
            idx_sorted = sorted(idx)
            pos_map = {orig: new_i for new_i, orig in enumerate(idx_sorted)}
            keep_mask = [a in pos_map for a in ca["at"]]
            ca["at"] = [pos_map[a] for a, fl in zip(ca["at"], keep_mask) if fl]
            ca["labels"] = [lb for lb, fl in zip(ca["labels"], keep_mask) if fl]
            ca["link_gp"] = subset_gp(ca["link_gp"], keep_mask)
            ca["labels_gp"] = subset_gp(ca["labels_gp"], keep_mask)
            return anno_mark(**ca)

        cp = self.copy_all()
        for var, rule in cp.subset_rule.items():
            if var in cp.var_env:
                cp.var_env[var] = rule(cp.var_env[var], idx)
        cp.n = len(idx)
        return cp

    def nobs(self) -> int | None:
        if self.n is None or self.n <= 0:
            return None
        return int(self.n)

    @property
    def size(self) -> Unit:
        return self.width if self.which == "row" else self.height

    @size.setter
    def size(self, val: Any) -> None:
        u = as_unit(val, "mm")
        if self.which == "row":
            self.width = u
        else:
            self.height = u

    def draw(
        self,
        index: Sequence[int] | None = None,
        k: int = 1,
        n: int = 1,
        test: bool | str = False,
        **vp_kwargs: Any,
    ) -> None:
        if isinstance(test, str):
            is_test = True
            test_label = test
        else:
            is_test = bool(test)
            test_label = ""
        if is_test:
            grid_newpage()
            pushViewport(Viewport(width=unit(0.8, "npc"), height=unit(0.8, "npc")))
            if self.n is None:
                self.n = 1
        if index is None:
            cnt = self.n if (self.n is not None and self.n > 0) else 1
            index = list(range(cnt))

        pushViewport(Viewport(width=self.width, height=self.height, **vp_kwargs))
        try:
            if self._takes_env:
                self.fun(index, k, n, env=self.var_env)
            elif self._n_pos_params <= 1:
                self.fun(index)
            elif self._n_pos_params == 2:
                self.fun(index, k)
            else:
                self.fun(index, k, n)

            if is_test:
                if test_label:
                    grid_text(test_label, y=unit(1, "npc") + unit(2, "mm"), just="bottom")
                ext = self.extended.to_mm("x")
                if ext[0] > 0:
                    grid_rect(y=unit(1, "npc"), height=unit(1, "npc") + unit(ext[0], "mm"), just="top", gp=gpar(fill="transparent", col="red", lty=2))
                elif ext[1] > 0:
                    grid_rect(x=unit(1, "npc"), width=unit(1, "npc") + unit(ext[1], "mm"), just="right", gp=gpar(fill="transparent", col="red", lty=2))
                elif ext[2] > 0:
                    grid_rect(y=unit(0, "npc"), height=unit(1, "npc") + unit(ext[2], "mm"), just="bottom", gp=gpar(fill="transparent", col="red", lty=2))
                elif ext[3] > 0:
                    grid_rect(x=unit(0, "npc"), width=unit(1, "npc") + unit(ext[3], "mm"), just="left", gp=gpar(fill="transparent", col="red", lty=2))
        finally:
            popViewport()
            if is_test:
                popViewport()

    def _repr_png_(self) -> bytes:
        import io
        import matplotlib.pyplot as plt
        dev = grid_newpage(width_in=6.0, height_in=4.0)
        self.draw(test=self.fun_name or "AnnotationFunction")
        buf = io.BytesIO()
        dev.fig.savefig(buf, format="png", dpi=120)
        plt.close(dev.fig)
        return buf.getvalue()

    def __repr__(self) -> str:
        fn = f"{self.fun_name}()" if self.fun_name else "user-defined"
        return (
            f"An AnnotationFunction object\n"
            f"  function: {fn}\n"
            f"  position: {self.which}\n"
            f"  items: {self.n if self.n is not None else 'unknown'}\n"
            f"  width: {self.width}\n"
            f"  height: {self.height}\n"
            f"  subsettable: {self.subsettable}"
        )


def _normalize_anno_matrix_or_vec(val: Any) -> tuple[Any, bool, list[str] | None]:
    """Normalize atomic vector / DataFrame / 2D array for simple annotation."""
    if isinstance(val, pd.Series):
        arr = val.to_numpy()
        return arr, False, None
    if isinstance(val, pd.DataFrame):
        cnames = [str(c) for c in val.columns]
        arr = val.to_numpy()
        if arr.shape[1] == 1:
            return arr[:, 0], False, None
        return arr, True, cnames
    arr = np.asarray(val)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0], False, None
        return arr, True, None
    return arr, False, None


class SingleAnnotation:
    """Single track or complex annotation with name, optional ColorMapping, and legend params."""

    def __init__(
        self,
        name: str | None = None,
        value: Any = None,
        col: Any = None,
        fun: AnnotationFunction | Callable | None = None,
        label: Any = None,
        na_col: str = "grey",
        which: str = "column",
        show_legend: bool = True,
        gp: dict[str, Any] | None = None,
        border: bool = False,
        legend_param: dict[str, Any] | None = None,
        show_name: bool = True,
        name_gp: dict[str, Any] | None = None,
        name_offset: Any = None,
        name_side: str | None = None,
        name_rot: float | None = None,
        simple_anno_size: Unit | None = None,
        width: Unit | None = None,
        height: Unit | None = None,
    ) -> None:
        from .anno_funcs import anno_simple

        self.which = which.lower()
        if name is None:
            name = f"anno{next(_ENV.anno_counter)}"
        self.name = str(name)
        self.label = self.name if label is None else label
        self.show_legend = bool(show_legend)
        self.color_mapping: ColorMapping | None = None
        self.color_is_random = False
        self.is_anno_matrix = False
        self.anno_matrix_colnames: list[str] | None = None
        self.anno_matrix_k: int = 1
        self.name_to_data_vp = False

        # Merge global legend defaults
        lp = dict(legend_param) if legend_param else {}
        for opt_n in ("title_gp", "title_position", "labels_gp", "grid_width", "grid_height", "border"):
            if opt_n not in lp:
                gv = getattr(ht_opt, f"legend_{opt_n}")
                if gv is not None:
                    lp[opt_n] = gv
        self.legend_param = lp

        if simple_anno_size is None:
            simple_anno_size = ht_opt.simple_anno_size
        if name_side is None:
            name_side = "right" if self.which == "column" else "bottom"
        name_gp_d = check_gp(name_gp or gpar(fontsize=12))
        if "fontsize" not in name_gp_d:
            name_gp_d["fontsize"] = 12.0

        is_offset_called = name_offset is not None
        is_rot_called = name_rot is not None

        if fun is not None:
            if isinstance(fun, AnnotationFunction):
                af = fun.reorient(self.which)
            elif callable(fun):
                af = AnnotationFunction(fun=fun, which=self.which, width=width, height=height)
            else:
                raise TypeError("`fun` must be an AnnotationFunction or callable.")
            self.fun = af
            self.show_legend = False
            if not af.show_name:
                show_name = False
            self.extended = af.extended
            self.width = af.width
            self.height = af.height
            self.subsettable = af.subsettable
        else:
            if value is None:
                raise ValueError("Either `value` or `fun` must be provided to SingleAnnotation.")
            norm_val, is_mat, cnames = _normalize_anno_matrix_or_vec(value)
            if np.asarray(norm_val).dtype == bool:
                norm_val = np.where(norm_val, "TRUE", "FALSE")
            self.is_anno_matrix = is_mat
            self.anno_matrix_colnames = cnames
            self.anno_matrix_k = norm_val.shape[1] if is_mat else 1

            # Resolve col -> ColorMapping
            if isinstance(col, ColorMapping):
                cm = col
            elif col is None:
                self.color_is_random = True
                inferred = default_col(norm_val)
                if callable(inferred):
                    cm = ColorMapping(name=self.name, col_fun=inferred, na_col=na_col)
                else:
                    cm = ColorMapping(name=self.name, colors=inferred, na_col=na_col)
            elif callable(col):
                cm = ColorMapping(name=self.name, col_fun=col, na_col=na_col)
            elif isinstance(col, dict):
                eff_na = col.get("_NA_", na_col)
                clean_col = {k: v for k, v in col.items() if k != "_NA_"}
                cm = ColorMapping(name=self.name, colors=clean_col, na_col=eff_na, full_col=clean_col)
            else:
                # Sequence of colors without names
                col_seq = list(col)
                arr_v = np.asarray(norm_val)
                is_num = np.issubdtype(arr_v.dtype, np.number)
                uniq = list(dict.fromkeys(v for v in arr_v.ravel() if v is not None and not (isinstance(v, float) and np.isnan(v))))
                if len(col_seq) == len(uniq):
                    keys = [str(u) for u in sorted(uniq)]
                    cm = ColorMapping(name=self.name, colors=dict(zip(keys, col_seq)), na_col=na_col)
                elif is_num:
                    from .colors import colorRamp2
                    vals = arr_v.astype(float)
                    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                    if vmin == vmax:
                        vmax = vmin + 1e-6
                    cf = colorRamp2(np.linspace(vmin, vmax, len(col_seq)), col_seq)
                    cm = ColorMapping(name=self.name, col_fun=cf, na_col=na_col)
                else:
                    raise ValueError("Unnamed `col` vector length does not match unique levels.")

            self.color_mapping = cm
            self.fun = anno_simple(
                norm_val,
                col=cm,
                na_col=na_col,
                which=self.which,
                border=border,
                gp=gp or gpar(col=None),
                simple_anno_size=simple_anno_size,
                width=width,
                height=height,
            )
            self.width = self.fun.width
            self.height = self.fun.height
            self.extended = self.fun.extended
            self.subsettable = self.fun.subsettable

        # Compute name placement parameters
        ext_mm = self.extended.to_mm("x")
        off_u = unit(1.0, "mm") if name_offset is None else as_unit(name_offset, "mm")
        rot = (0.0 if self.which == "column" else 90.0) if name_rot is None else float(name_rot)
        rot = rot % 360.0

        if self.which == "column":
            if name_side not in ("left", "right"):
                raise ValueError(f"{self.name}: `name_side` must be 'left' or 'right' for column annotation.")
            if name_side == "left":
                if ext_mm[1] > 0:
                    if not is_offset_called:
                        off_u = off_u + unit(ext_mm[1], "mm")
                    if not is_rot_called:
                        rot = 90.0
                if self.is_anno_matrix and self.anno_matrix_colnames:
                    nc = self.anno_matrix_k
                    nx = unit(np.zeros(nc), "npc") - off_u
                    ny = unit((nc - np.arange(nc) - 0.5) / nc, "npc")
                    lbl_txt = self.anno_matrix_colnames
                else:
                    nx = unit(0.0, "npc") - off_u
                    ny = unit(0.5, "npc")
                    lbl_txt = self.label
                just = {0: "right", 90: "bottom", 180: "left", 270: "top"}.get(rot, "right")
            else:
                if ext_mm[3] > 0:
                    if not is_offset_called:
                        off_u = off_u + unit(ext_mm[3], "mm")
                    if not is_rot_called:
                        rot = 90.0
                if self.is_anno_matrix and self.anno_matrix_colnames:
                    nc = self.anno_matrix_k
                    nx = unit(np.ones(nc), "npc") + off_u
                    ny = unit((nc - np.arange(nc) - 0.5) / nc, "npc")
                    lbl_txt = self.anno_matrix_colnames
                else:
                    nx = unit(1.0, "npc") + off_u
                    ny = unit(0.5, "npc")
                    lbl_txt = self.label
                just = {0: "left", 90: "top", 180: "right", 270: "bottom"}.get(rot, "left")
        else:
            if name_side not in ("top", "bottom"):
                raise ValueError(f"{self.name}: `name_side` must be 'top' or 'bottom' for row annotation.")
            if name_side == "top":
                if ext_mm[2] > 0:
                    if not is_offset_called:
                        off_u = off_u + unit(ext_mm[2], "mm")
                    if not is_rot_called:
                        rot = 0.0
                if self.is_anno_matrix and self.anno_matrix_colnames:
                    nc = self.anno_matrix_k
                    nx = unit((np.arange(nc) + 0.5) / nc, "npc")
                    ny = unit(np.ones(nc), "npc") + off_u
                    lbl_txt = self.anno_matrix_colnames
                else:
                    nx = unit(0.5, "npc")
                    ny = unit(1.0, "npc") + off_u
                    lbl_txt = self.label
                just = {0: "bottom", 90: "left", 180: "top", 270: "right"}.get(rot, "bottom")
            else:
                if ext_mm[0] > 0:
                    if not is_offset_called:
                        off_u = off_u + unit(ext_mm[0], "mm")
                    if not is_rot_called:
                        rot = 0.0
                if self.is_anno_matrix and self.anno_matrix_colnames:
                    nc = self.anno_matrix_k
                    nx = unit((np.arange(nc) + 0.5) / nc, "npc")
                    ny = unit(np.zeros(nc), "npc") - off_u
                    lbl_txt = self.anno_matrix_colnames
                else:
                    nx = unit(0.5, "npc")
                    ny = unit(0.0, "npc") - off_u
                    lbl_txt = self.label
                just = {0: "top", 90: "right", 180: "bottom", 270: "left"}.get(rot, "top")

        self.name_param = {
            "show": bool(show_name),
            "x": nx,
            "y": ny,
            "just": just,
            "rot": rot,
            "gp": name_gp_d,
            "side": name_side,
            "offset": off_u,
            "label": lbl_txt,
        }

    def copy_all(self) -> "SingleAnnotation":
        cp = copy.copy(self)
        cp.fun = self.fun.copy_all()
        cp.name_param = copy.deepcopy(self.name_param)
        cp.legend_param = copy.deepcopy(self.legend_param)
        return cp

    def nobs(self) -> int | None:
        return self.fun.nobs()

    @property
    def size(self) -> Unit:
        return self.width if self.which == "row" else self.height

    @size.setter
    def size(self, val: Unit) -> None:
        u = as_unit(val, "mm")
        if self.which == "row":
            self.width = u
            self.fun.width = u
        else:
            self.height = u
            self.fun.height = u

    def __getitem__(self, i: Any) -> "SingleAnnotation":
        cp = self.copy_all()
        cp.fun = cp.fun[i]
        cp.width = cp.fun.width
        cp.height = cp.fun.height
        return cp

    def draw(
        self,
        index: Sequence[int] | None = None,
        k: int = 1,
        n: int = 1,
        test: bool | str = False,
        anno_mark_param: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(test, str):
            is_test = True
            test_label = test
        else:
            is_test = bool(test)
            test_label = ""
        if is_test:
            grid_newpage()
            if self.which == "column":
                pushViewport(Viewport(width=unit(1, "npc") - unit(3, "cm"), height=self.height))
            else:
                pushViewport(Viewport(height=unit(1, "npc") - unit(3, "cm"), width=self.width))

        if index is None:
            cnt = self.fun.n if (self.fun.n is not None and self.fun.n > 0) else 1
            index = list(range(cnt))

        amp = anno_mark_param or {}
        is_mark_or_zoom = self.fun.fun_name in ("anno_mark", "anno_zoom") and bool(amp)
        if is_mark_or_zoom and k > 1:
            if is_test:
                popViewport()
            return

        if is_mark_or_zoom:
            # Draw in spanning viewport covering all slices
            self.fun.var_env[".pos"] = amp.get(".pos")
            if ".scale" in amp:
                self.fun.var_env[".scale"] = amp[".scale"]
            if self.which == "row":
                vp = Viewport(
                    x=unit(0.5, "npc"), y=amp.get("y", unit(0.5, "npc")),
                    width=self.width, height=amp.get("height", unit(1.0, "npc")),
                    just=amp.get("just", "centre"),
                    xscale=self.fun.data_scale, yscale=(0.5, len(index) + 0.5),
                    name=f"annotation_{self.name}_{k}",
                )
            else:
                vp = Viewport(
                    x=amp.get("x", unit(0.5, "npc")), y=unit(0.5, "npc"),
                    width=amp.get("width", unit(1.0, "npc")), height=self.height,
                    just=amp.get("just", "centre"),
                    xscale=(0.5, len(index) + 0.5), yscale=self.fun.data_scale,
                    name=f"annotation_{self.name}_{k}",
                )
            pushViewport(vp)
            try:
                idx_arg = amp.get("index", index)
                if self.fun._takes_env:
                    self.fun.fun(idx_arg, 1, 1, env=self.fun.var_env)
                elif self.fun._n_pos_params <= 1:
                    self.fun.fun(idx_arg)
                else:
                    self.fun.fun(idx_arg, 1, 1)
            finally:
                popViewport()
                if is_test:
                    popViewport()
            return

        m_obs = max(1, len(index))
        if self.which == "column":
            vp = Viewport(
                width=self.width, height=self.height,
                xscale=(0.5, m_obs + 0.5), yscale=self.fun.data_scale,
                name=f"annotation_{self.name}_{k}",
            )
        else:
            vp = Viewport(
                width=self.width, height=self.height,
                xscale=self.fun.data_scale, yscale=(0.5, m_obs + 0.5),
                name=f"annotation_{self.name}_{k}",
            )

        pushViewport(vp)
        try:
            if self.fun._takes_env:
                self.fun.fun(index, k, n, env=self.fun.var_env)
            elif self.fun._n_pos_params <= 1:
                self.fun.fun(index)
            elif self.fun._n_pos_params == 2:
                self.fun.fun(index, k)
            else:
                self.fun.fun(index, k, n)

            if self.name_param["show"]:
                side = self.name_param["side"]
                show_this_slice = (n == 1) or (k == 1 and side in ("left", "top")) or (k == n and side in ("right", "bottom"))
                if show_this_slice:
                    grid_text(
                        self.name_param["label"],
                        x=self.name_param["x"],
                        y=self.name_param["y"],
                        just=self.name_param["just"],
                        rot=self.name_param["rot"],
                        gp=self.name_param["gp"],
                    )
        finally:
            popViewport()
            if is_test:
                if test_label:
                    grid_text(test_label, y=unit(1, "npc") + unit(2, "mm"), just="bottom")
                popViewport()

    def _repr_png_(self) -> bytes:
        import io
        import matplotlib.pyplot as plt
        dev = grid_newpage(width_in=6.0, height_in=3.5)
        self.draw(test=self.name)
        buf = io.BytesIO()
        dev.fig.savefig(buf, format="png", dpi=120)
        plt.close(dev.fig)
        return buf.getvalue()

    def __repr__(self) -> str:
        return f"A SingleAnnotation object named {self.name!r} (which={self.which!r}, nobs={self.nobs()})"


def _recycle_param(val: Any, names: list[str], default: Any = None, as_list: bool = False) -> Any:
    n = len(names)
    if val is None:
        return [default] * n
    if isinstance(val, dict):
        return [val.get(nm, default) for nm in names]
    if isinstance(val, (str, bool, int, float, np.number, Unit)):
        if isinstance(val, Unit) and len(val) > 1:
            return [val[i] for i in range(n)]
        return [val] * n
    seq = list(val)
    if len(seq) == 1:
        return seq * n
    if len(seq) != n:
        raise ValueError(f"Parameter length ({len(seq)}) must be 1 or match annotation count ({n}).")
    return seq


class HeatmapAnnotation(AdditiveUnit):
    """Collection of row or column annotations aligned to a Heatmap."""

    def __init__(
        self,
        _annotations: dict[str, Any] | None = None,
        *,
        df: pd.DataFrame | dict[str, Any] | None = None,
        name: str | None = None,
        col: dict[str, Any] | None = None,
        na_col: str = "grey",
        annotation_legend_param: Any = None,
        show_legend: Any = True,
        which: str = "column",
        gp: dict[str, Any] | None = None,
        border: Any = None,
        gap: Any = unit(1, "points"),
        show_annotation_name: Any = True,
        annotation_label: Any = None,
        annotation_name_gp: Any = None,
        annotation_name_offset: Any = None,
        annotation_name_side: Any = None,
        annotation_name_rot: Any = None,
        annotation_name_align: bool = False,
        annotation_height: Any = None,
        annotation_width: Any = None,
        height: Any = None,
        width: Any = None,
        simple_anno_size: Unit | None = None,
        simple_anno_size_adjust: bool = False,
        **kwargs: Any,
    ) -> None:
        self.which = which.lower()
        if self.which not in ("column", "row"):
            raise ValueError("`which` must be 'column' or 'row'.")
        if name is None:
            name = f"heatmap_annotation_{next(_ENV.anno_counter)}"
        self.name = str(name)
        if simple_anno_size is None:
            simple_anno_size = ht_opt.simple_anno_size

        is_height_set = height is not None
        is_width_set = width is not None
        is_anno_h_set = annotation_height is not None
        is_anno_w_set = annotation_width is not None

        anno_values: dict[str, Any] = {}
        if _annotations:
            anno_values.update(_annotations)
        for k_arg, v_arg in kwargs.items():
            if k_arg == "anno_simple_size":
                raise ValueError("Use `simple_anno_size` instead of `anno_simple_size`.")
            anno_values[k_arg] = v_arg

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            for cname in df.columns:
                sc = str(cname)
                if sc in anno_values:
                    raise ValueError(f"Annotation name {sc!r} duplicates a column in `df`.")
                anno_values[sc] = df[cname].to_numpy()

        if not anno_values:
            raise ValueError("HeatmapAnnotation requires at least one named annotation or `df`.")

        an = list(anno_values.keys())
        n_total = len(an)
        is_simple = [not (isinstance(anno_values[k], AnnotationFunction) or callable(anno_values[k])) for k in an]
        simple_names = [nm for nm, fl in zip(an, is_simple) if fl]
        n_simple = len(simple_names)

        # Normalize show_legend for simple annotations
        if isinstance(show_legend, dict):
            sl_list = [bool(show_legend.get(nm, True)) for nm in simple_names]
        elif isinstance(show_legend, (bool, np.bool_)):
            sl_list = [bool(show_legend)] * n_simple
        else:
            seq_sl = list(show_legend)
            if len(seq_sl) == n_total and not all(is_simple):
                sl_list = [bool(s) for s, fl in zip(seq_sl, is_simple) if fl]
            elif len(seq_sl) == n_simple:
                sl_list = [bool(s) for s in seq_sl]
            else:
                sl_list = [bool(seq_sl[i % len(seq_sl)]) for i in range(n_simple)]

        # Normalize annotation_legend_param
        if not annotation_legend_param:
            alp_list: list[dict[str, Any]] = [{} for _ in range(n_simple)]
        elif isinstance(annotation_legend_param, dict):
            if annotation_legend_param and all(isinstance(v, dict) for v in annotation_legend_param.values()):
                bad = set(annotation_legend_param.keys()) - set(simple_names)
                if bad:
                    raise ValueError(f"Unknown simple annotation names in `annotation_legend_param`: {bad}")
                alp_list = [dict(annotation_legend_param.get(nm, {})) for nm in simple_names]
            else:
                alp_list = [dict(annotation_legend_param) for _ in range(n_simple)]
        else:
            seq_alp = list(annotation_legend_param)
            alp_list = [dict(seq_alp[i]) if i < len(seq_alp) else {} for i in range(n_simple)]

        if border is None:
            border = ht_opt.annotation_border if ht_opt.annotation_border is not None else False
        if annotation_name_side is None:
            annotation_name_side = "right" if self.which == "column" else "bottom"

        show_name_rec = _recycle_param(show_annotation_name, an, True)
        side_rec = _recycle_param(annotation_name_side, an, annotation_name_side)
        off_rec = _recycle_param(annotation_name_offset, an, None)
        rot_rec = _recycle_param(annotation_name_rot, an, None)
        border_rec = _recycle_param(border, an, False)
        name_gp_rec = recycle_gp(annotation_name_gp or gpar(), n_total)
        lbl_rec = _recycle_param(annotation_label, an, None)

        # Validate col dict
        col_dict = col or {}
        if col_dict and not isinstance(col_dict, dict):
            raise TypeError("`col` must be a dict mapping simple annotation names to colors/functions.")

        # Reorient flexible AnnotationFunctions to `self.which` before validating observation lengths
        for nm in an:
            val = anno_values[nm]
            if isinstance(val, AnnotationFunction) and val.which != self.which:
                anno_values[nm] = val.reorient(self.which)

        # Validate consistent observation counts
        obs_lens = []
        for nm, val in anno_values.items():
            if isinstance(val, AnnotationFunction):
                nb = val.nobs()
                if nb is not None:
                    obs_lens.append(nb)
            elif callable(val):
                pass
            else:
                arr = np.asarray(val)
                obs_lens.append(arr.shape[0])
        if len(set(obs_lens)) > 1:
            raise ValueError(f"Length of annotations differs: {obs_lens}")

        self.anno_list: dict[str, SingleAnnotation] = {}
        i_simple = 0
        for idx, nm in enumerate(an):
            val = anno_values[nm]
            common = dict(
                name=nm,
                which=self.which,
                label=lbl_rec[idx],
                show_name=bool(show_name_rec[idx]),
                name_gp=subset_gp(name_gp_rec, idx),
                name_offset=off_rec[idx],
                name_side=side_rec[idx],
                name_rot=rot_rec[idx],
                border=border_rec[idx],
            )
            if isinstance(val, AnnotationFunction) or callable(val):
                self.anno_list[nm] = SingleAnnotation(fun=val, **common)
            else:
                sa = SingleAnnotation(
                    value=val,
                    col=col_dict.get(nm),
                    na_col=na_col,
                    gp=gp or gpar(col=None),
                    show_legend=sl_list[i_simple],
                    legend_param=alp_list[i_simple],
                    simple_anno_size=simple_anno_size,
                    **common,
                )
                self.anno_list[nm] = sa
                i_simple += 1

        # Zoomed anno_empty check
        if n_total > 1:
            for sa in self.anno_list.values():
                if sa.fun.fun_name == "anno_empty" and sa.fun.var_env.get("zoom", False):
                    raise ValueError("`anno_empty(zoom=True)` must be the sole annotation in HeatmapAnnotation.")

        # Gap setup
        if gap is None or gap == 0:
            gap_u = unit(np.zeros(n_total), "mm")
        else:
            gap_raw = as_unit(gap, "points")
            if len(gap_raw) == 1:
                gap_u = unit(np.full(n_total, convertWidth(gap_raw, "mm", valueOnly=True)), "mm")
            elif len(gap_raw) == n_total - 1:
                gap_u = unit_c(gap_raw, unit(0.0, "mm"))
            elif len(gap_raw) >= n_total:
                gap_u = gap_raw[:n_total]
            else:
                raise ValueError("Length of `gap` does not match number of annotations.")
        self.gap: Unit = unit(gap_u.to_mm("x"), "mm")

        # Initial sizing
        if self.which == "column":
            sizes_mm = [convertHeight(sa.height, "mm", valueOnly=True) for sa in self.anno_list.values()]
            self.anno_size = unit(sizes_mm, "mm")
            self.height = unit(sum(sizes_mm) + sum(self.gap.to_mm("y")[:-1]), "mm") if height is None else as_unit(height, "mm")
            self.width = unit(1.0, "npc") if width is None else as_unit(width, "mm")
            for sa in self.anno_list.values():
                sa.width = self.width
                sa.fun.width = self.width
        else:
            sizes_mm = [convertWidth(sa.width, "mm", valueOnly=True) for sa in self.anno_list.values()]
            self.anno_size = unit(sizes_mm, "mm")
            self.width = unit(sum(sizes_mm) + sum(self.gap.to_mm("x")[:-1]), "mm") if width is None else as_unit(width, "mm")
            self.height = unit(1.0, "npc") if height is None else as_unit(height, "mm")
            for sa in self.anno_list.values():
                sa.height = self.height
                sa.fun.height = self.height

        if annotation_name_align:
            self._align_annotation_names()

        self.subsettable = all(sa.subsettable for sa in self.anno_list.values())
        self.extended = self._compute_extended()
        self.param = {
            "simple_anno_size": simple_anno_size,
            "simple_anno_size_adjust": simple_anno_size_adjust,
            "is_height_set": is_height_set,
            "is_width_set": is_width_set,
            "is_annotation_height_set": is_anno_h_set,
            "is_annotation_width_set": is_anno_w_set,
        }

        if self.which == "column":
            self.re_size(height=self.height if is_height_set else None, annotation_height=annotation_height)
        else:
            self.re_size(width=self.width if is_width_set else None, annotation_width=annotation_width)

    def _compute_extended(self) -> Unit:
        ext = [0.0, 0.0, 0.0, 0.0]
        for sa in self.anno_list.values():
            v = sa.extended.to_mm("x")
            for i in range(4):
                ext[i] = max(ext[i], float(v[i]))
        return unit(ext, "mm")

    def _align_annotation_names(self) -> None:
        for side, coord_key, sign, base_val in (
            ("bottom", "y", -1.0, 0.0),
            ("top", "y", 1.0, 1.0),
            ("left", "x", -1.0, 0.0),
            ("right", "x", 1.0, 1.0),
        ):
            targets = [sa for sa in self.anno_list.values() if sa.name_param["side"] == side and sa.name_param["show"] and sa.fun.fun_name != "anno_simple"]
            if len(targets) > 1:
                mx = max(convertWidth(sa.name_param["offset"], "mm", valueOnly=True) for sa in targets)
                u_mx = unit(mx, "mm")
                for sa in targets:
                    sa.name_param["offset"] = u_mx
                    sa.name_param[coord_key] = unit(base_val, "npc") + sign * u_mx

    def re_size(
        self,
        annotation_height: Any = None,
        annotation_width: Any = None,
        height: Any = None,
        width: Any = None,
        simple_anno_size: Unit | None = None,
        simple_anno_size_adjust: bool | None = None,
    ) -> "HeatmapAnnotation":
        if simple_anno_size_adjust is None:
            simple_anno_size_adjust = self.param.get("simple_anno_size_adjust", False)
        annos = list(self.anno_list.values())
        n = len(annos)
        all_simple = all(sa.fun.fun_name == "anno_simple" for sa in annos)
        gaps_mm = self.gap.to_mm("x")
        gap_sum = float(np.sum(gaps_mm[:-1])) if n > 1 else 0.0

        anno_size_arg = annotation_height if self.which == "column" else annotation_width
        tot_size_arg = height if self.which == "column" else width

        if anno_size_arg is not None:
            if isinstance(anno_size_arg, Unit):
                if is_abs_unit(anno_size_arg):
                    s_mm = anno_size_arg.to_mm("x")
                    if len(s_mm) == 1:
                        s_mm = np.repeat(s_mm[0], n)
                    for sa, v in zip(annos, s_mm):
                        sa.size = unit(float(v), "mm")
                    self.anno_size = unit(s_mm, "mm")
                    tot = float(np.sum(s_mm)) + gap_sum
                    if self.which == "column":
                        self.height = unit(tot, "mm")
                    else:
                        self.width = unit(tot, "mm")
                    return self
                else:
                    # Unit with null weights
                    weights = anno_size_arg._values if anno_size_arg._op is None else np.ones(n)
            else:
                arr = np.atleast_1d(np.asarray(anno_size_arg, dtype=float))
                weights = np.repeat(arr[0], n) if len(arr) == 1 else arr
            if tot_size_arg is None:
                tot_size_arg = self.height if self.which == "column" else self.width
            tot_mm = convertHeight(as_unit(tot_size_arg, "mm"), "mm", valueOnly=True)
            avail = max(0.0, tot_mm - gap_sum)
            s_mm = avail * (weights / float(np.sum(weights)))
            for sa, v in zip(annos, s_mm):
                sa.size = unit(float(v), "mm")
            self.anno_size = unit(s_mm, "mm")
            if self.which == "column":
                self.height = unit(tot_mm, "mm")
            else:
                self.width = unit(tot_mm, "mm")
            return self

        if tot_size_arg is None:
            return self

        if not simple_anno_size_adjust and all_simple:
            cur_mm = self.anno_size.to_mm("x")
            tot = float(np.sum(cur_mm)) + gap_sum
            if self.which == "column":
                self.height = unit(tot, "mm")
            else:
                self.width = unit(tot, "mm")
            return self

        tot_mm = convertHeight(as_unit(tot_size_arg, "mm"), "mm", valueOnly=True)
        cur_mm = self.anno_size.to_mm("x")
        if simple_anno_size_adjust or not any(sa.fun.fun_name == "anno_simple" for sa in annos):
            avail = max(0.0, tot_mm - gap_sum)
            denom = float(np.sum(cur_mm))
            s_mm = cur_mm * (avail / denom) if denom > 0 else np.full(n, avail / n)
        else:
            simple_mask = np.array([sa.fun.fun_name == "anno_simple" for sa in annos])
            fixed_mm = float(np.sum(cur_mm[simple_mask]))
            rem_avail = max(0.0, tot_mm - gap_sum - fixed_mm)
            comp_sum = float(np.sum(cur_mm[~simple_mask]))
            s_mm = cur_mm.copy()
            if comp_sum > 0:
                s_mm[~simple_mask] = cur_mm[~simple_mask] * (rem_avail / comp_sum)

        for sa, v in zip(annos, s_mm):
            sa.size = unit(float(v), "mm")
        self.anno_size = unit(s_mm, "mm")
        if self.which == "column":
            self.height = unit(tot_mm, "mm")
        else:
            self.width = unit(tot_mm, "mm")
        return self

    @property
    def size(self) -> Unit:
        return self.width if self.which == "row" else self.height

    @size.setter
    def size(self, val: Unit) -> None:
        if self.which == "row":
            self.re_size(width=val, simple_anno_size_adjust=True)
        else:
            self.re_size(height=val, simple_anno_size_adjust=True)

    @property
    def names(self) -> list[str]:
        return list(self.anno_list.keys())

    @names.setter
    def names(self, value: Sequence[str]) -> None:
        new_nms = [str(v) for v in value]
        if len(new_nms) != len(self.anno_list):
            raise ValueError("Length of new names must match number of annotations.")
        if len(set(new_nms)) != len(new_nms):
            raise ValueError("Annotation names must be unique.")
        new_dict: dict[str, SingleAnnotation] = {}
        for old_sa, nm in zip(self.anno_list.values(), new_nms):
            if old_sa.label == old_sa.name:
                old_sa.label = nm
            old_sa.name = nm
            old_sa.name_param["label"] = nm
            if old_sa.color_mapping is not None:
                old_sa.color_mapping.name = nm
            new_dict[nm] = old_sa
        self.anno_list = new_dict

    def __len__(self) -> int:
        return len(self.anno_list)

    def nobs(self) -> int | None:
        for sa in self.anno_list.values():
            n = sa.nobs()
            if n is not None:
                return n
        return None

    def copy_all(self) -> "HeatmapAnnotation":
        cp = copy.copy(self)
        cp.anno_list = {k: sa.copy_all() for k, sa in self.anno_list.items()}
        cp.param = dict(self.param)
        return cp

    def get_color_mapping_list(self) -> list[ColorMapping]:
        return [sa.color_mapping for sa in self.anno_list.values() if sa.show_legend and sa.color_mapping is not None]

    def get_legend_param_list(self) -> list[dict[str, Any]]:
        return [sa.legend_param for sa in self.anno_list.values() if sa.show_legend and sa.color_mapping is not None]

    def __getitem__(self, key: Any) -> "HeatmapAnnotation":
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("HeatmapAnnotation expects 1D or 2D indexing.")
            obs_idx, anno_idx = key
        else:
            obs_idx, anno_idx = key, slice(None)

        cp = self.copy_all()
        all_names = list(cp.anno_list.keys())
        if isinstance(anno_idx, slice):
            sel_indices = list(range(*anno_idx.indices(len(all_names))))
        elif isinstance(anno_idx, str):
            sel_indices = [all_names.index(anno_idx)]
        elif isinstance(anno_idx, (int, np.integer)):
            sel_indices = [int(anno_idx)]
        else:
            sel_indices = []
            for item in anno_idx:
                if isinstance(item, str):
                    sel_indices.append(all_names.index(item))
                elif isinstance(item, (bool, np.bool_)):
                    pass
                else:
                    sel_indices.append(int(item))
            if not sel_indices and all(isinstance(x, (bool, np.bool_)) for x in anno_idx):
                sel_indices = [k for k, fl in enumerate(anno_idx) if fl]

        sel_names = [all_names[k] for k in sel_indices]
        cp.anno_list = {nm: cp.anno_list[nm] for nm in sel_names}
        gaps_mm = self.gap.to_mm("x")
        new_gaps = [gaps_mm[k] for k in sel_indices]
        if new_gaps:
            new_gaps[-1] = 0.0
        cp.gap = unit(new_gaps, "mm")

        if not (isinstance(obs_idx, slice) and obs_idx == slice(None)):
            for nm in sel_names:
                cp.anno_list[nm] = cp.anno_list[nm][obs_idx]

        sizes_mm = [convertWidth(sa.size, "mm", valueOnly=True) for sa in cp.anno_list.values()]
        cp.anno_size = unit(sizes_mm, "mm")
        tot_mm = sum(sizes_mm) + (sum(new_gaps[:-1]) if len(new_gaps) > 1 else 0.0)
        if cp.which == "column":
            cp.height = unit(tot_mm, "mm")
        else:
            cp.width = unit(tot_mm, "mm")
        cp.extended = cp._compute_extended()
        cp.subsettable = all(sa.subsettable for sa in cp.anno_list.values())
        return cp

    def draw(
        self,
        index: Sequence[int] | None = None,
        k: int = 1,
        n: int = 1,
        test: bool | str = False,
        anno_mark_param: dict[str, Any] | None = None,
        **vp_kwargs: Any,
    ) -> None:
        if isinstance(test, str):
            is_test = True
            test_label = test
        else:
            is_test = bool(test)
            test_label = ""

        if is_test:
            grid_newpage()
            ext = self.extended.to_mm("x")
            if self.which == "column":
                pushViewport(Viewport(width=unit(1, "npc") - unit(3, "cm") - unit(ext[1] + ext[3], "mm"), height=self.height))
            else:
                pushViewport(Viewport(height=unit(1, "npc") - unit(3, "cm") - unit(ext[0] + ext[2], "mm"), width=self.width))
        else:
            pushViewport(Viewport(**vp_kwargs))

        try:
            if index is None:
                nb = self.nobs()
                if nb is None:
                    raise ValueError("Cannot infer number of observations for HeatmapAnnotation.")
                index = list(range(nb))

            annos = list(self.anno_list.values())
            n_anno = len(annos)
            sizes_mm = self.anno_size.to_mm("x")
            gaps_mm = self.gap.to_mm("x")

            if self.which == "column":
                for i in range(n_anno):
                    y_top = float(np.sum(sizes_mm[i:]) + np.sum(gaps_mm[i:]) - gaps_mm[-1])
                    pushViewport(Viewport(y=unit(y_top, "mm"), height=unit(float(sizes_mm[i]), "mm"), just=("center", "top")))
                    try:
                        annos[i].draw(index, k=k, n=n, anno_mark_param=anno_mark_param)
                    finally:
                        popViewport()
            else:
                for i in range(n_anno):
                    x_right = float(np.sum(sizes_mm[: i + 1]) + np.sum(gaps_mm[: i + 1]) - gaps_mm[i])
                    pushViewport(Viewport(x=unit(x_right, "mm"), width=unit(float(sizes_mm[i]), "mm"), just=("right", "center")))
                    try:
                        annos[i].draw(index, k=k, n=n, anno_mark_param=anno_mark_param)
                    finally:
                        popViewport()

            if is_test and test_label:
                grid_text(test_label, y=unit(1, "npc") + unit(2, "mm"), just="bottom")
        finally:
            popViewport()

    def _repr_png_(self) -> bytes:
        import io
        import matplotlib.pyplot as plt
        dev = grid_newpage(width_in=7.0, height_in=3.5)
        self.draw(test=self.name)
        buf = io.BytesIO()
        dev.fig.savefig(buf, format="png", dpi=120)
        plt.close(dev.fig)
        return buf.getvalue()

    def __repr__(self) -> str:
        lines = [
            f"A HeatmapAnnotation object with {len(self.anno_list)} annotation(s)",
            f"  name: {self.name}",
            f"  position: {self.which}",
            f"  items: {self.nobs() or 'unknown'}",
            f"  width: {self.width}",
            f"  height: {self.height}",
            f"  annotations: {list(self.anno_list.keys())}",
        ]
        return "\n".join(lines)


def rowAnnotation(*args: Any, **kwargs: Any) -> HeatmapAnnotation:
    """Create a row HeatmapAnnotation (`which='row'`)."""
    kwargs.setdefault("which", "row")
    if args:
        if len(args) == 1 and isinstance(args[0], dict):
            return HeatmapAnnotation(_annotations=args[0], **kwargs)
        raise TypeError("rowAnnotation accepts named keyword arguments or a single dict.")
    return HeatmapAnnotation(**kwargs)


def columnAnnotation(*args: Any, **kwargs: Any) -> HeatmapAnnotation:
    """Create a column HeatmapAnnotation (`which='column'`)."""
    kwargs.setdefault("which", "column")
    if args:
        if len(args) == 1 and isinstance(args[0], dict):
            return HeatmapAnnotation(_annotations=args[0], **kwargs)
        raise TypeError("columnAnnotation accepts named keyword arguments or a single dict.")
    return HeatmapAnnotation(**kwargs)


def c_HeatmapAnnotation(*annos: HeatmapAnnotation | None, gap: Any = unit(1, "points")) -> HeatmapAnnotation | None:
    """Concatenate multiple `HeatmapAnnotation` objects with identical `which`."""
    items = [a for a in annos if a is not None]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    whiches = {a.which for a in items}
    if len(whiches) != 1:
        raise ValueError("All concatenated HeatmapAnnotations must have the same `which`.")

    gap_u = as_unit(gap, "points")
    n = len(items)
    if len(gap_u) == 1:
        g_vals = [convertWidth(gap_u, "mm", valueOnly=True)] * n
    elif len(gap_u) == n - 1:
        g_vals = list(gap_u.to_mm("x")) + [0.0]
    else:
        g_vals = list(gap_u.to_mm("x")[:n])

    out = items[0].copy_all()
    all_sizes = list(out.anno_size.to_mm("x"))
    all_gaps = list(out.gap.to_mm("x"))
    all_gaps[-1] = g_vals[0]

    existing_names = list(out.anno_list.keys())
    new_list = dict(out.anno_list)

    for idx in range(1, n):
        other = items[idx].copy_all()
        o_gaps = list(other.gap.to_mm("x"))
        o_gaps[-1] = g_vals[idx]
        all_sizes.extend(other.anno_size.to_mm("x"))
        all_gaps.extend(o_gaps)
        for nm, sa in other.anno_list.items():
            new_nm = nm
            if new_nm in new_list:
                suffix = 2
                while f"{nm}_{suffix}" in new_list:
                    suffix += 1
                new_nm = f"{nm}_{suffix}"
                if sa.label == sa.name:
                    sa.label = new_nm
                sa.name = new_nm
                sa.name_param["label"] = new_nm
                if sa.color_mapping is not None:
                    sa.color_mapping.name = new_nm
            new_list[new_nm] = sa

    all_gaps[-1] = 0.0
    out.anno_list = new_list
    out.anno_size = unit(all_sizes, "mm")
    out.gap = unit(all_gaps, "mm")
    tot = sum(all_sizes) + (sum(all_gaps[:-1]) if len(all_gaps) > 1 else 0.0)
    if out.which == "column":
        out.height = unit(tot, "mm")
    else:
        out.width = unit(tot, "mm")
    out.extended = out._compute_extended()
    out.subsettable = all(sa.subsettable for sa in out.anno_list.values())
    return out


from .anno_funcs import (  # noqa: E402, F401
    anno_barplot,
    anno_block,
    anno_boxplot,
    anno_customize,
    anno_density,
    anno_empty,
    anno_histogram,
    anno_horizon,
    anno_image,
    anno_joyplot,
    anno_lines,
    anno_link,
    anno_mark,
    anno_numeric,
    anno_oncoprint_barplot,
    anno_points,
    anno_simple,
    anno_summary,
    anno_text,
    anno_textbox,
    anno_zoom,
    row_anno_barplot,
    row_anno_boxplot,
    row_anno_density,
    row_anno_histogram,
    row_anno_points,
    row_anno_text,
)
