"""Matplotlib-backed Grid Graphics, Viewports, Grobs, and Primitives."""

from __future__ import annotations

from collections import OrderedDict
import copy
import math
from typing import Any, Callable, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np

from .colors import grid_pretty, pretty_breaks, to_hex_rgba, to_rgba_tuple
from .units import (
    ABS_UNITS,
    Gpar,
    Unit,
    _MM_PER_BIGPT,
    _MM_PER_INCH,
    _text_dims_mm,
    as_unit,
    check_gp,
    gpar,
    recycle_gp,
    subset_gp,
    unit,
)

# Justification helper
_JUST_MAP = {
    "center": (0.5, 0.5),
    "centre": (0.5, 0.5),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "bottom": (0.5, 0.0),
    "top": (0.5, 1.0),
}


def parse_just(just: Any) -> tuple[float, float]:
    if just is None:
        return (0.5, 0.5)
    if isinstance(just, str):
        k = just.strip().lower()
        if k in _JUST_MAP:
            return _JUST_MAP[k]
        raise ValueError(f"Unknown just value: {just!r}")
    if isinstance(just, (int, float, np.number)):
        v = float(just)
        return (v, v)
    if isinstance(just, (list, tuple, np.ndarray)):
        vals = list(just)
        if len(vals) == 1:
            return parse_just(vals[0])
        if len(vals) >= 2:
            jx = _JUST_MAP[vals[0].lower()][0] if isinstance(vals[0], str) else float(vals[0])
            jy = _JUST_MAP[vals[1].lower()][1] if isinstance(vals[1], str) else float(vals[1])
            return (jx, jy)
    return (0.5, 0.5)


def _just_to_ha_va(jx: float, jy: float) -> tuple[str, str]:
    ha = "left" if jx < 0.25 else ("right" if jx > 0.75 else "center")
    va = "bottom" if jy < 0.25 else ("top" if jy > 0.75 else "center")
    return ha, va


def _linestyle_for_mpl(lty: Any) -> Any:
    if lty is None or lty == "solid" or lty == 1:
        return "-"
    if lty == "dashed" or lty == 2:
        return (0, (4, 4))
    if lty == "dotted" or lty == 3:
        return (0, (1, 2))
    if lty == "dotdash" or lty == 4:
        return (0, (1, 2, 4, 2))
    if lty == "longdash" or lty == 5:
        return (0, (7, 3))
    if lty == "twodash" or lty == 6:
        return (0, (2, 2, 6, 2))
    if lty in ("none", "blank", 0):
        return "None"
    return lty


def _unwrap_scalar(v: Any) -> Any:
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
        return v[0]
    return v


def _font_props(gp: dict[str, Any]) -> tuple[float, str, str, str]:
    fs = float(_unwrap_scalar(gp.get("fontsize", 12.0))) * float(_unwrap_scalar(gp.get("cex", 1.0)))
    ff = _unwrap_scalar(gp.get("fontfamily", "sans-serif")) or "sans-serif"
    face = _unwrap_scalar(gp.get("fontface", "plain"))
    if face in (2, "bold"):
        weight, style = "bold", "normal"
    elif face in (3, "italic"):
        weight, style = "normal", "italic"
    elif face in (4, "bold.italic", "bold_italic", "bold italic"):
        weight, style = "bold", "italic"
    elif face == "oblique":
        weight, style = "normal", "oblique"
    else:
        weight, style = "normal", "normal"
    return fs, str(ff), weight, style


class Viewport:
    """Grid viewport representing a rectangular millimeter region and native coordinate scales."""

    def __init__(
        self,
        x: Any = None,
        y: Any = None,
        width: Any = None,
        height: Any = None,
        just: Any = "centre",
        xscale: Sequence[float] = (0.0, 1.0),
        yscale: Sequence[float] = (0.0, 1.0),
        clip: bool | str = "inherit",
        name: str | None = None,
        gp: dict[str, Any] | None = None,
        layout: Any = None,
        layout_pos_row: int | tuple[int, int] | None = None,
        layout_pos_col: int | tuple[int, int] | None = None,
    ) -> None:
        self.x = unit(0.5, "npc") if x is None else as_unit(x, "npc")
        self.y = unit(0.5, "npc") if y is None else as_unit(y, "npc")
        self.width = unit(1.0, "npc") if width is None else as_unit(width, "npc")
        self.height = unit(1.0, "npc") if height is None else as_unit(height, "npc")
        self.just = just
        self.xscale = (float(xscale[0]), float(xscale[1]))
        self.yscale = (float(yscale[0]), float(yscale[1]))
        self.clip = clip
        self.name = name
        self.gp = check_gp(gp)
        self.layout = layout
        self.layout_pos_row = layout_pos_row
        self.layout_pos_col = layout_pos_col

        # Resolved geometry in figure millimeters (populated upon pushViewport)
        self.x0_mm: float = 0.0
        self.y0_mm: float = 0.0
        self.w_mm: float = 200.0
        self.h_mm: float = 200.0
        self.clip_rect_mm: tuple[float, float, float, float] | None = None

    @property
    def width_mm(self) -> float:
        return self.w_mm

    @property
    def height_mm(self) -> float:
        return self.h_mm


def viewport(
    x: Any = None,
    y: Any = None,
    width: Any = None,
    height: Any = None,
    just: Any = "centre",
    xscale: Sequence[float] = (0.0, 1.0),
    yscale: Sequence[float] = (0.0, 1.0),
    clip: bool | str = "inherit",
    name: str | None = None,
    gp: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Viewport:
    return Viewport(
        x=x,
        y=y,
        width=width,
        height=height,
        just=just,
        xscale=xscale,
        yscale=yscale,
        clip=clip,
        name=name,
        gp=gp,
        **kwargs,
    )


class Grob:
    """Base graphical object with intrinsic millimeter size."""

    def __init__(self, name: str | None = None, gp: dict[str, Any] | None = None, vp: Viewport | None = None) -> None:
        self.name = name
        self.gp = check_gp(gp)
        self.vp = vp

    def width_mm(self, vp: Viewport | None = None) -> float:
        return 0.0

    def height_mm(self, vp: Viewport | None = None) -> float:
        return 0.0

    def draw(self, device: "GridDevice | None" = None) -> None:
        raise NotImplementedError


class TextGrob(Grob):
    def __init__(
        self,
        label: Any,
        x: Any = None,
        y: Any = None,
        just: Any = "centre",
        rot: float = 0.0,
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
        name: str | None = None,
        vp: Viewport | None = None,
    ) -> None:
        super().__init__(name=name, gp=gp, vp=vp)
        self.label = label
        self.x = unit(0.5, "npc") if x is None else as_unit(x, default_units)
        self.y = unit(0.5, "npc") if y is None else as_unit(y, default_units)
        self.just = just
        self.rot = float(rot)

    def width_mm(self, vp: Viewport | None = None) -> float:
        from .text import GridText, _richtext_dims_mm

        if isinstance(self.label, GridText):
            w, _ = _richtext_dims_mm(self.label[0] if len(self.label) else "", self.gp, self.label.param, self.rot)
            return w
        w, _ = _text_dims_mm(str(self.label), self.gp, self.rot)
        return w

    def height_mm(self, vp: Viewport | None = None) -> float:
        from .text import GridText, _richtext_dims_mm

        if isinstance(self.label, GridText):
            _, h = _richtext_dims_mm(self.label[0] if len(self.label) else "", self.gp, self.label.param, self.rot)
            return h
        _, h = _text_dims_mm(str(self.label), self.gp, self.rot)
        return h

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            dev.draw_text(self.label, x=self.x, y=self.y, just=self.just, rot=self.rot, gp=self.gp)
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


class RectGrob(Grob):
    def __init__(
        self,
        x: Any = None,
        y: Any = None,
        width: Any = None,
        height: Any = None,
        just: Any = "centre",
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
        name: str | None = None,
        vp: Viewport | None = None,
    ) -> None:
        super().__init__(name=name, gp=gp, vp=vp)
        self.x = unit(0.5, "npc") if x is None else as_unit(x, default_units)
        self.y = unit(0.5, "npc") if y is None else as_unit(y, default_units)
        self.width = unit(1.0, "npc") if width is None else as_unit(width, default_units)
        self.height = unit(1.0, "npc") if height is None else as_unit(height, default_units)
        self.just = just

    def width_mm(self, vp: Viewport | None = None) -> float:
        return float(np.max(self.width.to_mm("x", vp)))

    def height_mm(self, vp: Viewport | None = None) -> float:
        return float(np.max(self.height.to_mm("y", vp)))

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            dev.draw_rect(x=self.x, y=self.y, width=self.width, height=self.height, just=self.just, gp=self.gp)
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


class RoundRectGrob(RectGrob):
    def __init__(self, *args: Any, r: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.r = unit(0.1, "snpc") if r is None else as_unit(r, "snpc")

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            dev.draw_roundrect(
                x=self.x, y=self.y, width=self.width, height=self.height,
                r=self.r, just=self.just, gp=self.gp,
            )
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


class LinesGrob(Grob):
    def __init__(self, x: Any, y: Any, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> None:
        super().__init__(name=name, gp=gp, vp=vp)
        self.x = as_unit(x, default_units)
        self.y = as_unit(y, default_units)

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            dev.draw_lines(self.x, self.y, gp=self.gp)
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


class SegmentsGrob(Grob):
    def __init__(self, x0: Any, y0: Any, x1: Any, y1: Any, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> None:
        super().__init__(name=name, gp=gp, vp=vp)
        self.x0 = as_unit(x0, default_units)
        self.y0 = as_unit(y0, default_units)
        self.x1 = as_unit(x1, default_units)
        self.y1 = as_unit(y1, default_units)

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            dev.draw_segments(self.x0, self.y0, self.x1, self.y1, gp=self.gp)
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


class PolygonGrob(Grob):
    def __init__(self, x: Any, y: Any, id: Any = None, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> None:
        super().__init__(name=name, gp=gp, vp=vp)
        self.x = as_unit(x, default_units)
        self.y = as_unit(y, default_units)
        self.id = id

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            dev.draw_polygon(self.x, self.y, id=self.id, gp=self.gp)
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


class PointsGrob(Grob):
    def __init__(self, x: Any, y: Any, pch: Any = 1, size: Any = None, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> None:
        super().__init__(name=name, gp=gp, vp=vp)
        self.x = as_unit(x, default_units)
        self.y = as_unit(y, default_units)
        self.pch = pch
        self.size = unit(2.0, "mm") if size is None else as_unit(size, "mm")

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            dev.draw_points(self.x, self.y, pch=self.pch, size=self.size, gp=self.gp)
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


class GTree(Grob):
    """Composite container of child Grobs inside an optional viewport."""

    def __init__(
        self,
        children: Sequence[Grob] = (),
        vp: Viewport | None = None,
        cl: str | None = None,
        width: Unit | None = None,
        height: Unit | None = None,
        name: str | None = None,
        gp: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, gp=gp, vp=vp)
        self.children: list[Grob] = [c for c in children if c is not None]
        self.cl = cl
        self._width = width
        self._height = height

    def width_mm(self, vp: Viewport | None = None) -> float:
        if self._width is not None:
            return self._width.to_mm_scalar("x", vp)
        if self.vp is not None and self.vp.width is not None:
            return self.vp.width.to_mm_scalar("x", vp)
        return max((c.width_mm(vp) for c in self.children), default=0.0)

    def height_mm(self, vp: Viewport | None = None) -> float:
        if self._height is not None:
            return self._height.to_mm_scalar("y", vp)
        if self.vp is not None and self.vp.height is not None:
            return self.vp.height.to_mm_scalar("y", vp)
        return max((c.height_mm(vp) for c in self.children), default=0.0)

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        if self.vp is not None:
            dev.push_viewport(self.vp)
        try:
            for ch in self.children:
                ch.draw(dev)
        finally:
            if self.vp is not None:
                dev.pop_viewport(1)


def _scale_abs_unit(u: Unit, factor: float) -> Unit:
    """Scale absolute-unit subexpressions by `factor` while leaving relative (`npc`, `native`, `snpc`) intact."""
    if abs(factor - 1.0) < 1e-6:
        return u
    if u._op is None:
        if u._unit in ABS_UNITS:
            return Unit(u._values * factor, u._unit, _refs=u._refs)
        return u
    if u._op in ("add", "sub"):
        left = _scale_abs_unit(u._left, factor)
        right = _scale_abs_unit(u._right, factor)
        return Unit(_op=u._op, _left=left, _right=right)
    if u._op in ("mul", "div"):
        return Unit(_op=u._op, _left=_scale_abs_unit(u._left, factor), _scalar=u._scalar)
    if u._op in ("concat", "pmax", "pmin", "max", "min"):
        return Unit(_op=u._op, _items=[_scale_abs_unit(it, factor) for it in (u._items or [])])
    if u._op == "sum":
        return Unit(_op="sum", _left=_scale_abs_unit(u._left, factor))
    return u


def _scale_coord(val: Any, default_units: str, factor: float) -> Any:
    if val is None or abs(factor - 1.0) < 1e-6:
        return val
    return _scale_abs_unit(as_unit(val, default_units), factor)


class CapturedGrob(Grob):
    """Replayable grob captured via `grid.grabExpr`."""

    def __init__(self, commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]], width_mm: float, height_mm: float) -> None:
        super().__init__()
        self._commands = commands
        self._w_mm = float(width_mm)
        self._h_mm = float(height_mm)

    def width_mm(self, vp: Viewport | None = None) -> float:
        return self._w_mm

    def height_mm(self, vp: Viewport | None = None) -> float:
        return self._h_mm

    def draw(self, device: "GridDevice | None" = None) -> None:
        dev = device or current_device()
        cur = dev.current_vp
        sx = cur.w_mm / self._w_mm if self._w_mm > 0 else 1.0
        sy = cur.h_mm / self._h_mm if self._h_mm > 0 else 1.0
        smin = min(sx, sy)
        wrapper = Viewport(
            x=unit(0.5, "npc"), y=unit(0.5, "npc"),
            width=unit(1.0, "npc"), height=unit(1.0, "npc"),
            clip=cur.clip,
        )
        dev.push_viewport(wrapper)
        base_depth = len(dev.vp_stack)
        try:
            for kind, args, kwargs in self._commands:
                if kind == "push_vp":
                    vp_copy = copy.deepcopy(args[0])
                    vp_copy.x = _scale_abs_unit(vp_copy.x, sx)
                    vp_copy.y = _scale_abs_unit(vp_copy.y, sy)
                    vp_copy.width = _scale_abs_unit(vp_copy.width, sx)
                    vp_copy.height = _scale_abs_unit(vp_copy.height, sy)
                    if vp_copy.gp and abs(smin - 1.0) > 1e-6:
                        if "fontsize" in vp_copy.gp and isinstance(vp_copy.gp["fontsize"], (int, float, np.number)):
                            vp_copy.gp["fontsize"] = float(vp_copy.gp["fontsize"]) * smin
                    dev.push_viewport(vp_copy)
                elif kind == "pop_vp":
                    n = min(args[0], len(dev.vp_stack) - base_depth)
                    if n > 0:
                        dev.pop_viewport(n)
                elif kind == "seek_vp":
                    if args[0] in dev.named_viewports:
                        dev.seek_viewport(args[0])
                elif kind == "rect":
                    x, y, width, height, just, gp, du = args
                    dev.draw_rect(
                        x=_scale_coord(x, du, sx), y=_scale_coord(y, du, sy),
                        width=_scale_coord(width, du, sx), height=_scale_coord(height, du, sy),
                        just=just, gp=gp, default_units=du, **kwargs,
                    )
                elif kind == "roundrect":
                    x, y, width, height, r, just, gp, du = args
                    dev.draw_roundrect(
                        x=_scale_coord(x, du, sx), y=_scale_coord(y, du, sy),
                        width=_scale_coord(width, du, sx), height=_scale_coord(height, du, sy),
                        r=_scale_coord(r, "snpc", smin), just=just, gp=gp, default_units=du, **kwargs,
                    )
                elif kind == "lines":
                    x, y, gp, du = args
                    dev.draw_lines(_scale_coord(x, du, sx), _scale_coord(y, du, sy), gp=gp, default_units=du, **kwargs)
                elif kind == "segments":
                    x0, y0, x1, y1, gp, du = args
                    dev.draw_segments(
                        _scale_coord(x0, du, sx), _scale_coord(y0, du, sy),
                        _scale_coord(x1, du, sx), _scale_coord(y1, du, sy),
                        gp=gp, default_units=du, **kwargs,
                    )
                elif kind == "polygon":
                    x, y, gid, gp, du = args
                    dev.draw_polygon(_scale_coord(x, du, sx), _scale_coord(y, du, sy), id=gid, gp=gp, default_units=du, **kwargs)
                elif kind == "points":
                    x, y, pch, sz, gp, du = args
                    dev.draw_points(
                        _scale_coord(x, du, sx), _scale_coord(y, du, sy),
                        pch=pch, size=_scale_coord(sz, "mm", smin), gp=gp, default_units=du, **kwargs,
                    )
                elif kind == "circle":
                    x, y, r, gp, du = args
                    dev.draw_circle(
                        x=_scale_coord(x, du, sx), y=_scale_coord(y, du, sy),
                        r=_scale_coord(r, du, smin), gp=gp, default_units=du, **kwargs,
                    )
                elif kind == "text":
                    lbl, x, y, just, rot, gp, du = args
                    dev.draw_text(lbl, x=_scale_coord(x, du, sx), y=_scale_coord(y, du, sy), just=just, rot=rot, gp=gp, default_units=du, **kwargs)
                elif kind == "raster":
                    img, x, y, width, height, just, interp, du = args
                    dev.draw_raster(
                        img, x=_scale_coord(x, du, sx), y=_scale_coord(y, du, sy),
                        width=_scale_coord(width, du, sx), height=_scale_coord(height, du, sy),
                        just=just, interpolate=interp, default_units=du, **kwargs,
                    )
        finally:
            while len(dev.vp_stack) > base_depth:
                dev.pop_viewport(1)
            dev.pop_viewport(1)


class GridDevice:
    """Active Matplotlib figure + viewport stack acting as an R grid graphics device."""

    def __init__(
        self,
        width_in: float = 8.0,
        height_in: float = 8.0,
        dpi: float = 100.0,
        fig: matplotlib.figure.Figure | None = None,
        record: bool = False,
    ) -> None:
        self.dpi = float(dpi)
        self.record = record
        self.commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        if fig is None:
            self.fig = plt.figure(figsize=(width_in, height_in), dpi=self.dpi)
        else:
            self.fig = fig
            self.fig.clf()
            width_in, height_in = self.fig.get_size_inches()
        self.w_mm = float(width_in) * _MM_PER_INCH
        self.h_mm = float(height_in) * _MM_PER_INCH
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.set_xlim(0.0, self.w_mm)
        self.ax.set_ylim(0.0, self.h_mm)
        self.ax.set_axis_off()
        self.ax.set_aspect("auto")

        root = Viewport(
            x=unit(0.5, "npc"), y=unit(0.5, "npc"),
            width=unit(self.w_mm, "mm"), height=unit(self.h_mm, "mm"),
            clip=False, name="ROOT",
        )
        root.x0_mm = 0.0
        root.y0_mm = 0.0
        root.w_mm = self.w_mm
        root.h_mm = self.h_mm
        root.clip_rect_mm = None

        self.vp_stack: list[Viewport] = [root]
        self.named_viewports: OrderedDict[str, Viewport] = OrderedDict()

    @property
    def current_vp(self) -> Viewport:
        return self.vp_stack[-1]

    @property
    def width_mm(self) -> float:
        return self.w_mm

    @property
    def height_mm(self) -> float:
        return self.h_mm

    def resize(self, width_in: float, height_in: float) -> None:
        self.fig.set_size_inches(width_in, height_in, forward=True)
        self.w_mm = float(width_in) * _MM_PER_INCH
        self.h_mm = float(height_in) * _MM_PER_INCH
        self.ax.set_xlim(0.0, self.w_mm)
        self.ax.set_ylim(0.0, self.h_mm)
        root = self.vp_stack[0]
        root.w_mm = self.w_mm
        root.h_mm = self.h_mm
        root.width = unit(self.w_mm, "mm")
        root.height = unit(self.h_mm, "mm")

    def _clip_bbox(self, vp: Viewport) -> Any:
        if vp.clip_rect_mm is None:
            return None
        x0, y0, w, h = vp.clip_rect_mm
        if w <= 0 or h <= 0:
            return TransformedBbox(Bbox.from_bounds(x0, y0, 0.001, 0.001), self.ax.transData)
        return TransformedBbox(Bbox.from_bounds(x0, y0, w, h), self.ax.transData)

    def push_viewport(self, vp: Viewport) -> Viewport:
        if self.record:
            self.commands.append(("push_vp", (vp,), {}))
        parent = self.current_vp
        vp_copy = copy.copy(vp)
        if not vp_copy.gp:
            vp_copy.gp = parent.gp.copy()
        else:
            merged = parent.gp.copy()
            merged.update(vp_copy.gp)
            vp_copy.gp = merged

        w_mm = vp_copy.width.to_mm_scalar("x", parent, pos_mode=False)
        h_mm = vp_copy.height.to_mm_scalar("y", parent, pos_mode=False)
        ax_mm = parent.x0_mm + vp_copy.x.to_mm_scalar("x", parent, pos_mode=True)
        ay_mm = parent.y0_mm + vp_copy.y.to_mm_scalar("y", parent, pos_mode=True)
        jx, jy = parse_just(vp_copy.just)
        vp_copy.w_mm = w_mm
        vp_copy.h_mm = h_mm
        vp_copy.x0_mm = ax_mm - jx * w_mm
        vp_copy.y0_mm = ay_mm - jy * h_mm

        if vp_copy.clip is True or vp_copy.clip == "on":
            vp_copy.clip_rect_mm = (vp_copy.x0_mm, vp_copy.y0_mm, vp_copy.w_mm, vp_copy.h_mm)
        elif vp_copy.clip is False or vp_copy.clip == "off":
            vp_copy.clip_rect_mm = None
        else:  # "inherit"
            vp_copy.clip_rect_mm = parent.clip_rect_mm

        self.vp_stack.append(vp_copy)
        if vp_copy.name:
            self.named_viewports[vp_copy.name] = vp_copy
        return vp_copy

    def pop_viewport(self, n: int = 1) -> None:
        if self.record:
            self.commands.append(("pop_vp", (n,), {}))
        if n == 0:
            self.vp_stack = self.vp_stack[:1]
            return
        for _ in range(n):
            if len(self.vp_stack) > 1:
                self.vp_stack.pop()

    def seek_viewport(self, name: str) -> Viewport:
        if self.record:
            self.commands.append(("seek_vp", (name,), {}))
        if name == "ROOT":
            self.vp_stack = self.vp_stack[:1]
            return self.current_vp
        if name not in self.named_viewports:
            raise KeyError(f"Viewport {name!r} not found.")
        target = self.named_viewports[name]
        # Replace top with a clean stack [root, target]
        self.vp_stack = [self.vp_stack[0], target]
        return target

    def _xy_mm(self, x: Any, y: Any, default_units: str = "npc") -> tuple[np.ndarray, np.ndarray]:
        vp = self.current_vp
        ux = as_unit(x, default_units)
        uy = as_unit(y, default_units)
        xmm = vp.x0_mm + ux.to_mm("x", vp, pos_mode=True)
        ymm = vp.y0_mm + uy.to_mm("y", vp, pos_mode=True)
        xmm, ymm = np.broadcast_arrays(xmm, ymm)
        return np.atleast_1d(xmm), np.atleast_1d(ymm)

    def draw_rect(
        self,
        x: Any = None,
        y: Any = None,
        width: Any = None,
        height: Any = None,
        just: Any = "centre",
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("rect", (x, y, width, height, just, gp, default_units), {}))
        vp = self.current_vp
        ux = unit(0.5, "npc") if x is None else as_unit(x, default_units)
        uy = unit(0.5, "npc") if y is None else as_unit(y, default_units)
        uw = unit(1.0, "npc") if width is None else as_unit(width, default_units)
        uh = unit(1.0, "npc") if height is None else as_unit(height, default_units)

        xmm = vp.x0_mm + ux.to_mm("x", vp, pos_mode=True)
        ymm = vp.y0_mm + uy.to_mm("y", vp, pos_mode=True)
        wmm = uw.to_mm("x", vp, pos_mode=False)
        hmm = uh.to_mm("y", vp, pos_mode=False)
        xmm, ymm, wmm, hmm = np.broadcast_arrays(xmm, ymm, wmm, hmm)
        n = len(xmm)
        if n == 0:
            return

        jx, jy = parse_just(just)
        x0 = xmm - jx * wmm
        y0 = ymm - jy * hmm

        gp_n = recycle_gp(gp or {}, n)
        fills = gp_n.get("fill", ["#00000000"] * n)
        # Note: in R grid, default rect border col is "black" if unspecified,
        # unless gp has col=NA / col=None. When gp is empty gpar(), R draws black outline & transparent fill!
        has_col = "col" in (gp or {})
        cols = gp_n.get("col", ["#000000FF"] * n if has_col or "fill" not in (gp or {}) else ["#000000FF"] * n)
        if not has_col and "fill" in (gp or {}):
            # In R grid, grid.rect(gp = gpar(fill="red")) still has black default col unless col=NA.
            # Keep black border when col is omitted, matching R grid!
            cols = ["#000000FF"] * n
        lwds = gp_n.get("lwd", [1.0] * n)
        ltys = gp_n.get("lty", ["solid"] * n)
        alphas = gp_n.get("alpha", [None] * n)

        verts = []
        fc = []
        ec = []
        lw_list = []
        ls_list = []
        for k in range(n):
            xk, yk, wk, hk = float(x0[k]), float(y0[k]), float(wmm[k]), float(hmm[k])
            if wk < 0:
                xk += wk
                wk = -wk
            if hk < 0:
                yk += hk
                hk = -hk
            verts.append([(xk, yk), (xk + wk, yk), (xk + wk, yk + hk), (xk, yk + hk)])
            f_rgba = list(to_rgba_tuple(fills[k]))
            c_rgba = list(to_rgba_tuple(cols[k]))
            if alphas[k] is not None:
                f_rgba[3] *= float(alphas[k])
                c_rgba[3] *= float(alphas[k])
            fc.append(tuple(f_rgba))
            ec.append(tuple(c_rgba))
            lwd_k = float(lwds[k]) * 0.75
            if c_rgba[3] == 0.0 or ltys[k] in ("none", "blank", 0):
                lwd_k = 0.0
            lw_list.append(lwd_k)
            ls_list.append(_linestyle_for_mpl(ltys[k]))

        pc = PolyCollection(verts, facecolors=fc, edgecolors=ec, linewidths=lw_list, linestyles=ls_list, antialiaseds=True)
        cb = self._clip_bbox(vp)
        if cb is not None:
            pc.set_clip_path(cb.bounds if False else None)
            pc.set_clip_box(cb)
        self.ax.add_collection(pc)

    def draw_roundrect(
        self,
        x: Any = None,
        y: Any = None,
        width: Any = None,
        height: Any = None,
        r: Any = None,
        just: Any = "centre",
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("roundrect", (x, y, width, height, r, just, gp, default_units), {}))
        vp = self.current_vp
        ux = unit(0.5, "npc") if x is None else as_unit(x, default_units)
        uy = unit(0.5, "npc") if y is None else as_unit(y, default_units)
        uw = unit(1.0, "npc") if width is None else as_unit(width, default_units)
        uh = unit(1.0, "npc") if height is None else as_unit(height, default_units)
        ur = unit(0.1, "snpc") if r is None else as_unit(r, "snpc")

        xmm = vp.x0_mm + ux.to_mm("x", vp, pos_mode=True)
        ymm = vp.y0_mm + uy.to_mm("y", vp, pos_mode=True)
        wmm = uw.to_mm("x", vp, pos_mode=False)
        hmm = uh.to_mm("y", vp, pos_mode=False)
        rmm = ur.to_mm("x", vp, pos_mode=False)
        xmm, ymm, wmm, hmm, rmm = np.broadcast_arrays(xmm, ymm, wmm, hmm, rmm)
        jx, jy = parse_just(just)
        n = len(xmm)
        gp_n = recycle_gp(gp or {}, n)
        fills = gp_n.get("fill", ["#00000000"] * n)
        cols = gp_n.get("col", ["#000000FF"] * n)
        lwds = gp_n.get("lwd", [1.0] * n)
        cb = self._clip_bbox(vp)
        for k in range(n):
            wk, hk = float(wmm[k]), float(hmm[k])
            xk = float(xmm[k]) - jx * wk
            yk = float(ymm[k]) - jy * hk
            if wk < 0:
                xk += wk
                wk = -wk
            if hk < 0:
                yk += hk
                hk = -hk
            rk = min(float(rmm[k]), wk * 0.5, hk * 0.5)
            c_rgba = to_rgba_tuple(cols[k])
            f_rgba = to_rgba_tuple(fills[k])
            patch = FancyBboxPatch(
                (xk, yk), wk, hk,
                boxstyle=f"round,pad=0,rounding_size={max(0.0, rk)}",
                facecolor=f_rgba,
                edgecolor=c_rgba,
                linewidth=float(lwds[k]) * 0.75 if c_rgba[3] > 0 else 0.0,
            )
            if cb is not None:
                patch.set_clip_box(cb)
            self.ax.add_patch(patch)

    def draw_lines(
        self,
        x: Any,
        y: Any,
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("lines", (x, y, gp, default_units), {}))
        xmm, ymm = self._xy_mm(x, y, default_units)
        if len(xmm) < 2:
            return
        gp_d = check_gp(gp)
        col_val = _unwrap_scalar(gp_d.get("col", "#000000FF"))
        col = to_rgba_tuple(col_val)
        if col[3] == 0.0:
            return
        lwd = float(_unwrap_scalar(gp_d.get("lwd", 1.0))) * 0.75
        ls = _linestyle_for_mpl(_unwrap_scalar(gp_d.get("lty", "solid")))
        alpha = _unwrap_scalar(gp_d.get("alpha", None))
        if alpha is not None:
            col = (col[0], col[1], col[2], col[3] * float(alpha))
        # Split on NaN
        valid = np.isfinite(xmm) & np.isfinite(ymm)
        if np.all(valid):
            line = Line2D(xmm, ymm, color=col, linewidth=lwd, linestyle=ls, solid_capstyle="butt")
            cb = self._clip_bbox(self.current_vp)
            if cb is not None:
                line.set_clip_box(cb)
            self.ax.add_line(line)
        else:
            # Draw contiguous runs
            i = 0
            n = len(xmm)
            cb = self._clip_bbox(self.current_vp)
            while i < n:
                while i < n and not valid[i]:
                    i += 1
                j = i
                while j < n and valid[j]:
                    j += 1
                if j - i >= 2:
                    line = Line2D(xmm[i:j], ymm[i:j], color=col, linewidth=lwd, linestyle=ls)
                    if cb is not None:
                        line.set_clip_box(cb)
                    self.ax.add_line(line)
                i = j

    def draw_segments(
        self,
        x0: Any,
        y0: Any,
        x1: Any,
        y1: Any,
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("segments", (x0, y0, x1, y1, gp, default_units), {}))
        vp = self.current_vp
        x0_mm = vp.x0_mm + as_unit(x0, default_units).to_mm("x", vp, pos_mode=True)
        y0_mm = vp.y0_mm + as_unit(y0, default_units).to_mm("y", vp, pos_mode=True)
        x1_mm = vp.x0_mm + as_unit(x1, default_units).to_mm("x", vp, pos_mode=True)
        y1_mm = vp.y0_mm + as_unit(y1, default_units).to_mm("y", vp, pos_mode=True)
        x0_mm, y0_mm, x1_mm, y1_mm = np.broadcast_arrays(x0_mm, y0_mm, x1_mm, y1_mm)
        n = len(x0_mm)
        if n == 0:
            return
        gp_n = recycle_gp(gp or {}, n)
        cols = gp_n.get("col", ["#000000FF"] * n)
        lwds = gp_n.get("lwd", [1.0] * n)
        ltys = gp_n.get("lty", ["solid"] * n)
        segs = []
        colors = []
        widths = []
        styles = []
        for k in range(n):
            c = to_rgba_tuple(cols[k])
            if c[3] == 0.0 or ltys[k] in ("none", "blank", 0):
                continue
            segs.append([(float(x0_mm[k]), float(y0_mm[k])), (float(x1_mm[k]), float(y1_mm[k]))])
            colors.append(c)
            widths.append(float(lwds[k]) * 0.75)
            styles.append(_linestyle_for_mpl(ltys[k]))
        if not segs:
            return
        lc = LineCollection(segs, colors=colors, linewidths=widths, linestyles=styles)
        cb = self._clip_bbox(vp)
        if cb is not None:
            lc.set_clip_box(cb)
        self.ax.add_collection(lc)

    def draw_polygon(
        self,
        x: Any,
        y: Any,
        id: Any = None,
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("polygon", (x, y, id, gp, default_units), {}))
        xmm, ymm = self._xy_mm(x, y, default_units)
        if len(xmm) < 3:
            return
        if id is None:
            groups = [np.arange(len(xmm))]
        else:
            id_arr = np.asarray(id)
            seen: OrderedDict[Any, list[int]] = OrderedDict()
            for idx, gid in enumerate(id_arr):
                seen.setdefault(gid, []).append(idx)
            groups = [np.asarray(idxs, dtype=int) for idxs in seen.values()]

        ng = len(groups)
        gp_n = recycle_gp(gp or {}, ng)
        fills = gp_n.get("fill", ["#00000000"] * ng)
        cols = gp_n.get("col", ["#000000FF"] * ng)
        lwds = gp_n.get("lwd", [1.0] * ng)
        ltys = gp_n.get("lty", ["solid"] * ng)

        verts = []
        fc = []
        ec = []
        lws = []
        lss = []
        for k, idxs in enumerate(groups):
            pts = list(zip(xmm[idxs], ymm[idxs]))
            verts.append(pts)
            f_c = to_rgba_tuple(fills[k])
            e_c = to_rgba_tuple(cols[k])
            fc.append(f_c)
            ec.append(e_c)
            lws.append(float(lwds[k]) * 0.75 if e_c[3] > 0 else 0.0)
            lss.append(_linestyle_for_mpl(ltys[k]))

        pc = PolyCollection(verts, facecolors=fc, edgecolors=ec, linewidths=lws, linestyles=lss, antialiaseds=True)
        cb = self._clip_bbox(self.current_vp)
        if cb is not None:
            pc.set_clip_box(cb)
        self.ax.add_collection(pc)

    def draw_circle(
        self,
        x: Any = None,
        y: Any = None,
        r: Any = None,
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("circle", (x, y, r, gp, default_units), {}))
        vp = self.current_vp
        ux = unit(0.5, "npc") if x is None else as_unit(x, default_units)
        uy = unit(0.5, "npc") if y is None else as_unit(y, default_units)
        ur = unit(0.5, "snpc") if r is None else as_unit(r, default_units)
        xmm = vp.x0_mm + ux.to_mm("x", vp, pos_mode=True)
        ymm = vp.y0_mm + uy.to_mm("y", vp, pos_mode=True)
        rmm = ur.to_mm("x", vp, pos_mode=False)
        xmm, ymm, rmm = np.broadcast_arrays(xmm, ymm, rmm)
        n = len(xmm)
        gp_n = recycle_gp(gp or {}, n)
        fills = gp_n.get("fill", ["#00000000"] * n)
        cols = gp_n.get("col", ["#000000FF"] * n)
        lwds = gp_n.get("lwd", [1.0] * n)
        cb = self._clip_bbox(vp)
        for k in range(n):
            f_c = to_rgba_tuple(fills[k])
            e_c = to_rgba_tuple(cols[k])
            patch = Circle(
                (float(xmm[k]), float(ymm[k])),
                radius=max(0.0, float(rmm[k])),
                facecolor=f_c,
                edgecolor=e_c,
                linewidth=float(lwds[k]) * 0.75 if e_c[3] > 0 else 0.0,
            )
            if cb is not None:
                patch.set_clip_box(cb)
            self.ax.add_patch(patch)

    def draw_points(
        self,
        x: Any,
        y: Any,
        pch: Any = 1,
        size: Any = None,
        gp: dict[str, Any] | None = None,
        default_units: str = "native",
    ) -> None:
        if self.record:
            self.commands.append(("points", (x, y, pch, size, gp, default_units), {}))
        vp = self.current_vp
        xmm, ymm = self._xy_mm(x, y, default_units)
        n = len(xmm)
        if n == 0:
            return
        sz_u = unit(1.0, "char") if size is None else as_unit(size, "mm")
        sz_mm = np.broadcast_to(sz_u.to_mm("x", vp, pos_mode=False), (n,))
        if isinstance(pch, (int, np.integer, str)) or pch is None:
            pchs = [pch] * n
        else:
            p_list = list(pch)
            pchs = [p_list[i % len(p_list)] for i in range(n)]

        gp_n = recycle_gp(gp or {}, n)
        cols = gp_n.get("col", ["#000000FF"] * n)
        fills = gp_n.get("fill", ["#00000000"] * n)
        lwds = gp_n.get("lwd", [1.0] * n)
        cb = self._clip_bbox(vp)

        # Group simple markers or draw per-point accurately in mm
        for k in range(n):
            xk, yk, sk = float(xmm[k]), float(ymm[k]), float(sz_mm[k])
            if not (math.isfinite(xk) and math.isfinite(yk)):
                continue
            pk = pchs[k]
            if pk is None or (isinstance(pk, float) and math.isnan(pk)):
                continue
            col_k = to_rgba_tuple(cols[k])
            fill_k = to_rgba_tuple(fills[k])
            lwd_k = float(lwds[k]) * 0.75

            if isinstance(pk, str):
                fs = max(4.0, sk / _MM_PER_BIGPT)
                txt = self.ax.text(xk, yk, pk, fontsize=fs, color=col_k, ha="center", va="center")
                if cb is not None:
                    txt.set_clip_box(cb)
                continue

            code = int(pk)
            r = sk * 0.45
            if code in (26, 27, 28):
                # Custom ComplexHeatmap diagonal markers
                segs = []
                if code in (26, 28):
                    segs.append([(xk - r, yk - r), (xk + r, yk + r)])
                if code in (27, 28):
                    segs.append([(xk - r, yk + r), (xk + r, yk - r)])
                lc = LineCollection(segs, colors=[col_k] * len(segs), linewidths=lwd_k)
                if cb is not None:
                    lc.set_clip_box(cb)
                self.ax.add_collection(lc)
                continue

            # Matplotlib scatter uses area in pt^2
            s_pt = max(1.0, sk / _MM_PER_BIGPT)
            if code == 20:
                s_pt *= 0.67
            area = s_pt ** 2

            filled_no_edge = {15: "s", 16: "o", 17: "^", 18: "D", 19: "o", 20: "o"}
            hollow_map = {0: "s", 1: "o", 2: "^", 5: "D", 6: "v"}
            line_map = {3: "+", 4: "x", 8: "*"}
            filled_edge_map = {21: "o", 22: "s", 23: "D", 24: "^", 25: "v"}

            if code in filled_no_edge:
                sc = self.ax.scatter([xk], [yk], s=area, marker=filled_no_edge[code], c=[col_k], edgecolors="none")
                if cb is not None:
                    sc.set_clip_box(cb)
            elif code in hollow_map:
                sc = self.ax.scatter([xk], [yk], s=area, marker=hollow_map[code], facecolors="none", edgecolors=[col_k], linewidths=lwd_k)
                if cb is not None:
                    sc.set_clip_box(cb)
            elif code in line_map:
                sc = self.ax.scatter([xk], [yk], s=area, marker=line_map[code], c=[col_k], linewidths=lwd_k)
                if cb is not None:
                    sc.set_clip_box(cb)
            elif code in filled_edge_map:
                sc = self.ax.scatter([xk], [yk], s=area, marker=filled_edge_map[code], facecolors=[fill_k], edgecolors=[col_k], linewidths=lwd_k)
                if cb is not None:
                    sc.set_clip_box(cb)
            else:
                # Composite markers (7, 9, 10, 11, 12, 13, 14)
                base = {7: "s", 9: "D", 10: "o", 12: "s", 13: "o", 14: "s"}.get(code, "o")
                sc = self.ax.scatter([xk], [yk], s=area, marker=base, facecolors="none", edgecolors=[col_k], linewidths=lwd_k)
                if cb is not None:
                    sc.set_clip_box(cb)
                extra_segs = []
                if code in (7, 13):
                    extra_segs = [[(xk - r, yk - r), (xk + r, yk + r)], [(xk - r, yk + r), (xk + r, yk - r)]]
                elif code in (9, 10, 12):
                    extra_segs = [[(xk - r, yk), (xk + r, yk)], [(xk, yk - r), (xk, yk + r)]]
                elif code == 11:
                    sc2 = self.ax.scatter([xk], [yk], s=area, marker="^", facecolors="none", edgecolors=[col_k], linewidths=lwd_k)
                    sc3 = self.ax.scatter([xk], [yk], s=area, marker="v", facecolors="none", edgecolors=[col_k], linewidths=lwd_k)
                    if cb is not None:
                        sc2.set_clip_box(cb)
                        sc3.set_clip_box(cb)
                if extra_segs:
                    lc = LineCollection(extra_segs, colors=[col_k] * len(extra_segs), linewidths=lwd_k)
                    if cb is not None:
                        lc.set_clip_box(cb)
                    self.ax.add_collection(lc)

    def draw_text(
        self,
        label: Any,
        x: Any = None,
        y: Any = None,
        just: Any = "centre",
        rot: float = 0.0,
        gp: dict[str, Any] | None = None,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("text", (label, x, y, just, rot, gp, default_units), {}))
        from .text import GridText, draw_richtext

        if isinstance(label, GridText):
            draw_richtext(self, label, x=x, y=y, just=just, rot=rot, gp=gp, default_units=default_units)
            return

        vp = self.current_vp
        labels = [label] if isinstance(label, (str, int, float, np.number)) or label is None else list(label)
        n = len(labels)
        if n == 0:
            return
        ux = unit(0.5, "npc") if x is None else as_unit(x, default_units)
        uy = unit(0.5, "npc") if y is None else as_unit(y, default_units)
        xmm = vp.x0_mm + ux.to_mm("x", vp, pos_mode=True)
        ymm = vp.y0_mm + uy.to_mm("y", vp, pos_mode=True)
        xmm, ymm = np.broadcast_arrays(xmm, ymm, np.zeros(n))[:2]
        jx, jy = parse_just(just)
        ha, va = _just_to_ha_va(jx, jy)
        gp_n = recycle_gp(gp or {}, n)
        cb = self._clip_bbox(vp)
        for k in range(n):
            lbl = labels[k]
            if lbl is None or (isinstance(lbl, float) and math.isnan(lbl)):
                continue
            gpk = subset_gp(gp_n, k)
            col = to_rgba_tuple(gpk.get("col", "#000000FF"))
            if col[3] == 0.0:
                continue
            fs, ff, weight, style = _font_props(gpk)
            t = self.ax.text(
                float(xmm[k]),
                float(ymm[k]),
                str(lbl),
                fontsize=fs,
                fontfamily=ff,
                fontweight=weight,
                fontstyle=style,
                color=col,
                rotation=float(rot),
                rotation_mode="anchor",
                ha=ha,
                va=va,
            )
            if cb is not None:
                t.set_clip_box(cb)

    def draw_raster(
        self,
        image: np.ndarray,
        x: Any = None,
        y: Any = None,
        width: Any = None,
        height: Any = None,
        just: Any = "centre",
        interpolate: bool = False,
        default_units: str = "npc",
    ) -> None:
        if self.record:
            self.commands.append(("raster", (image, x, y, width, height, just, interpolate, default_units), {}))
        vp = self.current_vp
        ux = unit(0.5, "npc") if x is None else as_unit(x, default_units)
        uy = unit(0.5, "npc") if y is None else as_unit(y, default_units)
        uw = unit(1.0, "npc") if width is None else as_unit(width, default_units)
        uh = unit(1.0, "npc") if height is None else as_unit(height, default_units)
        cx = vp.x0_mm + ux.to_mm("x", vp, pos_mode=True)[0]
        cy = vp.y0_mm + uy.to_mm("y", vp, pos_mode=True)[0]
        w = uw.to_mm("x", vp, pos_mode=False)[0]
        h = uh.to_mm("y", vp, pos_mode=False)[0]
        jx, jy = parse_just(just)
        x0 = cx - jx * w
        y0 = cy - jy * h
        im = self.ax.imshow(
            image,
            extent=(x0, x0 + w, y0, y0 + h),
            aspect="auto",
            interpolation="bilinear" if interpolate else "nearest",
            origin="upper",
        )
        cb = self._clip_bbox(vp)
        if cb is not None:
            im.set_clip_box(cb)


_DEVICE_STACK: list[GridDevice] = []


def current_device(create: bool = True) -> GridDevice:
    if not _DEVICE_STACK:
        if not create:
            raise RuntimeError("No active GridDevice.")
        _DEVICE_STACK.append(GridDevice())
    return _DEVICE_STACK[-1]


def set_current_device(dev: GridDevice) -> None:
    if not _DEVICE_STACK:
        _DEVICE_STACK.append(dev)
    else:
        _DEVICE_STACK[-1] = dev


def grid_newpage(width_in: float = 8.0, height_in: float = 8.0, dpi: float = 100.0, fig: matplotlib.figure.Figure | None = None) -> GridDevice:
    dev = GridDevice(width_in=width_in, height_in=height_in, dpi=dpi, fig=fig)
    set_current_device(dev)
    return dev


def pushViewport(vp: Viewport) -> Viewport:
    return current_device().push_viewport(vp)


def popViewport(n: int = 1) -> None:
    current_device().pop_viewport(n)


def upViewport(n: int = 1) -> None:
    current_device().pop_viewport(n)


def seekViewport(name: str) -> Viewport:
    return current_device().seek_viewport(name)


def current_viewport() -> Viewport:
    return current_device().current_vp


def list_components(pattern: str | None = None) -> list[str]:
    """List named viewports on the active device, optionally filtered by regex."""
    import re

    dev = current_device(create=False) if _DEVICE_STACK else None
    if dev is None:
        return []
    names = [nm for nm in dev.named_viewports.keys() if not re.fullmatch(r"\d+", nm) and not nm.startswith("GRID.VP")]
    if pattern is not None:
        rx = re.compile(pattern)
        names = [nm for nm in names if rx.search(nm)]
    return names


def getXY_in_parent_vp(u: dict[str, Unit] | tuple[Unit, Unit]) -> dict[str, Unit]:
    """Convert point `(x, y)` in the current viewport into parent viewport's `mm` units."""
    dev = current_device()
    if len(dev.vp_stack) < 2:
        raise RuntimeError("Current viewport is ROOT; no parent viewport.")
    cur = dev.vp_stack[-1]
    parent = dev.vp_stack[-2]
    if isinstance(u, dict):
        ux, uy = as_unit(u["x"], "npc"), as_unit(u["y"], "npc")
    else:
        ux, uy = as_unit(u[0], "npc"), as_unit(u[1], "npc")
    px = (cur.x0_mm + ux.to_mm("x", cur, pos_mode=True)) - parent.x0_mm
    py = (cur.y0_mm + uy.to_mm("y", cur, pos_mode=True)) - parent.y0_mm
    return {"x": unit(px[0] if len(px) == 1 else px, "mm"), "y": unit(py[0] if len(py) == 1 else py, "mm")}


def textGrob(label: Any, x: Any = None, y: Any = None, just: Any = "centre", rot: float = 0.0, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> TextGrob:
    return TextGrob(label=label, x=x, y=y, just=just, rot=rot, gp=gp, default_units=default_units, name=name, vp=vp)


def rectGrob(x: Any = None, y: Any = None, width: Any = None, height: Any = None, just: Any = "centre", gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> RectGrob:
    return RectGrob(x=x, y=y, width=width, height=height, just=just, gp=gp, default_units=default_units, name=name, vp=vp)


def roundrectGrob(x: Any = None, y: Any = None, width: Any = None, height: Any = None, r: Any = None, just: Any = "centre", gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> RoundRectGrob:
    return RoundRectGrob(x=x, y=y, width=width, height=height, r=r, just=just, gp=gp, default_units=default_units, name=name, vp=vp)


def linesGrob(x: Any, y: Any, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> LinesGrob:
    return LinesGrob(x=x, y=y, gp=gp, default_units=default_units, name=name, vp=vp)


def segmentsGrob(x0: Any, y0: Any, x1: Any, y1: Any, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> SegmentsGrob:
    return SegmentsGrob(x0=x0, y0=y0, x1=x1, y1=y1, gp=gp, default_units=default_units, name=name, vp=vp)


def polygonGrob(x: Any, y: Any, id: Any = None, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> PolygonGrob:
    return PolygonGrob(x=x, y=y, id=id, gp=gp, default_units=default_units, name=name, vp=vp)


def pointsGrob(x: Any, y: Any, pch: Any = 1, size: Any = None, gp: dict[str, Any] | None = None, default_units: str = "npc", name: str | None = None, vp: Viewport | None = None) -> PointsGrob:
    return PointsGrob(x=x, y=y, pch=pch, size=size, gp=gp, default_units=default_units, name=name, vp=vp)


def grobWidth(grob: Grob) -> Unit:
    return Unit([1.0], "grobwidth", _refs=[grob])


def grobHeight(grob: Grob) -> Unit:
    return Unit([1.0], "grobheight", _refs=[grob])


def stringWidth(text: str, gp: dict[str, Any] | None = None, rot: float = 0.0) -> Unit:
    return Unit([1.0], "strwidth", _refs=[(text, gp or {}, rot)])


def stringHeight(text: str, gp: dict[str, Any] | None = None, rot: float = 0.0) -> Unit:
    return Unit([1.0], "strheight", _refs=[(text, gp or {}, rot)])


def max_text_width(text: Any, gp: dict[str, Any] | None = None, rot: float = 0.0) -> Unit:
    """Maximum rendered width of text items in mm."""
    if text is None:
        return unit(0.0, "mm")
    from .text import GridText

    if isinstance(text, GridText):
        n = len(text)
        if n == 0:
            return unit(0.0, "mm")
        gp_n = recycle_gp(gp or {}, n)
        mx = 0.0
        for i in range(n):
            g = textGrob(text[i], gp=subset_gp(gp_n, i), rot=rot)
            mx = max(mx, g.width_mm())
        return unit(mx, "mm")
    items = [text] if isinstance(text, (str, int, float, np.number)) else list(text)
    if not items:
        return unit(0.0, "mm")
    gp_n = recycle_gp(gp or {}, len(items))
    mx = max(textGrob(items[i], gp=subset_gp(gp_n, i), rot=rot).width_mm() for i in range(len(items)))
    return unit(mx, "mm")


def max_text_height(text: Any, gp: dict[str, Any] | None = None, rot: float = 0.0) -> Unit:
    """Maximum rendered height of text items in mm."""
    if text is None:
        return unit(0.0, "mm")
    from .text import GridText

    if isinstance(text, GridText):
        n = len(text)
        if n == 0:
            return unit(0.0, "mm")
        gp_n = recycle_gp(gp or {}, n)
        mx = 0.0
        for i in range(n):
            g = textGrob(text[i], gp=subset_gp(gp_n, i), rot=rot)
            mx = max(mx, g.height_mm())
        return unit(mx, "mm")
    items = [text] if isinstance(text, (str, int, float, np.number)) else list(text)
    if not items:
        return unit(0.0, "mm")
    gp_n = recycle_gp(gp or {}, len(items))
    mx = max(textGrob(items[i], gp=subset_gp(gp_n, i), rot=rot).height_mm() for i in range(len(items)))
    return unit(mx, "mm")


def grid_rect(x: Any = None, y: Any = None, width: Any = None, height: Any = None, just: Any = "centre", gp: dict[str, Any] | None = None, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_rect(x=x, y=y, width=width, height=height, just=just, gp=gp, default_units=default_units)


def grid_roundrect(x: Any = None, y: Any = None, width: Any = None, height: Any = None, r: Any = None, just: Any = "centre", gp: dict[str, Any] | None = None, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_roundrect(x=x, y=y, width=width, height=height, r=r, just=just, gp=gp, default_units=default_units)


def grid_lines(x: Any, y: Any, gp: dict[str, Any] | None = None, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_lines(x=x, y=y, gp=gp, default_units=default_units)


def grid_segments(x0: Any, y0: Any, x1: Any, y1: Any, gp: dict[str, Any] | None = None, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_segments(x0=x0, y0=y0, x1=x1, y1=y1, gp=gp, default_units=default_units)


def grid_polygon(x: Any, y: Any, id: Any = None, gp: dict[str, Any] | None = None, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_polygon(x=x, y=y, id=id, gp=gp, default_units=default_units)


def grid_circle(x: Any = None, y: Any = None, r: Any = None, gp: dict[str, Any] | None = None, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_circle(x=x, y=y, r=r, gp=gp, default_units=default_units)


def grid_points(x: Any, y: Any, pch: Any = 1, size: Any = None, gp: dict[str, Any] | None = None, default_units: str = "native", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_points(x=x, y=y, pch=pch, size=size, gp=gp, default_units=default_units)


def grid_text(label: Any, x: Any = None, y: Any = None, just: Any = "centre", rot: float = 0.0, gp: dict[str, Any] | None = None, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_text(label=label, x=x, y=y, just=just, rot=rot, gp=gp, default_units=default_units)


def grid_raster(image: np.ndarray, x: Any = None, y: Any = None, width: Any = None, height: Any = None, just: Any = "centre", interpolate: bool = False, default_units: str = "npc", **kwargs: Any) -> None:
    default_units = kwargs.pop("default.units", default_units)
    current_device().draw_raster(image=image, x=x, y=y, width=width, height=height, just=just, interpolate=interpolate, default_units=default_units)


def grid_xaxis(at: Sequence[float] | None = None, label: Sequence[str] | bool = True, main: bool = True, gp: dict[str, Any] | None = None) -> None:
    vp = current_viewport()
    scale = vp.xscale
    ticks = grid_pretty(scale) if at is None else np.asarray(at, dtype=float)
    y_pos = 0.0 if main else 1.0
    tick_dy = unit(-1.5 if main else 1.5, "mm")
    lbl_dy = unit(-2.5 if main else 2.5, "mm")
    grid_lines([scale[0], scale[1]], [y_pos, y_pos], gp=gp, default_units="native")
    # Actually y is npc, x is native:
    grid_segments(unit(scale[0], "native"), unit(y_pos, "npc"), unit(scale[1], "native"), unit(y_pos, "npc"), gp=gp)
    if len(ticks) > 0:
        grid_segments(unit(ticks, "native"), unit(y_pos, "npc"), unit(ticks, "native"), unit(y_pos, "npc") + tick_dy, gp=gp)
        if label is not False:
            lbls = [f"{v:g}" for v in ticks] if label is True else [str(v) for v in label]
            grid_text(lbls, unit(ticks, "native"), unit(y_pos, "npc") + lbl_dy, just="top" if main else "bottom", gp=gp)


def grid_yaxis(at: Sequence[float] | None = None, label: Sequence[str] | bool = True, main: bool = True, gp: dict[str, Any] | None = None) -> None:
    vp = current_viewport()
    scale = vp.yscale
    ticks = grid_pretty(scale) if at is None else np.asarray(at, dtype=float)
    x_pos = 0.0 if main else 1.0
    tick_dx = unit(-1.5 if main else 1.5, "mm")
    lbl_dx = unit(-2.5 if main else 2.5, "mm")
    grid_segments(unit(x_pos, "npc"), unit(scale[0], "native"), unit(x_pos, "npc"), unit(scale[1], "native"), gp=gp)
    if len(ticks) > 0:
        grid_segments(unit(x_pos, "npc"), unit(ticks, "native"), unit(x_pos, "npc") + tick_dx, unit(ticks, "native"), gp=gp)
        if label is not False:
            lbls = [f"{v:g}" for v in ticks] if label is True else [str(v) for v in label]
            grid_text(lbls, unit(x_pos, "npc") + lbl_dx, unit(ticks, "native"), just="right" if main else "left", gp=gp)


def grid_draw(obj: Any, **kwargs: Any) -> None:
    if hasattr(obj, "draw") and callable(obj.draw):
        # Check if it's a Legends or Grob object
        obj.draw(**kwargs) if not isinstance(obj, Grob) else obj.draw()
    else:
        raise TypeError(f"Cannot draw object of type {type(obj).__name__}.")


def grid_grabExpr(expr_fn: Callable[[], Any], width_in: float = 8.0, height_in: float = 8.0) -> CapturedGrob:
    """Execute `expr_fn()` on an offscreen recording GridDevice and return a replayable CapturedGrob."""
    rec = GridDevice(width_in=width_in, height_in=height_in, record=True)
    _DEVICE_STACK.append(rec)
    try:
        expr_fn()
    finally:
        _DEVICE_STACK.pop()
        plt.close(rec.fig)
    return CapturedGrob(rec.commands, width_mm=rec.w_mm, height_mm=rec.h_mm)


def grid_boxplot(
    value: Sequence[float],
    pos: float = 0.5,
    outline: bool = True,
    box_width: float = 0.6,
    pch: Any = 1,
    size: Any = None,
    gp: dict[str, Any] | None = None,
    direction: str = "vertical",
) -> None:
    """Draw a single boxplot at `pos` in native coordinates inside the current viewport."""
    arr = np.asarray(value, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    in_whisk = arr[(arr >= lo_fence) & (arr <= hi_fence)]
    w_lo = float(np.min(in_whisk)) if len(in_whisk) else float(q1)
    w_hi = float(np.max(in_whisk)) if len(in_whisk) else float(q3)
    outliers = arr[(arr < w_lo) | (arr > w_hi)]

    gp_d = gpar(fill="#CCCCCC", col="black")
    if gp:
        gp_d.update(check_gp(gp))
    line_gp = subset_gp(gp_d, 0)
    line_gp.pop("fill", None)
    sz = unit(2.0, "mm") if size is None else as_unit(size, "mm")
    hw = 0.5 * float(box_width)

    if direction == "vertical":
        grid_rect(x=pos, y=q1, width=box_width, height=max(0.0, q3 - q1), just="bottom", gp=gp_d, default_units="native")
        grid_segments([pos - hw, pos, pos, pos - hw * 0.5, pos - hw * 0.5], [med, q3, q1, w_hi, w_lo],
                      [pos + hw, pos, pos, pos + hw * 0.5, pos + hw * 0.5], [med, w_hi, w_lo, w_hi, w_lo],
                      gp=line_gp, default_units="native")
        if outline and len(outliers) > 0:
            grid_points(np.full(len(outliers), pos), outliers, pch=pch, size=sz, gp=line_gp, default_units="native")
    else:
        grid_rect(x=q1, y=pos, width=max(0.0, q3 - q1), height=box_width, just="left", gp=gp_d, default_units="native")
        grid_segments([med, q3, q1, w_hi, w_lo], [pos - hw, pos, pos, pos - hw * 0.5, pos - hw * 0.5],
                      [med, w_hi, w_lo, w_hi, w_lo], [pos + hw, pos, pos, pos + hw * 0.5, pos + hw * 0.5],
                      gp=line_gp, default_units="native")
        if outline and len(outliers) > 0:
            grid_points(outliers, np.full(len(outliers), pos), pch=pch, size=sz, gp=line_gp, default_units="native")


class _GridNamespace:
    """R-style `grid.*` namespace object (supports `grid.rect(...)`, `grid.newpage()`, etc.)."""

    newpage = staticmethod(grid_newpage)
    rect = staticmethod(grid_rect)
    roundrect = staticmethod(grid_roundrect)
    lines = staticmethod(grid_lines)
    segments = staticmethod(grid_segments)
    polygon = staticmethod(grid_polygon)
    circle = staticmethod(grid_circle)
    points = staticmethod(grid_points)
    text = staticmethod(grid_text)
    raster = staticmethod(grid_raster)
    xaxis = staticmethod(grid_xaxis)
    yaxis = staticmethod(grid_yaxis)
    draw = staticmethod(grid_draw)
    grabExpr = staticmethod(grid_grabExpr)
    boxplot = staticmethod(grid_boxplot)
    pretty = staticmethod(grid_pretty)
    ls = staticmethod(list_components)
    viewport = staticmethod(viewport)
    pushViewport = staticmethod(pushViewport)
    popViewport = staticmethod(popViewport)
    upViewport = staticmethod(upViewport)
    seekViewport = staticmethod(seekViewport)
    current_viewport = staticmethod(current_viewport)

    @property
    def dendrogram(self):
        from .dendrogram import grid_dendrogram
        return grid_dendrogram

    @property
    def textbox(self):
        from .text import grid_textbox
        return grid_textbox

    @property
    def annotation_axis(self):
        from .annotations import grid_annotation_axis
        return grid_annotation_axis


grid = _GridNamespace()
