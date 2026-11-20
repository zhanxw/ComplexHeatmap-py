"""Grid-compatible Unit system and graphical parameters (`gpar`)."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

import numpy as np

from .colors import to_hex_rgba

# Unit conversion factors to millimeters
_POINTS_PER_INCH = 72.27
_BIGPTS_PER_INCH = 72.0
_MM_PER_INCH = 25.4
_MM_PER_PT = _MM_PER_INCH / _POINTS_PER_INCH
_MM_PER_BIGPT = _MM_PER_INCH / _BIGPTS_PER_INCH

_UNIT_ALIASES: dict[str, str] = {
    "mm": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "in": "inches",
    "inch": "inches",
    "inches": "inches",
    "pt": "points",
    "point": "points",
    "points": "points",
    "bigpts": "bigpts",
    "bigpt": "bigpts",
    "picas": "picas",
    "pica": "picas",
    "dida": "dida",
    "cicero": "cicero",
    "scaledpts": "scaledpts",
    "line": "lines",
    "lines": "lines",
    "myline": "lines",
    "mylines": "lines",
    "char": "char",
    "mychar": "char",
    "npc": "npc",
    "snpc": "snpc",
    "native": "native",
    "null": "null",
    "grobwidth": "grobwidth",
    "grobheight": "grobheight",
    "strwidth": "strwidth",
    "mystrwidth": "strwidth",
    "strheight": "strheight",
    "mystrheight": "strheight",
    "strascent": "strascent",
    "strdescent": "strdescent",
}

ABS_UNITS = frozenset({
    "cm", "inches", "mm", "points", "picas", "bigpts", "dida", "cicero",
    "scaledpts", "lines", "char", "strwidth", "strheight", "grobwidth",
    "grobheight", "strascent", "strdescent",
})

_LTY_MAP = {
    0: "none",
    1: "solid",
    2: "dashed",
    3: "dotted",
    4: "dotdash",
    5: "longdash",
    6: "twodash",
    "blank": "none",
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dotdash": "dotdash",
    "longdash": "longdash",
    "twodash": "twodash",
    "none": "none",
}

_FONTFACE_MAP = {
    1: ("normal", "normal"),
    2: ("bold", "normal"),
    3: ("normal", "italic"),
    4: ("bold", "italic"),
    "plain": ("normal", "normal"),
    "normal": ("normal", "normal"),
    "bold": ("bold", "normal"),
    "italic": ("normal", "italic"),
    "oblique": ("normal", "oblique"),
    "bold.italic": ("bold", "italic"),
    "bold_italic": ("bold", "italic"),
    "bold italic": ("bold", "italic"),
}


def _norm_unit_name(name: str) -> str:
    key = name.strip().lower()
    if key not in _UNIT_ALIASES:
        raise ValueError(f"Unsupported unit type: {name!r}")
    return _UNIT_ALIASES[key]


def _text_dims_mm(text: str, gp: dict[str, Any] | None = None, rot: float = 0.0) -> tuple[float, float]:
    """Measure width and height in mm of a text string using Matplotlib's TextPath."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    import math

    if text is None or text == "":
        return (0.0, 0.0)
    s = str(text)
    gp = gp or {}
    fontsize = gp.get("fontsize", 12.0)
    if isinstance(fontsize, (list, tuple, np.ndarray)):
        fontsize = float(fontsize[0])
    cex = gp.get("cex", 1.0)
    if isinstance(cex, (list, tuple, np.ndarray)):
        cex = float(cex[0])
    eff_fs = float(fontsize) * float(cex)
    fontfamily = gp.get("fontfamily", "sans-serif")
    if isinstance(fontfamily, (list, tuple, np.ndarray)):
        fontfamily = str(fontfamily[0])
    if not fontfamily:
        fontfamily = "sans-serif"
    fontface = gp.get("fontface", "plain")
    if isinstance(fontface, (list, tuple, np.ndarray)):
        fontface = fontface[0]
    weight, style = _FONTFACE_MAP.get(fontface, ("normal", "normal"))
    prop = FontProperties(family=fontfamily, weight=weight, style=style, size=eff_fs)

    lines = s.split("\n")
    max_w_pt = 0.0
    total_h_pt = 0.0
    line_gap_pt = eff_fs * 0.25
    for idx, ln in enumerate(lines):
        if ln.strip() == "":
            w_pt = len(ln) * eff_fs * 0.28
            h_pt = eff_fs * 0.72
        else:
            tp = TextPath((0, 0), ln, prop=prop)
            bb = tp.get_extents()
            w_pt = float(bb.width)
            h_pt = max(float(bb.height), eff_fs * 0.72)
        if w_pt > max_w_pt:
            max_w_pt = w_pt
        total_h_pt += h_pt + (line_gap_pt if idx > 0 else 0.0)

    w_mm = max_w_pt * _MM_PER_BIGPT
    h_mm = total_h_pt * _MM_PER_BIGPT

    if rot % 360 != 0:
        rad = math.radians(rot)
        c = abs(math.cos(rad))
        sn = abs(math.sin(rad))
        w_rot = w_mm * c + h_mm * sn
        h_rot = w_mm * sn + h_mm * c
        return (w_rot, h_rot)
    return (w_mm, h_mm)


class Unit:
    """Vectorized unit expression tree analogous to R `grid::unit`."""

    __slots__ = ("_op", "_values", "_unit", "_refs", "_left", "_right", "_scalar", "_items")

    def __init__(
        self,
        values: Any = None,
        units: str = "npc",
        *,
        _op: str | None = None,
        _refs: list[Any] | None = None,
        _left: "Unit | None" = None,
        _right: "Unit | None" = None,
        _scalar: np.ndarray | float | None = None,
        _items: list["Unit"] | None = None,
    ) -> None:
        self._op = _op
        self._refs = _refs
        self._left = _left
        self._right = _right
        self._scalar = _scalar
        self._items = _items
        if _op is None:
            arr = np.atleast_1d(np.asarray(values, dtype=float))
            if arr.ndim != 1:
                arr = arr.ravel()
            self._values = arr
            self._unit = _norm_unit_name(units)
        else:
            self._values = None
            self._unit = None

    def __len__(self) -> int:
        if self._op is None:
            return len(self._values)
        if self._op in ("add", "sub", "pmax", "pmin"):
            if self._items is not None:
                return max(len(it) for it in self._items)
            return max(len(self._left), len(self._right))
        if self._op in ("mul", "div"):
            l_len = len(self._left)
            if isinstance(self._scalar, np.ndarray) and self._scalar.ndim >= 1:
                return max(l_len, len(self._scalar))
            return l_len
        if self._op == "concat":
            return sum(len(it) for it in self._items)
        if self._op in ("sum", "max", "min"):
            return 1
        return 1

    def __getitem__(self, idx: Any) -> "Unit":
        if self._op is None:
            vals = self._values[idx]
            refs = None
            if self._refs is not None:
                if isinstance(idx, slice):
                    refs = self._refs[idx]
                elif isinstance(idx, (int, np.integer)):
                    refs = [self._refs[int(idx)]]
                elif isinstance(idx, (list, tuple, np.ndarray)):
                    arr_idx = np.asarray(idx)
                    if arr_idx.dtype == bool:
                        refs = [r for r, flag in zip(self._refs, arr_idx) if flag]
                    else:
                        refs = [self._refs[int(i)] for i in arr_idx]
            return Unit(vals, self._unit, _refs=refs)

        if self._op == "concat":
            # Flatten elements to index accurately
            flat: list[Unit] = []
            for it in self._items:
                for k in range(len(it)):
                    flat.append(it[k])
            selected = flat[idx] if isinstance(idx, slice) else (
                [flat[int(idx)]] if isinstance(idx, (int, np.integer)) else None
            )
            if selected is None:
                arr_idx = np.asarray(idx)
                if arr_idx.dtype == bool:
                    selected = [u for u, f in zip(flat, arr_idx) if f]
                else:
                    selected = [flat[int(i)] for i in arr_idx]
            if not selected:
                return unit([], "mm")
            if len(selected) == 1:
                return selected[0]
            return Unit(_op="concat", _items=selected)

        if self._op in ("add", "sub", "pmax", "pmin"):
            n = len(self)
            def _sub_with_broadcast(u: Unit) -> Unit:
                if len(u) == 1:
                    return u
                return u[idx]
            if self._items is not None:
                return Unit(_op=self._op, _items=[_sub_with_broadcast(it) for it in self._items])
            return Unit(_op=self._op, _left=_sub_with_broadcast(self._left), _right=_sub_with_broadcast(self._right))

        if self._op in ("mul", "div"):
            left_s = self._left if len(self._left) == 1 else self._left[idx]
            sc = self._scalar
            if isinstance(sc, np.ndarray) and sc.ndim >= 1 and len(sc) > 1:
                sc = sc[idx]
            return Unit(_op=self._op, _left=left_s, _scalar=sc)

        return self

    def __add__(self, other: Any) -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit(_op="add", _left=self, _right=other)

    def __radd__(self, other: Any) -> "Unit":
        if other == 0:
            return self
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit(_op="add", _left=other, _right=self)

    def __sub__(self, other: Any) -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit(_op="sub", _left=self, _right=other)

    def __neg__(self) -> "Unit":
        return Unit(_op="mul", _left=self, _scalar=-1.0)

    def __mul__(self, other: Any) -> "Unit":
        if isinstance(other, (int, float, np.number, list, tuple, np.ndarray)):
            sc = float(other) if np.ndim(other) == 0 else np.asarray(other, dtype=float)
            return Unit(_op="mul", _left=self, _scalar=sc)
        return NotImplemented

    def __rmul__(self, other: Any) -> "Unit":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Unit":
        if isinstance(other, (int, float, np.number, list, tuple, np.ndarray)):
            sc = float(other) if np.ndim(other) == 0 else np.asarray(other, dtype=float)
            return Unit(_op="div", _left=self, _scalar=sc)
        return NotImplemented

    def _unit_types(self) -> set[str]:
        if self._op is None:
            return {self._unit}
        out: set[str] = set()
        if self._left is not None:
            out |= self._left._unit_types()
        if self._right is not None:
            out |= self._right._unit_types()
        if self._items is not None:
            for it in self._items:
                out |= it._unit_types()
        return out

    def is_abs(self) -> bool:
        return self._unit_types().issubset(ABS_UNITS)

    def _eval_mm(self, axis: str, vp: Any = None, pos_mode: bool = False) -> np.ndarray:
        w_mm = getattr(vp, "w_mm", 200.0) if vp is not None else 200.0
        h_mm = getattr(vp, "h_mm", 200.0) if vp is not None else 200.0
        xscale = getattr(vp, "xscale", (0.0, 1.0)) if vp is not None else (0.0, 1.0)
        yscale = getattr(vp, "yscale", (0.0, 1.0)) if vp is not None else (0.0, 1.0)
        vp_gp = getattr(vp, "gp", {}) if vp is not None else {}
        fs = float(vp_gp.get("fontsize", 12.0)) * float(vp_gp.get("cex", 1.0))

        if self._op is None:
            u = self._unit
            vals = self._values
            if u == "mm":
                return vals.copy()
            if u == "cm":
                return vals * 10.0
            if u == "inches":
                return vals * _MM_PER_INCH
            if u == "points":
                return vals * _MM_PER_PT
            if u == "bigpts":
                return vals * _MM_PER_BIGPT
            if u == "picas":
                return vals * (12.0 * _MM_PER_PT)
            if u == "dida":
                return vals * 0.376
            if u == "cicero":
                return vals * (12.0 * 0.376)
            if u == "scaledpts":
                return vals * (_MM_PER_PT / 65536.0)
            if u == "lines":
                return vals * (fs * 1.2 * _MM_PER_BIGPT)
            if u == "char":
                return vals * (fs * _MM_PER_BIGPT)
            if u == "npc":
                dim = w_mm if axis == "x" else h_mm
                return vals * dim
            if u == "snpc":
                return vals * min(w_mm, h_mm)
            if u == "native":
                if axis == "x":
                    s0, s1 = float(xscale[0]), float(xscale[1])
                    span = s1 - s0
                    if abs(span) < 1e-15:
                        return np.zeros_like(vals)
                    if pos_mode:
                        return ((vals - s0) / span) * w_mm
                    return (vals / abs(span)) * w_mm
                else:
                    s0, s1 = float(yscale[0]), float(yscale[1])
                    span = s1 - s0
                    if abs(span) < 1e-15:
                        return np.zeros_like(vals)
                    if pos_mode:
                        return ((vals - s0) / span) * h_mm
                    return (vals / abs(span)) * h_mm
            if u == "null":
                return np.zeros_like(vals)
            if u in ("grobwidth", "grobheight", "strwidth", "strheight", "strascent", "strdescent"):
                out = np.zeros(len(vals), dtype=float)
                refs = self._refs or []
                for idx in range(len(vals)):
                    ref = refs[idx] if idx < len(refs) else None
                    if ref is None:
                        out[idx] = 0.0
                    elif u in ("strwidth", "strheight", "strascent", "strdescent"):
                        txt, tgp, rot = ref
                        tw, th = _text_dims_mm(txt, tgp, rot)
                        out[idx] = vals[idx] * (tw if u == "strwidth" else th)
                    elif hasattr(ref, "width_mm") and hasattr(ref, "height_mm"):
                        gw = ref.width_mm(vp)
                        gh = ref.height_mm(vp)
                        out[idx] = vals[idx] * (gw if u == "grobwidth" else gh)
                return out
            return vals.copy()

        if self._op == "add":
            a = self._left._eval_mm(axis, vp, pos_mode=pos_mode)
            # In R grid, if both sides are native in position mode, pos_mode applies to sum;
            # if left is native/npc and right is offset (mm/pt), right is an offset.
            right_pos = False if pos_mode and not self._right._unit_types().intersection({"npc", "native"}) else pos_mode
            if pos_mode and self._left._unit == "native" and self._right._unit == "native":
                # R evaluates (v1 + v2 - s0)/span * dim
                raw = self._left._values + self._right._values
                return Unit(raw, "native")._eval_mm(axis, vp, pos_mode=True)
            b = self._right._eval_mm(axis, vp, pos_mode=right_pos)
            return a + b

        if self._op == "sub":
            a = self._left._eval_mm(axis, vp, pos_mode=pos_mode)
            b = self._right._eval_mm(axis, vp, pos_mode=False)
            return a - b

        if self._op == "mul":
            a = self._left._eval_mm(axis, vp, pos_mode=pos_mode)
            return a * self._scalar

        if self._op == "div":
            a = self._left._eval_mm(axis, vp, pos_mode=pos_mode)
            return a / self._scalar

        if self._op == "concat":
            parts = [it._eval_mm(axis, vp, pos_mode=pos_mode) for it in self._items]
            return np.concatenate(parts) if parts else np.array([], dtype=float)

        if self._op == "sum":
            vals = self._left._eval_mm(axis, vp, pos_mode=pos_mode)
            return np.array([float(np.sum(vals))])

        if self._op == "max":
            vals = np.concatenate([it._eval_mm(axis, vp, pos_mode=pos_mode) for it in self._items])
            return np.array([float(np.max(vals)) if len(vals) else 0.0])

        if self._op == "min":
            vals = np.concatenate([it._eval_mm(axis, vp, pos_mode=pos_mode) for it in self._items])
            return np.array([float(np.min(vals)) if len(vals) else 0.0])

        if self._op == "pmax":
            arrs = np.broadcast_arrays(*[it._eval_mm(axis, vp, pos_mode=pos_mode) for it in self._items])
            return np.maximum.reduce(arrs)

        if self._op == "pmin":
            arrs = np.broadcast_arrays(*[it._eval_mm(axis, vp, pos_mode=pos_mode) for it in self._items])
            return np.minimum.reduce(arrs)

        raise ValueError(f"Unknown unit op: {self._op}")

    def to_mm(self, axis: str = "x", vp: Any = None, pos_mode: bool = False) -> np.ndarray:
        return self._eval_mm(axis=axis, vp=vp, pos_mode=pos_mode)

    def to_mm_scalar(self, axis: str = "x", vp: Any = None, pos_mode: bool = False) -> float:
        arr = self._eval_mm(axis=axis, vp=vp, pos_mode=pos_mode)
        return float(arr[0]) if len(arr) else 0.0

    def __repr__(self) -> str:
        if self._op is None:
            if len(self._values) == 1:
                return f"{self._values[0]:g}{self._unit}"
            vals = ", ".join(f"{v:g}{self._unit}" for v in self._values[:6])
            if len(self._values) > 6:
                vals += ", ..."
            return f"unit([{vals}])"
        if self._op == "add":
            return f"({self._left!r} + {self._right!r})"
        if self._op == "sub":
            return f"({self._left!r} - {self._right!r})"
        if self._op == "mul":
            return f"({self._left!r} * {self._scalar})"
        if self._op == "div":
            return f"({self._left!r} / {self._scalar})"
        if self._op == "concat":
            return "unit.c(" + ", ".join(repr(it) for it in self._items) + ")"
        return f"unit_expr({self._op})"


def unit(x: Any, units: str = "npc") -> Unit:
    """Create a Unit object (equivalent to R `grid::unit`)."""
    if isinstance(x, Unit):
        return x
    return Unit(x, units)


def is_unit(x: Any) -> bool:
    return isinstance(x, Unit)


def as_unit(x: Any, default_units: str = "npc") -> Unit:
    if isinstance(x, Unit):
        return x
    return Unit(x, default_units)


def unit_c(*args: Any) -> Unit:
    """Concatenate multiple Unit objects into a single vector Unit."""
    items = [a if isinstance(a, Unit) else unit(a, "npc") for a in args if a is not None]
    if not items:
        return unit([], "mm")
    if len(items) == 1:
        return items[0]
    return Unit(_op="concat", _items=items)


def unit_sum(u: Unit) -> Unit:
    return Unit(_op="sum", _left=u)


def unit_max(*args: Unit) -> Unit:
    return Unit(_op="max", _items=[as_unit(a) for a in args])


def unit_min(*args: Unit) -> Unit:
    return Unit(_op="min", _items=[as_unit(a) for a in args])


def unit_pmax(*args: Unit) -> Unit:
    return Unit(_op="pmax", _items=[as_unit(a) for a in args])


def unit_pmin(*args: Unit) -> Unit:
    return Unit(_op="pmin", _items=[as_unit(a) for a in args])


def unit_rep(u: Unit, times: int) -> Unit:
    if len(u) == 1:
        if u._op is None and u._refs is None:
            return Unit(np.repeat(u._values, times), u._unit)
        return Unit(_op="concat", _items=[u] * times)
    return Unit(_op="concat", _items=[u] * times)


def is_abs_unit(u: Any) -> bool:
    """Return True if `u` is a Unit composed exclusively of absolute units."""
    if not isinstance(u, Unit):
        return False
    return u.is_abs()


def _resolve_vp(vp: Any) -> Any:
    if vp is not None:
        return vp
    from .grid import _DEVICE_STACK
    return _DEVICE_STACK[-1].current_vp if _DEVICE_STACK else None


def convert_unit(u: Any, unit_to: str = "mm", axis: str = "x", vp: Any = None, value_only: bool = False) -> Any:
    vp = _resolve_vp(vp)
    u_obj = as_unit(u, "mm")
    mm_vals = u_obj.to_mm(axis=axis, vp=vp, pos_mode=False)
    target = _norm_unit_name(unit_to)
    scale = Unit(1.0, target).to_mm_scalar(axis=axis, vp=vp, pos_mode=False)
    if scale == 0:
        converted = np.zeros_like(mm_vals)
    else:
        converted = mm_vals / scale
    if value_only:
        return float(converted[0]) if len(converted) == 1 else converted
    return Unit(converted[0] if len(converted) == 1 else converted, target)


def convertWidth(u: Any, unit_to: str = "mm", valueOnly: bool = False, vp: Any = None) -> Any:
    return convert_unit(u, unit_to=unit_to, axis="x", vp=vp, value_only=valueOnly)


def convertHeight(u: Any, unit_to: str = "mm", valueOnly: bool = False, vp: Any = None) -> Any:
    return convert_unit(u, unit_to=unit_to, axis="y", vp=vp, value_only=valueOnly)


def convertX(u: Any, unit_to: str = "mm", valueOnly: bool = False, vp: Any = None) -> Any:
    vp = _resolve_vp(vp)
    u_obj = as_unit(u, "npc")
    mm_vals = u_obj.to_mm(axis="x", vp=vp, pos_mode=True)
    target = _norm_unit_name(unit_to)
    scale = Unit(1.0, target).to_mm_scalar(axis="x", vp=vp, pos_mode=False)
    converted = mm_vals / scale if scale != 0 else np.zeros_like(mm_vals)
    if valueOnly:
        return float(converted[0]) if len(converted) == 1 else converted
    return Unit(converted[0] if len(converted) == 1 else converted, target)


def convertY(u: Any, unit_to: str = "mm", valueOnly: bool = False, vp: Any = None) -> Any:
    vp = _resolve_vp(vp)
    u_obj = as_unit(u, "npc")
    mm_vals = u_obj.to_mm(axis="y", vp=vp, pos_mode=True)
    target = _norm_unit_name(unit_to)
    scale = Unit(1.0, target).to_mm_scalar(axis="y", vp=vp, pos_mode=False)
    converted = mm_vals / scale if scale != 0 else np.zeros_like(mm_vals)
    if valueOnly:
        return float(converted[0]) if len(converted) == 1 else converted
    return Unit(converted[0] if len(converted) == 1 else converted, target)


# ---- Graphical parameters (gpar) ----

class Gpar(dict):
    """Graphical parameters dictionary compatible with R `grid::gpar`."""

    def copy(self) -> "Gpar":
        return Gpar(**{k: copy.deepcopy(v) for k, v in self.items()})

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _normalize_gp_value(key: str, val: Any) -> Any:
    if key in ("col", "fill", "box_fill", "box_col", "pos_fill", "neg_fill"):
        if val is None:
            return "#00000000"
        if isinstance(val, (str, int, np.integer, float, np.floating)):
            if isinstance(val, str) and val.lower() == "asis":
                return "asis"
            return to_hex_rgba(val)
        if isinstance(val, tuple) and len(val) in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in val):
            return to_hex_rgba(val)
        if isinstance(val, (list, tuple, np.ndarray)):
            return [
                "asis" if (isinstance(v, str) and v.lower() == "asis") else to_hex_rgba(v)
                for v in val
            ]
    if key == "lty":
        if isinstance(val, (list, tuple, np.ndarray)):
            return [_LTY_MAP.get(v, v) for v in val]
        return _LTY_MAP.get(val, val)
    if key == "fontface":
        return val
    return val


def gpar(**kwargs: Any) -> Gpar:
    """Create graphical parameters (col, fill, lwd, lty, fontsize, fontface, fontfamily, cex, alpha, ...)."""
    norm = {k: _normalize_gp_value(k, v) for k, v in kwargs.items()}
    return Gpar(norm)


def check_gp(gp: Any) -> Gpar:
    if gp is None:
        return Gpar()
    if isinstance(gp, Gpar):
        return gp.copy()
    if isinstance(gp, dict):
        return gpar(**gp)
    raise TypeError("Graphical parameters must be created by `gpar(...)` or `dict`.")


def _gp_elem_len(v: Any) -> int:
    if isinstance(v, (list, tuple, np.ndarray)) and not (
        isinstance(v, tuple) and len(v) in (3, 4) and all(isinstance(x, (int, float, np.number)) for x in v)
    ):
        return len(v)
    if isinstance(v, Unit):
        return len(v)
    return 1


def subset_gp(gp: dict[str, Any], i: Any) -> Gpar:
    """Subset vectorized entries of a `gpar` object while keeping scalar entries intact."""
    gp = check_gp(gp)
    out = Gpar()
    for k, v in gp.items():
        if _gp_elem_len(v) <= 1:
            if isinstance(v, (list, tuple, np.ndarray)) and not (
                isinstance(v, tuple) and len(v) in (3, 4) and all(isinstance(x, (int, float, np.number)) for x in v)
            ) and len(v) == 1:
                out[k] = v[0]
            else:
                out[k] = v
        elif isinstance(v, Unit):
            out[k] = v[i]
        elif isinstance(v, np.ndarray):
            out[k] = v[i].tolist()
        else:
            if isinstance(i, slice):
                out[k] = list(v)[i]
            elif isinstance(i, (int, np.integer)):
                out[k] = v[int(i)]
            else:
                arr_i = np.asarray(i)
                if arr_i.dtype == bool:
                    out[k] = [val for val, flag in zip(v, arr_i) if flag]
                else:
                    out[k] = [v[int(idx)] for idx in arr_i]
    return out


def recycle_gp(gp: dict[str, Any], n: int = 1) -> Gpar:
    """Recycle every entry of a `gpar` object to length `n`."""
    gp = check_gp(gp)
    out = Gpar()
    for k, v in gp.items():
        elen = _gp_elem_len(v)
        if elen == 1:
            scalar = v[0] if isinstance(v, (list, tuple, np.ndarray)) and not (
                isinstance(v, tuple) and len(v) in (3, 4) and all(isinstance(x, (int, float, np.number)) for x in v)
            ) else v
            out[k] = [scalar] * n
        elif elen == n:
            out[k] = v if isinstance(v, Unit) else list(v)
        else:
            seq = list(v)
            out[k] = [seq[idx % len(seq)] for idx in range(n)]
    return out


def merge_gp(*gps: dict[str, Any] | None) -> Gpar:
    """Merge multiple `gpar` dicts left-to-right (later overrides earlier)."""
    merged = Gpar()
    for gp in gps:
        if gp:
            for k, v in check_gp(gp).items():
                merged[k] = v
    return merged
