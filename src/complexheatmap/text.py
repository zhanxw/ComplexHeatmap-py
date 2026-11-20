"""Rich text (`gt_render`), multi-phrase `textbox_grob`, and interval alignment (`smartAlign2`)."""

from __future__ import annotations

from builtins import range as _range
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .colors import add_transparency, rand_color, to_hex_rgba, to_rgba_tuple
from .grid import (
    GTree,
    Grob,
    RectGrob,
    RoundRectGrob,
    TextGrob,
    Viewport,
    current_device,
    grid_newpage,
    parse_just,
    rectGrob,
    roundrectGrob,
    textGrob,
)
from .units import (
    Gpar,
    Unit,
    _MM_PER_BIGPT,
    _text_dims_mm,
    as_unit,
    check_gp,
    convertHeight,
    convertWidth,
    gpar,
    recycle_gp,
    subset_gp,
    unit,
)


class GridText:
    """Wrapper marking labels for rich-text (HTML/Markdown) rendering via `gt_render`."""

    def __init__(self, x: Any, **param: Any) -> None:
        if isinstance(x, str):
            self._items = [x]
            self._scalar = True
        else:
            self._items = [str(v) for v in x]
            self._scalar = False
        self.param = dict(param)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, (int, np.integer)):
            return GridText([self._items[int(idx)]], **self.param)
        if isinstance(idx, slice):
            return GridText(self._items[idx], **self.param)
        arr = np.asarray(idx)
        if arr.dtype == bool:
            sel = [v for v, f in zip(self._items, arr) if f]
        else:
            sel = [self._items[int(i)] for i in arr]
        return GridText(sel, **self.param)

    def __iter__(self):
        return iter(self._items)

    def as_list(self) -> list[str]:
        return list(self._items)

    def __repr__(self) -> str:
        return f"GridText({self._items!r}, param={self.param!r})"


def gt_render(x: Any, **kwargs: Any) -> GridText:
    """Mark text string(s) for Markdown/HTML rendering (ComplexHeatmap `gt_render`)."""
    return GridText(x, **kwargs)


@dataclass
class _Run:
    text: str
    bold: bool = False
    italic: bool = False
    color: str | None = None
    fontsize: float | None = None


_STYLE_RE = re.compile(r"([a-zA-Z\-]+)\s*:\s*([^;]+)")


def _parse_richtext_lines(s: str) -> list[list[_Run]]:
    """Parse simplified HTML/Markdown (`<br>`, `<span style='...'>`, `**bold**`, `*italic*`)."""
    # Normalize <br> tags to newline
    s = re.sub(r"<br\s*/?>", "\n", str(s), flags=re.IGNORECASE)
    raw_lines = s.split("\n")
    parsed_lines: list[list[_Run]] = []

    token_re = re.compile(
        r"(<span\s+style=['\"][^'\"]*['\"]\s*>|</span>|\*\*[^*]+\*\*|\*[^*]+\*)",
        flags=re.IGNORECASE,
    )

    for ln in raw_lines:
        runs: list[_Run] = []
        style_stack: list[dict[str, Any]] = [{}]
        pos = 0
        for m in token_re.finditer(ln):
            if m.start() > pos:
                chunk = ln[pos:m.start()]
                cur = style_stack[-1]
                runs.append(_Run(chunk, bold=cur.get("bold", False), italic=cur.get("italic", False), color=cur.get("color"), fontsize=cur.get("fontsize")))
            tok = m.group(0)
            low = tok.lower()
            if low.startswith("<span"):
                cur = dict(style_stack[-1])
                sm = re.search(r"style=['\"]([^'\"]*)['\"]", tok, flags=re.IGNORECASE)
                if sm:
                    for k, v in _STYLE_RE.findall(sm.group(1)):
                        kl = k.strip().lower()
                        vl = v.strip()
                        if kl == "color":
                            cur["color"] = vl
                        elif kl == "font-size":
                            fs_m = re.match(r"([0-9.]+)\s*(pt|px)?", vl, flags=re.IGNORECASE)
                            if fs_m:
                                cur["fontsize"] = float(fs_m.group(1))
                        elif kl == "font-weight" and "bold" in vl.lower():
                            cur["bold"] = True
                style_stack.append(cur)
            elif low == "</span>":
                if len(style_stack) > 1:
                    style_stack.pop()
            elif tok.startswith("**") and tok.endswith("**"):
                inner = tok[2:-2]
                cur = style_stack[-1]
                runs.append(_Run(inner, bold=True, italic=cur.get("italic", False), color=cur.get("color"), fontsize=cur.get("fontsize")))
            elif tok.startswith("*") and tok.endswith("*"):
                inner = tok[1:-1]
                cur = style_stack[-1]
                runs.append(_Run(inner, bold=cur.get("bold", False), italic=True, color=cur.get("color"), fontsize=cur.get("fontsize")))
            pos = m.end()
        if pos < len(ln):
            cur = style_stack[-1]
            runs.append(_Run(ln[pos:], bold=cur.get("bold", False), italic=cur.get("italic", False), color=cur.get("color"), fontsize=cur.get("fontsize")))
        if not runs:
            runs.append(_Run(""))
        parsed_lines.append(runs)
    return parsed_lines


def _extract_padding_mm(param: dict[str, Any]) -> tuple[float, float, float, float]:
    pad = param.get("padding")
    if pad is None:
        return (0.0, 0.0, 0.0, 0.0)
    u = as_unit(pad, "pt")
    vals = u.to_mm("x")
    if len(vals) == 1:
        v = float(vals[0])
        return (v, v, v, v)
    if len(vals) == 2:
        return (float(vals[0]), float(vals[1]), float(vals[0]), float(vals[1]))
    # [top, right, bottom, left] in CSS / gridtext order
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _richtext_dims_mm(text: Any, gp: dict[str, Any], param: dict[str, Any], rot: float = 0.0) -> tuple[float, float]:
    s = text._items[0] if isinstance(text, GridText) and len(text) else str(text)
    lines = _parse_richtext_lines(s)
    base_fs = float(gp.get("fontsize", 12.0))
    base_face = gp.get("fontface", "plain")
    ff = gp.get("fontfamily", "sans-serif")
    max_w = 0.0
    total_h = 0.0
    for idx, runs in enumerate(lines):
        lw = 0.0
        lh = base_fs * 0.75 * _MM_PER_BIGPT
        for run in runs:
            fs = run.fontsize if run.fontsize is not None else base_fs
            face = "bold.italic" if (run.bold and run.italic) else ("bold" if run.bold else ("italic" if run.italic else base_face))
            rw, rh = _text_dims_mm(run.text, {"fontsize": fs, "fontface": face, "fontfamily": ff}, 0.0)
            lw += rw
            lh = max(lh, rh)
        max_w = max(max_w, lw)
        total_h += lh + (base_fs * 0.25 * _MM_PER_BIGPT if idx > 0 else 0.0)
    pt, pr, pb, pl = _extract_padding_mm(param)
    w_mm = max_w + pl + pr
    h_mm = total_h + pt + pb
    if rot % 360 != 0:
        rad = math.radians(rot)
        c, sn = abs(math.cos(rad)), abs(math.sin(rad))
        return (w_mm * c + h_mm * sn, w_mm * sn + h_mm * c)
    return (w_mm, h_mm)


def draw_richtext(
    dev: Any,
    label: GridText,
    x: Any = None,
    y: Any = None,
    just: Any = "centre",
    rot: float = 0.0,
    gp: dict[str, Any] | None = None,
    default_units: str = "npc",
) -> None:
    """Render `GridText` items with optional box background and inline formatting."""
    gp_raw = check_gp(gp)
    box_gp = Gpar()
    clean_gp = Gpar()
    for k, v in gp_raw.items():
        if k.startswith("box_"):
            box_gp[k[4:]] = v
        else:
            clean_gp[k] = v

    items = label._items
    n = len(items)
    if n == 0:
        return
    vp = dev.current_vp
    ux = unit(0.5, "npc") if x is None else as_unit(x, default_units)
    uy = unit(0.5, "npc") if y is None else as_unit(y, default_units)
    xmm = vp.x0_mm + ux.to_mm("x", vp, pos_mode=True)
    ymm = vp.y0_mm + uy.to_mm("y", vp, pos_mode=True)
    xmm, ymm = np.broadcast_arrays(xmm, ymm, np.zeros(n))[:2]
    jx, jy = parse_just(just)
    gp_n = recycle_gp(clean_gp, n)
    box_gp_n = recycle_gp(box_gp, n) if box_gp else None
    pt, pr, pb, pl = _extract_padding_mm(label.param)
    rad = math.radians(rot)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    align_w = 0.0
    if label.param.get("align_widths", False):
        align_w = max(_richtext_dims_mm(it, subset_gp(gp_n, k), label.param, 0.0)[0] for k, it in enumerate(items))

    cb = dev._clip_bbox(vp)
    for k in range(n):
        gpk = subset_gp(gp_n, k)
        bgpk = subset_gp(box_gp_n, k) if box_gp_n is not None else {}
        lines = _parse_richtext_lines(items[k])
        base_fs = float(gpk.get("fontsize", 12.0))
        base_col = gpk.get("col", "#000000FF")
        base_face = gpk.get("fontface", "plain")
        ff = gpk.get("fontfamily", "sans-serif") or "sans-serif"

        line_data = []
        content_w = 0.0
        content_h = 0.0
        line_gap = base_fs * 0.25 * _MM_PER_BIGPT
        for li, runs in enumerate(lines):
            lw = 0.0
            lh = base_fs * 0.75 * _MM_PER_BIGPT
            run_metrics = []
            for r in runs:
                fs = r.fontsize if r.fontsize is not None else base_fs
                face = "bold.italic" if (r.bold and r.italic) else ("bold" if r.bold else ("italic" if r.italic else base_face))
                rw, rh = _text_dims_mm(r.text, {"fontsize": fs, "fontface": face, "fontfamily": ff}, 0.0)
                run_metrics.append((r, rw, rh, fs, face))
                lw += rw
                lh = max(lh, rh)
            content_w = max(content_w, lw)
            content_h += lh + (line_gap if li > 0 else 0.0)
            line_data.append((lw, lh, run_metrics))

        box_w = max(align_w, content_w + pl + pr)
        box_h = content_h + pt + pb
        ax_k, ay_k = float(xmm[k]), float(ymm[k])

        # Local frame: bottom-left of unrotated box relative to anchor (ax_k, ay_k)
        bx0 = -jx * box_w
        by0 = -jy * box_h

        def _rot_pt(lx: float, ly: float) -> tuple[float, float]:
            return (ax_k + lx * cos_r - ly * sin_r, ay_k + lx * sin_r + ly * cos_r)

        if bgpk:
            corners = [_rot_pt(bx0, by0), _rot_pt(bx0 + box_w, by0), _rot_pt(bx0 + box_w, by0 + box_h), _rot_pt(bx0, by0 + box_h)]
            from matplotlib.patches import Polygon as MplPolygon
            fc = to_rgba_tuple(bgpk.get("fill", "#00000000"))
            ec = to_rgba_tuple(bgpk.get("col", "#00000000" if "fill" in bgpk else "#000000FF"))
            lwd = float(bgpk.get("lwd", 1.0)) * 0.75 if ec[3] > 0 else 0.0
            poly = MplPolygon(corners, closed=True, facecolor=fc, edgecolor=ec, linewidth=lwd)
            poly.set_zorder(dev._next_zorder())
            if cb is not None:
                poly.set_clip_box(cb)
            dev.ax.add_patch(poly)

        # Draw lines top-to-bottom inside local box
        cur_top = by0 + box_h - pt
        for li, (lw, lh, run_metrics) in enumerate(line_data):
            cur_y = cur_top - lh * 0.5
            cur_x = bx0 + pl
            for r, rw, rh, fs, face in run_metrics:
                if r.text:
                    px, py = _rot_pt(cur_x, cur_y)
                    weight = "bold" if "bold" in str(face) else "normal"
                    style = "italic" if "italic" in str(face) else "normal"
                    col = to_rgba_tuple(r.color if r.color else base_col)
                    t = dev.ax.text(
                        px, py, r.text,
                        fontsize=fs, fontfamily=ff, fontweight=weight, fontstyle=style,
                        color=col, rotation=float(rot), rotation_mode="anchor",
                        ha="left", va="center",
                    )
                    t.set_zorder(dev._next_zorder())
                    if cb is not None:
                        t.set_clip_box(cb)
                cur_x += rw
            cur_top -= lh + line_gap


def textbox_grob(
    text: Sequence[str],
    x: Any = None,
    y: Any = None,
    just: Any = "centre",
    gp: dict[str, Any] | None = None,
    background_gp: dict[str, Any] | None = None,
    round_corners: bool = False,
    r: Any = None,
    line_space: Any = None,
    text_space: Any = None,
    max_width: Any = None,
    padding: Any = None,
    first_text_from: str = "top",
    add_new_line: bool = False,
    word_wrap: bool = False,
) -> GTree:
    """Create a multi-line packed word-cloud / phrase box grob (`textbox_grob`)."""
    items = [str(t) for t in text]
    n_text = len(items)
    if n_text == 0:
        return GTree(children=[], cl="textbox", width=unit(0.0, "mm"), height=unit(0.0, "mm"))

    gp_d = check_gp(gp)
    if "col" not in gp_d:
        gp_d["col"] = rand_color(n_text, friendly=True)
    if "fontsize" not in gp_d:
        gp_d["fontsize"] = [12.0] * n_text
    if "fontfamily" not in gp_d:
        gp_d["fontfamily"] = ["sans-serif"] * n_text
    if "fontface" not in gp_d:
        gp_d["fontface"] = ["plain"] * n_text
    gp_d = recycle_gp(gp_d, n_text)

    bg_gp = gpar(col="black", fill="transparent") if background_gp is None else check_gp(background_gp)
    vp_x = unit(0.5, "npc") if x is None else as_unit(x, "npc")
    vp_y = unit(0.5, "npc") if y is None else as_unit(y, "npc")
    r_u = unit(0.1, "snpc") if r is None else as_unit(r, "snpc")
    ls_mm = 4.0 * _MM_PER_BIGPT if line_space is None else (float(line_space) if not isinstance(line_space, Unit) else convertHeight(line_space, "mm", valueOnly=True))
    ts_mm = 4.0 * _MM_PER_BIGPT if text_space is None else (float(text_space) if not isinstance(text_space, Unit) else convertWidth(text_space, "mm", valueOnly=True))
    mw_mm = 100.0 if max_width is None else (float(max_width) if not isinstance(max_width, Unit) else convertWidth(max_width, "mm", valueOnly=True))

    if word_wrap:
        if first_text_from == "bottom":
            items = list(reversed(items))
            gp_d = subset_gp(gp_d, list(range(n_text - 1, -1, -1)))
            first_text_from = "top"
        tokens: list[str] = []
        space_after: list[bool] = []
        new_line_after: list[bool] = []
        col_list, fs_list, ff_list, face_list = [], [], [], []
        for idx, phrase in enumerate(items):
            words = phrase.split()
            if not words:
                words = [""]
            w_tokens: list[str] = []
            for wi, wd in enumerate(words):
                if wi > 0:
                    w_tokens.append(" ")
                w_tokens.append(wd)
            tokens.extend(w_tokens)
            for wi in range(len(w_tokens)):
                is_last = wi == len(w_tokens) - 1
                space_after.append(is_last)
                new_line_after.append(bool(add_new_line and is_last))
            sub = subset_gp(gp_d, idx)
            for _ in w_tokens:
                col_list.append(sub["col"])
                fs_list.append(sub["fontsize"])
                ff_list.append(sub["fontfamily"])
                face_list.append(sub["fontface"])
        gp_d = gpar(col=col_list, fontsize=fs_list, fontfamily=ff_list, fontface=face_list)
    else:
        tokens = items
        space_after = [True] * n_text
        new_line_after = [bool(add_new_line)] * n_text

    n = len(tokens)
    t_widths = np.array([textGrob(tokens[i], gp=subset_gp(gp_d, i)).width_mm() for i in range(n)], dtype=float)
    t_heights = np.array([textGrob(tokens[i], gp=subset_gp(gp_d, i)).height_mm() for i in range(n)], dtype=float)

    mw_mm = max(mw_mm, float(np.max(t_widths)))
    xs = np.zeros(n, dtype=float)
    ys = np.zeros(n, dtype=float)
    w = float(t_widths[0])

    if first_text_from == "bottom":
        cur_w = float(t_widths[0])
        h = float(t_heights[0])
        for i in range(1, n):
            if cur_w + t_widths[i] + ts_mm > mw_mm or new_line_after[i - 1]:
                xs[i] = 0.0
                ys[i] = h + ls_mm
                cur_w = float(t_widths[i])
                w = max(w, cur_w)
                h = ys[i] + float(t_heights[i])
            else:
                xs[i] = cur_w + (ts_mm if space_after[i - 1] else 0.0)
                ys[i] = ys[i - 1]
                cur_w = xs[i] + float(t_widths[i])
                w = max(w, cur_w)
                h = max(h, ys[i] + float(t_heights[i]))
    elif first_text_from == "top":
        cur_w = float(t_widths[0])
        h = -float(t_heights[0])
        prev_line = [0]
        for i in range(1, n):
            if cur_w + t_widths[i] + ts_mm > mw_mm or new_line_after[i - 1]:
                for pi in prev_line:
                    ys[pi] = h
                prev_line = [i]
                xs[i] = 0.0
                ys[i] = h - ls_mm
                cur_w = float(t_widths[i])
                w = max(w, cur_w)
                h = ys[i] - float(t_heights[i])
            else:
                xs[i] = cur_w + (ts_mm if space_after[i - 1] else 0.0)
                ys[i] = ys[i - 1]
                cur_w = xs[i] + float(t_widths[i])
                w = max(w, cur_w)
                h = min(h, ys[i] - float(t_heights[i]))
                prev_line.append(i)
        for pi in prev_line:
            ys[pi] = h
        ys = ys - h
        h = -h
    else:
        raise ValueError("`first_text_from` must be 'top' or 'bottom'.")

    pad_u = unit(4.0, "pt") if padding is None else as_unit(padding, "pt")
    pad_vals = pad_u.to_mm("x")
    if len(pad_vals) == 1:
        pb = pl = pt = pr = float(pad_vals[0])
    elif len(pad_vals) == 2:
        pb = pt = float(pad_vals[0])
        pl = pr = float(pad_vals[1])
    else:
        pb, pl, pt, pr = (float(pad_vals[0]), float(pad_vals[1]), float(pad_vals[2]), float(pad_vals[3]))

    w_tot = w + pl + pr
    h_tot = h + pb + pt
    xs = xs + pl
    ys = ys + pb

    bg_grob: Grob = roundrectGrob(gp=bg_gp, r=r_u) if round_corners else rectGrob(gp=bg_gp)
    txt_grob = textGrob(tokens, x=xs, y=ys, gp=gp_d, default_units="mm", just=(0.0, 0.0))
    vp = Viewport(x=vp_x, y=vp_y, just=just, width=unit(w_tot, "mm"), height=unit(h_tot, "mm"))
    return GTree(children=[bg_grob, txt_grob], vp=vp, cl="textbox", width=unit(w_tot, "mm"), height=unit(h_tot, "mm"))


def grid_textbox(text: Sequence[str], x: Any = None, y: Any = None, gp: dict[str, Any] | None = None, **kwargs: Any) -> GTree:
    gb = textbox_grob(text, x=x, y=y, gp=gp, **kwargs)
    gb.draw()
    return gb


# ---- Interval alignment (`smartAlign` & `smartAlign2`) ----

class _Box:
    def __init__(self, start: float, end: float, orig_idx: int) -> None:
        self.start = float(start)
        self.end = float(end)
        self.new_start = float(start)
        self.new_end = float(end)
        self.height = float(end - start)
        self.mid = 0.5 * (start + end)
        self.orig_idx = orig_idx


class _BoxCluster:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0
        self.height = 0.0
        self.mid = 0.0
        self.boxes: list[_Box] = []

    def add_box(self, box_or_cluster: Any) -> None:
        if isinstance(box_or_cluster, _Box):
            self.boxes.append(box_or_cluster)
        else:
            self.boxes.extend(box_or_cluster.boxes)
        self.start = min(b.start for b in self.boxes)
        self.end = max(b.end for b in self.boxes)
        self.mid = 0.5 * (self.start + self.end)
        self.height = self.end - self.start

    def flatten(self, rng: tuple[float, float]) -> None:
        s = min(b.start for b in self.boxes)
        e = max(b.end for b in self.boxes)
        self.mid = 0.5 * (s + e)
        self.height = sum(b.height for b in self.boxes)
        s2 = self.mid - self.height * 0.5
        e2 = self.mid + self.height * 0.5
        if s2 < rng[0]:
            s2 = rng[0]
            e2 = s2 + self.height
        elif e2 > rng[1]:
            e2 = rng[1]
            s2 = e2 - self.height
        self.start = s2
        self.end = e2
        self.update_box_new_pos()

    def update_box_new_pos(self) -> None:
        cur = self.start
        for b in self.boxes:
            b.new_start = cur
            b.new_end = cur + b.height
            cur = b.new_end

    def move(self, dx: float) -> None:
        self.start += dx
        self.end += dx
        self.mid += dx
        self.update_box_new_pos()


def _clusters_overlap(c1: Any, c2: Any) -> bool:
    return not (c1.end < c2.start - 1e-12 or c2.end < c1.start - 1e-12)


def smartAlign2(
    start: Any,
    end: Any = None,
    range: Sequence[float] | None = None,
    plot: bool = False,
) -> np.ndarray:
    """Resolve overlapping 1D intervals `[start, end]` inside `range` (port of `smartAlign2`)."""
    if end is None:
        arr = np.asarray(start, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("When `end` is None, `start` must be an (N, 2) array.")
        s_arr = arr[:, 0].copy()
        e_arr = arr[:, 1].copy()
    else:
        s_arr = np.atleast_1d(np.asarray(start, dtype=float)).copy()
        e_arr = np.atleast_1d(np.asarray(end, dtype=float)).copy()

    n = len(s_arr)
    if n == 0:
        return np.zeros((0, 2), dtype=float)

    if range is None:
        rng = (float(min(np.min(s_arr), np.min(e_arr))), float(max(np.max(s_arr), np.max(e_arr))))
    else:
        rng = (float(range[0]), float(range[1]))

    if n == 1:
        h = e_arr[0] - s_arr[0]
        ns = max(rng[0], min(s_arr[0], rng[1] - h))
        res = np.array([[ns, ns + h]], dtype=float)
    elif float(np.sum(e_arr - s_arr)) > (rng[1] - rng[0]) + 1e-12:
        mid = 0.5 * (s_arr + e_arr)
        od = np.argsort(mid, kind="mergesort")
        h = e_arr[od] - s_arr[od]
        mid_diff = np.zeros(n, dtype=float)
        mid_diff[1:] = 0.5 * h[1:] + 0.5 * h[:-1]
        mid_radius = float(np.sum(h)) - 0.5 * h[-1] - 0.5 * h[0]
        a_1 = rng[0] + 0.5 * h[0]
        a_n = rng[1] - 0.5 * h[-1]
        if mid_radius <= 1e-12:
            a = np.linspace(a_1, a_n, n)
        else:
            a = a_1 + np.cumsum(mid_diff) / mid_radius * (a_n - a_1)
        ns = a - 0.5 * h
        ne = a + 0.5 * h
        res = np.empty((n, 2), dtype=float)
        res[od, 0] = ns
        res[od, 1] = ne
    else:
        boxes = [_Box(s_arr[i], e_arr[i], i) for i in _range(n)]
        boxes.sort(key=lambda b: b.mid)
        clusters: list[_BoxCluster] = []
        c0 = _BoxCluster()
        c0.add_box(boxes[0])
        clusters.append(c0)
        for b in boxes[1:]:
            if _clusters_overlap(clusters[-1], b):
                clusters[-1].add_box(b)
            else:
                nc = _BoxCluster()
                nc.add_box(b)
                clusters.append(nc)
        for c in clusters:
            c.flatten(rng)

        def _merge(cl: list[_BoxCluster]) -> list[_BoxCluster]:
            while True:
                merged = False
                i = 1
                while i < len(cl):
                    if _clusters_overlap(cl[i], cl[i - 1]):
                        cl[i - 1].add_box(cl[i])
                        cl.pop(i)
                        merged = True
                    else:
                        i += 1
                if not merged:
                    break
                for c in cl:
                    c.flatten(rng)
            return cl

        clusters = _merge(clusters)
        for c in clusters:
            if c.start < rng[0]:
                c.move(rng[0] - c.start)
            if c.end > rng[1]:
                c.move(rng[1] - c.end)
        clusters = _merge(clusters)

        res = np.empty((n, 2), dtype=float)
        for c in clusters:
            for b in c.boxes:
                res[b.orig_idx, 0] = b.new_start
                res[b.orig_idx, 1] = b.new_end

    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        cols = rand_color(n, transparency=0.5)
        for idx in _range(n):
            col = to_rgba_tuple(cols[idx])
            ax.add_patch(plt.Rectangle((0.5, s_arr[idx]), 1.0, e_arr[idx] - s_arr[idx], facecolor=col, edgecolor="black"))
            ax.add_patch(plt.Rectangle((2.5, res[idx, 0]), 1.0, res[idx, 1] - res[idx, 0], facecolor=col, edgecolor="black"))
            ax.plot([1.5, 2.5], [0.5 * (s_arr[idx] + e_arr[idx]), 0.5 * (res[idx, 0] + res[idx, 1])], color="black", lw=0.8)
        ax.set_xlim(0, 4)
        ax.set_ylim(min(rng[0], float(np.min(s_arr))) - 0.5, max(rng[1], float(np.max(e_arr))) + 0.5)
        ax.set_xticks([1, 3])
        ax.set_xticklabels(["original", "adjusted"])

    return res


def smartAlign(x1: Any, x2: Any, xlim: Sequence[float]) -> np.ndarray:
    """Alias for `smartAlign2(x1, x2, xlim)` matching `circlize::smartAlign`."""
    return smartAlign2(x1, x2, range=xlim, plot=False)
