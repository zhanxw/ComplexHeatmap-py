"""HeatmapList orchestration, multi-heatmap alignment, legend packing, and 7x7 canvas layout."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._base import AdditiveUnit
from ._options import GLOBAL_PADDING, _ENV, ht_opt
from .annotations import AnnotationFunction, HeatmapAnnotation, c_HeatmapAnnotation
from .colors import ColorMapping, c_ColorMapping
from .dendrogram import Dendrogram
from .grid import (
    Grob,
    Viewport,
    current_device,
    grid_newpage,
    grid_rect,
    grid_text,
    max_text_height,
    max_text_width,
    popViewport,
    pushViewport,
)
from .heatmap import (
    HEATMAP_LAYOUT_COLUMN_COMPONENTS,
    HEATMAP_LAYOUT_ROW_COMPONENTS,
    Heatmap,
    _calc_slice_layout,
    _get_text_just,
    _normalize_side,
    _normalize_title_list,
    _subset_heatmap_cols,
    _subset_heatmap_rows,
    _title_padding_unit,
)
from .legend import Legends, packLegend
from .units import (
    Unit,
    as_unit,
    check_gp,
    convertHeight,
    convertWidth,
    gpar,
    is_abs_unit,
    unit,
    unit_c,
    unit_sum,
)


HEATMAP_LIST_COLUMN_COMPONENTS = (
    "annotation_legend_top",
    "heatmap_legend_top",
    "column_title_top",
    "heatmap_list",
    "column_title_bottom",
    "heatmap_legend_bottom",
    "annotation_legend_bottom",
)

HEATMAP_LIST_ROW_COMPONENTS = (
    "annotation_legend_left",
    "heatmap_legend_left",
    "row_title_left",
    "heatmap_list",
    "row_title_right",
    "heatmap_legend_right",
    "annotation_legend_right",
)


def _normalize_ht_gap(ht_gap: Any, n_ht: int) -> Unit:
    gap_u = unit(2, "mm") if ht_gap is None else as_unit(ht_gap, "mm")
    vals = gap_u.to_mm("x")
    if n_ht <= 1:
        return unit([0.0], "mm")
    out = [float(vals[i % len(vals)]) for i in range(n_ht - 1)]
    return unit(out, "mm")


def _normalize_padding(padding: Any) -> list[float]:
    """Return `[bottom, left, top, right]` padding in mm."""
    if padding is None:
        raw = ht_opt.PAGE_PADDING if hasattr(ht_opt, "PAGE_PADDING") and ht_opt.PAGE_PADDING is not None else GLOBAL_PADDING
    else:
        raw = padding
    u = as_unit(raw, "mm")
    vals = [float(v) for v in u.to_mm("x")]
    if len(vals) == 1:
        return vals * 4
    if len(vals) == 2:
        return [vals[0], vals[1], vals[0], vals[1]]
    if len(vals) == 4:
        return vals
    raise ValueError("`padding` must have length 1, 2, or 4.")


def add_heatmap(object: Any, x: Any, direction: str = "horizontal") -> "HeatmapList":
    """Concatenate Heatmaps, HeatmapAnnotations, or HeatmapLists horizontally (`+`) or vertically (`%v%`)."""
    direction = _normalize_side(direction, ("horizontal", "vertical"), "direction")

    if object is None and x is None:
        return HeatmapList([], direction=direction)
    if object is None:
        if isinstance(x, HeatmapList):
            cp = copy.copy(x)
            return cp
        return add_heatmap(HeatmapList([], direction=direction), x, direction=direction)
    if x is None:
        if isinstance(object, HeatmapList):
            return copy.copy(object)
        return add_heatmap(HeatmapList([], direction=direction), object, direction=direction)

    # Coerce left object into a HeatmapList
    if isinstance(object, HeatmapList):
        ht_list = copy.copy(object)
        ht_list.ht_list = list(object.ht_list)
        ht_list.ht_list_param = copy.deepcopy(object.ht_list_param)
        ht_list.layout = copy.deepcopy(object.layout)
        if ht_list.ht_list and ht_list.direction != direction:
            raise ValueError(
                f"Cannot concatenate {direction!r} item onto a {ht_list.direction!r} HeatmapList. "
                "Wrap the first group explicitly or keep directions consistent."
            )
        ht_list.direction = direction
    elif isinstance(object, (Heatmap, HeatmapAnnotation)):
        if isinstance(object, HeatmapAnnotation):
            expected = "row" if direction == "horizontal" else "column"
            if object.which != expected:
                raise ValueError(
                    f"Cannot concatenate `{object.which}` HeatmapAnnotation {direction}ly; "
                    f"expected `which={expected!r}`."
                )
        ht_list = HeatmapList([object], direction=direction)
    else:
        raise TypeError(f"Cannot add {type(object).__name__} to HeatmapList.")

    # Extract items from right operand x
    if isinstance(x, HeatmapList):
        if x.ht_list and len(ht_list.ht_list) > 0 and x.direction != direction:
            raise ValueError(f"Cannot combine a {x.direction!r} HeatmapList with {direction!r} concatenation.")
        new_items = list(x.ht_list)
    elif isinstance(x, (Heatmap, HeatmapAnnotation)):
        if isinstance(x, HeatmapAnnotation):
            expected = "row" if direction == "horizontal" else "column"
            if x.which != expected:
                raise ValueError(
                    f"Cannot concatenate `{x.which}` HeatmapAnnotation {direction}ly; "
                    f"expected `which={expected!r}` (use `rowAnnotation` for `+`, `HeatmapAnnotation` for `%v%`)."
                )
        new_items = [x]
    else:
        raise TypeError(f"Cannot concatenate {type(x).__name__} onto HeatmapList.")

    # Validate compatible row/column counts
    for item in new_items:
        ht_list._validate_and_append(item)

    return ht_list


def attach_annotation(object: Heatmap, ha: HeatmapAnnotation, side: str = "top", gap: Unit | None = None) -> Heatmap:
    """Attach a HeatmapAnnotation onto an existing side of a Heatmap."""
    return object.attach_annotation(ha, side=side, gap=gap)


def _merge_cm_and_params(cm_list: list[ColorMapping], lp_list: list[dict[str, Any]]) -> list[Legends]:
    """Deduplicate ColorMappings by name (merging discrete mappings) and render Legends grobs."""
    if not cm_list:
        return []
    grouped: dict[str, list[tuple[ColorMapping, dict[str, Any]]]] = {}
    order_names: list[str] = []
    for cm, lp in zip(cm_list, lp_list):
        if cm is None:
            continue
        nm = str(cm.name)
        if nm not in grouped:
            grouped[nm] = []
            order_names.append(nm)
        grouped[nm].append((cm, dict(lp or {})))

    out_legends: list[Legends] = []
    for nm in order_names:
        pairs = grouped[nm]
        if len(pairs) == 1:
            cm, lp = pairs[0]
            out_legends.append(cm.color_mapping_legend(plot=False, **lp))
        else:
            cms = [p[0] for p in pairs]
            if all(c.type == "discrete" for c in cms):
                merged = c_ColorMapping(*cms, name=nm)
                lp0 = pairs[0][1]
                out_legends.append(merged.color_mapping_legend(plot=False, **lp0))
            elif all(c.type == "continuous" for c in cms):
                # Keep first if identical breakpoints/colors, else keep distinct
                c0, lp0 = pairs[0]
                out_legends.append(c0.color_mapping_legend(plot=False, **lp0))
                for ck, lpk in pairs[1:]:
                    if not (np.array_equal(ck.levels, c0.levels) and list(ck.colors) == list(c0.colors)):
                        out_legends.append(ck.color_mapping_legend(plot=False, **lpk))
            else:
                for ck, lpk in pairs:
                    out_legends.append(ck.color_mapping_legend(plot=False, **lpk))
    return out_legends


class HeatmapList(AdditiveUnit):
    """Collection of vertically or horizontally concatenated Heatmaps and HeatmapAnnotations."""

    def __init__(self, ht_list: Sequence[Heatmap | HeatmapAnnotation] = (), direction: str = "horizontal") -> None:
        self.direction = _normalize_side(direction, ("horizontal", "vertical"), "direction")
        self.ht_list: list[Heatmap | HeatmapAnnotation] = []
        self.ht_list_param: dict[str, Any] = {
            "ht_gap": unit(2, "mm"),
            "main_heatmap": None,
            "merge_legends": False,
            "legend_grouping": "adjusted",
            "padding": None,
            "auto_adjust": True,
            "initialized": False,
        }
        self.row_title: list[Any] = []
        self.row_title_param: dict[str, Any] = {}
        self.column_title: list[Any] = []
        self.column_title_param: dict[str, Any] = {}
        self.heatmap_legend_param: dict[str, Any] = {"show": False, "side": "right", "size": unit([0, 0], "mm"), "list": []}
        self.annotation_legend_param: dict[str, Any] = {"show": False, "side": "right", "size": unit([0, 0], "mm"), "list": []}
        self.layout: dict[str, Any] = {
            "initialized": False,
            "layout_size": {
                "annotation_legend_top_height": unit(0, "mm"),
                "heatmap_legend_top_height": unit(0, "mm"),
                "column_title_top_height": unit(0, "mm"),
                "column_title_bottom_height": unit(0, "mm"),
                "heatmap_legend_bottom_height": unit(0, "mm"),
                "annotation_legend_bottom_height": unit(0, "mm"),
                "annotation_legend_left_width": unit(0, "mm"),
                "heatmap_legend_left_width": unit(0, "mm"),
                "row_title_left_width": unit(0, "mm"),
                "row_title_right_width": unit(0, "mm"),
                "heatmap_legend_right_width": unit(0, "mm"),
                "annotation_legend_right_width": unit(0, "mm"),
            },
            "heatmap_list_padding": [0.0, 0.0, 0.0, 0.0],
        }
        self._device: Any = None
        for item in ht_list:
            self._validate_and_append(item)

    def _item_axis_n(self, item: Heatmap | HeatmapAnnotation) -> int | None:
        if isinstance(item, Heatmap):
            return item.nrow if self.direction == "horizontal" else item.ncol
        return item.nobs()

    def _validate_and_append(self, item: Heatmap | HeatmapAnnotation) -> None:
        if isinstance(item, HeatmapAnnotation):
            expected = "row" if self.direction == "horizontal" else "column"
            if item.which != expected:
                raise ValueError(f"HeatmapAnnotation with `which={item.which!r}` cannot be in {self.direction!r} HeatmapList.")
        elif not isinstance(item, Heatmap):
            raise TypeError(f"Expected Heatmap or HeatmapAnnotation, got {type(item).__name__}.")

        new_n = self._item_axis_n(item)
        if new_n is not None:
            for existing in self.ht_list:
                ex_n = self._item_axis_n(existing)
                if ex_n is not None and ex_n != new_n:
                    dim_nm = "rows" if self.direction == "horizontal" else "columns"
                    raise ValueError(f"Number of {dim_nm} ({new_n}) must match existing items ({ex_n}).")
        self.ht_list.append(copy.deepcopy(item))

    def copy_all(self) -> "HeatmapList":
        return copy.deepcopy(self)

    def __len__(self) -> int:
        return len(self.ht_list)

    @property
    def names(self) -> list[str]:
        return [item.name for item in self.ht_list]

    def __getitem__(self, key: Any) -> "HeatmapList":
        if self.layout["initialized"]:
            raise RuntimeError("Subsetting is only allowed before layout initialization.")
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("HeatmapList expects 1D or 2D indexing.")
            i_key, j_key = key
        else:
            i_key, j_key = key, slice(None)

        def _select_components(items: list[Any], sel: Any) -> list[Any]:
            if isinstance(sel, slice):
                if sel == slice(None):
                    return list(items)
                return list(items[sel])
            if isinstance(sel, (int, np.integer)):
                return [items[int(sel)]]
            if isinstance(sel, str):
                matched = []
                for item in items:
                    if isinstance(item, Heatmap):
                        if item.name == sel:
                            matched.append(item)
                    else:
                        if sel in item.names:
                            matched.append(item[:, [sel]])
                        elif item.name == sel:
                            matched.append(item)
                if not matched:
                    raise KeyError(f"Component {sel!r} not found in HeatmapList.")
                return matched
            seq = list(sel)
            if seq and all(isinstance(v, (bool, np.bool_)) for v in seq):
                if len(seq) != len(items):
                    raise IndexError("Boolean component mask length does not match HeatmapList length.")
                return [it for it, flag in zip(items, seq) if flag]
            if seq and all(isinstance(v, str) for v in seq):
                sel_set = set(seq)
                matched = []
                found_names: set[str] = set()
                for item in items:
                    if isinstance(item, Heatmap):
                        if item.name in sel_set:
                            matched.append(item)
                            found_names.add(item.name)
                    else:
                        sub_names = [nm for nm in item.names if nm in sel_set]
                        if sub_names:
                            matched.append(item[:, sub_names])
                            found_names.update(sub_names)
                        elif item.name in sel_set:
                            matched.append(item)
                            found_names.add(item.name)
                missing = [s for s in seq if s not in found_names]
                if missing:
                    raise KeyError(f"Components not found in HeatmapList: {missing}")
                return matched
            return [items[int(idx)] for idx in seq]

        cp = self.copy_all()
        if self.direction == "horizontal":
            if not (isinstance(j_key, slice) and j_key == slice(None)):
                cp.ht_list = _select_components(cp.ht_list, j_key)
            if not (isinstance(i_key, slice) and i_key == slice(None)):
                new_items = []
                for item in cp.ht_list:
                    if isinstance(item, Heatmap):
                        new_items.append(_subset_heatmap_rows(item, i_key))
                    else:
                        new_items.append(item[i_key])
                cp.ht_list = new_items
        else:
            if not (isinstance(i_key, slice) and i_key == slice(None)):
                cp.ht_list = _select_components(cp.ht_list, i_key)
            if not (isinstance(j_key, slice) and j_key == slice(None)):
                new_items = []
                for item in cp.ht_list:
                    if isinstance(item, Heatmap):
                        new_items.append(_subset_heatmap_cols(item, j_key))
                    else:
                        new_items.append(item[j_key])
                cp.ht_list = new_items
        return cp

    # ---- Layout & Alignment ----

    def _find_main_heatmap_index(self, main_heatmap: int | str | None) -> int | None:
        ht_indices = [i for i, item in enumerate(self.ht_list) if isinstance(item, Heatmap)]
        if not ht_indices:
            return None
        if main_heatmap is None:
            for i in ht_indices:
                ht = self.ht_list[i]
                assert isinstance(ht, Heatmap)
                if ht.nrow > 0 and ht.ncol > 0:
                    return i
            return ht_indices[0]
        if isinstance(main_heatmap, (int, np.integer)):
            idx = int(main_heatmap)
            if idx < 0 or idx >= len(self.ht_list) or not isinstance(self.ht_list[idx], Heatmap):
                raise IndexError(f"`main_heatmap={main_heatmap}` does not index a Heatmap.")
            return idx
        for i in ht_indices:
            if self.ht_list[i].name == str(main_heatmap):
                return i
        raise KeyError(f"Heatmap named {main_heatmap!r} not found in HeatmapList.")

    def _collect_legends(self, which: str) -> list[Legends]:
        """Collect either `'heatmap'` or `'annotation'` Legends (respecting merge_legends & legend_grouping)."""
        merge_legends = bool(self.ht_list_param["merge_legends"])
        legend_grouping = self.ht_list_param["legend_grouping"]
        cm_list: list[ColorMapping] = []
        lp_list: list[dict[str, Any]] = []

        want_heatmap = which == "heatmap"
        if not want_heatmap and merge_legends:
            return []

        for item in self.ht_list:
            if isinstance(item, Heatmap):
                # Left / right / top / bottom annotations
                side_map = [
                    ("left", item.left_annotation),
                    ("top", item.top_annotation),
                    ("bottom", item.bottom_annotation),
                    ("right", item.right_annotation),
                ]
                for side, anno in side_map:
                    if anno is None:
                        continue
                    is_row_anno = side in ("left", "right")
                    belongs_to_main = (
                        merge_legends
                        or (legend_grouping == "adjusted" and (
                            (self.direction == "horizontal" and is_row_anno)
                            or (self.direction == "vertical" and not is_row_anno)
                        ))
                    )
                    if (want_heatmap and belongs_to_main) or (not want_heatmap and not belongs_to_main):
                        cm_list.extend(anno.get_color_mapping_list())
                        lp_list.extend(anno.get_legend_param_list())
                # Main matrix legend
                if want_heatmap and item.heatmap_param["show_heatmap_legend"] and item.matrix_color_mapping is not None:
                    cm_list.append(item.matrix_color_mapping)
                    lp_list.append(item.matrix_legend_param)
            elif isinstance(item, HeatmapAnnotation):
                belongs_to_main = merge_legends or (legend_grouping == "adjusted")
                if (want_heatmap and belongs_to_main) or (not want_heatmap and not belongs_to_main):
                    cm_list.extend(item.get_color_mapping_list())
                    lp_list.extend(item.get_legend_param_list())

        return _merge_cm_and_params(cm_list, lp_list)

    def make_layout(
        self,
        row_title: Any = None,
        row_title_side: str = "left",
        row_title_gp: dict[str, Any] | None = None,
        column_title: Any = None,
        column_title_side: str = "top",
        column_title_gp: dict[str, Any] | None = None,

        heatmap_legend_side: str = "right",
        merge_legends: bool = False,
        show_heatmap_legend: bool = True,
        heatmap_legend_list: Sequence[Legends | Grob] = (),
        annotation_legend_side: str = "right",
        show_annotation_legend: bool = True,
        annotation_legend_list: Sequence[Legends | Grob] = (),
        align_heatmap_legend: str | None = None,
        align_annotation_legend: str | None = None,
        legend_grouping: str = "adjusted",

        ht_gap: Unit | None = None,
        main_heatmap: int | str | None = None,
        padding: Any = None,
        auto_adjust: bool = True,

        row_dend_side: str = "original",
        row_sub_title_side: str = "original",
        column_dend_side: str = "original",
        column_sub_title_side: str = "original",

        # Overrides forwarded to main_heatmap
        row_km: int | None = None,
        row_km_repeats: int | None = None,
        row_split: Any = None,
        row_gap: Unit | None = None,
        cluster_rows: Any = None,
        cluster_row_slices: bool | None = None,
        clustering_distance_rows: Any = None,
        clustering_method_rows: str | None = None,
        row_dend_width: Unit | None = None,
        show_row_dend: bool | None = None,
        row_dend_reorder: Any = None,
        row_dend_gp: dict[str, Any] | None = None,
        row_order: Any = None,
        height: Unit | None = None,
        heatmap_height: Unit | None = None,

        column_km: int | None = None,
        column_km_repeats: int | None = None,
        column_split: Any = None,
        column_gap: Unit | None = None,
        cluster_columns: Any = None,
        cluster_column_slices: bool | None = None,
        clustering_distance_columns: Any = None,
        clustering_method_columns: str | None = None,
        column_dend_height: Unit | None = None,
        show_column_dend: bool | None = None,
        column_dend_reorder: Any = None,
        column_dend_gp: dict[str, Any] | None = None,
        column_order: Any = None,
        width: Unit | None = None,
        heatmap_width: Unit | None = None,

        use_raster: bool | None = None,
        page_size_mm: tuple[float, float] = (200.0, 200.0),
    ) -> "HeatmapList":
        if self.ht_list_param["initialized"]:
            return self

        work = self.copy_all()
        if not work.ht_list:
            work.ht_list_param["initialized"] = True
            work.layout["initialized"] = True
            return work

        gap_arg = ht_opt.ht_gap if ht_gap is None and hasattr(ht_opt, "ht_gap") and ht_opt.ht_gap is not None else ht_gap
        eff_auto_adjust = bool(auto_adjust)
        legend_grouping = _normalize_side(legend_grouping, ("adjusted", "original"), "legend_grouping")
        hm_leg_side = _normalize_side(heatmap_legend_side, ("right", "left", "bottom", "top"), "heatmap_legend_side")
        an_leg_side = _normalize_side(annotation_legend_side, ("right", "left", "bottom", "top"), "annotation_legend_side")
        rt_side = _normalize_side(row_title_side, ("left", "right"), "row_title_side")
        ct_side = _normalize_side(column_title_side, ("top", "bottom"), "column_title_side")

        # Identify main heatmap
        i_main = work._find_main_heatmap_index(main_heatmap)
        has_heatmap = i_main is not None

        if has_heatmap:
            assert i_main is not None
            ht_main = work.ht_list[i_main]
            assert isinstance(ht_main, Heatmap)

            # Forward draw() overrides
            if use_raster is not None:
                for item in work.ht_list:
                    if isinstance(item, Heatmap):
                        item.heatmap_param["use_raster"] = bool(use_raster)

            if work.direction == "horizontal":
                if row_km is not None:
                    ht_main.matrix_param["row_km"] = int(row_km)
                if row_km_repeats is not None:
                    ht_main.matrix_param["row_km_repeats"] = int(row_km_repeats)
                if row_split is not None:
                    ht_main.matrix_param["row_split"] = row_split
                if row_gap is not None:
                    ht_main.matrix_param["row_gap"] = as_unit(row_gap, "mm")
                if cluster_rows is not None:
                    if isinstance(cluster_rows, (bool, np.bool_)):
                        ht_main.row_dend_param["cluster"] = bool(cluster_rows)
                        ht_main.row_dend_param["obj"] = None
                    elif callable(cluster_rows):
                        ht_main.row_dend_param["cluster"] = True
                        ht_main.row_dend_param["fun"] = cluster_rows
                    else:
                        from .dendrogram import as_dendrogram
                        ht_main.row_dend_param["cluster"] = True
                        ht_main.row_dend_param["obj"] = as_dendrogram(cluster_rows)
                if cluster_row_slices is not None:
                    ht_main.row_dend_param["cluster_slices"] = bool(cluster_row_slices)
                if clustering_distance_rows is not None:
                    ht_main.row_dend_param["distance"] = clustering_distance_rows
                if clustering_method_rows is not None:
                    ht_main.row_dend_param["method"] = clustering_method_rows
                if row_dend_width is not None:
                    pad_d = convertWidth(as_unit(ht_opt.DENDROGRAM_PADDING, "mm"), "mm", valueOnly=True)
                    rd_mm = convertWidth(as_unit(row_dend_width, "mm"), "mm", valueOnly=True)
                    ht_main.row_dend_param["width"] = unit(0.0 if rd_mm == 0 else rd_mm + pad_d, "mm")
                if show_row_dend is not None:
                    ht_main.row_dend_param["show"] = bool(show_row_dend)
                if row_dend_reorder is not None:
                    ht_main.row_dend_param["reorder"] = row_dend_reorder
                if row_dend_gp is not None:
                    ht_main.row_dend_param["gp"] = check_gp(row_dend_gp)
                if row_order is not None:
                    ht_main.row_order = [int(x) for x in row_order]
                    ht_main.row_dend_param["cluster"] = False
                    ht_main.row_dend_param["show"] = False
                if height is not None:
                    ht_main.matrix_param["height"] = as_unit(height, "mm")
                    ht_main.heatmap_param["height"] = unit(1.0, "npc")
                elif heatmap_height is not None:
                    ht_main.matrix_param["height"] = unit(1.0, "npc")
                    ht_main.heatmap_param["height"] = as_unit(heatmap_height, "mm")
            else:
                if column_km is not None:
                    ht_main.matrix_param["column_km"] = int(column_km)
                if column_km_repeats is not None:
                    ht_main.matrix_param["column_km_repeats"] = int(column_km_repeats)
                if column_split is not None:
                    ht_main.matrix_param["column_split"] = column_split
                if column_gap is not None:
                    ht_main.matrix_param["column_gap"] = as_unit(column_gap, "mm")
                if cluster_columns is not None:
                    if isinstance(cluster_columns, (bool, np.bool_)):
                        ht_main.column_dend_param["cluster"] = bool(cluster_columns)
                        ht_main.column_dend_param["obj"] = None
                    elif callable(cluster_columns):
                        ht_main.column_dend_param["cluster"] = True
                        ht_main.column_dend_param["fun"] = cluster_columns
                    else:
                        from .dendrogram import as_dendrogram
                        ht_main.column_dend_param["cluster"] = True
                        ht_main.column_dend_param["obj"] = as_dendrogram(cluster_columns)
                if cluster_column_slices is not None:
                    ht_main.column_dend_param["cluster_slices"] = bool(cluster_column_slices)
                if clustering_distance_columns is not None:
                    ht_main.column_dend_param["distance"] = clustering_distance_columns
                if clustering_method_columns is not None:
                    ht_main.column_dend_param["method"] = clustering_method_columns
                if column_dend_height is not None:
                    pad_d = convertHeight(as_unit(ht_opt.DENDROGRAM_PADDING, "mm"), "mm", valueOnly=True)
                    cd_mm = convertHeight(as_unit(column_dend_height, "mm"), "mm", valueOnly=True)
                    ht_main.column_dend_param["height"] = unit(0.0 if cd_mm == 0 else cd_mm + pad_d, "mm")
                if show_column_dend is not None:
                    ht_main.column_dend_param["show"] = bool(show_column_dend)
                if column_dend_reorder is not None:
                    ht_main.column_dend_param["reorder"] = column_dend_reorder
                if column_dend_gp is not None:
                    ht_main.column_dend_param["gp"] = check_gp(column_dend_gp)
                if column_order is not None:
                    ht_main.column_order = [int(x) for x in column_order]
                    ht_main.column_dend_param["cluster"] = False
                    ht_main.column_dend_param["show"] = False
                if width is not None:
                    ht_main.matrix_param["width"] = as_unit(width, "mm")
                    ht_main.heatmap_param["width"] = unit(1.0, "npc")
                elif heatmap_width is not None:
                    ht_main.matrix_param["width"] = unit(1.0, "npc")
                    ht_main.heatmap_param["width"] = as_unit(heatmap_width, "mm")

            # Cluster main heatmap on shared axis
            if work.direction == "horizontal":
                ht_main.make_row_cluster()
            else:
                ht_main.make_column_cluster()

            n_ht = len(work.ht_list)
            eff_ht_gap = _normalize_ht_gap(gap_arg, n_ht)

            # Outer zero-dim Heatmap scaffolding if dendrograms/subtitles must sit outside a leading/trailing HeatmapAnnotation
            rd_side_opt = _normalize_side(row_dend_side, ("original", "left", "right"), "row_dend_side")
            rs_side_opt = _normalize_side(row_sub_title_side, ("original", "left", "right"), "row_sub_title_side")
            cd_side_opt = _normalize_side(column_dend_side, ("original", "top", "bottom"), "column_dend_side")
            cs_side_opt = _normalize_side(column_sub_title_side, ("original", "top", "bottom"), "column_sub_title_side")

            if len(work.ht_list) > 1 and eff_auto_adjust:
                if work.direction == "horizontal":
                    if (rd_side_opt == "left" or rs_side_opt == "left") and isinstance(work.ht_list[0], HeatmapAnnotation):
                        dummy = Heatmap(np.empty((ht_main.nrow, 0)), name=f"_scaffold_l_{_ENV.next_heatmap_index()}", show_heatmap_legend=False)
                        dummy.row_order = list(ht_main.row_order)
                        dummy.row_order_list = [list(s) for s in ht_main.row_order_list]
                        work.ht_list.insert(0, dummy)
                        eff_ht_gap = unit_c(unit(0.0, "mm"), eff_ht_gap)
                        i_main += 1
                    if (rd_side_opt == "right" or rs_side_opt == "right") and isinstance(work.ht_list[-1], HeatmapAnnotation):
                        dummy = Heatmap(np.empty((ht_main.nrow, 0)), name=f"_scaffold_r_{_ENV.next_heatmap_index()}", show_heatmap_legend=False)
                        dummy.row_order = list(ht_main.row_order)
                        dummy.row_order_list = [list(s) for s in ht_main.row_order_list]
                        work.ht_list.append(dummy)
                        eff_ht_gap = unit_c(eff_ht_gap, unit(0.0, "mm"))
                else:
                    if (cd_side_opt == "top" or cs_side_opt == "top") and isinstance(work.ht_list[0], HeatmapAnnotation):
                        dummy = Heatmap(np.empty((0, ht_main.ncol)), name=f"_scaffold_t_{_ENV.next_heatmap_index()}", show_heatmap_legend=False)
                        dummy.column_order = list(ht_main.column_order)
                        dummy.column_order_list = [list(s) for s in ht_main.column_order_list]
                        work.ht_list.insert(0, dummy)
                        eff_ht_gap = unit_c(unit(0.0, "mm"), eff_ht_gap)
                        i_main += 1
                    if (cd_side_opt == "bottom" or cs_side_opt == "bottom") and isinstance(work.ht_list[-1], HeatmapAnnotation):
                        dummy = Heatmap(np.empty((0, ht_main.ncol)), name=f"_scaffold_b_{_ENV.next_heatmap_index()}", show_heatmap_legend=False)
                        dummy.column_order = list(ht_main.column_order)
                        dummy.column_order_list = [list(s) for s in ht_main.column_order_list]
                        work.ht_list.append(dummy)
                        eff_ht_gap = unit_c(eff_ht_gap, unit(0.0, "mm"))

            n_ht = len(work.ht_list)

            # Propagate shared ordering, splits, gaps, auto-adjustment, and outer dends/subtitles
            if work.direction == "horizontal":
                for i, item in enumerate(work.ht_list):
                    if isinstance(item, Heatmap) and i != i_main:
                        item.row_order_list = [list(s) for s in ht_main.row_order_list]
                        item.row_order = list(ht_main.row_order)
                        item.row_dend_param["show"] = False
                        item.row_dend_param["cluster"] = False
                        item.matrix_param["row_gap"] = ht_main.matrix_param["row_gap"]
                        item.matrix_param["_row_split_df"] = ht_main.matrix_param.get("_row_split_df")
                        item.matrix_param["height"] = ht_main.matrix_param["height"]
                        item.heatmap_param["height"] = ht_main.heatmap_param["height"]
                        item.row_title = []

                if eff_auto_adjust and n_ht > 1:
                    for i, item in enumerate(work.ht_list):
                        if isinstance(item, Heatmap):
                            keep_left = (i == 0 and item.row_names_param["side"] == "left")
                            keep_right = (i == n_ht - 1 and item.row_names_param["side"] == "right")
                            if not (keep_left or keep_right):
                                item.row_names_param["show"] = False
                                item.row_names_param["anno"] = None

                    if rd_side_opt in ("left", "right"):
                        target_i = 0 if rd_side_opt == "left" else n_ht - 1
                        if isinstance(work.ht_list[target_i], Heatmap) and target_i != i_main:
                            tgt = work.ht_list[target_i]
                            assert isinstance(tgt, Heatmap)
                            tgt.row_dend_list = ht_main.row_dend_list
                            tgt.row_dend_slice = ht_main.row_dend_slice
                            tgt.row_dend_param = dict(ht_main.row_dend_param)
                            tgt.row_dend_param["side"] = rd_side_opt
                            ht_main.row_dend_param["show"] = False
                        elif target_i == i_main:
                            ht_main.row_dend_param["side"] = rd_side_opt

                    if rs_side_opt in ("left", "right"):
                        target_i = 0 if rs_side_opt == "left" else n_ht - 1
                        if isinstance(work.ht_list[target_i], Heatmap) and target_i != i_main:
                            tgt = work.ht_list[target_i]
                            assert isinstance(tgt, Heatmap)
                            tgt.row_title = list(ht_main.row_title)
                            tgt.row_title_param = dict(ht_main.row_title_param)
                            tgt.row_title_param["side"] = rs_side_opt
                            tgt.row_title_param["just"] = _get_text_just(tgt.row_title_param["rot"], rs_side_opt)
                            ht_main.row_title = []
                        elif target_i == i_main:
                            ht_main.row_title_param["side"] = rs_side_opt
                            ht_main.row_title_param["just"] = _get_text_just(ht_main.row_title_param["rot"], rs_side_opt)

                for item in work.ht_list:
                    if isinstance(item, Heatmap):
                        item.prepare(process_rows=False, process_columns=True)
            else:
                for i, item in enumerate(work.ht_list):
                    if isinstance(item, Heatmap) and i != i_main:
                        item.column_order_list = [list(s) for s in ht_main.column_order_list]
                        item.column_order = list(ht_main.column_order)
                        item.column_dend_param["show"] = False
                        item.column_dend_param["cluster"] = False
                        item.matrix_param["column_gap"] = ht_main.matrix_param["column_gap"]
                        item.matrix_param["_column_split_df"] = ht_main.matrix_param.get("_column_split_df")
                        item.matrix_param["width"] = ht_main.matrix_param["width"]
                        item.heatmap_param["width"] = ht_main.heatmap_param["width"]
                        item.column_title = []

                if eff_auto_adjust and n_ht > 1:
                    for i, item in enumerate(work.ht_list):
                        if isinstance(item, Heatmap):
                            keep_top = (i == 0 and item.column_names_param["side"] == "top")
                            keep_bottom = (i == n_ht - 1 and item.column_names_param["side"] == "bottom")
                            if not (keep_top or keep_bottom):
                                item.column_names_param["show"] = False
                                item.column_names_param["anno"] = None

                    if cd_side_opt in ("top", "bottom"):
                        target_i = 0 if cd_side_opt == "top" else n_ht - 1
                        if isinstance(work.ht_list[target_i], Heatmap) and target_i != i_main:
                            tgt = work.ht_list[target_i]
                            assert isinstance(tgt, Heatmap)
                            tgt.column_dend_list = ht_main.column_dend_list
                            tgt.column_dend_slice = ht_main.column_dend_slice
                            tgt.column_dend_param = dict(ht_main.column_dend_param)
                            tgt.column_dend_param["side"] = cd_side_opt
                            ht_main.column_dend_param["show"] = False
                        elif target_i == i_main:
                            ht_main.column_dend_param["side"] = cd_side_opt

                    if cs_side_opt in ("top", "bottom"):
                        target_i = 0 if cs_side_opt == "top" else n_ht - 1
                        if isinstance(work.ht_list[target_i], Heatmap) and target_i != i_main:
                            tgt = work.ht_list[target_i]
                            assert isinstance(tgt, Heatmap)
                            tgt.column_title = list(ht_main.column_title)
                            tgt.column_title_param = dict(ht_main.column_title_param)
                            tgt.column_title_param["side"] = cs_side_opt
                            tgt.column_title_param["just"] = _get_text_just(tgt.column_title_param["rot"], cs_side_opt)
                            ht_main.column_title = []
                        elif target_i == i_main:
                            ht_main.column_title_param["side"] = cs_side_opt
                            ht_main.column_title_param["just"] = _get_text_just(ht_main.column_title_param["rot"], cs_side_opt)

                for item in work.ht_list:
                    if isinstance(item, Heatmap):
                        item.prepare(process_rows=True, process_columns=False)

            work.ht_list_param["main_heatmap"] = i_main
            work.ht_list_param["ht_gap"] = eff_ht_gap
        else:
            # Annotation-only HeatmapList
            work.ht_list_param["main_heatmap"] = None
            work.ht_list_param["ht_gap"] = _normalize_ht_gap(gap_arg, len(work.ht_list))

        work.ht_list_param["merge_legends"] = bool(merge_legends)
        work.ht_list_param["legend_grouping"] = legend_grouping
        work.ht_list_param["auto_adjust"] = eff_auto_adjust
        work.ht_list_param["padding"] = padding

        # Equalize perpendicular layout components across heatmaps
        work._adjust_heatmap_list()

        # Global titles
        ct_gp = check_gp(gpar(fontsize=14) if column_title_gp is None else column_title_gp)
        rt_gp = check_gp(gpar(fontsize=14) if row_title_gp is None else row_title_gp)
        work.column_title = _normalize_title_list(column_title)
        work.column_title_param = {"side": ct_side, "gp": ct_gp}
        work.row_title = _normalize_title_list(row_title)
        work.row_title_param = {"side": rt_side, "gp": rt_gp}

        ls = work.layout["layout_size"]
        if work.column_title:
            ct_str = work.column_title[0]
            ct_h = max_text_height(ct_str, gp=ct_gp) + unit_sum(_title_padding_unit(ct_gp))
            ls[f"column_title_{ct_side}_height"] = ct_h
        if work.row_title:
            rt_str = work.row_title[0]
            rt_w = max_text_width(rt_str, gp=rt_gp, rot=90.0 if rt_side == "left" else 270.0) + unit_sum(_title_padding_unit(rt_gp))
            ls[f"row_title_{rt_side}_width"] = rt_w

        # Legends
        hm_leg_list_items = work._collect_legends("heatmap") if show_heatmap_legend else []
        if heatmap_legend_list:
            extra_hm = [heatmap_legend_list] if isinstance(heatmap_legend_list, (Legends, Grob)) else list(heatmap_legend_list)
            hm_leg_list_items.extend(extra_hm)

        an_leg_list_items = [] if merge_legends or not show_annotation_legend else work._collect_legends("annotation")
        if annotation_legend_list:
            extra_an = [annotation_legend_list] if isinstance(annotation_legend_list, (Legends, Grob)) else list(annotation_legend_list)
            if merge_legends and show_heatmap_legend:
                hm_leg_list_items.extend(extra_an)
            elif show_annotation_legend:
                an_leg_list_items.extend(extra_an)

        max_leg_w = unit(page_size_mm[0] * 0.9, "mm")
        max_leg_h = unit(page_size_mm[1] * 0.9, "mm")

        if hm_leg_list_items:
            hm_pad_u = as_unit(ht_opt.HEATMAP_LEGEND_PADDING, "mm")
            hm_pad_mm = convertWidth(hm_pad_u, "mm", valueOnly=True)
            hm_dir = "vertical" if hm_leg_side in ("left", "right") else "horizontal"
            packed_hm = packLegend(
                *hm_leg_list_items,
                direction=hm_dir,
                max_height=max_leg_h if hm_dir == "vertical" else None,
                max_width=max_leg_w if hm_dir == "horizontal" else None,
            )
            w_mm = convertWidth(packed_hm.width, "mm", valueOnly=True)
            h_mm = convertHeight(packed_hm.height, "mm", valueOnly=True)
            work.heatmap_legend_param = {
                "show": True,
                "side": hm_leg_side,
                "packed": packed_hm,
                "align": align_heatmap_legend or "heatmap_center",
                "padding_mm": hm_pad_mm,
                "size": unit([w_mm + (hm_pad_mm if hm_leg_side in ("left", "right") else 0.0), h_mm + (hm_pad_mm if hm_leg_side in ("top", "bottom") else 0.0)], "mm"),
            }
            if hm_leg_side in ("left", "right"):
                ls[f"heatmap_legend_{hm_leg_side}_width"] = unit(w_mm + hm_pad_mm, "mm")
            else:
                ls[f"heatmap_legend_{hm_leg_side}_height"] = unit(h_mm + hm_pad_mm, "mm")
        else:
            work.heatmap_legend_param = {"show": False, "side": hm_leg_side, "packed": None, "size": unit([0, 0], "mm")}

        if an_leg_list_items:
            an_pad_u = as_unit(ht_opt.ANNOTATION_LEGEND_PADDING, "mm")
            an_pad_mm = convertWidth(an_pad_u, "mm", valueOnly=True)
            an_dir = "vertical" if an_leg_side in ("left", "right") else "horizontal"
            packed_an = packLegend(
                *an_leg_list_items,
                direction=an_dir,
                max_height=max_leg_h if an_dir == "vertical" else None,
                max_width=max_leg_w if an_dir == "horizontal" else None,
            )
            w_mm = convertWidth(packed_an.width, "mm", valueOnly=True)
            h_mm = convertHeight(packed_an.height, "mm", valueOnly=True)
            work.annotation_legend_param = {
                "show": True,
                "side": an_leg_side,
                "packed": packed_an,
                "align": align_annotation_legend or "heatmap_center",
                "padding_mm": an_pad_mm,
                "size": unit([w_mm + (an_pad_mm if an_leg_side in ("left", "right") else 0.0), h_mm + (an_pad_mm if an_leg_side in ("top", "bottom") else 0.0)], "mm"),
            }
            if an_leg_side in ("left", "right"):
                ls[f"annotation_legend_{an_leg_side}_width"] = unit(w_mm + an_pad_mm, "mm")
            else:
                ls[f"annotation_legend_{an_leg_side}_height"] = unit(h_mm + an_pad_mm, "mm")
        else:
            work.annotation_legend_param = {"show": False, "side": an_leg_side, "packed": None, "size": unit([0, 0], "mm")}

        work.ht_list_param["initialized"] = True
        work.layout["initialized"] = True
        return work

    def _adjust_heatmap_list(self) -> None:
        ht_indices = [i for i, item in enumerate(self.ht_list) if isinstance(item, Heatmap)]
        if not ht_indices:
            self.layout["heatmap_list_padding"] = [0.0, 0.0, 0.0, 0.0]
            return

        if self.direction == "horizontal":
            col_pad_mm = convertHeight(as_unit(ht_opt.COLUMN_ANNO_PADDING, "mm"), "mm", valueOnly=True)
            # Resize all top & bottom annotations to common max height
            top_hs = [convertHeight(self.ht_list[i].top_annotation.height, "mm", valueOnly=True) for i in ht_indices if self.ht_list[i].top_annotation is not None]
            bot_hs = [convertHeight(self.ht_list[i].bottom_annotation.height, "mm", valueOnly=True) for i in ht_indices if self.ht_list[i].bottom_annotation is not None]
            max_top_a = max(top_hs, default=0.0)
            max_bot_a = max(bot_hs, default=0.0)
            for i in ht_indices:
                ht = self.ht_list[i]
                assert isinstance(ht, Heatmap)
                if ht.top_annotation is not None and max_top_a > 0:
                    ht.top_annotation.re_size(height=unit(max_top_a, "mm"))
                    ht.layout["layout_size"]["column_anno_top_height"] = unit(max_top_a + col_pad_mm, "mm")
                if ht.bottom_annotation is not None and max_bot_a > 0:
                    ht.bottom_annotation.re_size(height=unit(max_bot_a, "mm"))
                    ht.layout["layout_size"]["column_anno_bottom_height"] = unit(max_bot_a + col_pad_mm, "mm")

            max_ct_top = max(convertHeight(self.ht_list[i].layout["layout_size"]["column_title_top_height"], "mm", valueOnly=True) for i in ht_indices)
            max_ct_bot = max(convertHeight(self.ht_list[i].layout["layout_size"]["column_title_bottom_height"], "mm", valueOnly=True) for i in ht_indices)
            max_top_comp = max(
                sum(convertHeight(self.ht_list[i].layout["layout_size"][k], "mm", valueOnly=True) for k in ("column_dend_top_height", "column_names_top_height", "column_anno_top_height"))
                for i in ht_indices
            )
            max_bot_comp = max(
                sum(convertHeight(self.ht_list[i].layout["layout_size"][k], "mm", valueOnly=True) for k in ("column_dend_bottom_height", "column_names_bottom_height", "column_anno_bottom_height"))
                for i in ht_indices
            )

            for i in ht_indices:
                ls_i = self.ht_list[i].layout["layout_size"]
                ls_i["column_title_top_height"] = unit(max_ct_top, "mm")
                ls_i["column_title_bottom_height"] = unit(max_ct_bot, "mm")
                used_top = convertHeight(ls_i["column_names_top_height"], "mm", valueOnly=True) + convertHeight(ls_i["column_anno_top_height"], "mm", valueOnly=True)
                used_bot = convertHeight(ls_i["column_names_bottom_height"], "mm", valueOnly=True) + convertHeight(ls_i["column_anno_bottom_height"], "mm", valueOnly=True)
                ls_i["column_dend_top_height"] = unit(max(0.0, max_top_comp - used_top), "mm")
                ls_i["column_dend_bottom_height"] = unit(max(0.0, max_bot_comp - used_bot), "mm")

            # Compute outer padding needed when row-annotation extended labels extend past top/bottom components
            ext_top = 0.0
            ext_bot = 0.0
            for item in self.ht_list:
                if isinstance(item, HeatmapAnnotation):
                    e = item.extended.to_mm("x")
                    ext_bot = max(ext_bot, float(e[0]))
                    ext_top = max(ext_top, float(e[2]))
                elif isinstance(item, Heatmap):
                    for ann in (item.left_annotation, item.right_annotation):
                        if ann is not None:
                            e = ann.extended.to_mm("x")
                            ext_bot = max(ext_bot, float(e[0]))
                            ext_top = max(ext_top, float(e[2]))
            pad_top = max(0.0, ext_top - max_top_comp - max_ct_top)
            pad_bot = max(0.0, ext_bot - max_bot_comp - max_ct_bot)

            ext_left = 0.0
            ext_right = 0.0
            first_item = self.ht_list[0]
            if isinstance(first_item, Heatmap):
                left_used = sum(convertWidth(first_item.layout["layout_size"][f"{c}_width"], "mm", valueOnly=True) for c in ("row_title_left", "row_dend_left", "row_names_left", "row_anno_left"))
                e_l = 0.0
                for ann in (first_item.top_annotation, first_item.bottom_annotation):
                    if ann is not None:
                        e_l = max(e_l, float(ann.extended.to_mm("x")[1]))
                ext_left = max(0.0, e_l - left_used)
            last_item = self.ht_list[-1]
            if isinstance(last_item, Heatmap):
                right_used = sum(convertWidth(last_item.layout["layout_size"][f"{c}_width"], "mm", valueOnly=True) for c in ("row_title_right", "row_dend_right", "row_names_right", "row_anno_right"))
                e_r = 0.0
                for ann in (last_item.top_annotation, last_item.bottom_annotation):
                    if ann is not None:
                        e_r = max(e_r, float(ann.extended.to_mm("x")[3]))
                ext_right = max(0.0, e_r - right_used)
            self.layout["heatmap_list_padding"] = [pad_bot, ext_left, pad_top, ext_right]
        else:
            row_pad_mm = convertWidth(as_unit(ht_opt.ROW_ANNO_PADDING, "mm"), "mm", valueOnly=True)
            left_ws = [convertWidth(self.ht_list[i].left_annotation.width, "mm", valueOnly=True) for i in ht_indices if self.ht_list[i].left_annotation is not None]
            right_ws = [convertWidth(self.ht_list[i].right_annotation.width, "mm", valueOnly=True) for i in ht_indices if self.ht_list[i].right_annotation is not None]
            max_left_a = max(left_ws, default=0.0)
            max_right_a = max(right_ws, default=0.0)
            for i in ht_indices:
                ht = self.ht_list[i]
                assert isinstance(ht, Heatmap)
                if ht.left_annotation is not None and max_left_a > 0:
                    ht.left_annotation.re_size(width=unit(max_left_a, "mm"))
                    ht.layout["layout_size"]["row_anno_left_width"] = unit(max_left_a + row_pad_mm, "mm")
                if ht.right_annotation is not None and max_right_a > 0:
                    ht.right_annotation.re_size(width=unit(max_right_a, "mm"))
                    ht.layout["layout_size"]["row_anno_right_width"] = unit(max_right_a + row_pad_mm, "mm")

            max_rt_left = max(convertWidth(self.ht_list[i].layout["layout_size"]["row_title_left_width"], "mm", valueOnly=True) for i in ht_indices)
            max_rt_right = max(convertWidth(self.ht_list[i].layout["layout_size"]["row_title_right_width"], "mm", valueOnly=True) for i in ht_indices)
            max_left_comp = max(
                sum(convertWidth(self.ht_list[i].layout["layout_size"][k], "mm", valueOnly=True) for k in ("row_dend_left_width", "row_names_left_width", "row_anno_left_width"))
                for i in ht_indices
            )
            max_right_comp = max(
                sum(convertWidth(self.ht_list[i].layout["layout_size"][k], "mm", valueOnly=True) for k in ("row_dend_right_width", "row_names_right_width", "row_anno_right_width"))
                for i in ht_indices
            )
            for i in ht_indices:
                ls_i = self.ht_list[i].layout["layout_size"]
                ls_i["row_title_left_width"] = unit(max_rt_left, "mm")
                ls_i["row_title_right_width"] = unit(max_rt_right, "mm")
                used_l = convertWidth(ls_i["row_names_left_width"], "mm", valueOnly=True) + convertWidth(ls_i["row_anno_left_width"], "mm", valueOnly=True)
                used_r = convertWidth(ls_i["row_names_right_width"], "mm", valueOnly=True) + convertWidth(ls_i["row_anno_right_width"], "mm", valueOnly=True)
                ls_i["row_dend_left_width"] = unit(max(0.0, max_left_comp - used_l), "mm")
                ls_i["row_dend_right_width"] = unit(max(0.0, max_right_comp - used_r), "mm")
            self.layout["heatmap_list_padding"] = [0.0, 0.0, 0.0, 0.0]

    def _compute_heatmap_list_spans(self, avail_w_mm: float, avail_h_mm: float) -> list[float]:
        """Allocate exact mm widths (horizontal) or heights (vertical) for each item in `ht_list`."""
        n = len(self.ht_list)
        if n == 0:
            return []
        gaps_mm = self.ht_list_param["ht_gap"].to_mm("x")
        total_gap = sum(float(gaps_mm[i]) for i in range(max(0, n - 1)))

        if self.direction == "horizontal":
            fixed_total = 0.0
            flex_weights: list[float] = []
            item_fixed_w: list[float | None] = []
            for item in self.ht_list:
                if isinstance(item, HeatmapAnnotation):
                    w_mm = convertWidth(item.width, "mm", valueOnly=True)
                    item_fixed_w.append(w_mm)
                    fixed_total += w_mm
                else:
                    if is_abs_unit(item.heatmap_param["width"]):
                        w_mm = convertWidth(item.heatmap_param["width"], "mm", valueOnly=True)
                        item_fixed_w.append(w_mm)
                        fixed_total += w_mm
                    elif is_abs_unit(item.matrix_param["width"]):
                        ws = item.component_widths_mm()
                        w_mm = sum(x or 0.0 for x in ws)
                        item_fixed_w.append(w_mm)
                        fixed_total += w_mm
                    else:
                        # Flexible body: track fixed non-body components + weight for body
                        ws = item.component_widths_mm()
                        fixed_non_body = sum(x or 0.0 for x in ws)
                        fixed_total += fixed_non_body
                        item_fixed_w.append(None)
                        flex_weights.append(float(max(1, item.ncol)))

            rem = max(0.0, avail_w_mm - total_gap - fixed_total)
            tot_wt = sum(flex_weights)
            out_w: list[float] = []
            f_idx = 0
            for i, item in enumerate(self.ht_list):
                if item_fixed_w[i] is not None:
                    out_w.append(float(item_fixed_w[i]))
                else:
                    wt = flex_weights[f_idx]
                    f_idx += 1
                    ws = item.component_widths_mm()
                    fixed_non_body = sum(x or 0.0 for x in ws)
                    body_w = rem * (wt / tot_wt) if tot_wt > 0 else 0.0
                    out_w.append(fixed_non_body + body_w)
            return out_w

        # vertical
        fixed_total = 0.0
        flex_weights = []
        item_fixed_h: list[float | None] = []
        for item in self.ht_list:
            if isinstance(item, HeatmapAnnotation):
                h_mm = convertHeight(item.height, "mm", valueOnly=True)
                item_fixed_h.append(h_mm)
                fixed_total += h_mm
            else:
                if is_abs_unit(item.heatmap_param["height"]):
                    h_mm = convertHeight(item.heatmap_param["height"], "mm", valueOnly=True)
                    item_fixed_h.append(h_mm)
                    fixed_total += h_mm
                elif is_abs_unit(item.matrix_param["height"]):
                    hs = item.component_heights_mm()
                    h_mm = sum(x or 0.0 for x in hs)
                    item_fixed_h.append(h_mm)
                    fixed_total += h_mm
                else:
                    hs = item.component_heights_mm()
                    fixed_non_body = sum(x or 0.0 for x in hs)
                    fixed_total += fixed_non_body
                    item_fixed_h.append(None)
                    flex_weights.append(float(max(1, item.nrow)))

        rem = max(0.0, avail_h_mm - total_gap - fixed_total)
        tot_wt = sum(flex_weights)
        out_h: list[float] = []
        f_idx = 0
        for i, item in enumerate(self.ht_list):
            if item_fixed_h[i] is not None:
                out_h.append(float(item_fixed_h[i]))
            else:
                wt = flex_weights[f_idx]
                f_idx += 1
                hs = item.component_heights_mm()
                fixed_non_body = sum(x or 0.0 for x in hs)
                body_h = rem * (wt / tot_wt) if tot_wt > 0 else 0.0
                out_h.append(fixed_non_body + body_h)
        return out_h

    def _draw_standalone_annotation(self, anno: HeatmapAnnotation, body_start_mm: float, body_len_mm: float) -> None:
        i_main = self.ht_list_param["main_heatmap"]
        ht_main = self.ht_list[i_main] if i_main is not None else None
        prev_ht = _ENV.current_heatmap
        _ENV.current_heatmap = ht_main
        try:
            if self.direction == "horizontal":
                # rowAnnotation placed aligned vertically with heatmap_body band
                if ht_main is not None and isinstance(ht_main, Heatmap):
                    ro_lt = ht_main.row_order_list
                    slice_y, slice_h = _calc_slice_layout([len(s) for s in ro_lt], ht_main.matrix_param["row_gap"], "y")
                    pushViewport(Viewport(x=unit(0, "npc"), y=unit(body_start_mm, "mm"), width=unit(1, "npc"), height=unit(body_len_mm, "mm"), just=("left", "top")))
                    try:
                        for kr, idx_s in enumerate(ro_lt, start=1):
                            anno.draw(index=idx_s, k=kr, n=len(ro_lt), x=unit(0.5, "npc"), y=slice_y[kr - 1], width=unit(1, "npc"), height=slice_h[kr - 1], just=("center", "top"))
                    finally:
                        popViewport()
                else:
                    n = anno.nobs() or 1
                    anno.draw(index=list(range(n)), k=1, n=1, x=unit(0.5, "npc"), y=unit(0.5, "npc"), width=unit(1, "npc"), height=unit(1, "npc"))
            else:
                # columnAnnotation placed aligned horizontally with heatmap_body band
                if ht_main is not None and isinstance(ht_main, Heatmap):
                    co_lt = ht_main.column_order_list
                    slice_x, slice_w = _calc_slice_layout([len(s) for s in co_lt], ht_main.matrix_param["column_gap"], "x")
                    pushViewport(Viewport(x=unit(body_start_mm, "mm"), y=unit(0, "npc"), width=unit(body_len_mm, "mm"), height=unit(1, "npc"), just=("left", "bottom")))
                    try:
                        for kc, idx_s in enumerate(co_lt, start=1):
                            anno.draw(index=idx_s, k=kc, n=len(co_lt), x=slice_x[kc - 1], y=unit(0.5, "npc"), width=slice_w[kc - 1], height=unit(1, "npc"), just=("left", "center"))
                    finally:
                        popViewport()
                else:
                    n = anno.nobs() or 1
                    anno.draw(index=list(range(n)), k=1, n=1, x=unit(0.5, "npc"), y=unit(0.5, "npc"), width=unit(1, "npc"), height=unit(1, "npc"))
        finally:
            _ENV.current_heatmap = prev_ht

    # ---- Canvas Drawing ----

    def draw(
        self,
        *,
        newpage: bool = True,
        background: str = "white",
        padding: Any = None,
        width: float | None = None,
        height: float | None = None,
        fig: Any = None,
        dpi: float = 100.0,
        save: str | None = None,
        show: bool = False,
        post_fun: Callable[..., Any] | None = None,
        **layout_kwargs: Any,
    ) -> "HeatmapList":
        # Default figure dimensions in inches (~ 8x7 in unless user provides size or existing device)
        if fig is not None:
            dev = grid_newpage(dpi=dpi, fig=fig) if newpage else current_device()
        elif newpage:
            w_in = float(width) if width is not None else 8.0
            h_in = float(height) if height is not None else 6.5
            dev = grid_newpage(width_in=w_in, height_in=h_in, dpi=dpi)
        else:
            dev = current_device()

        page_w_mm = dev.width_mm
        page_h_mm = dev.height_mm

        prepared = self.make_layout(padding=padding, page_size_mm=(page_w_mm, page_h_mm), **layout_kwargs)
        prepared._device = dev

        # Fill canvas background
        if background and str(background).lower() != "transparent":
            grid_rect(gp=gpar(fill=background, col=None))

        pad_mm = _normalize_padding(prepared.ht_list_param["padding"])  # [bottom, left, top, right]
        global_w_mm = max(10.0, page_w_mm - pad_mm[1] - pad_mm[3])
        global_h_mm = max(10.0, page_h_mm - pad_mm[0] - pad_mm[2])

        pushViewport(
            Viewport(
                name="global",
                x=unit(pad_mm[1], "mm"),
                y=unit(pad_mm[0], "mm"),
                width=unit(global_w_mm, "mm"),
                height=unit(global_h_mm, "mm"),
                just=("left", "bottom"),
            )
        )
        try:
            ls = prepared.layout["layout_size"]
            hl_pad = prepared.layout["heatmap_list_padding"]

            # 7 outer column widths (left -> right) and 7 outer row heights (top -> bottom)
            col_fixed = [
                convertWidth(ls[f"{c}_width"], "mm", valueOnly=True) if c != "heatmap_list" else None
                for c in HEATMAP_LIST_ROW_COMPONENTS
            ]
            row_fixed = [
                convertHeight(ls[f"{r}_height"], "mm", valueOnly=True) if r != "heatmap_list" else None
                for r in HEATMAP_LIST_COLUMN_COMPONENTS
            ]

            hl_w_mm = max(10.0, global_w_mm - sum(v for v in col_fixed if v is not None))
            hl_h_mm = max(10.0, global_h_mm - sum(v for v in row_fixed if v is not None))
            final_cols = [hl_w_mm if v is None else v for v in col_fixed]
            final_rows = [hl_h_mm if v is None else v for v in row_fixed]

            x_cols = [0.0] * 7
            cx = 0.0
            for c_i in range(7):
                x_cols[c_i] = cx
                cx += final_cols[c_i]

            y_rows_top = [0.0] * 7
            cy = sum(final_rows)
            for r_i in range(7):
                y_rows_top[r_i] = cy
                cy -= final_rows[r_i]

            # Inner heatmap_list box (row=3, col=3) minus annotation extended padding
            hl_x = x_cols[3] + hl_pad[1]
            hl_y_top = y_rows_top[3] - hl_pad[2]
            inner_w = max(5.0, final_cols[3] - hl_pad[1] - hl_pad[3])
            inner_h = max(5.0, final_rows[3] - hl_pad[0] - hl_pad[2])

            # Locate heatmap_body perpendicular band for aligning standalone annotations & centering legends
            i_main = prepared.ht_list_param["main_heatmap"]
            body_top_rel = inner_h
            body_h_mm = inner_h
            body_left_rel = 0.0
            body_w_mm = inner_w

            # Draw heatmap_list items
            pushViewport(
                Viewport(
                    name="heatmap_list",
                    x=unit(hl_x, "mm"),
                    y=unit(hl_y_top, "mm"),
                    width=unit(inner_w, "mm"),
                    height=unit(inner_h, "mm"),
                    just=("left", "top"),
                )
            )
            try:
                if prepared.ht_list:
                    spans = prepared._compute_heatmap_list_spans(inner_w, inner_h)
                    gaps_mm = prepared.ht_list_param["ht_gap"].to_mm("x")

                    if i_main is not None:
                        ht_main = prepared.ht_list[i_main]
                        assert isinstance(ht_main, Heatmap)
                        if prepared.direction == "horizontal":
                            hs_main = ht_main.component_heights_mm()
                            fixed_h = sum(x for x in hs_main if x is not None)
                            body_h_mm = max(0.0, inner_h - fixed_h) if hs_main[4] is None else hs_main[4]
                            body_top_rel = inner_h - sum(x or 0.0 for x in hs_main[:4])
                        else:
                            ws_main = ht_main.component_widths_mm()
                            fixed_w = sum(x for x in ws_main if x is not None)
                            body_w_mm = max(0.0, inner_w - fixed_w) if ws_main[4] is None else ws_main[4]
                            body_left_rel = sum(x or 0.0 for x in ws_main[:4])

                    if prepared.direction == "horizontal":
                        cur_x = 0.0
                        for idx_it, item in enumerate(prepared.ht_list):
                            sp_w = spans[idx_it]
                            pushViewport(Viewport(x=unit(cur_x, "mm"), y=unit(inner_h, "mm"), width=unit(sp_w, "mm"), height=unit(inner_h, "mm"), just=("left", "top")))
                            try:
                                if isinstance(item, Heatmap):
                                    item.draw_in_viewport()
                                else:
                                    prepared._draw_standalone_annotation(item, body_top_rel, body_h_mm)
                            finally:
                                popViewport()
                            if idx_it < len(prepared.ht_list) - 1:
                                cur_x += sp_w + float(gaps_mm[idx_it])
                    else:
                        cur_y_top = inner_h
                        for idx_it, item in enumerate(prepared.ht_list):
                            sp_h = spans[idx_it]
                            pushViewport(Viewport(x=unit(0.0, "mm"), y=unit(cur_y_top, "mm"), width=unit(inner_w, "mm"), height=unit(sp_h, "mm"), just=("left", "top")))
                            try:
                                if isinstance(item, Heatmap):
                                    item.draw_in_viewport()
                                else:
                                    prepared._draw_standalone_annotation(item, body_left_rel, body_w_mm)
                            finally:
                                popViewport()
                            if idx_it < len(prepared.ht_list) - 1:
                                cur_y_top -= sp_h + float(gaps_mm[idx_it])
            finally:
                popViewport()

            # Draw global column title
            if prepared.column_title:
                ct_side = prepared.column_title_param["side"]
                r_ct = 2 if ct_side == "top" else 4
                pushViewport(Viewport(name=f"global_column_title_{ct_side}", x=unit(x_cols[3], "mm"), y=unit(y_rows_top[r_ct], "mm"), width=unit(final_cols[3], "mm"), height=unit(final_rows[r_ct], "mm"), just=("left", "top")))
                try:
                    ct_gp = prepared.column_title_param["gp"]
                    pad_c = _title_padding_unit(ct_gp)
                    if ct_side == "top":
                        grid_text(prepared.column_title[0], x=unit(0.5, "npc"), y=pad_c[0], just="bottom", gp=ct_gp)
                    else:
                        grid_text(prepared.column_title[0], x=unit(0.5, "npc"), y=unit(1, "npc") - pad_c[0], just="top", gp=ct_gp)
                finally:
                    popViewport()

            # Draw global row title
            if prepared.row_title:
                rt_side = prepared.row_title_param["side"]
                c_rt = 2 if rt_side == "left" else 4
                pushViewport(Viewport(name=f"global_row_title_{rt_side}", x=unit(x_cols[c_rt], "mm"), y=unit(y_rows_top[3], "mm"), width=unit(final_cols[c_rt], "mm"), height=unit(final_rows[3], "mm"), just=("left", "top")))
                try:
                    rt_gp = prepared.row_title_param["gp"]
                    pad_r = _title_padding_unit(rt_gp)
                    rot_r = 90.0 if rt_side == "left" else 270.0
                    if rt_side == "left":
                        grid_text(prepared.row_title[0], x=unit(1, "npc") - pad_r[0], y=unit(0.5, "npc"), rot=rot_r, just="center", gp=rt_gp)
                    else:
                        grid_text(prepared.row_title[0], x=pad_r[0], y=unit(0.5, "npc"), rot=rot_r, just="center", gp=rt_gp)
                finally:
                    popViewport()

            # Draw heatmap & annotation legends in outer slots
            def _draw_outer_legend(leg_param: dict[str, Any], is_heatmap: bool) -> None:
                if not leg_param.get("show") or leg_param.get("packed") is None:
                    return
                side = leg_param["side"]
                packed: Legends = leg_param["packed"]
                pad_l = float(leg_param.get("padding_mm", 2.0))
                if side == "right":
                    c_idx = 5 if is_heatmap else 6
                    r_idx = 3
                    y_ref = y_rows_top[3] - inner_h + body_top_rel - 0.5 * body_h_mm
                    x_ref = x_cols[c_idx] + pad_l
                    pushViewport(Viewport(x=unit(x_ref, "mm"), y=unit(y_ref, "mm"), width=unit(max(1.0, final_cols[c_idx] - pad_l), "mm"), height=unit(final_rows[r_idx], "mm"), just=("left", "center")))
                    try:
                        packed.draw(x=unit(0, "npc"), y=unit(0.5, "npc"), just="left")
                    finally:
                        popViewport()
                elif side == "left":
                    c_idx = 1 if is_heatmap else 0
                    r_idx = 3
                    y_ref = y_rows_top[3] - inner_h + body_top_rel - 0.5 * body_h_mm
                    x_ref = x_cols[c_idx] + final_cols[c_idx] - pad_l
                    pushViewport(Viewport(x=unit(x_ref, "mm"), y=unit(y_ref, "mm"), width=unit(max(1.0, final_cols[c_idx] - pad_l), "mm"), height=unit(final_rows[r_idx], "mm"), just=("right", "center")))
                    try:
                        packed.draw(x=unit(1, "npc"), y=unit(0.5, "npc"), just="right")
                    finally:
                        popViewport()
                elif side == "top":
                    c_idx = 3
                    r_idx = 1 if is_heatmap else 0
                    x_ref = x_cols[3] + body_left_rel + 0.5 * body_w_mm
                    y_ref = y_rows_top[r_idx] - final_rows[r_idx] + pad_l
                    pushViewport(Viewport(x=unit(x_ref, "mm"), y=unit(y_ref, "mm"), width=unit(final_cols[c_idx], "mm"), height=unit(max(1.0, final_rows[r_idx] - pad_l), "mm"), just=("center", "bottom")))
                    try:
                        packed.draw(x=unit(0.5, "npc"), y=unit(0, "npc"), just="bottom")
                    finally:
                        popViewport()
                else:  # bottom
                    c_idx = 3
                    r_idx = 5 if is_heatmap else 6
                    x_ref = x_cols[3] + body_left_rel + 0.5 * body_w_mm
                    y_ref = y_rows_top[r_idx] - pad_l
                    pushViewport(Viewport(x=unit(x_ref, "mm"), y=unit(y_ref, "mm"), width=unit(final_cols[c_idx], "mm"), height=unit(max(1.0, final_rows[r_idx] - pad_l), "mm"), just=("center", "top")))
                    try:
                        packed.draw(x=unit(0.5, "npc"), y=unit(1, "npc"), just="top")
                    finally:
                        popViewport()

            _draw_outer_legend(prepared.heatmap_legend_param, is_heatmap=True)
            _draw_outer_legend(prepared.annotation_legend_param, is_heatmap=False)

            # Run any per-heatmap post_fun and top-level post_fun
            for item in prepared.ht_list:
                if isinstance(item, Heatmap) and item.heatmap_param.get("post_fun") is not None:
                    item.heatmap_param["post_fun"](prepared)
            if post_fun is not None:
                post_fun(prepared)
        finally:
            popViewport()

        _ENV.last = prepared
        if save is not None:
            dev.fig.savefig(save, dpi=dpi, facecolor=dev.fig.get_facecolor())
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return prepared

    def show(self) -> "HeatmapList":
        return self.draw()

    def summary(self) -> str:
        lines = [f"A HeatmapList ({self.direction}) with {len(self.ht_list)} component(s):"]
        for i, item in enumerate(self.ht_list):
            if isinstance(item, Heatmap):
                lines.append(f"  [{i}] Heatmap {item.name!r}: {item.nrow} x {item.ncol}")
            else:
                lines.append(f"  [{i}] HeatmapAnnotation {item.name!r}: which={item.which}, len={len(item)}")
        txt = "\n".join(lines)
        print(txt)
        return txt

    def _repr_png_(self) -> bytes:
        import io
        import matplotlib.pyplot as plt
        res = self.draw(show=False)
        buf = io.BytesIO()
        res._device.fig.savefig(buf, format="png", dpi=120)
        plt.close(res._device.fig)
        return buf.getvalue()

    def __repr__(self) -> str:
        return f"HeatmapList(direction={self.direction!r}, components={self.names})"


# ---- Top-level Dispatchers & Accessors ----

def draw(object: Any, **kwargs: Any) -> Any:
    """Draw a Heatmap, HeatmapList, HeatmapAnnotation, SingleAnnotation, AnnotationFunction, Legends, or Grob."""
    if isinstance(object, (Heatmap, HeatmapList)):
        return object.draw(**kwargs)
    if hasattr(object, "draw") and callable(object.draw):
        return object.draw(**kwargs)
    raise TypeError(f"Object of type {type(object).__name__} cannot be drawn.")


def ht_size(ht_list: Heatmap | HeatmapList, unit_name: str = "mm", valueOnly: bool = False) -> Any:
    """Return total width and height of a Heatmap or HeatmapList."""
    hl = HeatmapList([ht_list]) if isinstance(ht_list, Heatmap) else ht_list
    prep = hl if hl.layout["initialized"] else hl.make_layout()
    ls = prep.layout["layout_size"]
    hl_pad = prep.layout["heatmap_list_padding"]
    w_outer = sum(convertWidth(ls[f"{c}_width"], "mm", valueOnly=True) for c in HEATMAP_LIST_ROW_COMPONENTS if c != "heatmap_list")
    h_outer = sum(convertHeight(ls[f"{r}_height"], "mm", valueOnly=True) for r in HEATMAP_LIST_COLUMN_COMPONENTS if r != "heatmap_list")

    gaps_mm = prep.ht_list_param["ht_gap"].to_mm("x")
    n = len(prep.ht_list)
    gap_sum = sum(float(gaps_mm[i]) for i in range(max(0, n - 1)))

    if prep.direction == "horizontal":
        w_in = gap_sum + hl_pad[1] + hl_pad[3]
        h_in = hl_pad[0] + hl_pad[2]
        for item in prep.ht_list:
            if isinstance(item, Heatmap):
                ws = item.component_widths_mm()
                hs = item.component_heights_mm()
                w_in += sum(x if x is not None else max(10.0, item.ncol * 4.0) for x in ws)
                h_in = max(h_in, sum(x if x is not None else max(10.0, item.nrow * 4.0) for x in hs) + hl_pad[0] + hl_pad[2])
            else:
                w_in += convertWidth(item.width, "mm", valueOnly=True)
    else:
        w_in = hl_pad[1] + hl_pad[3]
        h_in = gap_sum + hl_pad[0] + hl_pad[2]
        for item in prep.ht_list:
            if isinstance(item, Heatmap):
                ws = item.component_widths_mm()
                hs = item.component_heights_mm()
                w_in = max(w_in, sum(x if x is not None else max(10.0, item.ncol * 4.0) for x in ws) + hl_pad[1] + hl_pad[3])
                h_in += sum(x if x is not None else max(10.0, item.nrow * 4.0) for x in hs)
            else:
                h_in += convertHeight(item.height, "mm", valueOnly=True)

    tot_w = w_outer + w_in
    tot_h = h_outer + h_in
    u = unit([tot_w, tot_h], "mm")
    if valueOnly:
        return [convertWidth(unit(tot_w, "mm"), unit_name, valueOnly=True), convertHeight(unit(tot_h, "mm"), unit_name, valueOnly=True)]
    if unit_name == "mm":
        return u
    return unit(
        [convertWidth(unit(tot_w, "mm"), unit_name, valueOnly=True), convertHeight(unit(tot_h, "mm"), unit_name, valueOnly=True)],
        unit_name,
    )


def _get_target_heatmap(obj: Heatmap | HeatmapList, name: str | int | None = None) -> list[Heatmap]:
    if isinstance(obj, Heatmap):
        if not obj.layout["initialized"]:
            obj.prepare()
        return [obj]
    if isinstance(obj, HeatmapList):
        if not obj.layout["initialized"]:
            obj = obj.make_layout()
        hms = [x for x in obj.ht_list if isinstance(x, Heatmap) and not x.name.startswith("_scaffold_")]
        if name is not None:
            if isinstance(name, (int, np.integer)):
                return [hms[int(name)]]
            return [h for h in hms if h.name == str(name)]
        return hms
    raise TypeError(f"Expected Heatmap or HeatmapList, got {type(obj).__name__}.")


def row_order(object: Heatmap | HeatmapList, name: str | int | None = None) -> Any:
    """Return ordered 0-based row indices from a Heatmap or HeatmapList."""
    if isinstance(object, Heatmap):
        if not object.layout["initialized"]:
            object.prepare()
        return object.row_order_list[0] if len(object.row_order_list) == 1 else list(object.row_order_list)

    if isinstance(object, HeatmapList):
        hms = _get_target_heatmap(object, name=name)
        if not hms:
            return []
        if object.direction == "horizontal" or name is not None or len(hms) == 1:
            hm = hms[0]
            return hm.row_order_list[0] if len(hm.row_order_list) == 1 else list(hm.row_order_list)
        return {hm.name: (hm.row_order_list[0] if len(hm.row_order_list) == 1 else list(hm.row_order_list)) for hm in hms}
    raise TypeError(f"Unsupported type for `row_order`: {type(object).__name__}")


def column_order(object: Heatmap | HeatmapList, name: str | int | None = None) -> Any:
    """Return ordered 0-based column indices from a Heatmap or HeatmapList."""
    if isinstance(object, Heatmap):
        if not object.layout["initialized"]:
            object.prepare()
        return object.column_order_list[0] if len(object.column_order_list) == 1 else list(object.column_order_list)

    if isinstance(object, HeatmapList):
        hms = _get_target_heatmap(object, name=name)
        if not hms:
            return []
        if object.direction == "vertical" or name is not None or len(hms) == 1:
            hm = hms[0]
            return hm.column_order_list[0] if len(hm.column_order_list) == 1 else list(hm.column_order_list)
        return {hm.name: (hm.column_order_list[0] if len(hm.column_order_list) == 1 else list(hm.column_order_list)) for hm in hms}
    raise TypeError(f"Unsupported type for `column_order`: {type(object).__name__}")


def row_dend(object: Heatmap | HeatmapList, name: str | int | None = None, on_slice: bool = False) -> Any:
    """Return row Dendrogram(s) from a Heatmap or HeatmapList."""
    if isinstance(object, Heatmap):
        if not object.layout["initialized"]:
            object.prepare()
        if on_slice:
            return object.row_dend_slice
        return object.row_dend_list[0] if len(object.row_dend_list) == 1 else list(object.row_dend_list)

    hms = _get_target_heatmap(object, name=name)
    if not hms:
        return None
    if object.direction == "horizontal" or name is not None or len(hms) == 1:
        hm = hms[0]
        if on_slice:
            return hm.row_dend_slice
        return hm.row_dend_list[0] if len(hm.row_dend_list) == 1 else list(hm.row_dend_list)
    return {hm.name: (hm.row_dend_slice if on_slice else (hm.row_dend_list[0] if len(hm.row_dend_list) == 1 else list(hm.row_dend_list))) for hm in hms}


def column_dend(object: Heatmap | HeatmapList, name: str | int | None = None, on_slice: bool = False) -> Any:
    """Return column Dendrogram(s) from a Heatmap or HeatmapList."""
    if isinstance(object, Heatmap):
        if not object.layout["initialized"]:
            object.prepare()
        if on_slice:
            return object.column_dend_slice
        return object.column_dend_list[0] if len(object.column_dend_list) == 1 else list(object.column_dend_list)

    hms = _get_target_heatmap(object, name=name)
    if not hms:
        return None
    if object.direction == "vertical" or name is not None or len(hms) == 1:
        hm = hms[0]
        if on_slice:
            return hm.column_dend_slice
        return hm.column_dend_list[0] if len(hm.column_dend_list) == 1 else list(hm.column_dend_list)
    return {hm.name: (hm.column_dend_slice if on_slice else (hm.column_dend_list[0] if len(hm.column_dend_list) == 1 else list(hm.column_dend_list))) for hm in hms}
