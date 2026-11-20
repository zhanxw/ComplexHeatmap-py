"""Tests for units, gpar, colors, ColorMapping, SingleAnnotation, HeatmapAnnotation, grid, and rich text."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import complexheatmap as ch
from complexheatmap import (
    ColorMapping,
    HeatmapAnnotation,
    SingleAnnotation,
    Viewport,
    add_transparency,
    as_unit,
    c,
    c_ColorMapping,
    c_HeatmapAnnotation,
    colorRamp2,
    convertHeight,
    convertWidth,
    convertX,
    convertY,
    draw,
    gpar,
    grid,
    grid_newpage,
    grid_rect,
    grobWidth,
    gt_render,
    ht_opt,
    is_abs_unit,
    is_unit,
    map_to_colors,
    max_text_height,
    max_text_width,
    popViewport,
    pushViewport,
    recycle_gp,
    rowAnnotation,
    smartAlign2,
    subset_gp,
    textGrob,
    unit,
)


def test_unit_arithmetic_and_conversion():
    u_mm = unit(10, "mm")
    u_cm = unit(2, "cm")
    assert is_unit(u_mm)
    assert is_abs_unit(u_mm)
    assert not is_abs_unit(unit(0.5, "npc"))
    assert not is_abs_unit(unit(1, "mm") + unit(0.5, "npc"))
    assert is_abs_unit(grobWidth(textGrob("hello")))

    assert convertWidth(u_cm, "mm", valueOnly=True) == pytest.approx(20.0)
    assert convertWidth(u_mm + u_cm, "mm", valueOnly=True) == pytest.approx(30.0)
    assert convertHeight(unit(1, "inches"), "mm", valueOnly=True) == pytest.approx(25.4)

    # Native position vs non-native offset rule matching R grid
    grid_newpage(width_in=4.0, height_in=4.0)
    pushViewport(Viewport(x=0.5, y=0.5, width=unit(100, "mm"), height=unit(100, "mm"), xscale=(1.0, 11.0), yscale=(0.0, 10.0)))
    try:
        x_off = convertX(unit(1.0, "native") + unit(5.0, "mm"), "mm", valueOnly=True)
        assert x_off == pytest.approx(5.0)
        x_sum = convertX(unit(2.0, "native") + unit(3.0, "native"), "mm", valueOnly=True)
        assert x_sum == pytest.approx(40.0)
    finally:
        popViewport()


def test_gpar_recycle_subset_and_r_palette():
    gp = gpar(col=[1, 2, 3, 4], fill="red", lty="dashed")
    assert gp["col"][0] == "#000000FF"
    assert gp["fill"] == "#FF0000FF"

    rec = recycle_gp(gpar(col="blue", fontsize=10), 1)
    assert rec["col"] == ["#0000FFFF"]
    assert rec["fontsize"] == [10]

    sub = subset_gp(gp, [0, 2])
    assert len(sub["col"]) == 2
    assert sub["fill"] == "#FF0000FF"


def test_color_ramp_and_color_mapping():
    cf = colorRamp2([-1.0, 0.0, 1.0], ["blue", "white", "red"])
    assert cf(-1.0) == "#0000FFFF"
    assert cf(0.0) == "#FFFFFFFF"
    assert cf(1.0) == "#FF0000FF"
    assert list(cf.breaks) == [-1.0, 0.0, 1.0]

    cm1 = ColorMapping(name="cond", colors={"A": "red", "B": "black"}, na_col="grey")
    cols = map_to_colors(cm1, ["A", "B", None])
    assert cols[0] == "#FF0000FF"
    assert cols[1] == "#000000FF"

    cm2 = ColorMapping(name="cond", colors={"B": "blue", "C": "green"})
    merged = c_ColorMapping(cm1, cm2)
    assert set(merged.levels) == {"A", "B", "C"}
    assert add_transparency("#FF0000", 0.5).endswith("80")


def test_single_and_heatmap_annotation():
    sa = SingleAnnotation(value=np.arange(1, 11))
    assert sa.nobs() == 10
    sub_sa = sa[:5]
    assert sub_sa.nobs() == 5
    draw(sub_sa, test="SingleAnnotation subset")

    df = pd.DataFrame({"type": ["a", "b"] * 5, "score": np.linspace(0, 1, 10)})
    ha = HeatmapAnnotation(
        df=df,
        pts=ch.anno_points(np.arange(10, 0, -1)),
        col={"type": {"a": "tomato", "b": "steelblue"}},
    )
    assert ha.which == "column"
    assert ha.nobs() == 10
    assert ha.names == ["pts", "type", "score"]

    ha.names = ["P", "T", "S"]
    assert ha.names == ["P", "T", "S"]

    ha_sub = ha[2:7, ["T", "P"]]
    assert ha_sub.nobs() == 5
    assert ha_sub.names == ["T", "P"]

    # Auto-reorientation inside rowAnnotation when which is omitted
    ra = rowAnnotation(pts=ch.anno_points(np.arange(10)), simple=list("abcdefghij"))
    assert ra.which == "row"
    assert all(sa.which == "row" for sa in ra.anno_list.values())

    ha2 = HeatmapAnnotation(foo=np.arange(10))
    ha3 = HeatmapAnnotation(bar=ch.anno_barplot(np.arange(10)))
    combined = c_HeatmapAnnotation(ha2, ha3, gap=unit(3, "mm"))
    assert len(combined) == 2


def test_rich_text_textbox_and_smart_align():
    w = max_text_width(["a", "bb", "ccc"], gp=gpar(fontsize=10))
    h = max_text_height(["a", "b\nb"], gp=gpar(fontsize=10))
    assert convertWidth(w, "mm", valueOnly=True) > 0
    assert convertHeight(h, "mm", valueOnly=True) > 0

    gt = gt_render(["<span style='color:blue'>**bold**</span>", "plain *italic*"], align_widths=True)
    assert len(gt) == 2

    grid_newpage()
    grid.textbox(["alpha", "beta gamma", "delta"], word_wrap=True)

    res = smartAlign2(np.array([[1.0, 4.0], [2.0, 5.0]]), range=(0.0, 10.0), plot=False)
    assert res.shape == (2, 2)
    assert res[1, 0] >= res[0, 1] - 1e-6


def test_ht_opt_context_and_xy_in_parent():
    assert ht_opt.DENDROGRAM_PADDING is not None
    with ht_opt.local(DENDROGRAM_PADDING=unit(3, "mm")):
        assert convertWidth(ht_opt.DENDROGRAM_PADDING, "mm", valueOnly=True) == pytest.approx(3.0)

    grid_newpage(width_in=5.0, height_in=5.0)
    pushViewport(Viewport(x=0.5, y=0.5, width=0.5, height=0.5, just=["left", "bottom"]))
    grid_rect()
    u = ch.getXY_in_parent_vp({"x": unit(1, "cm"), "y": unit(1, "cm")})
    popViewport()
    assert is_unit(u["x"]) and is_unit(u["y"])
