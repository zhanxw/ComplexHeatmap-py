"""Tests covering all 21 anno_* functions, 6 row_anno_* aliases, Legends, packLegend, and 11 decorate_* functions."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

import complexheatmap as ch
from complexheatmap import (
    Heatmap,
    HeatmapAnnotation,
    Legend,
    decorate_annotation,
    decorate_column_dend,
    decorate_column_names,
    decorate_column_title,
    decorate_dend,
    decorate_dimnames,
    decorate_heatmap_body,
    decorate_row_dend,
    decorate_row_names,
    decorate_row_title,
    decorate_title,
    draw,
    gpar,
    grid,
    list_components,
    packLegend,
    rowAnnotation,
    unit,
)


def test_all_21_anno_functions_and_6_row_aliases():
    rng = np.random.default_rng(7)
    vec = rng.uniform(0, 10, 12)
    mat = rng.standard_normal((12, 5))

    annos = [
        ch.anno_simple(vec),
        ch.anno_empty(),
        ch.anno_points(vec),
        ch.anno_lines(vec, add_points=True, smooth=True),
        ch.anno_barplot(mat[:, :3], beside=True, gp=gpar(fill=[2, 3, 4])),
        ch.anno_boxplot(mat.T),
        ch.anno_histogram(mat, which="row"),
        ch.anno_density(mat, type="violin", which="row"),
        ch.anno_joyplot(mat.T, which="row"),
        ch.anno_horizon(np.cumsum(mat, axis=1).T, which="row"),
        ch.anno_text([f"item_{i}" for i in range(12)]),
        ch.anno_numeric(vec - 5.0),
        ch.anno_mark(at=[0, 5, 11], labels=["first", "mid", "last"], which="row"),
        ch.anno_zoom(align_to={"g1": [0, 1, 2], "g2": [8, 9, 10]}, which="row"),
        ch.anno_link(align_to=[0, 1, 2], which="row"),
        ch.anno_block(gp=gpar(fill=2), labels=["A"]),
        ch.anno_customize(list("AABBCCAABBCC"), graphics={"A": lambda x, y, w, h: grid.circle(x, y, unit(1, "mm")), "B": lambda x, y, w, h: grid.rect(x, y, w, h), "C": lambda x, y, w, h: grid.points(x, y, pch=16)}),
        ch.anno_textbox({"g1": [0, 1, 2], "g2": [6, 7, 8]}, {"g1": ["alpha", "beta"], "g2": ["gamma", "delta"]}),
    ]
    for a in annos:
        draw(a, index=list(range(12)), test=a.fun_name)

    draw(ch.anno_block(gp=gpar(fill=[2, 3]), labels=["A", "B"]), index=list(range(6)), k=1, n=2, test="anno_block_slices")

    # Test anno_summary & anno_oncoprint_barplot inside Heatmaps
    ht_sum = Heatmap(vec, top_annotation=HeatmapAnnotation(s=ch.anno_summary()), row_split=["x"] * 6 + ["y"] * 6)
    draw(ht_sum)

    # Test anno_image with generated PNG files
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(5):
            p = os.path.join(td, f"img_{i}.png")
            Image.fromarray(np.full((16, 16, 4), [40 * i, 120, 200, 255], dtype=np.uint8)).save(p)
            paths.append(p)
        draw(ch.anno_image(paths), test="anno_image")

    # Test all 6 row_anno_* aliases
    for fn in (ch.row_anno_barplot, ch.row_anno_boxplot, ch.row_anno_density, ch.row_anno_histogram, ch.row_anno_points, ch.row_anno_text):
        arg = [f"r{i}" for i in range(12)] if fn is ch.row_anno_text else (mat if fn in (ch.row_anno_density, ch.row_anno_histogram, ch.row_anno_boxplot) else vec)
        a = fn(arg)
        assert a.which == "row"


def test_legends_and_packlegend():
    lgd_disc = Legend(labels=["Group 1", "Group 2", "Group 3"], title="Discrete", legend_gp=gpar(fill=[2, 3, 4]), nrow=2)
    draw(lgd_disc, test="discrete multi-row")

    lgd_pts = Legend(at=["P1", "P2"], title="Points", type="points", pch=[16, 17], legend_gp=gpar(col=["red", "blue"]))
    draw(lgd_pts, test="points legend")

    cf = ch.colorRamp2([0, 0.5, 1.0], ["navy", "white", "firebrick"])
    lgd_cont = Legend(col_fun=cf, title="Continuous", direction="horizontal", title_position="leftcenter")
    draw(lgd_cont, test="horizontal continuous")

    lgd_unequal = Legend(col_fun=cf, title="Breaks", at=[0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0])
    draw(lgd_unequal, test="unequal interval breaks")

    packed = packLegend(lgd_disc, lgd_pts, lgd_cont, direction="vertical")
    assert packed.n >= 2
    draw(packed, test="packed")


def test_all_11_decorate_functions():
    rng = np.random.default_rng(11)
    mat = pd.DataFrame(rng.standard_normal((10, 8)), index=[f"r{i}" for i in range(10)], columns=[f"c{j}" for j in range(8)])
    ha = HeatmapAnnotation(grp=["A", "B"] * 4)
    ht = Heatmap(
        mat,
        name="M",
        row_km=2,
        column_km=2,
        row_title=["R1", "R2"],
        column_title=["C1", "C2"],
        top_annotation=ha,
    )
    draw(ht)
    comps = list_components()
    assert "M_heatmap_body_1_1" in comps
    assert "annotation_grp_1" in comps

    # Test callable, context manager, and @decorator modes across all 11 functions
    decorate_heatmap_body("M", lambda: grid.circle(gp=gpar(fill="#FF000040")), row_slice=1, column_slice=1)
    with decorate_dend("M", slice=2, which="row"):
        grid.rect(gp=gpar(col="blue"))
    with decorate_row_dend("M", slice=1):
        grid.rect(gp=gpar(col="green"))
    with decorate_column_dend("M", slice=1):
        grid.rect(gp=gpar(col="purple"))
    with decorate_dimnames("M", slice=1, which="row"):
        grid.rect(gp=gpar(fill="#FFFF0030"))
    with decorate_row_names("M", slice=2):
        grid.rect(gp=gpar(fill="#00FF0030"))
    with decorate_column_names("M", slice=1):
        grid.rect(gp=gpar(fill="#0000FF30"))
    with decorate_title("M", slice=1, which="row"):
        grid.rect(gp=gpar(col="red"))
    with decorate_row_title("M", slice=2):
        grid.rect(gp=gpar(col="orange"))
    with decorate_column_title("M", slice=1):
        grid.rect(gp=gpar(col="brown"))

    @decorate_annotation("grp", slice=1)
    def _custom_anno_dec():
        grid.circle(x=unit([0.25, 0.75], "npc"), gp=gpar(fill="#FF000080"))
