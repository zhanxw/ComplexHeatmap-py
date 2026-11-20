"""Faithful Python ports of all 102 executable examples in ComplexHeatmap/man/*.Rd."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import complexheatmap as ch
from complexheatmap import (
    ColorMapping,
    GRanges,
    Heatmap,
    HeatmapAnnotation,
    Heatmap3D,
    IRanges,
    Legend,
    SingleAnnotation,
    AnnotationFunction,
    UpSet,
    add_transparency,
    adjust_dend_by_x,
    alter_graphic,
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
    anno_mark,
    anno_numeric,
    anno_points,
    anno_simple,
    anno_summary,
    anno_text,
    anno_textbox,
    anno_zoom,
    annotation_axis_grob,
    as_dendrogram,
    attach_annotation,
    bar3D,
    bin_genome,
    c,
    c_ColorMapping,
    c_HeatmapAnnotation,
    cluster_between_groups,
    cluster_within_group,
    colorRamp2,
    color_branches,
    columnAnnotation,
    column_dend,
    column_order,
    comb_degree,
    comb_name,
    comb_size,
    compare_heatmap,
    compare_heatmap_2,
    compare_pheatmap,
    decorate_annotation,
    decorate_dend,
    decorate_dimnames,
    decorate_heatmap_body,
    decorate_title,
    default_axis_param,
    dend_xy,
    densityHeatmap,
    dist,
    dist2,
    draw,
    extract_comb,
    findOverlaps,
    frequencyHeatmap,
    full_comb_code,
    generateRandomBed,
    getXY_in_parent_vp,
    gpar,
    grid,
    grid_boxplot,
    grid_dendrogram,
    grid_draw,
    grid_newpage,
    grid_points,
    grid_rect,
    grid_segments,
    grid_text,
    grid_textbox,
    grobWidth,
    gt_render,
    hclust,
    height,
    ht_opt,
    is_abs_unit,
    list_to_matrix,
    make_comb_mat,
    map_to_colors,
    max_text_height,
    max_text_width,
    merge_dendrogram,
    names,
    nobs,
    normalize_genomic_signals_to_bins,
    oncoPrint,
    packLegend,
    pindex,
    popViewport,
    pushViewport,
    rand_color,
    restore_matrix,
    rowAnnotation,
    row_dend,
    row_order,
    set_name,
    set_names,
    set_size,
    size,
    smartAlign2,
    subset_gp,
    summary,
    t,
    test_alter_fun,
    textGrob,
    unit,
    upset_right_annotation,
    upset_top_annotation,
    v,
    viewport,
    width,
)

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _rd_AnnotationFunction():
    x = np.arange(1, 11, dtype=float)
    def _fun(index, k=1, n=1):
        m = len(index)
        pushViewport(viewport(xscale=(0.5, m + 0.5), yscale=(0, 10)))
        grid_rect()
        grid_points(np.arange(1, m + 1), x[index], default_units="native")
        if k == 1:
            grid.yaxis()
        popViewport()

    anno1 = AnnotationFunction(
        fun=_fun,
        var_import={"x": x},
        n=10,
        subsettable=True,
        height=unit(2, "cm"),
    )
    m = np.vstack([np.arange(1, 11), np.arange(11, 21)])
    draw(Heatmap(m, top_annotation=HeatmapAnnotation(foo=anno1)))
    draw(Heatmap(m, top_annotation=HeatmapAnnotation(foo=anno1), column_km=2))


def _rd_ColorMapping():
    cm = ColorMapping(colors={"A": "red", "B": "black"})
    repr(cm)
    col_fun = colorRamp2([0, 1], ["white", "red"])
    cm2 = ColorMapping(col_fun=col_fun)
    repr(cm2)


def _rd_Extract_AnnotationFunction():
    anno = anno_simple(np.arange(1, 11))
    sub = anno[:5]
    draw(sub, test="subset of column annotation")


def _rd_Extract_Heatmap():
    rng = np.random.default_rng(1)
    m = pd.DataFrame(rng.standard_normal((10, 10)), index=list("abcdefghij"), columns=list("ABCDEFGHIJ"))
    ht = Heatmap(m)
    _ = ht[:5, :]
    _ = ht[:5]
    _ = ht[:, :5]
    _ = ht[:5, :5]


def _rd_Extract_HeatmapAnnotation():
    ha = HeatmapAnnotation(foo=np.arange(1, 11), bar=anno_points(np.arange(10, 0, -1)), sth=np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)]))
    _ = ha[:5, :]
    _ = ha[:, ["foo", "bar"]]
    _ = ha[:, :2]
    _ = ha[:5, ["foo", "sth"]]


def _rd_Extract_HeatmapList():
    rng = np.random.default_rng(1)
    ht_list = Heatmap(rng.standard_normal((10, 10)), name="rnorm") + rowAnnotation(foo=np.arange(1, 11), bar=anno_points(np.arange(10, 0, -1))) + Heatmap(rng.uniform(size=(10, 10)), name="runif")
    _ = ht_list[:5, :]
    ht_list_v = Heatmap(rng.standard_normal((10, 10)), name="rnorm") % v % columnAnnotation(foo=np.arange(1, 11), bar=anno_points(np.arange(10, 0, -1))) % v % Heatmap(rng.uniform(size=(10, 10)), name="runif")
    _ = ht_list_v[:, :5]


def _rd_Extract_SingleAnnotation():
    ha = SingleAnnotation(value=np.arange(1, 11))
    sub = ha[:5]
    draw(sub, test="ha[1:5]")


def _rd_Extract_comb_mat():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    lt = {
        "a": rng.choice(letters, 10, replace=False),
        "b": rng.choice(letters, 15, replace=False),
        "c": rng.choice(letters, 20, replace=False),
    }
    m = make_comb_mat(lt)
    m2 = m[:, comb_size(m) >= 3]
    _ = comb_size(m2)


def _rd_Heatmap3D():
    rng = np.random.default_rng(1)
    m = rng.integers(1, 101, size=(6, 6))
    draw(Heatmap3D(m))


def _rd_Legend():
    lgd = Legend(labels=MONTH_NAMES[:6], title="foo", legend_gp=gpar(fill=list(range(1, 7))))
    draw(lgd, test="add labels and title")
    col_fun = colorRamp2([0, 0.5, 1], ["blue", "white", "red"])
    draw(Legend(col_fun=col_fun, title="foo"), test="only col_fun")
    draw(Legend(col_fun=col_fun, title="foo", at=[0, 0.1, 0.15, 0.5, 0.9, 0.95, 1]), test="unequal interval breaks")


def _rd_Legends_class():
    lgd = Legend(at=[1, 2, 3, 4])
    repr(lgd)
    _ = lgd.grob


def _rd_SingleAnnotation():
    draw(SingleAnnotation(value=np.arange(1, 11)), test="single column annotation")
    m = pd.DataFrame(np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)]), columns=["a", "b"])
    draw(SingleAnnotation(value=m), test="matrix as column annotation")
    anno = anno_barplot(np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)]))
    draw(SingleAnnotation(fun=anno), test="anno_barplot as input")
    val = np.arange(1, 11, dtype=float)
    def _custom(index, k=1, n=1):
        pushViewport(viewport(xscale=(0.5, len(index) + 0.5), yscale=(float(val.min()), float(val.max()))))
        grid_points(np.arange(1, len(index) + 1), val[index])
        grid_rect()
        if k == 1:
            grid.yaxis()
        popViewport()
    ha = SingleAnnotation(fun=_custom, height=unit(4, "cm"))
    draw(ha, index=list(range(10)), test="self-defined function")


def _rd_UpSet():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    lt = {
        "a": rng.choice(letters, 10, replace=False),
        "b": rng.choice(letters, 15, replace=False),
        "c": rng.choice(letters, 20, replace=False),
    }
    m = make_comb_mat(lt)
    draw(UpSet(m))
    draw(UpSet(t(m)))
    m_u = make_comb_mat(lt, mode="union")
    draw(UpSet(m_u))
    draw(UpSet(m_u, comb_col=[2, 2, 2, 3, 3, 3, 1]))
    lt2 = {"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)}
    m2 = make_comb_mat(lt2)
    max1 = float(max(set_size(m).max(), set_size(m2).max()))
    max2 = float(max(comb_size(m).max(), comb_size(m2).max()))
    draw(
        UpSet(m, top_annotation=upset_top_annotation(m, ylim=(0, max2)), right_annotation=upset_right_annotation(m, ylim=(0, max1)), column_title="UpSet1")
        + UpSet(m2, top_annotation=upset_top_annotation(m2, ylim=(0, max2)), right_annotation=upset_right_annotation(m2, ylim=(0, max1)), column_title="UpSet2")
    )


def _rd_adjust_dend_by_x():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((10, 10))
    dend = as_dendrogram(hclust(dist(m)))
    dend = adjust_dend_by_x(dend, np.sort(rng.uniform(size=10)))
    dend = adjust_dend_by_x(dend, unit(np.arange(1, 11), "cm"))
    return dend


def _rd_alter_graphic():
    df = pd.DataFrame(
        {"s1": ["snv;indel", "", "snv"], "s2": ["snv", "snv;indel", ""], "s3": ["indel", "snv", "indel;snv"]},
        index=["g1", "g2", "g3"],
    )
    col = {"snv": "red", "indel": "blue"}
    op = oncoPrint(
        df,
        alter_fun={
            "snv": alter_graphic("rect", width=0.9, height=0.9, fill=col["snv"]),
            "indel": alter_graphic("rect", width=0.9, height=0.9, fill=col["indel"]),
        },
        col=col,
    )
    draw(op)


def _rd_anno_barplot():
    draw(anno_barplot(np.arange(1, 11)), test="a vector")
    rng = np.random.default_rng(1)
    m = rng.uniform(size=(10, 4))
    m = m / m.sum(axis=1, keepdims=True)
    draw(anno_barplot(m, gp=gpar(fill=[2, 3, 4, 5]), bar_width=1, height=unit(6, "cm")), test="proportion matrix")


def _rd_anno_block():
    rng = np.random.default_rng(1)
    draw(
        Heatmap(
            rng.standard_normal((10, 10)),
            top_annotation=HeatmapAnnotation(foo=anno_block(gp=gpar(fill=[2, 3, 4]), labels=["group1", "group2", "group3"], labels_gp=gpar(col="white"))),
            column_km=3,
            left_annotation=rowAnnotation(foo=anno_block(gp=gpar(fill=[2, 3, 4]), labels=["group1", "group2", "group3"], labels_gp=gpar(col="white"))),
            row_km=3,
        )
    )
    col_d = {"1": "red", "2": "blue", "A": "green", "B": "orange"}
    def _pf(index, levels):
        lv = [str(levels)] if isinstance(levels, str) else [str(x) for x in levels]
        c_fill = col_d.get(lv[-1], "grey")
        c_txt = col_d.get(lv[0], "black")
        grid_rect(gp=gpar(fill=c_fill, col="black"))
        grid_text(",".join(lv), 0.5, 0.5, rot=90, gp=gpar(col=c_txt))
    draw(
        Heatmap(rng.standard_normal((10, 10)), row_km=2, row_split=rng.choice(["A", "B"], 10))
        + rowAnnotation(foo=anno_block(panel_fun=_pf, width=unit(3, "cm")))
    )
    col2 = {"foo": "red", "bar": "blue"}
    draw(
        Heatmap(rng.standard_normal((10, 10)), cluster_rows=False)
        + rowAnnotation(
            foo=anno_block(
                align_to={"foo": list(range(0, 4)), "bar": list(range(5, 10))},
                panel_fun=lambda index, nm: (grid_rect(gp=gpar(fill=col2[nm])), grid_text(nm, 0.5, 0.5)),
                width=unit(2, "cm"),
            )
        )
    )


def _rd_anno_boxplot():
    rng = np.random.default_rng(123)
    m = rng.standard_normal((10, 10))
    draw(anno_boxplot(m, height=unit(4, "cm")), test="anno_boxplot")
    draw(anno_boxplot(m, height=unit(4, "cm"), gp=gpar(fill=list(range(1, 11)))), test="anno_boxplot with gp")


def _rd_anno_customize():
    rng = np.random.default_rng(1)
    x = sorted(rng.choice(["a", "b", "c"], 10))
    graphics = {
        "a": lambda x, y, w, h: grid_points(x, y, pch=16),
        "b": lambda x, y, w, h: grid_rect(x, y, w * 0.8, h * 0.8, gp=gpar(fill="red")),
        "c": lambda x, y, w, h: grid_segments(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5, gp=gpar(lty=2)),
    }
    anno = anno_customize(x, graphics=graphics)
    m = rng.standard_normal((10, 10))
    ht = Heatmap(m, top_annotation=HeatmapAnnotation(bar=x, foo=anno))
    lgd = Legend(title="foo", at=list(graphics.keys()), graphics=graphics)
    draw(ht, annotation_legend_list=[lgd])


def _rd_anno_density():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((10, 10))
    draw(anno_density(m, which="row"), test="normal density")
    draw(anno_density(m, which="row", type="violin"), test="violin")
    draw(anno_density(m, which="row", type="heatmap"), test="heatmap")
    draw(anno_density(m, which="row", type="heatmap", heatmap_colors=["white", "orange"]), test="heatmap, colors")


def _rd_anno_empty():
    draw(anno_empty(), test="anno_empty")
    draw(anno_empty(border=False), test="anno_empty without border")


def _rd_anno_histogram():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((100, 10))
    draw(anno_histogram(m.T, which="row"), test="row histogram")
    draw(anno_histogram(m.T, which="row", gp=gpar(fill=list(range(1, 11)))), test="row histogram with color")
    draw(anno_histogram(m.T, which="row", n_breaks=20), test="row histogram breaks")


def _rd_anno_horizon():
    rng = np.random.default_rng(1)
    lt = [np.cumprod(1 + rng.uniform(-x / 100.0, x / 100.0, 100)) - 1.0 for x in range(1, 21)]
    draw(anno_horizon(lt, which="row"), test="horizon chart")
    draw(anno_horizon(lt, which="row", gp=gpar(pos_fill="orange", neg_fill="darkgreen")), test="horizon chart, col")
    draw(anno_horizon(lt, which="row", negative_from_top=True), test="horizon chart + negative_from_top")
    draw(anno_horizon(lt, which="row", gap=unit(1, "mm")), test="horizon chart + gap")


def _rd_anno_image():
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        imgs = []
        for idx in range(10):
            arr = np.zeros((32, 32, 4), dtype=np.uint8)
            arr[:, :, idx % 3] = 220
            arr[:, :, 3] = 255
            p = os.path.join(tmpdir, f"icon_{idx}.png")
            Image.fromarray(arr).save(p)
            imgs.append(p)
        draw(anno_image(imgs), test="png")
        imgs_partial = list(imgs)
        for k in range(5):
            imgs_partial[k] = ""
        draw(anno_image(imgs_partial), test="some of png")


def _rd_anno_joyplot():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((100, 10))
    draw(anno_joyplot(m, width=unit(4, "cm"), which="row"), test="joyplot")
    draw(anno_joyplot(m, width=unit(4, "cm"), which="row", gp=gpar(fill=list(range(1, 11)))), test="joyplot + col")
    draw(anno_joyplot(m, width=unit(4, "cm"), which="row", scale=1), test="joyplot + scale")


def _rd_anno_lines():
    rng = np.random.default_rng(1)
    draw(anno_lines(rng.uniform(size=10)), test="anno_lines")
    m = np.column_stack([np.concatenate([np.arange(1, 6), np.arange(1, 6)]), np.concatenate([np.arange(5, 0, -1), np.arange(5, 0, -1)])])
    draw(anno_lines(m, gp=gpar(col=[2, 3])), test="matrix")
    draw(anno_lines(m, gp=gpar(col=[2, 3]), add_points=True, pt_gp=gpar(col=[5, 6]), pch=[1, 16]), test="matrix+pts")


def _rd_anno_mark():
    at = [0, 1, 2, 3, 19, 59, 96, 97, 98, 99]
    anno = anno_mark(at=at, labels=MONTH_NAMES[:10], which="row")
    draw(anno, index=list(range(100)), test="anno_mark")
    m = np.arange(1, 1001).reshape(100, 10)
    draw(Heatmap(m, cluster_rows=False, cluster_columns=False) + rowAnnotation(mark=anno))
    draw(Heatmap(m) + rowAnnotation(mark=anno))


def _rd_anno_numeric():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((10, 10))
    x = rng.standard_normal(10)
    draw(Heatmap(m, right_annotation=rowAnnotation(numeric=anno_numeric(x))))


def _rd_anno_points():
    rng = np.random.default_rng(1)
    draw(anno_points(rng.uniform(size=10)), test="anno_points")
    draw(anno_points(rng.uniform(size=(10, 2)), pch=[1, 2]), test="matrix")


def _rd_anno_simple():
    rng = np.random.default_rng(1)
    draw(anno_simple(np.arange(1, 11)), test="a numeric vector")
    draw(anno_simple(np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)])), test="a matrix")
    draw(anno_simple(np.arange(1, 11), pch=[1, 2, 3, 4, None, 6, 7, 8, None, 10]), test="pch has NA")
    draw(anno_simple(np.arange(1, 11), pch=["A"] * 5 + [None] * 5), test="char pch")
    pch_mat = np.arange(1, 21, dtype=object).reshape(10, 2)
    pch_mat.ravel()[rng.choice(20, 10, replace=False)] = None
    draw(anno_simple(np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)]), pch=pch_mat), test="matrix pch")


def _rd_anno_summary():
    rng = np.random.default_rng(1)
    ha = HeatmapAnnotation(summary=anno_summary(height=unit(4, "cm")))
    v_cat = rng.choice(["a", "b"], 50)
    sp = rng.choice(["a", "b"], 50)
    draw(Heatmap(v_cat, top_annotation=ha, width=unit(1, "cm"), split=sp))
    ha2 = HeatmapAnnotation(summary=anno_summary(gp=gpar(fill=[2, 3]), height=unit(4, "cm")))
    draw(Heatmap(rng.standard_normal(50), top_annotation=ha2, width=unit(1, "cm"), split=sp))


def _rd_anno_text():
    draw(anno_text(MONTH_NAMES), test="month names")
    draw(anno_text(MONTH_NAMES, gp=gpar(fontsize=16)), test="month names fontsize")
    draw(anno_text(MONTH_NAMES, gp=gpar(fontsize=[i + 5 for i in range(12)])), test="changing fontsize")
    draw(anno_text(MONTH_NAMES, which="row"), test="row month names")
    draw(anno_text(MONTH_NAMES, location=0, rot=45, just="left", gp=gpar(col=list(range(1, 13)))), test="rot left")
    draw(anno_text(MONTH_NAMES, location=1, rot=45, just="right", gp=gpar(fontsize=[i + 5 for i in range(12)])), test="rot right")


def _rd_anno_textbox():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((100, 10))
    split = rng.choice(list("abcdefghij"), 100)
    uniq_sp = list(dict.fromkeys(split))
    text_dict = {
        k: pd.DataFrame({
            "word": MONTH_NAMES,
            "col": rand_color(12, friendly=True),
            "fontsize": rng.uniform(6, 14, 12),
        })
        for k in uniq_sp
    }
    draw(Heatmap(mat, cluster_rows=False, row_split=split, right_annotation=rowAnnotation(wc=anno_textbox(split, text_dict))))


def _rd_anno_zoom():
    rng = np.random.default_rng(123)
    m = rng.standard_normal((100, 10))
    subgroup = rng.choice(["a", "b", "c"], 100, p=[1 / 16, 5 / 16, 10 / 16])
    rg = (float(m.min()), float(m.max()))
    def _panel_fun(index, nm):
        pushViewport(viewport(xscale=rg, yscale=(0, 2)))
        grid_rect()
        grid.xaxis(gp=gpar(fontsize=8))
        grid_boxplot(m[index, :].ravel(), pos=1, direction="horizontal")
        grid_text(f"distribution of group {nm}", 0.5 * (rg[0] + rg[1]), y=1.9, just="top", default_units="native", gp=gpar(fontsize=10))
        popViewport()
    anno = anno_zoom(align_to=subgroup, which="row", panel_fun=_panel_fun, size=unit(2, "cm"), gap=unit(1, "cm"), width=unit(4, "cm"))
    draw(Heatmap(m, right_annotation=rowAnnotation(foo=anno), row_split=subgroup))


def _rd_annotation_axis_grob():
    for side in ("left", "right", "top", "bottom"):
        for facing in ("outside", "inside"):
            gb = annotation_axis_grob(at=[1, 2, 3], labels=MONTH_NAMES[:3], labels_rot=0, side=side, facing=facing)
            grid_newpage()
            pushViewport(viewport(xscale=(0, 4), yscale=(0, 6), width=0.6, height=0.6))
            grid_rect()
            grid_text(f"side={side}, facing={facing}")
            grid_draw(gb)
            popViewport()


def _rd_attach_annotation():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((10, 10))
    ht = Heatmap(m)
    ht = attach_annotation(ht, HeatmapAnnotation(foo=np.arange(1, 11)))
    ht = attach_annotation(ht, HeatmapAnnotation(bar=list("abcdefghij")))
    draw(ht)


def _rd_bar3D():
    grid_newpage()
    bar3D([0.3, 0.7], 0.5, 0.2, 0.2, 0.2, fill=[2, 3])


def _rd_c_ColorMapping():
    cm1 = ColorMapping(colors={"A": "red", "B": "black"})
    cm2 = ColorMapping(colors={"B": "blue", "C": "green"})
    return c_ColorMapping(cm1, cm2)


def _rd_c_HeatmapAnnotation():
    ha1 = HeatmapAnnotation(foo=np.arange(1, 11))
    ha2 = HeatmapAnnotation(bar=anno_points(np.arange(10, 0, -1)))
    ha = c_HeatmapAnnotation(ha1, ha2)
    ha3 = HeatmapAnnotation(sth=np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)]))
    return c_HeatmapAnnotation(ha1, ha2, ha3, gap=unit([1, 4], "mm"))


def _rd_cluster_between_groups():
    rng = np.random.default_rng(1)
    m = pd.DataFrame(rng.standard_normal((10, 12)), columns=list("abcdefghijkl"))
    fa = ["a"] * 2 + ["b"] * 4 + ["c"] * 6
    dend = cluster_between_groups(m, fa)
    grid_dendrogram(dend, test=True)


def _rd_cluster_within_group():
    rng = np.random.default_rng(1)
    m = pd.DataFrame(rng.standard_normal((10, 12)), columns=list("abcdefghijkl"))
    fa = ["a"] * 2 + ["b"] * 4 + ["c"] * 6
    dend = cluster_within_group(m, fa)
    grid_dendrogram(dend, test=True)


def _rd_column_dend_Heatmap():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((10, 10))
    ht = draw(Heatmap(mat))
    _ = column_dend(ht)
    ht2 = draw(Heatmap(mat, column_km=2))
    _ = column_dend(ht2)


def _rd_column_dend_HeatmapList():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((10, 10))
    hl1 = draw(Heatmap(mat) + Heatmap(mat))
    _ = column_dend(hl1)
    hl2 = draw(Heatmap(mat, column_km=2) + Heatmap(mat, column_km=2))
    _ = column_dend(hl2)
    _ = column_dend(hl2, on_slice=True)
    hl3 = draw(Heatmap(mat) % v % Heatmap(mat))
    _ = column_dend(hl3)
    hl4 = draw(Heatmap(mat, column_km=2) % v % Heatmap(mat))
    _ = column_dend(hl4)


def _rd_column_order_Heatmap():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((10, 10))
    ht = draw(Heatmap(mat))
    _ = column_order(ht)
    ht2 = draw(Heatmap(mat, column_km=2))
    _ = column_order(ht2)


def _rd_column_order_HeatmapList():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((10, 10))
    _ = column_order(draw(Heatmap(mat) + Heatmap(mat)))
    _ = column_order(draw(Heatmap(mat, column_km=2) + Heatmap(mat, column_km=2)))
    _ = column_order(draw(Heatmap(mat) % v % Heatmap(mat)))
    _ = column_order(draw(Heatmap(mat, column_km=2) % v % Heatmap(mat)))


def _rd_comb_degree():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    return comb_degree(m)


def _rd_comb_name():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    return comb_name(m), comb_name(m, readable=True)


def _rd_comb_size():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    return comb_size(m)


def _rd_compare_heatmap2():
    draw(compare_heatmap_2(np.random.default_rng(1).standard_normal((10, 10))))


def _rd_compare_heatmap():
    draw(compare_heatmap(np.random.default_rng(1).standard_normal((10, 10))))


def _rd_compare_pheatmap():
    draw(compare_pheatmap(np.random.default_rng(1).standard_normal((10, 10))))


def _rd_decorate_annotation():
    rng = np.random.default_rng(123)
    ha1 = HeatmapAnnotation(df=pd.DataFrame({"type": ["a", "b"] * 5}))
    ha2 = rowAnnotation(point=anno_points(rng.uniform(size=10), which="row"))
    draw(Heatmap(rng.standard_normal((10, 10)), name="mat", km=2, top_annotation=ha1) + ha2)
    decorate_annotation("type", lambda: grid.circle(x=unit([0.2, 0.4, 0.6, 0.8], "npc"), gp=gpar(fill="#FF000080")))
    decorate_annotation("point", lambda: grid_rect(gp=gpar(fill="#FF000080")), slice=2)


def _rd_decorate_dend():
    rng = np.random.default_rng(123)
    draw(Heatmap(rng.standard_normal((10, 10)), name="mat", km=2))
    decorate_dend("mat", lambda: grid_rect(gp=gpar(fill="#FF000080")), which="row", slice=2)


def _rd_decorate_dimnames():
    rng = np.random.default_rng(123)
    mat = pd.DataFrame(rng.standard_normal((10, 10)), index=list("abcdefghij"), columns=list("ABCDEFGHIJ"))
    draw(Heatmap(mat, name="mat", km=2))
    decorate_dimnames("mat", lambda: grid_rect(gp=gpar(fill="#FF000080")), which="row", slice=2)


def _rd_decorate_heatmap_body():
    rng = np.random.default_rng(123)
    draw(Heatmap(rng.standard_normal((10, 10)), name="mat"))
    decorate_heatmap_body("mat", lambda: grid.circle(gp=gpar(fill="#FF000080")))


def _rd_decorate_title():
    rng = np.random.default_rng(123)
    draw(Heatmap(rng.standard_normal((10, 10)), name="mat", km=2, row_title=["S1", "S2"]))
    decorate_title("mat", lambda: grid_rect(gp=gpar(fill="#FF000080")), which="row", slice=2)


def _rd_default_axis_param():
    return default_axis_param("column"), default_axis_param("row")


def _rd_dend_xy():
    rng = np.random.default_rng(1)
    dend1 = as_dendrogram(hclust(dist(rng.standard_normal((10, 10)))))
    _ = dend_xy(dend1)
    dend1 = adjust_dend_by_x(dend1, np.sort(rng.uniform(size=10)))
    _ = dend_xy(dend1)
    dend1 = adjust_dend_by_x(dend1, unit(np.arange(1, 11), "cm"))
    return dend_xy(dend1)


def _rd_densityHeatmap():
    rng = np.random.default_rng(1)
    matrix = pd.DataFrame(rng.standard_normal((10, 10)), columns=list("abcdefghij"))
    draw(densityHeatmap(matrix))
    draw(densityHeatmap([rng.standard_normal(10), rng.standard_normal(10)]))
    ha = HeatmapAnnotation(points=anno_points(rng.uniform(size=10)), anno=["A"] * 5 + ["B"] * 5, col={"anno": {"A": "red", "B": "blue"}})
    draw(densityHeatmap(matrix, anno=ha))
    draw(densityHeatmap(matrix, anno=ha) % v % Heatmap(matrix, height=unit(6, "cm")))


def _rd_dist2():
    rng = np.random.default_rng(1)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    lt = [rng.choice(letters, int(rng.integers(6, 11)), replace=False) for _ in range(10)]
    return dist2(lt, lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y)))


def _rd_draw_Legends():
    lgd = Legend(at=[1, 2, 3, 4], title="foo")
    grid_newpage()
    lgd.draw(x=unit(0, "npc"), y=unit(0, "npc"), just=["left", "bottom"])
    pushViewport(viewport(x=unit(0, "npc"), y=unit(0, "npc"), just=["left", "bottom"]))
    grid_draw(lgd.grob)
    popViewport()


def _rd_extract_comb():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    return extract_comb(m, "110")


def _rd_frequencyHeatmap():
    rng = np.random.default_rng(1)
    mat = pd.DataFrame(rng.standard_normal((10, 10)), columns=list("abcdefghij"))
    draw(frequencyHeatmap(mat))
    draw(frequencyHeatmap(mat, use_3d=True))


def _rd_full_comb_code():
    return full_comb_code(2), full_comb_code(3), full_comb_code(4), full_comb_code(4, True)


def _rd_getXY_in_parent_vp():
    grid_newpage()
    pushViewport(viewport(x=0.5, y=0.5, width=0.5, height=0.5, just=["left", "bottom"]))
    grid_rect()
    grid_points(x=unit(2, "cm"), y=unit(2, "cm"), pch=1)
    u2 = getXY_in_parent_vp({"x": unit(2, "cm"), "y": unit(2, "cm")})
    popViewport()
    grid_rect(gp=gpar(col="red"))
    grid_points(x=u2["x"], y=u2["y"], pch=2)


def _rd_grid_boxplot():
    rng = np.random.default_rng(1)
    lt = [rng.standard_normal(100), rng.standard_normal(100)]
    grid_newpage()
    pushViewport(viewport(xscale=(0.5, 2.5), yscale=(float(min(lt[0].min(), lt[1].min())), float(max(lt[0].max(), lt[1].max())))))
    grid_boxplot(lt[0], pos=1, gp=gpar(fill="red"))
    grid_boxplot(lt[1], pos=2, gp=gpar(fill="green"))
    popViewport()


def _rd_grid_dendrogram():
    rng = np.random.default_rng(1)
    dend = as_dendrogram(hclust(dist(rng.standard_normal((10, 10)))))
    grid_newpage()
    from complexheatmap import dend_heights
    pushViewport(viewport(xscale=(0, 10.5), yscale=(0, float(dend_heights(dend))), width=0.9, height=0.9))
    grid_dendrogram(dend)
    popViewport()
    grid_dendrogram(dend, test=True)
    dend_c = color_branches(dend, k=2)
    dend_c = adjust_dend_by_x(dend_c, unit(np.sort(rng.uniform(size=10) * 10), "cm"))
    grid_dendrogram(dend_c, test=True)


def _rd_grid_draw_Legends():
    lgd = Legend(at=[1, 2, 3, 4], title="foo")
    grid_newpage()
    pushViewport(viewport(x=unit(0, "npc"), y=unit(0, "npc"), just=["left", "bottom"]))
    grid_draw(lgd.grob)
    popViewport()


def _rd_gt_render():
    rng = np.random.default_rng(1)
    mat = pd.DataFrame(rng.standard_normal((10, 10)), index=list("abcdefghij"))
    ht = Heatmap(
        mat,
        column_title=gt_render("Some <span style='color:blue'>blue text **in bold.**</span><br>And *italics text.*", r=unit(2, "pt"), padding=unit([2, 2, 2, 2], "pt")),
        column_title_gp=gpar(box_fill="orange"),
        row_labels=gt_render(list("abcdefghij"), padding=unit([2, 10, 2, 10], "pt")),
        row_names_gp=gpar(box_col="red"),
        row_km=2,
        row_title=gt_render(["title1", "title2"]),
        row_title_gp=gpar(box_fill="yellow"),
        heatmap_legend_param={
            "title": gt_render("<span style='color:orange'>**Legend title**</span>"),
            "title_gp": gpar(box_fill="grey"),
            "at": [-3, 0, 3],
            "labels": gt_render(["*negative* three", "zero", "*positive* three"]),
        },
    )
    hl = rowAnnotation(foo=anno_text(gt_render([c * 10 for c in "ABCDEFGHIJ"], align_widths=True), gp=gpar(box_col="blue", box_lwd=2), just="right", location=unit(1, "npc"))) + ht
    draw(hl)


def _rd_height_AnnotationFunction():
    a1 = anno_points(np.arange(1, 11))
    _ = height(a1)
    a2 = anno_points(np.arange(1, 11), which="row")
    _ = height(a2)


def _rd_height_Legends():
    lgd = Legend(labels=list(range(1, 11)), title="foo", legend_gp=gpar(fill="red"))
    return height(lgd)


def _rd_ht_opt():
    return repr(ht_opt)


def _rd_is_abs_unit():
    assert is_abs_unit(unit(1, "mm")) is True
    assert is_abs_unit(unit(1, "npc")) is False
    assert is_abs_unit(grobWidth(textGrob("foo"))) is True
    assert is_abs_unit(unit(1, "mm") + unit(1, "npc")) is False


def _rd_list_to_matrix():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    lt = {"a": rng.choice(letters, 5, replace=False), "b": rng.choice(letters, 10, replace=False), "c": rng.choice(letters, 15, replace=False)}
    _ = list_to_matrix(lt)
    return list_to_matrix(lt, universal_set=letters)


def _rd_make_comb_mat():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    lt = {"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)}
    _ = make_comb_mat(lt)
    mat = list_to_matrix(lt)
    _ = make_comb_mat(mat)
    # Also GRanges case from \dontrun
    gr_dict = {}
    for nm in ("a", "b", "c", "d"):
        bed = generateRandomBed(nr=25, nc=0, seed=ord(nm))
        gr_dict[nm] = GRanges(bed["chr"].tolist(), IRanges(bed["start"].to_numpy(), bed["end"].to_numpy()))
    return make_comb_mat(gr_dict)


def _rd_map_to_colors():
    cm = ColorMapping(colors={"A": "red", "B": "black"})
    _ = map_to_colors(cm, ["A", "B", "A", "B"])
    cm2 = ColorMapping(col_fun=colorRamp2([0, 1], ["white", "red"]))
    return map_to_colors(cm2, np.linspace(0, 1, 10))


def _rd_max_text_height():
    return max_text_height(["a", "b\nb", "c\nc\nc"], gp=gpar(fontsize=10))


def _rd_max_text_width():
    return max_text_width(["a", "bb", "ccc"], gp=gpar(fontsize=10))


def _rd_merge_dendrogram():
    rng = np.random.default_rng(1)
    m1 = rng.standard_normal((10, 10))
    m2 = rng.standard_normal((8, 10))
    m3 = rng.standard_normal((5, 10))
    dend1 = as_dendrogram(hclust(dist(m1)))
    dend2 = as_dendrogram(hclust(dist(m2)))
    dend3 = as_dendrogram(hclust(dist(m3)))
    dend_p = as_dendrogram(hclust(dist(np.vstack([m1.mean(axis=0), m2.mean(axis=0), m3.mean(axis=0)]))))
    grid_dendrogram(merge_dendrogram(dend_p, [dend1, dend2, dend3]), test=True)
    grid_dendrogram(merge_dendrogram(dend_p, [dend1, dend2, dend3], only_parent=True), test=True)
    dend_m = merge_dendrogram(
        color_branches(dend_p, k=1, col="orange"),
        [color_branches(dend1, k=1, col="red"), color_branches(dend2, k=1, col="blue"), color_branches(dend3, k=1, col="green")],
    )
    grid_dendrogram(dend_m, test=True)


def _rd_names_HeatmapAnnotation():
    ha = HeatmapAnnotation(foo=np.arange(1, 11), bar=anno_points(np.arange(10, 0, -1)))
    return names(ha)


def _rd_namesAssign_HeatmapAnnotation():
    ha = HeatmapAnnotation(foo=np.arange(1, 11), bar=anno_points(np.arange(10, 0, -1)))
    ha.names = ["A", "B"]
    return names(ha)


def _rd_nobs_AnnotationFunction():
    return nobs(anno_points(np.arange(1, 11)))


def _rd_normalize_genomic_signals_to_bins():
    rng = np.random.default_rng(42)
    chr_window = bin_genome("hg19", bin_size=10_000_000)
    bed1 = generateRandomBed(nr=200, nc=6, seed=1)
    gr1 = GRanges(bed1["chr"].tolist(), IRanges(bed1["start"].to_numpy(), bed1["end"].to_numpy()), **{c: bed1[c].to_numpy() for c in bed1.columns[3:]})
    num_gr = normalize_genomic_signals_to_bins(gr1, value_column=list(bed1.columns[3:]), bins=chr_window, empty_value=0.0)
    num_mat = num_gr.mcols.to_numpy(dtype=float)

    bed2 = generateRandomBed(nr=50, nc=2, seed=2)
    gr2 = GRanges(bed2["chr"].tolist(), IRanges(bed2["start"].to_numpy(), bed2["end"].to_numpy()), v1=bed2["value1"].to_numpy(), v2=bed2["value2"].to_numpy())
    v_gr = normalize_genomic_signals_to_bins(gr2, ["v1", "v2"], bins=chr_window, empty_value=0.0)
    v_mat = v_gr.mcols.to_numpy(dtype=float)

    bed3 = generateRandomBed(nr=20, nc=0, seed=3)
    gr3 = GRanges(bed3["chr"].tolist(), IRanges(bed3["start"].to_numpy(), bed3["start"].to_numpy()), gene=[f"gene_{k+1}" for k in range(len(bed3))])
    qh, sh = findOverlaps(chr_window, gr3)
    chr_col = pd.Categorical(chr_window.seqnames, categories=list(dict.fromkeys(chr_window.seqnames)), ordered=True)
    subgroup = ["A"] * 3 + ["B"] * 3
    with ht_opt.local(TITLE_PADDING=unit([4, 4], "pt")):
        ht_list = (
            Heatmap(num_mat, name="mat", col=colorRamp2([-1, 0, 1], ["green", "white", "red"]), row_split=chr_col, cluster_rows=False, show_column_dend=False, column_split=subgroup, cluster_column_slices=False, column_title="numeric matrix", top_annotation=HeatmapAnnotation(subgroup=subgroup, annotation_name_side="left"), row_title_rot=0, row_title_gp=gpar(fontsize=9), border=True, row_gap=unit(0, "pt"))
            + rowAnnotation(label=anno_mark(at=qh.tolist() or [0], labels=gr3.mcols["gene"].iloc[sh].tolist() if len(sh) else ["gene_1"]))
            + rowAnnotation(pt=anno_points(v_mat, gp=gpar(col=[4, 5]), pch=[1, 16]), width=unit(2, "cm"))
            + rowAnnotation(bar=anno_barplot(v_mat[:, 0], gp=gpar(col=[2 if val > 0 else 3 for val in v_mat[:, 0]])), width=unit(2, "cm"))
        )
        draw(ht_list, merge_legends=True)


def _rd_packLegend():
    col_fun = colorRamp2([0, 0.5, 1], ["blue", "white", "red"])
    lgd1 = Legend(at=list(range(1, 7)), legend_gp=gpar(fill=list(range(1, 7))), title="legend1")
    lgd2 = Legend(col_fun=col_fun, title="legend2", at=[0, 0.25, 0.5, 0.75, 1])
    draw(packLegend(lgd1, lgd2), test="two legends")
    draw(packLegend(lgd1, lgd2, direction="horizontal"), test="two legends packed horizontally")


def _rd_pindex():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((10, 10))
    rs, cs = np.where(m > 0)
    assert np.allclose(pindex(m, rs, cs), m[rs, cs])
    arr = np.arange(1, 28).reshape(3, 3, 3)
    res = pindex(arr, [0, 1], [1, 2])
    assert res.shape == (2, 3)


def _rd_restore_matrix():
    rng = np.random.default_rng(123)
    mat = np.round(rng.standard_normal((9, 9)), 2)
    def _lf(j, i, x, y, width, height, fill):
        ind_mat = restore_matrix(j, i, x, y)
        if ind_mat.shape[0] >= 2 and ind_mat.shape[1] >= 3:
            ind = np.unique(np.concatenate([ind_mat[1, :], ind_mat[:, 2]]))
            grid_text([f"{v:.2f}" for v in pindex(mat, i[ind], j[ind])], x[ind], y[ind], gp=gpar(fontsize=8))
    draw(Heatmap(mat, row_km=2, column_km=2, layer_fun=_lf))


def _rd_row_dend_Heatmap():
    mat = np.random.default_rng(1).standard_normal((10, 10))
    _ = row_dend(draw(Heatmap(mat)))
    _ = row_dend(draw(Heatmap(mat, row_km=2)))


def _rd_row_dend_HeatmapList():
    mat = np.random.default_rng(1).standard_normal((10, 10))
    _ = row_dend(draw(Heatmap(mat) + Heatmap(mat)))
    hl = draw(Heatmap(mat, row_km=2) + Heatmap(mat))
    _ = row_dend(hl)
    _ = row_dend(hl, on_slice=True)
    _ = row_dend(draw(Heatmap(mat, row_km=2) % v % Heatmap(mat)))


def _rd_row_order_Heatmap():
    mat = np.random.default_rng(1).standard_normal((10, 10))
    _ = row_order(draw(Heatmap(mat)))
    _ = row_order(draw(Heatmap(mat, row_km=2)))


def _rd_row_order_HeatmapList():
    mat = np.random.default_rng(1).standard_normal((10, 10))
    _ = row_order(draw(Heatmap(mat) + Heatmap(mat)))
    _ = row_order(draw(Heatmap(mat, row_km=2) + Heatmap(mat)))
    _ = row_order(draw(Heatmap(mat, row_km=2) % v % Heatmap(mat)))


def _rd_set_name():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    return set_name(m)


def _rd_set_nameAssign():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    m2 = set_name(m, ["A", "B", "C"])
    assert set_name(m2) == ["A", "B", "C"]


def _rd_set_size():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    return set_size(m)


def _rd_size_AnnotationFunction():
    a1 = anno_points(np.arange(1, 11))
    _ = size(a1)
    a2 = anno_points(np.arange(1, 11), which="row")
    return size(a2)


def _rd_sizeAssign_AnnotationFunction():
    anno = anno_points(np.arange(1, 11))
    anno.size = unit(4, "cm")
    return size(anno)


def _rd_smartAlign2():
    smartAlign2(np.array([[1.0, 2.0], [5.0, 7.0]]), range=(0, 10), plot=True)
    smartAlign2(np.array([[-0.5, 2.0], [5.0, 7.0]]), range=(0, 10), plot=True)
    smartAlign2(np.array([[3.0, 6.0], [4.0, 7.0]]), range=(0, 10), plot=True)
    smartAlign2(np.array([[1.0, 8.0], [3.0, 10.0]]), range=(0, 10), plot=True)


def _rd_subset_gp():
    gp = gpar(col=list(range(1, 11)), fill=1)
    return subset_gp(gp, slice(0, 5))


def _rd_t_comb_mat():
    rng = np.random.default_rng(123)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    m = make_comb_mat({"a": rng.choice(letters, 10, replace=False), "b": rng.choice(letters, 15, replace=False), "c": rng.choice(letters, 20, replace=False)})
    return t(m)


def _rd_test_alter_fun():
    alter_fun = {
        "mut1": lambda x, y, w, h: grid_rect(x, y, w, h, gp=gpar(fill="red", col=None)),
        "mut2": lambda x, y, w, h: grid_rect(x, y, w, h, gp=gpar(fill="blue", col=None)),
        "mut3": lambda x, y, w, h: grid_rect(x, y, w, h, gp=gpar(fill="yellow", col=None)),
        "mut4": lambda x, y, w, h: grid_rect(x, y, w, h, gp=gpar(fill="purple", col=None)),
        "mut5": lambda x, y, w, h: grid_rect(x, y, w, h, gp=gpar(lwd=2)),
        "mut6": lambda x, y, w, h: grid_points(x, y, pch=16),
        "mut7": lambda x, y, w, h: grid_segments(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5, gp=gpar(lwd=2)),
    }
    test_alter_fun(alter_fun)


def _rd_textbox_grob():
    rng = np.random.default_rng(1)
    words = [rng.choice(list("abcdefghijklmnopqrstuvwxyz")) * int(rng.integers(3, 11)) for _ in range(30)]
    grid_newpage()
    grid_textbox(words, gp=gpar(fontsize=rng.uniform(5, 30, 30)))
    sentences = ["This is sentense 1", "This is a long long long long long long long sentense."]
    grid_newpage()
    grid_textbox(sentences, word_wrap=True, add_new_line=True)


def _rd_width_AnnotationFunction():
    a1 = anno_points(np.arange(1, 11))
    _ = width(a1)
    a2 = anno_points(np.arange(1, 11), which="row")
    return width(a2)


def _rd_width_Legends():
    lgd = Legend(labels=list(range(1, 11)), title="foo", legend_gp=gpar(fill="red"))
    return width(lgd)


RD_EXAMPLES: dict[str, Callable[[], Any]] = {
    "AnnotationFunction.Rd": _rd_AnnotationFunction,
    "ColorMapping.Rd": _rd_ColorMapping,
    "Extract.AnnotationFunction.Rd": _rd_Extract_AnnotationFunction,
    "Extract.Heatmap.Rd": _rd_Extract_Heatmap,
    "Extract.HeatmapAnnotation.Rd": _rd_Extract_HeatmapAnnotation,
    "Extract.HeatmapList.Rd": _rd_Extract_HeatmapList,
    "Extract.SingleAnnotation.Rd": _rd_Extract_SingleAnnotation,
    "Extract.comb_mat.Rd": _rd_Extract_comb_mat,
    "Heatmap3D.Rd": _rd_Heatmap3D,
    "Legend.Rd": _rd_Legend,
    "Legends-class.Rd": _rd_Legends_class,
    "SingleAnnotation.Rd": _rd_SingleAnnotation,
    "UpSet.Rd": _rd_UpSet,
    "adjust_dend_by_x.Rd": _rd_adjust_dend_by_x,
    "alter_graphic.Rd": _rd_alter_graphic,
    "anno_barplot.Rd": _rd_anno_barplot,
    "anno_block.Rd": _rd_anno_block,
    "anno_boxplot.Rd": _rd_anno_boxplot,
    "anno_customize.Rd": _rd_anno_customize,
    "anno_density.Rd": _rd_anno_density,
    "anno_empty.Rd": _rd_anno_empty,
    "anno_histogram.Rd": _rd_anno_histogram,
    "anno_horizon.Rd": _rd_anno_horizon,
    "anno_image.Rd": _rd_anno_image,
    "anno_joyplot.Rd": _rd_anno_joyplot,
    "anno_lines.Rd": _rd_anno_lines,
    "anno_mark.Rd": _rd_anno_mark,
    "anno_numeric.Rd": _rd_anno_numeric,
    "anno_points.Rd": _rd_anno_points,
    "anno_simple.Rd": _rd_anno_simple,
    "anno_summary.Rd": _rd_anno_summary,
    "anno_text.Rd": _rd_anno_text,
    "anno_textbox.Rd": _rd_anno_textbox,
    "anno_zoom.Rd": _rd_anno_zoom,
    "annotation_axis_grob.Rd": _rd_annotation_axis_grob,
    "attach_annotation-Heatmap-method.Rd": _rd_attach_annotation,
    "bar3D.Rd": _rd_bar3D,
    "c.ColorMapping.Rd": _rd_c_ColorMapping,
    "c.HeatmapAnnotation.Rd": _rd_c_HeatmapAnnotation,
    "cluster_between_groups.Rd": _rd_cluster_between_groups,
    "cluster_within_group.Rd": _rd_cluster_within_group,
    "column_dend-Heatmap-method.Rd": _rd_column_dend_Heatmap,
    "column_dend-HeatmapList-method.Rd": _rd_column_dend_HeatmapList,
    "column_order-Heatmap-method.Rd": _rd_column_order_Heatmap,
    "column_order-HeatmapList-method.Rd": _rd_column_order_HeatmapList,
    "comb_degree.Rd": _rd_comb_degree,
    "comb_name.Rd": _rd_comb_name,
    "comb_size.Rd": _rd_comb_size,
    "compare_heatmap.2.Rd": _rd_compare_heatmap2,
    "compare_heatmap.Rd": _rd_compare_heatmap,
    "compare_pheatmap.Rd": _rd_compare_pheatmap,
    "decorate_annotation.Rd": _rd_decorate_annotation,
    "decorate_dend.Rd": _rd_decorate_dend,
    "decorate_dimnames.Rd": _rd_decorate_dimnames,
    "decorate_heatmap_body.Rd": _rd_decorate_heatmap_body,
    "decorate_title.Rd": _rd_decorate_title,
    "default_axis_param.Rd": _rd_default_axis_param,
    "dend_xy.Rd": _rd_dend_xy,
    "densityHeatmap.Rd": _rd_densityHeatmap,
    "dist2.Rd": _rd_dist2,
    "draw-Legends-method.Rd": _rd_draw_Legends,
    "extract_comb.Rd": _rd_extract_comb,
    "frequencyHeatmap.Rd": _rd_frequencyHeatmap,
    "full_comb_code.Rd": _rd_full_comb_code,
    "getXY_in_parent_vp.Rd": _rd_getXY_in_parent_vp,
    "grid.boxplot.Rd": _rd_grid_boxplot,
    "grid.dendrogram.Rd": _rd_grid_dendrogram,
    "grid.draw.Legends.Rd": _rd_grid_draw_Legends,
    "gt_render.Rd": _rd_gt_render,
    "height.AnnotationFunction.Rd": _rd_height_AnnotationFunction,
    "height.Legends.Rd": _rd_height_Legends,
    "ht_opt.Rd": _rd_ht_opt,
    "is_abs_unit.Rd": _rd_is_abs_unit,
    "list_to_matrix.Rd": _rd_list_to_matrix,
    "make_comb_mat.Rd": _rd_make_comb_mat,
    "map_to_colors-ColorMapping-method.Rd": _rd_map_to_colors,
    "max_text_height.Rd": _rd_max_text_height,
    "max_text_width.Rd": _rd_max_text_width,
    "merge_dendrogram.Rd": _rd_merge_dendrogram,
    "names.HeatmapAnnotation.Rd": _rd_names_HeatmapAnnotation,
    "namesAssign.HeatmapAnnotation.Rd": _rd_namesAssign_HeatmapAnnotation,
    "nobs.AnnotationFunction.Rd": _rd_nobs_AnnotationFunction,
    "normalize_genomic_signals_to_bins.Rd": _rd_normalize_genomic_signals_to_bins,
    "packLegend.Rd": _rd_packLegend,
    "pindex.Rd": _rd_pindex,
    "restore_matrix.Rd": _rd_restore_matrix,
    "row_dend-Heatmap-method.Rd": _rd_row_dend_Heatmap,
    "row_dend-HeatmapList-method.Rd": _rd_row_dend_HeatmapList,
    "row_order-Heatmap-method.Rd": _rd_row_order_Heatmap,
    "row_order-HeatmapList-method.Rd": _rd_row_order_HeatmapList,
    "set_name.Rd": _rd_set_name,
    "set_nameAssign.Rd": _rd_set_nameAssign,
    "set_size.Rd": _rd_set_size,
    "size.AnnotationFunction.Rd": _rd_size_AnnotationFunction,
    "sizeAssign.AnnotationFunction.Rd": _rd_sizeAssign_AnnotationFunction,
    "smartAlign2.Rd": _rd_smartAlign2,
    "subset_gp.Rd": _rd_subset_gp,
    "t.comb_mat.Rd": _rd_t_comb_mat,
    "test_alter_fun.Rd": _rd_test_alter_fun,
    "textbox_grob.Rd": _rd_textbox_grob,
    "width.AnnotationFunction.Rd": _rd_width_AnnotationFunction,
    "width.Legends.Rd": _rd_width_Legends,
}


def run_all() -> None:
    print(f"Running all {len(RD_EXAMPLES)} ComplexHeatmap .Rd examples...")
    for idx, (nm, fn) in enumerate(RD_EXAMPLES.items(), start=1):
        fn()
        plt.close("all")
        ht_opt(RESET=True)
        print(f"[{idx}/{len(RD_EXAMPLES)}] PASS {nm}")
    print("ALL 102 .Rd EXAMPLES PASSED CLEANLY!")


if __name__ == "__main__":
    run_all()
