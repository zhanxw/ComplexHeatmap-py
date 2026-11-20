"""Tests for clustering, dendrograms, Path A/B splitting, +/%v% alignment, subsetting, and pindex/restore_matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import complexheatmap as ch
from complexheatmap import (
    Heatmap,
    HeatmapAnnotation,
    as_dendrogram,
    cluster_between_groups,
    cluster_within_group,
    color_branches,
    column_dend,
    column_order,
    cut_dendrogram,
    cutree,
    dist,
    dist2,
    draw,
    gpar,
    hclust,
    ht_size,
    merge_dendrogram,
    nobs,
    order_dendrogram,
    pindex,
    restore_matrix,
    rowAnnotation,
    row_dend,
    row_order,
    unit,
    v,
)


def test_distances_and_dendrogram_utilities():
    rng = np.random.default_rng(5)
    m = rng.standard_normal((12, 6))
    for method in ("euclidean", "maximum", "manhattan", "canberra", "pearson", "spearman"):
        d = dist(m, method=method)
        assert d.n == 12

    d2 = dist2(m, lambda a, b: float(np.sum(np.abs(a - b))))
    assert d2.n == 12

    hc = hclust(dist(m), method="ward.D2")
    dend = as_dendrogram(hc)
    assert nobs(dend) == 12
    assert sorted(order_dendrogram(dend)) == list(range(12))

    ct = cut_dendrogram(dend, k=3)
    assert len(ct["lower"]) == 3
    assert sum(nobs(sub) for sub in ct["lower"]) == 12
    assert len(cutree(hc, k=3)) == 12

    colored = color_branches(dend, k=3, col=["red", "green", "blue"])
    assert nobs(colored) == 12

    fa = ["A"] * 2 + ["B"] * 4
    dw = cluster_within_group(m[:, :6], fa)
    db = cluster_between_groups(m[:, :6], fa)
    assert nobs(dw) == 6
    assert nobs(db) == 6

    dm = merge_dendrogram(as_dendrogram(hclust(dist(rng.standard_normal((2, 4))))), [as_dendrogram(hclust(dist(rng.standard_normal((3, 4))))), as_dendrogram(hclust(dist(rng.standard_normal((4, 4)))))])
    assert nobs(dm) == 7


def test_path_a_and_path_b_splitting():
    rng = np.random.default_rng(42)
    mat = pd.DataFrame(
        np.vstack([rng.normal(2.0, 0.3, (6, 8)), rng.normal(-2.0, 0.3, (6, 8))]),
        index=[f"G{i}" for i in range(12)],
        columns=[f"S{j}" for j in range(8)],
    )

    # Path A: integer cutree split
    ht_a = Heatmap(mat, name="A", row_split=2, column_split=2)
    res_a = draw(ht_a)
    ro_a = row_order(res_a)
    co_a = column_order(res_a)
    assert isinstance(ro_a, list) and len(ro_a) == 2
    assert sorted(idx for s in ro_a for idx in s) == list(range(12))
    assert isinstance(co_a, list) and len(co_a) == 2
    assert len(row_dend(res_a)) == 2
    assert row_dend(res_a, on_slice=True) is not None

    # Path B: consensus k-means + categorical split + cluster_slices
    ht_b = Heatmap(
        mat,
        name="B",
        row_km=2,
        row_km_repeats=3,
        row_split=["X", "Y"] * 6,
        row_title="Slice_%s",
    )
    res_b = draw(ht_b)
    ro_b = row_order(res_b)
    assert len(ro_b) >= 2
    assert sorted(idx for s in ro_b for idx in s) == list(range(12))


def test_horizontal_and_vertical_alignment_and_subsetting():
    rng = np.random.default_rng(7)
    m1 = pd.DataFrame(rng.standard_normal((10, 6)), index=list("abcdefghij"), columns=[f"c{j}" for j in range(6)])
    m2 = pd.DataFrame(rng.uniform(size=(10, 4)), index=list("abcdefghij"), columns=[f"d{j}" for j in range(4)])
    m3 = pd.DataFrame(rng.standard_normal((5, 6)), index=list("klmno"), columns=[f"c{j}" for j in range(6)])

    # Horizontal alignment (+)
    hl_h = (
        Heatmap(m1, name="M1", row_km=2, top_annotation=HeatmapAnnotation(bar=ch.anno_barplot(np.arange(6))))
        + Heatmap(m2, name="M2")
        + rowAnnotation(pt=ch.anno_points(np.arange(10)))
    )
    res_h = draw(hl_h, row_dend_side="right")
    sz = ht_size(res_h, unit_name="mm", valueOnly=True)
    assert sz[0] > 0 and sz[1] > 0
    ro_h = row_order(res_h)
    assert len(ro_h) == 2

    # Vertical alignment (%v% and /)
    hl_v = Heatmap(m1, name="M1", column_km=2) % v % Heatmap(m3, name="M3")
    res_v = draw(hl_v)
    assert len(column_order(res_v)) == 2
    hl_div = Heatmap(m1, name="M1") / Heatmap(m3, name="M3")
    assert len(hl_div) == 2

    # Subsetting before draw
    ht_sub = Heatmap(m1, name="sub")[1:6, 2:5]
    assert ht_sub.shape == (5, 3)

    hl_sub = hl_h[:4, "M2"]
    assert len(hl_sub) == 1
    assert hl_sub.ht_list[0].shape == (4, 4)


def test_pindex_restore_matrix_and_raster():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((8, 6))
    captured = {}

    def _lf(j, i, x, y, w, h, fill):
        ind_mat = restore_matrix(j, i, x, y)
        captured["shape"] = ind_mat.shape
        vals = pindex(m, i, j)
        captured["len"] = len(vals)

    draw(Heatmap(m, row_km=2, column_km=2, layer_fun=_lf, use_raster=True))
    assert captured["shape"][0] * captured["shape"][1] == captured["len"]
    assert pindex(np.arange(27).reshape(3, 3, 3), [0, 2], [1, 1]).shape == (2, 3)


def test_edge_case_shapes_and_cell_fun():
    # 1D vector, 1x1 matrix, zero-col matrix, and cell_fun
    draw(Heatmap(np.array([1.0, 2.0, 3.0]), name="vec"))
    draw(Heatmap(np.array([[42.0]]), name="one"))

    called = []
    m = np.arange(6, dtype=float).reshape(2, 3)
    draw(
        Heatmap(
            m,
            cluster_rows=False,
            cluster_columns=False,
            cell_fun=lambda j, i, x, y, w, h, fill: called.append((i, j)),
        )
    )
    assert len(called) == 6
