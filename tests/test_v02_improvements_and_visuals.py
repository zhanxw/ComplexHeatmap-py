"""Regression, performance, concurrency, composition, and visual checks for v0.2.0 improvements."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from scipy.stats import pearsonr, spearmanr

import complexheatmap as ch
from complexheatmap import (
    ColorMapping,
    GRanges,
    Heatmap,
    HeatmapAnnotation,
    IRanges,
    UpSet,
    Viewport,
    as_dist,
    colorRamp2,
    decorate_heatmap_body,
    dist,
    draw,
    findOverlaps,
    getXY_in_parent_vp,
    grid_newpage,
    grid_xaxis,
    hclust,
    ht_opt,
    make_comb_mat,
    popViewport,
    pushViewport,
    restore_matrix,
    seekViewport,
    unit,
    upViewport,
)
from complexheatmap.heatmap_list import _normalize_padding


def test_grid_xaxis_no_spurious_native_zero_line():
    dev = grid_newpage(width_in=4.0, height_in=4.0)
    pushViewport(Viewport(x=0.5, y=0.5, width=0.6, height=0.6, xscale=(0.0, 10.0), yscale=(-5.0, 5.0)))
    try:
        grid_xaxis()
        # Only LineCollection spine/ticks (no stray Line2D at y=0 native in mid-viewport)
        assert len(dev.ax.lines) == 0
        assert len(dev.ax.collections) >= 1
    finally:
        popViewport()


def test_ht_opt_keys_and_validate_names():
    with ht_opt.local(
        kmeans_seed=42,
        PAGE_PADDING=unit([4, 4, 4, 4], "mm"),
        ht_gap=unit(6, "mm"),
        merge_legends=True,
        show_vp=True,
    ):
        assert ht_opt.kmeans_seed == 42
        assert _normalize_padding(None) == pytest.approx([4.0, 4.0, 4.0, 4.0])

        rng = np.random.default_rng(1)
        m = rng.standard_normal((8, 6))
        hl = Heatmap(m, name="dup", row_km=2) + Heatmap(m, name="dup")
        with pytest.warns(UserWarning, match="unique"):
            res = draw(hl, show=False)
        assert res.ht_list_param["merge_legends"] is True


def test_restore_matrix_mixed_units_with_active_viewport():
    dev = grid_newpage(width_in=5.0, height_in=3.0)
    pushViewport(Viewport(x=0.5, y=0.5, width=unit(80, "mm"), height=unit(40, "mm")))
    try:
        nr, nc = 3, 4
        j = np.repeat(np.arange(nc), nr)
        i = np.tile(np.arange(nr), nc)
        x = unit(np.repeat((np.arange(nc) + 0.5) / nc, nr), "npc") + unit(2.0, "mm")
        y = unit(np.tile((nr - np.arange(nr) - 0.5) / nr, nc), "npc") + unit(1.0, "mm")
        ind = restore_matrix(j, i, x, y)
        assert ind.shape == (nr, nc)
        assert set(ind.ravel().tolist()) == set(range(nr * nc))
    finally:
        popViewport()


def test_seek_viewport_preserves_parent_stack():
    m = np.arange(20, dtype=float).reshape(5, 4)
    ht = Heatmap(m, name="P")
    res = draw(ht)
    with decorate_heatmap_body("P"):
        xy_parent = getXY_in_parent_vp((unit(0.5, "npc"), unit(0.5, "npc")))
        assert float(xy_parent["x"].to_mm("x")[0]) >= 0.0
        upViewport(1)
        cur = ch.current_viewport()
        assert cur.name != "ROOT"
    seekViewport("P_heatmap_body_1_1")
    assert len(ch.current_device().vp_stack) >= 3


def test_deep_dendrogram_and_ward_d_vs_d2_and_corr_dist():
    # Deep >1000 chain without RecursionError
    rng = np.random.default_rng(9)
    n_deep = 1500
    pts = np.linspace(0.0, 100.0, n_deep)[:, None]
    hc = hclust(dist(pts, method="euclidean"), method="single")
    assert len(hc.order) == n_deep

    # Ward.D vs Ward.D2 mathematical equivalence: h_ward_d == hclust(sqrt(d), "ward.D2").height ** 2
    sub = rng.standard_normal((16, 8))
    d_euc = dist(sub, method="euclidean")
    hc_d = hclust(d_euc, method="ward.D")
    hc_d2_sqrt = hclust(as_dist(np.sqrt(d_euc.condensed)), method="ward.D2")
    assert np.allclose(hc_d.height, hc_d2_sqrt.height ** 2)
    hc_d2 = hclust(d_euc, method="ward.D2")
    assert not np.allclose(hc_d.height, hc_d2.height)

    # Vectorized Pearson / Spearman including constant rows and NaNs
    mat_c = rng.standard_normal((12, 15))
    mat_c[3, :] = 2.5  # constant row
    for method, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
        got = dist(mat_c, method=method).to_square()
        for a in range(12):
            for b in range(a + 1, 12):
                if np.std(mat_c[a]) == 0 or np.std(mat_c[b]) == 0:
                    exp = 0.5
                else:
                    exp = 0.5 * (1.0 - float(fn(mat_c[a], mat_c[b])[0]))
                assert got[a, b] == pytest.approx(exp, abs=1e-10)

    mat_na = mat_c[:6, :10].copy()
    mat_na[0, [1, 4]] = np.nan
    mat_na[2, 7] = np.nan
    got_na = dist(mat_na, method="pearson").to_square()
    for a in range(6):
        for b in range(a + 1, 6):
            mask = np.isfinite(mat_na[a]) & np.isfinite(mat_na[b])
            if np.std(mat_na[a, mask]) == 0 or np.std(mat_na[b, mask]) == 0:
                exp = 0.5
            else:
                exp = 0.5 * (1.0 - float(pearsonr(mat_na[a, mask], mat_na[b, mask])[0]))
            assert got_na[a, b] == pytest.approx(exp, abs=1e-10)


def test_vectorized_map_to_rgba_and_raster_downsampling():
    cf = colorRamp2([-1.0, 0.0, 1.0], ["blue", "white", "red"])
    cm = ColorMapping(name="expr", col_fun=cf, na_col="#888888")
    arr = np.array([[np.nan, -1.0], [0.0, 1.0]])
    rgba = cm.map_to_rgba(arr)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 1] == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=0.02)
    assert rgba[1, 1] == pytest.approx([1.0, 0.0, 0.0, 1.0], abs=0.02)
    assert rgba[0, 0, :3] == pytest.approx([0.533, 0.533, 0.533], abs=0.02)

    big = np.random.default_rng(2).standard_normal((450, 12))
    ht = Heatmap(big, use_raster=True, raster_resize_mat=True, cluster_rows=False, cluster_columns=False)
    svg = ht.to_svg()
    assert "<svg" in svg


def test_fast_find_overlaps_correctness_and_scale():
    rng = np.random.default_rng(12)
    qs = rng.integers(0, 5000, size=200)
    qe = qs + rng.integers(5, 100, size=200)
    ss = rng.integers(0, 5000, size=200)
    se = ss + rng.integers(5, 100, size=200)

    got_q, got_s = findOverlaps(IRanges(qs, qe), IRanges(ss, se))
    got_pairs = set(zip(got_q.tolist(), got_s.tolist()))
    exp_pairs = {(i, j) for i in range(200) for j in range(200) if qs[i] <= se[j] and ss[j] <= qe[i]}
    assert got_pairs == exp_pairs

    # Scale check: 5000 x 5000 intervals under 1.0s
    qs_big = rng.integers(0, 1_000_000, size=5000)
    qe_big = qs_big + rng.integers(50, 500, size=5000)
    ss_big = rng.integers(0, 1_000_000, size=5000)
    se_big = ss_big + rng.integers(50, 500, size=5000)
    t0 = time.perf_counter()
    qh, sh = findOverlaps(
        GRanges(["chr1"] * 5000, start=qs_big, end=qe_big),
        GRanges(["chr1"] * 5000, start=ss_big, end=se_big),
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0
    assert len(qh) == len(sh) and len(qh) > 0


def test_no_pyplot_leak_and_subplot_subfigure_composition():
    plt.close("all")
    m = np.random.default_rng(3).standard_normal((6, 5))
    for k in range(25):
        draw(Heatmap(m, name=f"h{k}", show_heatmap_legend=False), show=False)
    assert len(plt.get_fignums()) == 0

    # Subplot ax= composition leaves sibling axes untouched
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot([0, 1], [0, 1], color="navy")
    draw(Heatmap(m, name="sub_ax"), ax=ax2)
    assert len(ax1.lines) == 1
    assert len(ax2.collections) > 0

    # SubFigure composition
    fig2 = plt.figure(figsize=(10, 4))
    subfigs = fig2.subfigures(1, 2)
    sub_ax = subfigs[0].subplots()
    sub_ax.bar([1, 2], [3, 4])
    res_sf = draw(Heatmap(m, name="subfig"), fig=subfigs[1])
    assert res_sf._device.fig is subfigs[1]
    assert len(sub_ax.patches) == 2
    plt.close("all")


def test_concurrent_threaded_rendering_isolation():
    rng = np.random.default_rng(5)

    def _render(idx: int) -> tuple[int, str, str]:
        mat = rng.standard_normal((6 + idx, 5))
        ht = Heatmap(mat, name=f"thr_{idx}")
        svg = ht.to_svg(width=5.0, height=4.0)
        return idx, svg, ch._ENV.last.ht_list[0].name

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_render, range(4)))
    for idx, svg, last_name in results:
        assert "<svg" in svg
        assert last_name == f"thr_{idx}"


def test_visual_svg_and_rendered_pixel_layout():
    rng = np.random.default_rng(21)
    cm = make_comb_mat({"A": ["g1", "g2", "g3", "g4"], "B": ["g2", "g3", "g5"], "C": ["g3", "g6"]})
    up = UpSet(cm, comb_col="firebrick")
    buf = io.BytesIO()
    res_up = draw(up, width=7.0, height=4.5, dpi=100)
    res_up._device.fig.savefig(buf, format="png", dpi=100, facecolor="white")
    img_up = np.asarray(Image.open(buf).convert("RGB"))
    # Center band (UpSet dot matrix) should contain firebrick combination dots/segments
    h, w, _ = img_up.shape
    dot_band = img_up[int(0.35 * h):int(0.95 * h), int(0.10 * w):int(0.80 * w)]
    red_pixels = int(np.sum((dot_band[..., 0] > 140) & (dot_band[..., 1] < 80) & (dot_band[..., 2] < 80)))
    assert red_pixels > 50

    # Heatmap3D vector SVG + rendered pixel check
    h3 = ch.Heatmap3D(rng.uniform(0.2, 1.0, size=(4, 5)), column_title="3D Check")
    svg_3d = h3.to_svg(width=6.0, height=5.0)
    assert "<svg" in svg_3d
    buf3 = io.BytesIO()
    res3 = draw(h3, width=6.0, height=5.0, dpi=100)
    res3._device.fig.savefig(buf3, format="png", dpi=100, facecolor="white")
    img3 = np.asarray(Image.open(buf3).convert("RGB"))
    center3 = img3[int(0.2 * img3.shape[0]):int(0.8 * img3.shape[0]), int(0.2 * img3.shape[1]):int(0.8 * img3.shape[1])]
    assert np.mean(np.min(center3, axis=2) < 220) > 0.05
