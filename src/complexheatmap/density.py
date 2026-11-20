"""Density heatmaps, frequency heatmaps, and KS-distance clustering."""

from __future__ import annotations

from builtins import range as _range
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp

from ._options import _ENV
from .annotations import HeatmapAnnotation, anno_empty, anno_lines, anno_points, rowAnnotation
from .colors import brewer_pal, colorRamp2
from .decorate import decorate_heatmap_body
from .dendrogram import DistMatrix, as_dist
from .grid import Viewport, grid_lines, grid_segments, grid_text, grid_yaxis, popViewport, pushViewport
from .heatmap import Heatmap
from .units import Unit, as_unit, check_gp, gpar, unit


def _to_column_series_list(data: Any) -> tuple[list[np.ndarray], list[str] | None]:
    if isinstance(data, dict):
        names = [str(k) for k in data.keys()]
        vecs = [np.asarray(v, dtype=float).ravel() for v in data.values()]
        return [v[np.isfinite(v)] for v in vecs], names
    if isinstance(data, pd.DataFrame):
        names = [str(c) for c in data.columns]
        vecs = [data[c].to_numpy(dtype=float).ravel() for c in data.columns]
        return [v[np.isfinite(v)] for v in vecs], names
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            v = data.astype(float)
            return [v[np.isfinite(v)]], None
        vecs = [data[:, c].astype(float) for c in range(data.shape[1])]
        return [v[np.isfinite(v)] for v in vecs], None
    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], (list, tuple, np.ndarray, pd.Series)):
            vecs = [np.asarray(v, dtype=float).ravel() for v in data]
            return [v[np.isfinite(v)] for v in vecs], None
        v = np.asarray(data, dtype=float).ravel()
        return [v[np.isfinite(v)]], None
    raise TypeError(f"Unsupported input type for densityHeatmap/frequencyHeatmap: {type(data)}")


def ks_dist(mat: Any, mc_cores: int = 1) -> DistMatrix:
    """Pairwise Kolmogorov-Smirnov distance matrix across columns of `mat` (`ks_dist.Rd`)."""
    vecs, labels = _to_column_series_list(mat)
    n = len(vecs)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if len(vecs[i]) == 0 or len(vecs[j]) == 0:
                val = 1.0
            else:
                val = float(ks_2samp(vecs[i], vecs[j]).statistic)
            d[i, j] = d[j, i] = val
    return as_dist(d, labels=labels, method="ks")


def densityHeatmap(
    data: Any,
    col: Sequence[str] | None = None,
    color_space: str = "LAB",
    anno: HeatmapAnnotation | None = None,
    ylab: str = "value",
    title: str | None = None,
    ylim: Sequence[float] | None = None,
    range: Sequence[float] | None = None,
    cluster_columns: Any = False,
    clustering_distance_columns: Any = "ks",
    clustering_method_columns: str = "complete",
    mc_cores: int = 1,
    show_quantiles: bool = True,
    quantile_gp: dict[str, Any] | None = None,
    n_grid: int = 200,
    name: str | None = None,
    **kwargs: Any,
) -> Heatmap:
    """Visualize distribution densities across columns or distributions (`densityHeatmap.Rd`)."""
    vecs, col_names = _to_column_series_list(data)
    if not vecs:
        raise ValueError("`data` cannot be empty.")
    n_cols = len(vecs)

    all_vals = np.concatenate([v for v in vecs if len(v) > 0]) if any(len(v) > 0 for v in vecs) else np.array([0.0, 1.0])
    if ylim is not None:
        ymin, ymax = float(ylim[0]), float(ylim[1])
    elif range is not None:
        ymin, ymax = float(range[0]), float(range[1])
    else:
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0

    y_grid = np.linspace(ymin, ymax, int(n_grid))
    # Reverse top-to-bottom so largest value is at top row 0
    y_desc = y_grid[::-1]
    dens_mat = np.zeros((int(n_grid), n_cols), dtype=float)
    quantiles_mat = np.full((5, n_cols), np.nan, dtype=float)  # 0, 25, 50, 75, 100

    for c_i, v in enumerate(vecs):
        if len(v) < 2 or float(np.std(v)) < 1e-12:
            if len(v) > 0:
                quantiles_mat[:, c_i] = float(v[0])
                nearest = int(np.argmin(np.abs(y_desc - float(v[0]))))
                dens_mat[nearest, c_i] = 1.0
        else:
            quantiles_mat[:, c_i] = np.quantile(v, [0.0, 0.25, 0.50, 0.75, 1.0])
            kde = gaussian_kde(v)
            dens_mat[:, c_i] = kde(y_desc)

    if col is None:
        cols_pal = brewer_pal(9, "Spectral")[::-1]
    else:
        cols_pal = list(col)

    dmax = float(np.nanmax(dens_mat))
    if dmax <= 0:
        dmax = 1.0
    col_fun = colorRamp2(np.linspace(0.0, dmax, len(cols_pal)), cols_pal, space=color_space)

    ht_name = name or f"density_{_ENV.next_heatmap_index()}"

    # Y-axis range mapping inside body viewport
    eff_distance = ks_dist(data) if clustering_distance_columns == "ks" else clustering_distance_columns
    # Notice if ks distance DF is passed, wrap clustering_distance_columns cleanly via callable returning matrix
    if clustering_distance_columns == "ks" and cluster_columns:
        ks_d = ks_dist(data)
        def _ks_fn(_sub: np.ndarray) -> DistMatrix:
            return ks_d
        eff_distance_arg: Any = _ks_fn
    else:
        eff_distance_arg = eff_distance

    q_gp = check_gp(gpar(lty="dashed", col="#222222", lwd=1.0) if quantile_gp is None else quantile_gp)

    def _layer_fun(j: Any, i: Any, x: Any, y: Any, w: Any, h: Any, fill: Any) -> None:
        if not show_quantiles:
            return
        # Order of columns in current slice
        uniq_cols = []
        seen_c: set[int] = set()
        for jj in j:
            ji = int(jj)
            if ji not in seen_c:
                seen_c.add(ji)
                uniq_cols.append(ji)
        m_c = len(uniq_cols)
        if m_c == 0:
            return
        xs_npc = (np.arange(m_c, dtype=float) + 0.5) / m_c
        span = max(1e-12, ymax - ymin)
        q_labels = ["min", "25%", "median", "75%", "max"]
        for qi in _range(5):
            ys_npc = np.array([(quantiles_mat[qi, ci] - ymin) / span for ci in uniq_cols], dtype=float)
            valid = np.isfinite(ys_npc)
            if np.sum(valid) >= 2:
                grid_lines(unit(xs_npc[valid], "npc"), unit(ys_npc[valid], "npc"), gp=q_gp)
            if np.any(valid):
                last_y = float(ys_npc[np.where(valid)[0][-1]])
                grid_text(q_labels[qi], x=unit(1.0, "npc") + unit(1.5, "mm"), y=unit(last_y, "npc"), just="left", gp=gpar(fontsize=8, col=q_gp.get("col", "#222222")))

    # Add y-axis left gap via empty/left annotation or post_fun
    user_post = kwargs.pop("post_fun", None)

    def _draw_yaxis_post(ht_list: Any) -> None:
        try:
            with decorate_heatmap_body(ht_name):
                pushViewport(Viewport(xscale=(0.0, 1.0), yscale=(ymin, ymax), clip=False))
                try:
                    grid_yaxis()
                    grid_text(ylab, x=unit(-11, "mm"), y=unit(0.5, "npc"), rot=90, gp=gpar(fontsize=10))
                finally:
                    popViewport()
        except KeyError:
            pass
        if user_post is not None:
            user_post(ht_list)

    df_dens = pd.DataFrame(dens_mat, columns=col_names)
    hm_kwargs: dict[str, Any] = dict(
        col=col_fun,
        name=ht_name,
        column_title=title if title is not None else [],
        top_annotation=anno,
        left_annotation=rowAnnotation(_yaxis_space=anno_empty(border=False, width=unit(12, "mm")), show_annotation_name=False),
        right_annotation=rowAnnotation(_qlabel_space=anno_empty(border=False, width=unit(11, "mm")), show_annotation_name=False) if show_quantiles else None,
        cluster_rows=False,
        cluster_columns=bool(cluster_columns),
        clustering_distance_columns=eff_distance_arg,
        clustering_method_columns=clustering_method_columns,
        show_row_names=False,
        show_column_names=col_names is not None,
        layer_fun=_layer_fun,
        post_fun=_draw_yaxis_post,
        heatmap_legend_param=kwargs.pop("heatmap_legend_param", {"title": "density"}),
    )
    hm_kwargs.update(kwargs)
    return Heatmap(df_dens, **hm_kwargs)


def frequencyHeatmap(
    data: Any,
    breaks: Any = "Sturges",
    stat: str = "count",
    use_3d: bool = False,
    col: Sequence[str] | None = None,
    ylim: Sequence[float] | None = None,
    title: str | None = None,
    ylab: str = "value",
    cluster_columns: Any = False,
    name: str | None = None,
    **kwargs: Any,
) -> Heatmap:
    """Visualize binned frequency/proportion matrix across columns or distributions (`frequencyHeatmap.Rd`)."""
    vecs, col_names = _to_column_series_list(data)
    if not vecs:
        raise ValueError("`data` cannot be empty.")
    all_vals = np.concatenate([v for v in vecs if len(v) > 0])
    if ylim is not None:
        range_vals = (float(ylim[0]), float(ylim[1]))
    else:
        range_vals = (float(np.min(all_vals)), float(np.max(all_vals)))
        if range_vals[0] == range_vals[1]:
            range_vals = (range_vals[0] - 0.5, range_vals[1] + 0.5)

    breaks_arg = breaks.lower() if isinstance(breaks, str) else breaks
    _, edges = np.histogram(all_vals, bins=breaks_arg, range=range_vals)
    n_bins = len(edges) - 1
    n_cols = len(vecs)
    freq_mat = np.zeros((n_bins, n_cols), dtype=float)
    for c_i, v in enumerate(vecs):
        cnts, _ = np.histogram(v, bins=edges)
        cnts_f = cnts.astype(float)
        if stat == "proportion" and len(v) > 0:
            cnts_f /= float(len(v))
        elif stat == "density" and len(v) > 0:
            cnts_f = cnts_f / (float(len(v)) * np.diff(edges))
        freq_mat[:, c_i] = cnts_f[::-1]  # largest bin at top

    row_labels = [f"[{edges[i]:.3g}, {edges[i+1]:.3g}]" for i in range(n_bins)][::-1]
    if col is None:
        cols_pal = brewer_pal(9, "Spectral")[::-1]
    else:
        cols_pal = list(col)

    vmax = float(np.nanmax(freq_mat))
    if vmax <= 0:
        vmax = 1.0
    col_fun = colorRamp2(np.linspace(0.0, vmax, len(cols_pal)), cols_pal)
    df_freq = pd.DataFrame(freq_mat, index=row_labels, columns=col_names)
    ht_name = name or f"freq_{_ENV.next_heatmap_index()}"

    if use_3d:
        from .three_d import Heatmap3D
        return Heatmap3D(
            df_freq,
            col=col_fun,
            name=ht_name,
            column_title=title if title is not None else [],
            cluster_rows=False,
            cluster_columns=cluster_columns,
            heatmap_legend_param=kwargs.pop("heatmap_legend_param", {"title": stat}),
            **kwargs,
        )

    return Heatmap(
        df_freq,
        col=col_fun,
        name=ht_name,
        column_title=title if title is not None else [],
        cluster_rows=False,
        cluster_columns=cluster_columns,
        heatmap_legend_param=kwargs.pop("heatmap_legend_param", {"title": stat}),
        **kwargs,
    )
