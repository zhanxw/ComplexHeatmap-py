"""Drop-in translations of pheatmap, stats::heatmap, and gplots::heatmap.2."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .annotations import HeatmapAnnotation, rowAnnotation
from .colors import brewer_pal, colorRamp2, hcl_colors
from .dendrogram import as_dendrogram, cutree, dist as calc_dist, hclust
from .grid import grid_text
from .heatmap import Heatmap
from .heatmap_list import HeatmapList, draw
from .units import Unit, check_gp, gpar, unit


def _scale_matrix(mat: np.ndarray, scale: str) -> np.ndarray:
    x = np.asarray(mat, dtype=float).copy()
    if scale == "row":
        m = np.nanmean(x, axis=1, keepdims=True)
        s = np.nanstd(x, axis=1, ddof=1, keepdims=True)
        s = np.where((s == 0) | np.isnan(s), 1.0, s)
        x = (x - m) / s
    elif scale == "column":
        m = np.nanmean(x, axis=0, keepdims=True)
        s = np.nanstd(x, axis=0, ddof=1, keepdims=True)
        s = np.where((s == 0) | np.isnan(s), 1.0, s)
        x = (x - m) / s
    return x


def pheatmap(
    mat: Any,
    color: Sequence[str] | None = None,
    kmeans_k: int | None = None,
    breaks: Sequence[float] | None = None,
    border_color: str | None = "#999999",
    cellwidth: float | None = None,
    cellheight: float | None = None,
    scale: str = "none",
    cluster_rows: Any = True,
    cluster_cols: Any = True,
    clustering_distance_rows: str = "euclidean",
    clustering_distance_cols: str = "euclidean",
    clustering_method: str = "complete",
    clustering_callback: Any = None,
    cutree_rows: int | None = None,
    cutree_cols: int | None = None,
    treeheight_row: float | None = None,
    treeheight_col: float | None = None,
    legend: bool = True,
    legend_breaks: Sequence[float] | None = None,
    legend_labels: Sequence[str] | None = None,
    annotation_row: pd.DataFrame | None = None,
    annotation_col: pd.DataFrame | None = None,
    annotation: pd.DataFrame | None = None,
    annotation_colors: dict[str, Any] | None = None,
    annotation_legend: bool = True,
    annotation_names_row: bool = True,
    annotation_names_col: bool = True,
    drop_levels: bool = True,
    show_rownames: bool = True,
    show_colnames: bool = True,
    main: str | None = None,
    fontsize: float = 10.0,
    fontsize_row: float | None = None,
    fontsize_col: float | None = None,
    angle_col: float = 270.0,
    display_numbers: Any = False,
    number_format: str = "{:.2f}",
    number_color: str = "#333333",
    fontsize_number: float | None = None,
    gaps_row: Sequence[int] | None = None,
    gaps_col: Sequence[int] | None = None,
    labels_row: Sequence[str] | None = None,
    labels_col: Sequence[str] | None = None,
    filename: str | None = None,
    width: float | None = None,
    height: float | None = None,
    silent: bool = False,
    na_col: str = "#DDDDDD",
    name: str | None = None,
    run_draw: bool = False,
    **kwargs: Any,
) -> Heatmap:
    """R `pheatmap::pheatmap()` API translated cleanly to ComplexHeatmap (`pheatmap.Rd`)."""
    if isinstance(mat, pd.DataFrame):
        rn = list(mat.index.astype(str))
        cn = list(mat.columns.astype(str))
        arr = mat.to_numpy(dtype=float)
    else:
        arr = np.asarray(mat, dtype=float)
        rn = [str(i) for i in range(arr.shape[0])]
        cn = [str(j) for j in range(arr.shape[1])]

    arr = _scale_matrix(arr, str(scale).lower())
    df_mat = pd.DataFrame(arr, index=rn, columns=cn)

    # Color scale
    if color is None:
        pal = hcl_colors(100, "RdYlBu", rev=True)
    else:
        pal = list(color)
    finite = arr[np.isfinite(arr)]
    vmin = float(np.min(finite)) if len(finite) else -1.0
    vmax = float(np.max(finite)) if len(finite) else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    if breaks is not None:
        br = np.asarray(breaks, dtype=float)
        if len(br) == len(pal):
            col_fun = colorRamp2(br, pal)
        else:
            col_fun = colorRamp2(np.linspace(float(br[0]), float(br[-1]), len(pal)), pal)
    else:
        col_fun = colorRamp2(np.linspace(vmin, vmax, len(pal)), pal)

    # Distance aliases ('correlation' -> 'pearson')
    d_row = "pearson" if clustering_distance_rows == "correlation" else clustering_distance_rows
    d_col = "pearson" if clustering_distance_cols == "correlation" else clustering_distance_cols

    # Row / col splits from gaps or cutree
    row_split: Any = None
    col_split: Any = None
    if not cluster_rows and gaps_row:
        grp = np.zeros(arr.shape[0], dtype=int)
        for idx_g, g in enumerate(sorted(set(gaps_row)), start=1):
            grp[int(g):] = idx_g
        row_split = pd.Categorical([f"gap_{k}" for k in grp], categories=[f"gap_{k}" for k in range(max(grp) + 1)], ordered=True)
    elif cluster_rows and cutree_rows is not None and int(cutree_rows) > 1:
        row_split = int(cutree_rows)

    if not cluster_cols and gaps_col:
        grp = np.zeros(arr.shape[1], dtype=int)
        for idx_g, g in enumerate(sorted(set(gaps_col)), start=1):
            grp[int(g):] = idx_g
        col_split = pd.Categorical([f"gap_{k}" for k in grp], categories=[f"gap_{k}" for k in range(max(grp) + 1)], ordered=True)
    elif cluster_cols and cutree_cols is not None and int(cutree_cols) > 1:
        col_split = int(cutree_cols)

    # Annotations
    top_anno = None
    ann_col_df = annotation_col if annotation_col is not None else annotation
    if ann_col_df is not None:
        ann_df = ann_col_df.loc[cn] if isinstance(ann_col_df, pd.DataFrame) and set(cn).issubset(ann_col_df.index) else pd.DataFrame(ann_col_df)
        top_anno = HeatmapAnnotation(
            df=ann_df,
            col=annotation_colors,
            show_legend=annotation_legend,
            show_annotation_name=annotation_names_col,
        )

    left_anno = None
    if annotation_row is not None:
        ann_r_df = annotation_row.loc[rn] if isinstance(annotation_row, pd.DataFrame) and set(rn).issubset(annotation_row.index) else pd.DataFrame(annotation_row)
        left_anno = rowAnnotation(
            df=ann_r_df,
            col=annotation_colors,
            show_legend=annotation_legend,
            show_annotation_name=annotation_names_row,
        )

    # Cell numbers
    cell_fun = None
    fs_num = fontsize_number if fontsize_number is not None else 0.8 * fontsize
    if isinstance(display_numbers, np.ndarray) or isinstance(display_numbers, pd.DataFrame):
        dn_arr = np.asarray(display_numbers, dtype=object)
        def _cell_fun_mat(j: int, i: int, x: Unit, y: Unit, w: Unit, h: Unit, fill: str) -> None:
            grid_text(str(dn_arr[i, j]), x, y, gp=gpar(fontsize=fs_num, col=number_color))
        cell_fun = _cell_fun_mat
    elif display_numbers is True:
        def _cell_fun_num(j: int, i: int, x: Unit, y: Unit, w: Unit, h: Unit, fill: str) -> None:
            v = arr[i, j]
            lbl = "" if np.isnan(v) else (number_format.format(v) if "{" in number_format else (number_format % v))
            grid_text(lbl, x, y, gp=gpar(fontsize=fs_num, col=number_color))
        cell_fun = _cell_fun_num

    # Angle mapping from pheatmap angle_col -> ComplexHeatmap column_names_rot
    col_rot = {0: 0.0, 45: 45.0, 90: 90.0, 270: 90.0, 315: 45.0}.get(int(angle_col), float(angle_col))

    hm_w = unit(cellwidth * arr.shape[1] * 25.4 / 72.0, "mm") if cellwidth is not None else None
    hm_h = unit(cellheight * arr.shape[0] * 25.4 / 72.0, "mm") if cellheight is not None else None

    rd_w = unit(0, "mm") if (treeheight_row is not None and treeheight_row == 0) else (unit(treeheight_row * 25.4 / 72.0, "mm") if treeheight_row is not None else None)
    cd_h = unit(0, "mm") if (treeheight_col is not None and treeheight_col == 0) else (unit(treeheight_col * 25.4 / 72.0, "mm") if treeheight_col is not None else None)

    hlp: dict[str, Any] = {"title": name or ""}
    if legend_breaks is not None:
        hlp["at"] = list(legend_breaks)
    if legend_labels is not None:
        hlp["labels"] = list(legend_labels)

    ht = Heatmap(
        df_mat,
        col=col_fun,
        name=name or "matrix",
        na_col=na_col,
        rect_gp=gpar(col=border_color if border_color not in (False, "none", "NA") else None),
        cell_fun=cell_fun,
        column_title=main if main is not None else [],
        column_title_gp=gpar(fontsize=fontsize * 1.3, fontface="bold"),
        cluster_rows=cluster_rows,
        cluster_columns=cluster_cols,
        clustering_distance_rows=d_row,
        clustering_distance_columns=d_col,
        clustering_method_rows=clustering_method,
        clustering_method_columns=clustering_method,
        row_km=kmeans_k if kmeans_k and kmeans_k > 1 else 1,
        row_split=row_split,
        column_split=col_split,
        row_dend_width=rd_w,
        column_dend_height=cd_h,
        show_row_dend=bool(cluster_rows and (treeheight_row is None or treeheight_row > 0)),
        show_column_dend=bool(cluster_cols and (treeheight_col is None or treeheight_col > 0)),
        show_row_names=show_rownames,
        show_column_names=show_colnames,
        row_labels=list(labels_row) if labels_row is not None else None,
        column_labels=list(labels_col) if labels_col is not None else None,
        row_names_gp=gpar(fontsize=fontsize_row if fontsize_row is not None else fontsize),
        column_names_gp=gpar(fontsize=fontsize_col if fontsize_col is not None else fontsize),
        column_names_rot=col_rot,
        top_annotation=top_anno,
        left_annotation=left_anno,
        width=hm_w,
        height=hm_h,
        show_heatmap_legend=legend,
        heatmap_legend_param=hlp,
        **kwargs,
    )
    if filename is not None or run_draw:
        ht.draw(save=filename, width=width, height=height, show=False)
    return ht


def heatmap(
    x: Any,
    Rowv: Any = True,
    Colv: Any = True,
    scale: str = "row",
    col: Sequence[str] | None = None,
    RowSideColors: Sequence[str] | None = None,
    ColSideColors: Sequence[str] | None = None,
    main: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    labRow: Sequence[str] | None = None,
    labCol: Sequence[str] | None = None,
    **kwargs: Any,
) -> Heatmap:
    """Translate R `stats::heatmap()` to ComplexHeatmap::Heatmap (`heatmap.Rd`)."""
    arr = np.asarray(x, dtype=float)
    arr = _scale_matrix(arr, str(scale).lower())
    if col is None:
        col = hcl_colors(12, "YlOrRd", rev=True)

    cl_r = False if Rowv is None or Rowv is False else Rowv
    cl_c = False if Colv is None or Colv is False else Colv

    left_a = rowAnnotation(RowSide=list(RowSideColors), show_annotation_name=False, show_legend=False) if RowSideColors is not None else None
    top_a = HeatmapAnnotation(ColSide=list(ColSideColors), show_annotation_name=False, show_legend=False) if ColSideColors is not None else None

    return Heatmap(
        arr,
        col=col,
        cluster_rows=cl_r,
        cluster_columns=cl_c,
        left_annotation=left_a,
        top_annotation=top_a,
        column_title=main or xlab or [],
        column_title_side="top" if main else "bottom",
        row_title=ylab or [],
        row_title_side="right",
        row_labels=list(labRow) if labRow is not None else None,
        column_labels=list(labCol) if labCol is not None else None,
        show_heatmap_legend=False,
        **kwargs,
    )


def heatmap_2(
    x: Any,
    Rowv: Any = True,
    Colv: Any = True,
    dendrogram: str = "both",
    scale: str = "none",
    col: Sequence[str] | None = None,
    RowSideColors: Sequence[str] | None = None,
    ColSideColors: Sequence[str] | None = None,
    key: bool = True,
    key_title: str = "Color Key",
    main: str | None = None,
    trace: str = "none",
    **kwargs: Any,
) -> Heatmap:
    """Translate R `gplots::heatmap.2()` to ComplexHeatmap::Heatmap (`heatmap.2.Rd`)."""
    arr = np.asarray(x, dtype=float)
    arr = _scale_matrix(arr, str(scale).lower())
    if col is None:
        col = hcl_colors(25, "RdBu", rev=True)

    d_opt = str(dendrogram).lower()
    show_r_dend = d_opt in ("both", "row")
    show_c_dend = d_opt in ("both", "column", "col")

    left_a = rowAnnotation(RowSide=list(RowSideColors), show_annotation_name=False, show_legend=False) if RowSideColors is not None else None
    top_a = HeatmapAnnotation(ColSide=list(ColSideColors), show_annotation_name=False, show_legend=False) if ColSideColors is not None else None

    return Heatmap(
        arr,
        col=col,
        cluster_rows=False if Rowv is None or Rowv is False else Rowv,
        cluster_columns=False if Colv is None or Colv is False else Colv,
        show_row_dend=show_r_dend,
        show_column_dend=show_c_dend,
        left_annotation=left_a,
        top_annotation=top_a,
        column_title=main or [],
        show_heatmap_legend=bool(key),
        heatmap_legend_param={"title": key_title, "direction": "horizontal"},
        **kwargs,
    )


def compare_pheatmap(mat: Any, **kwargs: Any) -> Heatmap:
    """Create and optionally draw ComplexHeatmap equivalent of `pheatmap(mat, ...)`."""
    return pheatmap(mat, **kwargs)


def compare_heatmap(x: Any, **kwargs: Any) -> Heatmap:
    """Create ComplexHeatmap equivalent of `stats::heatmap(x, ...)`."""
    return heatmap(x, **kwargs)


def compare_heatmap_2(x: Any, **kwargs: Any) -> Heatmap:
    """Create ComplexHeatmap equivalent of `gplots::heatmap.2(x, ...)`."""
    return heatmap_2(x, **kwargs)
