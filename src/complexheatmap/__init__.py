"""ComplexHeatmap-py: Python port of R/Bioconductor ComplexHeatmap (v2.25.3)."""

from __future__ import annotations

__version__ = "0.1.0"
__upstream_r_version__ = "2.25.3"

# Core options, base infix operators, units & graphics
from ._base import AdditiveUnit, v
from ._options import GLOBAL_PADDING, _ENV, get_last_ht, ht_global_opt, ht_opt
from .colors import (
    ColorMapping,
    add_luminance,
    add_transparency,
    brewer_pal,
    c_ColorMapping,
    colorRamp2,
    color_mapping_legend,
    default_col,
    grid_pretty,
    hcl_colors,
    map_to_colors,
    pretty_breaks,
    rand_color,
    to_hex_rgba,
    to_hex_rgba_list,
    to_rgba_tuple,
)
from .grid import (
    GTree,
    Grob,
    Viewport,
    current_device,
    current_viewport,
    getXY_in_parent_vp,
    grid,
    grid_boxplot,
    grid_circle,
    grid_draw,
    grid_grabExpr,
    grid_lines,
    grid_newpage,
    grid_points,
    grid_polygon,
    grid_raster,
    grid_rect,
    grid_roundrect,
    grid_segments,
    grid_text,
    grid_xaxis,
    grid_yaxis,
    grobHeight,
    grobWidth,
    linesGrob,
    list_components,
    max_text_height,
    max_text_width,
    pointsGrob,
    polygonGrob,
    popViewport,
    pushViewport,
    rectGrob,
    roundrectGrob,
    seekViewport,
    segmentsGrob,
    textGrob,
    upViewport,
    viewport,
)
from .text import GridText, draw_richtext, grid_textbox, gt_render, smartAlign, smartAlign2, textbox_grob
from .units import (
    Gpar,
    Unit,
    as_unit,
    check_gp,
    convertHeight,
    convertWidth,
    convertX,
    convertY,
    gpar,
    is_abs_unit,
    is_unit,
    merge_gp,
    recycle_gp,
    subset_gp,
    unit,
    unit_c,
    unit_max,
    unit_min,
    unit_pmax,
    unit_pmin,
    unit_rep,
    unit_sum,
)

# Dendrogram & clustering
from .dendrogram import (
    Dendrogram,
    DistMatrix,
    Hclust,
    adjust_dend_by_x,
    as_dendrogram,
    as_dist,
    cluster_between_groups,
    cluster_within_group,
    color_branches,
    cut_dendrogram,
    cutree,
    dend_heights,
    dend_xy,
    dendrogramGrob,
    dist,
    dist2,
    grid_dendrogram,
    hclust,
    merge_dendrogram,
    nobs,
    order_dendrogram,
    reorder_dendrogram,
)

# Legends
from .legend import Legend, Legends, packLegend

# Annotations (always import annotations before anno_funcs)
from .annotations import (
    AnnotationFunction,
    HeatmapAnnotation,
    SingleAnnotation,
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
    anno_link,
    anno_mark,
    anno_numeric,
    anno_oncoprint_barplot,
    anno_points,
    anno_simple,
    anno_summary,
    anno_text,
    anno_textbox,
    anno_zoom,
    annotation_axis_grob,
    c_HeatmapAnnotation,
    columnAnnotation,
    construct_axis_grob,
    default_axis_param,
    grid_annotation_axis,
    rowAnnotation,
    row_anno_barplot,
    row_anno_boxplot,
    row_anno_density,
    row_anno_histogram,
    row_anno_points,
    row_anno_text,
    validate_axis_param,
)

# Heatmap, HeatmapList & Decorations
from .decorate import (
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
)
from .heatmap import Heatmap
from .heatmap_list import (
    HeatmapList,
    add_heatmap,
    attach_annotation,
    column_dend,
    column_order,
    draw,
    ht_size,
    row_dend,
    row_order,
)

# High-level plots, genomic utilities, translations & helpers
from .density import densityHeatmap, frequencyHeatmap, ks_dist
from .genomic import GRanges, IRanges, bin_genome, findOverlaps, generateRandomBed, normalize_genomic_signals_to_bins
from .oncoprint import alter_graphic, default_get_type, oncoPrint, test_alter_fun, unify_mat_list
from .three_d import Heatmap3D, bar3D
from .translate import compare_heatmap, compare_heatmap_2, compare_pheatmap, heatmap, heatmap_2, pheatmap
from .upset import (
    CombMat,
    UpSet,
    comb_degree,
    comb_name,
    comb_size,
    complement_size,
    extract_comb,
    full_comb_code,
    list_to_matrix,
    make_comb_mat,
    normalize_comb_mat,
    order_comb_mat,
    set_name,
    set_size,
    upset_left_annotation,
    upset_right_annotation,
    upset_top_annotation,
)
from .utils import c, dim, height, load_dataset, names, ncol, nrow, pindex, restore_matrix, set_names, size, summary, t, width

__all__ = [name for name in globals() if not name.startswith("_")]
