# ComplexHeatmap-py

Complete standalone Python port of the R/Bioconductor **ComplexHeatmap** (v2.25.3) package, powered by NumPy, SciPy, Pandas, Pillow, and Matplotlib.

## Installation

```bash
pip install ComplexHeatmap-py
```

Or install from source:

```bash
git clone https://github.com/zhanxw/ComplexHeatmap-py.git
cd ComplexHeatmap-py
pip install -e .
```

Both `import complexheatmap as ch` and `import ComplexHeatmap as ch` are supported.

## Quick Example

```python
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import complexheatmap as ch

rng = np.random.default_rng(123)
mat = pd.DataFrame(
    rng.standard_normal((12, 10)),
    index=[f"gene_{i}" for i in range(12)],
    columns=[f"sample_{j}" for j in range(10)],
)

ha_top = ch.HeatmapAnnotation(
    Group=["Ctrl"] * 5 + ["Treat"] * 5,
    Score=ch.anno_barplot(rng.uniform(0, 1, 10)),
    col={"Group": {"Ctrl": "#4DAF4A", "Treat": "#E41A1C"}},
)

ht1 = ch.Heatmap(mat, name="Expr", row_km=2, top_annotation=ha_top, border=True)
ht2 = ch.Heatmap(mat.iloc[:, :4] * 2, name="Delta")
ch.draw(ht1 + ht2, save="heatmap.png", width=9, height=6.5, dpi=150)

# Post-draw decoration (just like R's decorate_heatmap_body)
with ch.decorate_heatmap_body("Expr", row_slice=1, column_slice=1):
    ch.grid.rect(gp=ch.gpar(fill="transparent", col="gold", lwd=2))
```

## Implemented Modules & Features

- **Core Graphics & Units (`units.py`, `grid.py`, `text.py`, `colors.py`)**: Vectorized `Unit` arithmetic (`mm`, `cm`, `inches`, `pt`, `npc`, `snpc`, `native`, `lines`, `char`, `grobwidth`/`grobheight`), `gpar`, nested Matplotlib viewports (`Viewport`, `pushViewport`, `popViewport`, `seekViewport`, `grid.*`), `GridText` (`gt_render` rich markup), `textbox_grob`, `smartAlign2`, and `colorRamp2` (LAB/XYZ/sRGB/HSV) + `ColorMapping`.
- **Clustering & Dendrograms (`dendrogram.py`)**: `dist`, `dist2`, `hclust`, `Dendrogram`, `cut_dendrogram`, `adjust_dend_by_x`, `merge_dendrogram`, `color_branches`, `cluster_within_group`, `cluster_between_groups`, `grid_dendrogram`, Path A cutree & Path B slice-first + consensus k-means clustering.
- **Annotations (`annotations.py`, `anno_funcs.py`)**: All 21 `anno_*` (`anno_simple`, `anno_empty`, `anno_points`, `anno_lines`, `anno_barplot`, `anno_boxplot`, `anno_histogram`, `anno_density`, `anno_joyplot`, `anno_horizon`, `anno_text`, `anno_numeric`, `anno_mark`, `anno_zoom`, `anno_link`, `anno_block`, `anno_summary`, `anno_image`, `anno_customize`, `anno_textbox`, `anno_oncoprint_barplot`) + 6 `row_anno_*` aliases, `HeatmapAnnotation`, `rowAnnotation`, `columnAnnotation`, `SingleAnnotation`, `AnnotationFunction`, `attach_annotation`.
- **Multi-Heatmap Lists & Legends (`heatmap.py`, `heatmap_list.py`, `legend.py`, `decorate.py`)**: Horizontal (`+`) and vertical (`%v%` / `/`) concatenation, `auto_adjust`, `main_heatmap` alignment, `Legend`/`packLegend` discrete + continuous + unequal-break legends, legend deduplication/`c_ColorMapping`, all 11 `decorate_*` viewport decorators, and slice/subsetting indexing (`ht[i, j]`, `ht_list[i, j]`).
- **High-Level Genomics & Plots (`oncoprint.py`, `upset.py`, `density.py`, `three_d.py`, `genomic.py`, `translate.py`)**: `oncoPrint` (with memoSort & alteration polygons), `make_comb_mat` / `UpSet` (`distinct`, `intersect`, `union`, genomic intervals, `t(m)`), `densityHeatmap`, `frequencyHeatmap`, `ks_dist`, `bar3D`, `Heatmap3D`, `IRanges`/`GRanges`/`bin_genome`/`normalize_genomic_signals_to_bins`, and drop-in translations `pheatmap`, `heatmap`, `heatmap_2`, `compare_*`.
- **Examples & Verification**: Complete ports of all **102 executable `.Rd` examples** ([examples/run_all_rd_examples.py](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/examples/run_all_rd_examples.py)), publication gallery ([examples/showcase_gallery.py](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/examples/showcase_gallery.py)), R-to-Python guide ([examples/r_to_python_guide.md](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/examples/r_to_python_guide.md)), and `pytest` suite.

## License & Attribution

Released under the [MIT License](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/LICENSE). Ported from the R/Bioconductor [ComplexHeatmap](https://github.com/jokergoo/ComplexHeatmap) package (v2.25.3), Copyright (c) 2020 Zuguang Gu (MIT License).
