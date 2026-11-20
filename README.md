<p align="center">
  <img src="https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/01_logo.png" alt="ComplexHeatmap-py Logo" width="220" />
</p>

# ComplexHeatmap-py

Complete standalone Python port of the R/Bioconductor **ComplexHeatmap** (v2.25.3) package, powered by NumPy, SciPy, Pandas, Pillow, and Matplotlib.

## Documentation

Full documentation and chapter-by-chapter tutorial book: **[ComplexHeatmap-py Complete Reference Book](https://zhanxw.github.io/ComplexHeatmap-reference-py/book/)**

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

ch.ht_opt(kmeans_seed=123)
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
res = ch.draw(ht1 + ht2, width=9.0, height=6.5, dpi=150)

# Post-draw decoration (just like R's decorate_heatmap_body)
with ch.decorate_heatmap_body("Expr", row_slice=1, column_slice=1):
    ch.grid.rect(gp=ch.gpar(fill="transparent", col="gold", lwd=2))

res._device.fig.savefig("heatmap.png", dpi=150, facecolor="white")
```

![Quick Example Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/00_quick_example.png)

## Gallery of Examples & Output Figures

All figures below are generated directly by [`examples/showcase_gallery.py`](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/examples/showcase_gallery.py).

### 1. Split Heatmap with Colored Dendrograms & Block Annotations

```python
dend_r = ch.color_branches(ch.hclust(ch.dist(mat)), k=2, col=["#E41A1C", "#377EB8"])
dend_c = ch.color_branches(ch.hclust(ch.dist(mat.T)), k=3, col=["#4DAF4A", "#984EA3", "#FF7F00"])
ht = ch.Heatmap(
    mat, name="Z-score",
    cluster_rows=dend_r, row_split=2,
    cluster_columns=dend_c, column_split=3,
    top_annotation=top_ha, left_annotation=left_ha, border=True,
)
ht.draw(save="02_single_heatmap_split.png")
```

![Split Heatmap Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/02_single_heatmap_split.png)

### 2. Multi-Track Annotations Zoo (`anno_points`, `anno_lines`, `anno_barplot`, `anno_boxplot`, `anno_density`, `anno_horizon`, `anno_mark`, `anno_textbox`)

```python
top_ha = ch.HeatmapAnnotation(
    Simple=list("AB" * 10),
    Points=ch.anno_points(pts),
    Lines=ch.anno_lines(line_vals, add_points=True),
    Bars=ch.anno_barplot(bar_vals),
    Boxplot=ch.anno_boxplot(box_mat),
)
right_ha = ch.rowAnnotation(
    Density=ch.anno_density(mat.to_numpy(), type="violin"),
    Horizon=ch.anno_horizon(horizon_series),
    Markers=ch.anno_mark(at=[2, 8, 15, 24, 33, 38], labels=["TP53", "KRAS", "EGFR", "MYC", "BRAF", "PTEN"]),
    Text=ch.anno_textbox(sp, pathways_dict),
)
ch.Heatmap(mat, row_split=sp, top_annotation=top_ha, right_annotation=right_ha).draw(...)
```

![Annotations Zoo Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/03_annotations_zoo.png)

### 3. Measles Incidence Heatmap & Post-Draw Viewport Decoration

```python
mat = ch.load_dataset("measles")
ht = ch.Heatmap(mat, name="Cases", col=col_fun, cluster_rows=False, cluster_columns=False,
                top_annotation=ha_col, right_annotation=ha_row, border=True)
ht.draw(width=11.0, height=8.0, dpi=140)
with ch.decorate_heatmap_body("Cases"):
    ch.grid_segments(ch.unit(pos, "npc"), ch.unit(0, "npc"), ch.unit(pos, "npc"), ch.unit(1, "npc"),
                     gp=ch.gpar(lty="dashed", lwd=2))
    ch.grid_text("Vaccine (1961)", x=ch.unit(pos + 0.01, "npc"), y=ch.unit(0.96, "npc"), just=["left", "top"])
```

![Measles Vaccine Heatmap Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/04_measles_vaccine.png)

### 4. TCGA OncoPrint (`oncoPrint`)

```python
df = ch.load_dataset("tcga_lung_adenocarcinoma").iloc[:25, :60]
op = ch.oncoPrint(df, alter_fun=alter_fun, col=alt_col,
                  remove_empty_columns=True, remove_empty_rows=True)
op.draw(save="05_tcga_oncoprint.png")
```

![TCGA OncoPrint Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/05_tcga_oncoprint.png)

### 5. Multi-Omics UpSet Plot (`make_comb_mat` + `UpSet`)

```python
cm = ch.make_comb_mat(sets)
cm = cm[:, ch.comb_size(cm) >= 2]
up = ch.UpSet(
    cm,
    comb_col=["#E41A1C" if d >= 3 else ("#377EB8" if d == 2 else "#333333") for d in ch.comb_degree(cm)],
    top_annotation=ch.upset_top_annotation(cm, add_numbers=True),
    right_annotation=ch.upset_right_annotation(cm, add_numbers=True),
)
up.draw(save="06_upset_plot.png")
```

![UpSet Plot Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/06_upset_plot.png)

### 6. Integrated Multi-Heatmap List (`+` & `rowAnnotation`)

```python
hl = ht_meth + ht_expr + ch.rowAnnotation(Corr=ch.anno_points(corr_vals))
ch.draw(hl, column_title="Integrated Methylation & Gene Expression", merge_legends=True)
```

![Multi-Heatmap List Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/07_multi_heatmap_list.png)

### 7. Density Heatmap & 3D Isometric Heatmap (`densityHeatmap`, `Heatmap3D`)

```python
ch.densityHeatmap(df_d, title="Distribution Density Across 10 Conditions", ylab="Signal").draw(...)
ch.Heatmap3D(mat_3d, name="Intensity", column_title="3D Isometric Heatmap (Heatmap3D)").draw(...)
```

![Density Heatmap Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/08_density_heatmap.png)
![Heatmap3D Output](https://raw.githubusercontent.com/zhanxw/ComplexHeatmap-py/main/examples/output/08_heatmap3d.png)

## Implemented Modules & Features

- **Core Graphics & Units (`units.py`, `grid.py`, `text.py`, `colors.py`)**: Vectorized `Unit` arithmetic (`mm`, `cm`, `inches`, `pt`, `npc`, `snpc`, `native`, `lines`, `char`, `grobwidth`/`grobheight`), `gpar`, nested Matplotlib viewports (`Viewport`, `pushViewport`, `popViewport`, `seekViewport`, `grid.*`), `ax=`/`SubFigure` composition, `to_svg()`, `GridText` (`gt_render` rich markup), `textbox_grob`, `smartAlign2`, and `colorRamp2` (LAB/XYZ/sRGB/HSV) + `ColorMapping`.
- **Clustering & Dendrograms (`dendrogram.py`)**: `dist`, `dist2`, `hclust` (including distinct `ward.D` and `ward.D2` + optional `fastcluster`), `Dendrogram`, `cut_dendrogram`, `adjust_dend_by_x`, `merge_dendrogram`, `color_branches`, `cluster_within_group`, `cluster_between_groups`, `grid_dendrogram`, Path A cutree & Path B slice-first + consensus k-means clustering.
- **Annotations (`annotations.py`, `anno_funcs.py`)**: All 21 `anno_*` (`anno_simple`, `anno_empty`, `anno_points`, `anno_lines`, `anno_barplot`, `anno_boxplot`, `anno_histogram`, `anno_density`, `anno_joyplot`, `anno_horizon`, `anno_text`, `anno_numeric`, `anno_mark`, `anno_zoom`, `anno_link`, `anno_block`, `anno_summary`, `anno_image`, `anno_customize`, `anno_textbox`, `anno_oncoprint_barplot`) + 6 `row_anno_*` aliases, `HeatmapAnnotation`, `rowAnnotation`, `columnAnnotation`, `SingleAnnotation`, `AnnotationFunction`, `attach_annotation`.
- **Multi-Heatmap Lists & Legends (`heatmap.py`, `heatmap_list.py`, `legend.py`, `decorate.py`)**: Horizontal (`+`) and vertical (`%v%` / `/`) concatenation, `auto_adjust`, `main_heatmap` alignment, `Legend`/`packLegend` discrete + continuous + unequal-break legends, legend deduplication/`c_ColorMapping`, all 11 `decorate_*` viewport decorators, and slice/subsetting indexing (`ht[i, j]`, `ht_list[i, j]`).
- **High-Level Genomics & Plots (`oncoprint.py`, `upset.py`, `density.py`, `three_d.py`, `genomic.py`, `translate.py`)**: `oncoPrint` (with memoSort & alteration polygons), `make_comb_mat` / `UpSet` (`distinct`, `intersect`, `union`, genomic intervals, `t(m)`), `densityHeatmap`, `frequencyHeatmap`, `ks_dist`, `bar3D`, `Heatmap3D`, `IRanges`/`GRanges`/`bin_genome`/`normalize_genomic_signals_to_bins`, and drop-in translations `pheatmap`, `heatmap`, `heatmap_2`, `compare_*`.
- **Examples & Verification**: Complete documentation in the **[ComplexHeatmap-py Complete Reference Book](https://zhanxw.github.io/ComplexHeatmap-reference-py/book/)**, ports of all **102 executable `.Rd` examples** ([examples/run_all_rd_examples.py](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/examples/run_all_rd_examples.py)), publication gallery ([examples/showcase_gallery.py](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/examples/showcase_gallery.py)), R-to-Python guide ([examples/r_to_python_guide.md](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/examples/r_to_python_guide.md)), and `pytest` suite (`135 passed`).

## License & Attribution

Released under the [MIT License](https://github.com/zhanxw/ComplexHeatmap-py/blob/main/LICENSE). Ported from the R/Bioconductor [ComplexHeatmap](https://github.com/jokergoo/ComplexHeatmap) package (v2.25.3), Copyright (c) 2020 Zuguang Gu (MIT License).
