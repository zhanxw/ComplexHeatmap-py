"""Publication showcase gallery porting flagship R ComplexHeatmap figures to PNG."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import complexheatmap as ch


OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_01_logo() -> None:
    """Port of ComplexHeatmap/logo.R."""
    lines = [
        "oooooo                  oooooo  oo          oo  oo",
        "oooooo                  oooooo  oo          oo  oo",
        "oo      oooooo  oo  oo  oo  oo  oo  oooooo  oooooo",
        "oo      oooooo  oooooo  oo  oo  oo  oooooo  oooooo",
        "oo      oo  oo  oooooo  oooo    oo  oo  oo    oo  ",
        "oo      oo  oo  oooooo  oooo    oo  oo  oo    oo  ",
        "oooooo  oooooo  oo  oo  oo      oo  oooo    oo  oo",
        "oooooo  oooooo  oo  oo  oo      oo  oooo    oo  oo",
    ]
    lines2 = [
        "oo  oo                    oo                    oooooo",
        "oo  oo                    oo                    oooooo",
        "oo  oo  oooooo    oooo  oooooo  oo  oo  oooooo  oo  oo",
        "oo  oo  oooooo    oooo  oooooo  oooooo  oooooo  oo  oo",
        "oooooo  oo  oo  oo  oo    oo    oooooo  oo  oo  oooo  ",
        "oooooo  oo  oo  oo  oo    oo    oooooo  oo  oo  oooo  ",
        "oo  oo  oooo    oooooo    oo    oo  oo  oooo    oo    ",
        "oo  oo  oooo    oooooo    oo    oo  oo  oooo    oo    ",
    ]
    mat1 = np.zeros((8, max(len(l) for l in lines)), dtype=int)
    for i, l in enumerate(lines):
        for j, c in enumerate(l):
            if not c.isspace():
                mat1[i, j] = 1
    mat1 = np.hstack([np.zeros((8, 2), dtype=int), mat1])

    mat2 = np.zeros((8, max(len(l) for l in lines2)), dtype=int)
    for i, l in enumerate(lines2):
        for j, c in enumerate(l):
            if not c.isspace():
                mat2[i, j] = 1

    if mat1.shape[1] < mat2.shape[1]:
        mat1 = np.hstack([mat1, np.zeros((8, mat2.shape[1] - mat1.shape[1]), dtype=int)])
    elif mat2.shape[1] < mat1.shape[1]:
        mat2 = np.hstack([mat2, np.zeros((8, mat1.shape[1] - mat2.shape[1]), dtype=int)])

    mat = np.vstack([mat1, np.zeros((2, mat1.shape[1]), dtype=int), mat2])
    mat = np.vstack([np.zeros((16, mat.shape[1]), dtype=int), mat, np.zeros((16, mat.shape[1]), dtype=int)])
    mat = np.hstack([np.zeros((mat.shape[0], 6), dtype=int), mat, np.zeros((mat.shape[0], 6), dtype=int)])
    mat[12:14, 5:-5] = 1
    mat[-14:-12, 5:-5] = 1

    rng = np.random.default_rng(123)
    def col_fun(x):
        arr = np.asarray(x, dtype=float)
        cols = np.empty(arr.shape, dtype=object)
        for idx, val in np.ndenumerate(arr):
            if val == 1:
                cols[idx] = ch.add_transparency("#4DAF4A", float(rng.uniform(0.0, 0.3)))
            else:
                cols[idx] = ch.rand_color(1, luminosity="light", transparency=0.82)[0]
        return cols
    col_fun.breaks = [0.0, 1.0]

    ht = ch.Heatmap(
        mat,
        name="logo",
        col=col_fun,
        rect_gp=ch.gpar(col="white", lwd=0.4),
        cluster_rows=False,
        cluster_columns=False,
        show_heatmap_legend=False,
    )
    g = ch.grid_grabExpr(lambda: ch.draw(ht, padding=ch.unit([0, 0, 0, 0], "mm"), newpage=False), width_in=6.0, height_in=4.5)

    dev = ch.grid_newpage(width_in=6.0, height_in=6.0, dpi=140)
    ch.pushViewport(ch.Viewport(xscale=(0, 2), yscale=(0, 2), width=ch.unit(0.92, "snpc"), height=ch.unit(0.92, "snpc")))
    ang = np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    hx = np.cos(ang) + 1.0
    hy = np.sin(ang) + 1.0
    ch.grid_polygon(hx, hy, default_units="native", gp=ch.gpar(fill="#F9FFF9", col="#4DAF4A", lwd=5))
    ch.pushViewport(ch.Viewport(x=0.5, y=0.5, width=0.70, height=0.70 * mat.shape[0] / mat.shape[1]))
    ch.grid_draw(g)
    ch.popViewport()
    ch.grid_polygon(hx, hy, default_units="native", gp=ch.gpar(fill="transparent", col="#4DAF4A", lwd=7))
    ch.popViewport()
    dev.fig.savefig(OUT_DIR / "01_logo.png", dpi=140, facecolor="white")
    plt.close(dev.fig)


def render_02_single_heatmap_split() -> None:
    """Split heatmap with colored dendrograms, anno_block, layer_fun, and decorations."""
    rng = np.random.default_rng(123)
    mat = pd.DataFrame(
        np.vstack([rng.normal(1.2, 0.7, (12, 16)), rng.normal(-1.0, 0.7, (12, 16))]),
        index=[f"Gene_{i+1}" for i in range(24)],
        columns=[f"Sample_{j+1}" for j in range(16)],
    )
    mat.iloc[:12, 8:] -= 1.8
    mat.iloc[12:, :8] += 1.6

    hc_r = ch.hclust(ch.dist(mat.to_numpy()))
    dend_r = ch.color_branches(ch.as_dendrogram(hc_r), k=2, col=["#E41A1C", "#377EB8"])
    hc_c = ch.hclust(ch.dist(mat.to_numpy().T))
    dend_c = ch.color_branches(ch.as_dendrogram(hc_c), k=3, col=["#4DAF4A", "#984EA3", "#FF7F00"])

    top_ha = ch.HeatmapAnnotation(
        Cluster=ch.anno_block(gp=ch.gpar(fill=["#4DAF4A", "#984EA3", "#FF7F00"]), labels=["C1", "C2", "C3"], labels_gp=ch.gpar(col="white", fontsize=10, fontface="bold")),
        Score=ch.anno_barplot(rng.uniform(0.2, 1.0, 16), gp=ch.gpar(fill="#5B8DBE"), height=ch.unit(14, "mm")),
        Group=["Ctrl"] * 8 + ["Treat"] * 8,
        col={"Group": {"Ctrl": "#66C2A5", "Treat": "#FC8D62"}},
    )
    left_ha = ch.rowAnnotation(
        Module=ch.anno_block(gp=ch.gpar(fill=["#E41A1C", "#377EB8"]), labels=["Up", "Down"], labels_gp=ch.gpar(col="white", fontsize=10, fontface="bold")),
    )

    ht = ch.Heatmap(
        mat,
        name="Z-score",
        col=ch.colorRamp2([-2.5, 0, 2.5], ["#2166AC", "#F7F7F7", "#B2182B"]),
        cluster_rows=dend_r,
        row_split=2,
        cluster_columns=dend_c,
        column_split=3,
        top_annotation=top_ha,
        left_annotation=left_ha,
        border=True,
        row_gap=ch.unit(2, "mm"),
        column_gap=ch.unit(2, "mm"),
        column_title="Split Heatmap with Dendrogram Coloring & Annotations",
        column_title_gp=ch.gpar(fontsize=13, fontface="bold"),
        row_names_gp=ch.gpar(fontsize=8),
        column_names_gp=ch.gpar(fontsize=8),
    )
    ht.draw(save=str(OUT_DIR / "02_single_heatmap_split.png"), width=9.0, height=7.5, dpi=140)
    plt.close("all")


def render_03_annotations_zoo() -> None:
    """Multi-track HeatmapAnnotation and rowAnnotation showcase."""
    rng = np.random.default_rng(7)
    nr, nc = 40, 20
    mat = pd.DataFrame(
        rng.standard_normal((nr, nc)),
        index=[f"G{i+1}" for i in range(nr)],
        columns=[f"S{j+1}" for j in range(nc)],
    )
    sp = ["Cluster A"] * 20 + ["Cluster B"] * 20

    top_ha = ch.HeatmapAnnotation(
        Simple=list("AB" * 10),
        Points=ch.anno_points(rng.normal(0, 1, (nc, 2)), pch=[16, 17], gp=ch.gpar(col=["#D95F02", "#7570B3"]), height=ch.unit(15, "mm")),
        Lines=ch.anno_lines(np.cumsum(rng.normal(0, 0.4, nc)), add_points=True, smooth=False, gp=ch.gpar(col="#1B9E77"), height=ch.unit(15, "mm")),
        Bars=ch.anno_barplot(rng.uniform(0, 10, nc), gp=ch.gpar(fill="#386CB0"), height=ch.unit(15, "mm")),
        Boxplot=ch.anno_boxplot(rng.standard_normal((12, nc)), gp=ch.gpar(fill="#F0027F"), height=ch.unit(16, "mm")),
        col={"Simple": {"A": "#66C2A5", "B": "#FC8D62"}},
    )
    right_ha = ch.rowAnnotation(
        Density=ch.anno_density(mat.to_numpy(), type="violin", gp=ch.gpar(fill="#8DA0CB"), width=ch.unit(22, "mm")),
        Horizon=ch.anno_horizon([np.cumsum(rng.normal(0, 0.3, 30)) for _ in range(nr)], width=ch.unit(22, "mm")),
        Markers=ch.anno_mark(at=[2, 8, 15, 24, 33, 38], labels=["TP53", "KRAS", "EGFR", "MYC", "BRAF", "PTEN"]),
        Text=ch.anno_textbox(sp, {"Cluster A": ["Pathway 1", "Cell Cycle", "p53 signaling"], "Cluster B": ["MAPK cascade", "Immunity", "Metabolism"]}),
    )

    ht = ch.Heatmap(
        mat,
        name="Expr",
        row_split=sp,
        top_annotation=top_ha,
        right_annotation=right_ha,
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        column_title="ComplexHeatmap Annotation Zoo",
        column_title_gp=ch.gpar(fontsize=13, fontface="bold"),
        border=True,
    )
    ht.draw(save=str(OUT_DIR / "03_annotations_zoo.png"), width=11.0, height=8.5, dpi=140, merge_legends=True, padding=ch.unit([8, 4, 4, 4], "mm"))
    plt.close("all")


def render_04_measles_vaccine() -> None:
    """Classic US Measles vaccine incidence heatmap using bundled measles.csv."""
    mat = ch.load_dataset("measles")
    ha_col = ch.HeatmapAnnotation(
        Total=ch.anno_barplot(mat.sum(axis=0).to_numpy(), gp=ch.gpar(fill="#666666", col=None), border=False, height=ch.unit(20, "mm")),
        show_annotation_name=False,
    )
    ha_row = ch.rowAnnotation(
        Total=ch.anno_barplot(mat.sum(axis=1).to_numpy(), gp=ch.gpar(fill="#666666", col=None), border=False, width=ch.unit(20, "mm")),
        show_annotation_name=False,
    )
    col_fun = ch.colorRamp2([0, 400, 800, 1200, 1600, 2000], ["white", "cornflowerblue", "yellow", "orange", "red", "darkred"])
    ht = ch.Heatmap(
        mat,
        name="Cases",
        col=col_fun,
        cluster_rows=False,
        cluster_columns=False,
        row_names_gp=ch.gpar(fontsize=6.5),
        column_names_gp=ch.gpar(fontsize=6.5),
        top_annotation=ha_col,
        right_annotation=ha_row,
        column_title="Measles Cases in US States (1930-2001) | Vaccine Introduced 1961",
        column_title_gp=ch.gpar(fontsize=12, fontface="bold"),
        border=True,
    )
    ht.draw(save=str(OUT_DIR / "04_measles_vaccine.png"), width=11.0, height=8.0, dpi=140)
    # Mark vaccine introduction year 1961
    cols_list = list(mat.columns.astype(str))
    if "1961" in cols_list:
        pos = (cols_list.index("1961") + 0.5) / len(cols_list)
        with ch.decorate_heatmap_body("Cases"):
            ch.grid_segments(ch.unit(pos, "npc"), ch.unit(0, "npc"), ch.unit(pos, "npc"), ch.unit(1, "npc"), gp=ch.gpar(lty="dashed", col="#111111", lwd=2))
            ch.grid_text("Vaccine (1961)", x=ch.unit(pos + 0.01, "npc"), y=ch.unit(0.96, "npc"), just=["left", "top"], gp=ch.gpar(fontsize=9, fontface="bold"))
    ch._ENV.last._device.fig.savefig(OUT_DIR / "04_measles_vaccine.png", dpi=140, facecolor="white")
    plt.close("all")


def render_05_tcga_oncoprint() -> None:
    """TCGA Lung Adenocarcinoma OncoPrint using bundled dataset."""
    df = ch.load_dataset("tcga_lung_adenocarcinoma").iloc[:25, :60]
    alt_col = {"HOMDEL": "blue", "AMP": "red", "MUT": "#008000"}
    alter_fun = {
        "background": ch.alter_graphic("rect", fill="#EEEEEE"),
        "HOMDEL": ch.alter_graphic("rect", fill=alt_col["HOMDEL"]),
        "AMP": ch.alter_graphic("rect", fill=alt_col["AMP"]),
        "MUT": ch.alter_graphic("rect", height=0.35, fill=alt_col["MUT"]),
    }
    op = ch.oncoPrint(
        df,
        alter_fun=alter_fun,
        col=alt_col,
        remove_empty_columns=True,
        remove_empty_rows=True,
        column_title="TCGA Lung Adenocarcinoma (Top 25 Genes)",
        column_title_gp=ch.gpar(fontsize=13, fontface="bold"),
        row_names_gp=ch.gpar(fontsize=8),
        pct_gp=ch.gpar(fontsize=8),
    )
    op.draw(save=str(OUT_DIR / "05_tcga_oncoprint.png"), width=10.5, height=7.0, dpi=140)
    plt.close("all")


def render_06_upset_plot() -> None:
    """UpSet plot of multi-set intersections."""
    rng = np.random.default_rng(2026)
    genes = [f"G{i}" for i in range(120)]
    sets = {
        "RNA-seq Up": rng.choice(genes, 45, replace=False),
        "ATAC Peak": rng.choice(genes, 52, replace=False),
        "ChIP-seq H3K27ac": rng.choice(genes, 38, replace=False),
        "GWAS Loci": rng.choice(genes, 28, replace=False),
        "CRISPR Hit": rng.choice(genes, 22, replace=False),
    }
    cm = ch.make_comb_mat(sets)
    cm = cm[:, ch.comb_size(cm) >= 2]
    up = ch.UpSet(
        cm,
        pt_size=ch.unit(3.2, "mm"),
        lwd=2.0,
        comb_col=["#E41A1C" if d >= 3 else ("#377EB8" if d == 2 else "#333333") for d in ch.comb_degree(cm)],
        top_annotation=ch.upset_top_annotation(cm, add_numbers=True, gp=ch.gpar(fill="#377EB8", col=None)),
        right_annotation=ch.upset_right_annotation(cm, add_numbers=True, gp=ch.gpar(fill="#4DAF4A", col=None)),
        column_title="Multi-Omics Overlap (UpSet Plot)",
        column_title_gp=ch.gpar(fontsize=13, fontface="bold"),
    )
    up.draw(save=str(OUT_DIR / "06_upset_plot.png"), width=10.0, height=5.8, dpi=140, padding=ch.unit([8, 4, 4, 4], "mm"))
    plt.close("all")


def render_07_multi_heatmap_list() -> None:
    """Multi-heatmap list aligning methylation, expression, and clinical annotations."""
    ds = ch.load_dataset("random_meth_expr_data")
    mat_meth = ds["mat_meth"].iloc[:60, :18]
    mat_expr = ds["mat_expr"].iloc[:60, :18]
    row_meta = ds["row_meta"].iloc[:60]

    ha_meth = ch.HeatmapAnnotation(
        Type=["Tumor"] * 9 + ["Normal"] * 9,
        col={"Type": {"Tumor": "#E41A1C", "Normal": "#377EB8"}},
        show_legend={"Type": True},
    )
    col_meth = ch.colorRamp2([0, 0.5, 1], ["#2B8CBE", "#FFFFBF", "#D7301F"])
    col_expr = ch.colorRamp2([-2, 0, 2], ["#2166AC", "white", "#B2182B"])

    ht_meth = ch.Heatmap(
        mat_meth,
        name="Methylation",
        col=col_meth,
        row_split=row_meta.iloc[:, 0].astype(str).tolist(),
        top_annotation=ha_meth,
        column_title="Methylation",
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=7),
        border=True,
    )
    ht_expr = ch.Heatmap(
        mat_expr,
        name="Expression",
        col=col_expr,
        column_title="Expression",
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=7),
        border=True,
    )
    anno_corr = ch.rowAnnotation(
        Corr=ch.anno_points(np.linspace(-0.8, 0.8, 60), gp=ch.gpar(col="#4DAF4A"), width=ch.unit(20, "mm")),
    )
    hl = ht_meth + ht_expr + anno_corr
    ch.draw(
        hl,
        save=str(OUT_DIR / "07_multi_heatmap_list.png"),
        width=11.0,
        height=7.5,
        dpi=140,
        column_title="Integrated Methylation & Gene Expression",
        column_title_gp=ch.gpar(fontsize=14, fontface="bold"),
        merge_legends=True,
    )
    plt.close("all")


def render_08_density_and_3d() -> None:
    """DensityHeatmap and Heatmap3D showcase."""
    rng = np.random.default_rng(99)
    mat_d = np.column_stack([rng.normal(loc=k * 0.35, scale=0.8 + 0.1 * (k % 3), size=80) for k in range(10)])
    df_d = pd.DataFrame(mat_d, columns=[f"C{k+1}" for k in range(10)])
    dh = ch.densityHeatmap(df_d, title="Distribution Density Across 10 Conditions", ylab="Signal")
    dh.draw(save=str(OUT_DIR / "08_density_heatmap.png"), width=8.5, height=5.5, dpi=140)
    plt.close("all")

    mat_3d = pd.DataFrame(
        rng.uniform(0.1, 1.0, size=(6, 8)),
        index=[f"Row_{i+1}" for i in range(6)],
        columns=[f"Col_{j+1}" for j in range(8)],
    )
    h3 = ch.Heatmap3D(
        mat_3d,
        col=ch.colorRamp2([0, 0.5, 1], ["#2B8CBE", "#FEE08B", "#D7301F"]),
        name="Intensity",
        column_title="3D Isometric Heatmap (Heatmap3D)",
        column_title_gp=ch.gpar(fontsize=13, fontface="bold"),
        top_annotation=ch.HeatmapAnnotation(_spacer=ch.anno_empty(border=False, height=ch.unit(8, "mm")), show_annotation_name=False),
        bar_max_length=ch.unit(7.5, "mm"),
    )
    h3.draw(save=str(OUT_DIR / "08_heatmap3d.png"), width=8.5, height=6.5, dpi=140, padding=ch.unit([6, 4, 6, 4], "mm"))
    plt.close("all")


def main() -> None:
    for fn in (
        render_01_logo,
        render_02_single_heatmap_split,
        render_03_annotations_zoo,
        render_04_measles_vaccine,
        render_05_tcga_oncoprint,
        render_06_upset_plot,
        render_07_multi_heatmap_list,
        render_08_density_and_3d,
    ):
        fn()
        ch.ht_opt(RESET=True)
        print(f"Rendered {fn.__name__}")


if __name__ == "__main__":
    main()
