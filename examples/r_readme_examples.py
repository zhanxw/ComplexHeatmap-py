"""Replicate the 6 flagship example figures from the upstream R ComplexHeatmap README.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import complexheatmap as ch


OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_r_readme_01_methylation_profile() -> None:
    """1. Visualize Methylation Profile with Complex Annotations (Sturm et al., 2012 / Ch 14.5)."""
    ch.ht_opt(kmeans_seed=123)
    rng = np.random.default_rng(101)

    subtypes = ["IDH", "K27", "G34", "RTK I", "Mesenchymal", "RTK II"]
    n_per = 20
    n_samples = len(subtypes) * n_per
    dkfz_subtype = np.repeat(subtypes, n_per)

    tcga_map = {
        "IDH": ("Proneural", 0.90),
        "K27": ("Proneural", 0.65),
        "G34": ("Proneural", 0.55),
        "RTK I": ("Proneural", 0.50),
        "Mesenchymal": ("Mesenchymal", 0.88),
        "RTK II": ("Classical", 0.85),
    }
    tcga_pool = ["Proneural", "Neural", "Classical", "Mesenchymal"]
    tcga_subtype = []
    for st in dkfz_subtype:
        dom, prob = tcga_map[st]
        if rng.random() < prob:
            tcga_subtype.append(dom)
        else:
            tcga_subtype.append(rng.choice(tcga_pool))

    age = np.zeros(n_samples)
    for i, st in enumerate(dkfz_subtype):
        if st in ("K27", "G34"):
            age[i] = np.clip(rng.normal(14, 5), 4, 30)
        elif st == "IDH":
            age[i] = np.clip(rng.normal(38, 9), 20, 62)
        else:
            age[i] = np.clip(rng.normal(58, 11), 28, 82)

    idh1 = ["Mutant" if (st == "IDH" and rng.random() < 0.95) or rng.random() < 0.04 else "WT" for st in dkfz_subtype]
    h3f3a = []
    for st in dkfz_subtype:
        if st == "K27" and rng.random() < 0.90:
            h3f3a.append("K27M")
        elif st == "G34" and rng.random() < 0.88:
            h3f3a.append("G34R")
        else:
            h3f3a.append("WT")
    tp53 = [
        "Mutant" if ((st in ("IDH", "G34") and rng.random() < 0.80) or (st == "K27" and rng.random() < 0.60) or rng.random() < 0.25) else "WT"
        for st in dkfz_subtype
    ]
    chr7_gain = ["Gain" if ((st in ("RTK I", "Mesenchymal", "RTK II") and rng.random() < 0.82) or rng.random() < 0.12) else "Neutral" for st in dkfz_subtype]
    chr10_loss = ["Loss" if ((st in ("RTK I", "Mesenchymal", "RTK II") and rng.random() < 0.78) or rng.random() < 0.15) else "Neutral" for st in dkfz_subtype]
    cdkn2a_del = ["Del" if ((st in ("RTK I", "RTK II", "Mesenchymal") and rng.random() < 0.62) or rng.random() < 0.10) else "Neutral" for st in dkfz_subtype]
    egfr_amp = ["Amp" if ((st == "RTK II" and rng.random() < 0.78) or (st == "Mesenchymal" and rng.random() < 0.25) or rng.random() < 0.05) else "Neutral" for st in dkfz_subtype]
    pdgfra_amp = ["Amp" if ((st == "RTK I" and rng.random() < 0.55) or rng.random() < 0.04) else "Neutral" for st in dkfz_subtype]

    # Simulate subtype-specific CpG methylation (240 probes x 120 samples) and matched expression (100 genes x 120)
    n_cpg = 240
    meth_means = np.full((n_cpg, len(subtypes)), 0.5)
    meth_means[:45, :] = np.array([0.92, 0.42, 0.38, 0.55, 0.48, 0.52])    # IDH G-CIMP hyper
    meth_means[45:85, :] = np.array([0.75, 0.16, 0.24, 0.52, 0.46, 0.50])   # K27/G34 hypo
    meth_means[85:125, :] = np.array([0.82, 0.76, 0.20, 0.44, 0.35, 0.40])  # G34 hypo block
    meth_means[125:165, :] = np.array([0.65, 0.50, 0.48, 0.22, 0.62, 0.68]) # RTK I hypo
    meth_means[165:205, :] = np.array([0.72, 0.45, 0.45, 0.58, 0.20, 0.55]) # Mesenchymal hypo
    meth_means[205:, :] = np.array([0.68, 0.52, 0.55, 0.54, 0.60, 0.18])    # RTK II hypo

    mat_meth = np.empty((n_cpg, n_samples), dtype=float)
    for k_st in range(len(subtypes)):
        c0, c1 = k_st * n_per, (k_st + 1) * n_per
        mat_meth[:, c0:c1] = np.clip(
            meth_means[:, k_st:k_st + 1] + rng.normal(0, 0.12, size=(n_cpg, n_per)),
            0.0, 1.0,
        )

    n_genes = 100
    expr_means = np.zeros((n_genes, len(subtypes)))
    for k_st in range(len(subtypes)):
        r0 = int(k_st * (n_genes / len(subtypes)))
        r1 = int((k_st + 1) * (n_genes / len(subtypes)))
        expr_means[r0:r1, k_st] = 1.65
        expr_means[r0:r1, (k_st + 3) % len(subtypes)] = -0.9
    mat_expr = np.empty((n_genes, n_samples), dtype=float)
    for k_st in range(len(subtypes)):
        c0, c1 = k_st * n_per, (k_st + 1) * n_per
        mat_expr[:, c0:c1] = expr_means[:, k_st:k_st + 1] + rng.normal(0, 0.65, size=(n_genes, n_per))

    col_map = {
        "DKFZ_subtype": {"IDH": "#E41A1C", "K27": "#377EB8", "G34": "#4DAF4A", "RTK I": "#984EA3", "Mesenchymal": "#FF7F00", "RTK II": "#A65628"},
        "TCGA_subtype": {"Proneural": "#1B9E77", "Neural": "#7570B3", "Classical": "#E7298A", "Mesenchymal": "#D95F02"},
        "IDH1": {"Mutant": "#111111", "WT": "#EEEEEE"},
        "H3F3A": {"WT": "#EEEEEE", "K27M": "#377EB8", "G34R": "#4DAF4A"},
        "TP53": {"Mutant": "#111111", "WT": "#EEEEEE"},
        "chr7_gain": {"Gain": "#D7301F", "Neutral": "#EEEEEE"},
        "chr10_loss": {"Loss": "#2B8CBE", "Neutral": "#EEEEEE"},
        "CDKN2A_del": {"Del": "#2B8CBE", "Neutral": "#EEEEEE"},
        "EGFR_amp": {"Amp": "#D7301F", "Neutral": "#EEEEEE"},
        "PDGFRA_amp": {"Amp": "#984EA3", "Neutral": "#EEEEEE"},
    }

    ha_clinical = ch.HeatmapAnnotation(
        Age=ch.anno_points(age, height=ch.unit(12, "mm"), size=ch.unit(1.4, "mm"), gp=ch.gpar(col="#333333")),
        DKFZ_subtype=dkfz_subtype,
        TCGA_subtype=tcga_subtype,
        IDH1=idh1,
        H3F3A=h3f3a,
        TP53=tp53,
        chr7_gain=chr7_gain,
        chr10_loss=chr10_loss,
        CDKN2A_del=cdkn2a_del,
        EGFR_amp=egfr_amp,
        PDGFRA_amp=pdgfra_amp,
        col=col_map,
        simple_anno_size=ch.unit(3.2, "mm"),
        annotation_name_gp=ch.gpar(fontsize=8),
        annotation_name_side="left",
        show_legend={"TP53": False, "chr10_loss": False, "EGFR_amp": False, "PDGFRA_amp": False},
    )

    ht_meth = ch.Heatmap(
        mat_meth,
        name="Methylation",
        col=ch.colorRamp2([0.0, 0.5, 1.0], ["#2166AC", "#F7F7F7", "#B2182B"]),
        column_split=pd.Categorical(dkfz_subtype, categories=subtypes, ordered=True),
        cluster_column_slices=False,
        column_title="%s",
        column_title_gp=ch.gpar(fontsize=10, fontface="bold"),
        row_title="DNA Methylation\n(240 Probes)",
        row_title_gp=ch.gpar(fontsize=10, fontface="bold"),
        show_row_names=False,
        show_column_names=False,
        show_column_dend=False,
        show_row_dend=False,
        top_annotation=ha_clinical,
        use_raster=True,
        border=True,
        column_gap=ch.unit(1.5, "mm"),
    )
    ht_expr = ch.Heatmap(
        mat_expr,
        name="Expression",
        col=ch.colorRamp2([-2.0, 0.0, 2.0], ["#1B7837", "#F7F7F7", "#762A83"]),
        height=ch.unit(38, "mm"),
        row_title="Expression\n(100 Genes)",
        row_title_gp=ch.gpar(fontsize=10, fontface="bold"),
        show_row_names=False,
        show_column_names=False,
        show_row_dend=False,
        use_raster=True,
        border=True,
    )
    ch.draw(
        ht_meth / ht_expr,
        save=str(OUT_DIR / "r_readme_01_methylation_profile.png"),
        width=13.0,
        height=9.5,
        dpi=140,
        merge_legends=True,
        column_title="Glioblastoma DNA Methylation & Matched Expression Across 6 Subtypes (Sturm et al., 2012 Style)",
        column_title_gp=ch.gpar(fontsize=12, fontface="bold"),
    )
    plt.close("all")


def render_r_readme_02_meth_expr_corr() -> None:
    """2. Correlations between methylation, expression and other genomic features (Ch 14.4 / meth.rds)."""
    ch.ht_opt(kmeans_seed=123)
    ds = ch.load_dataset("meth")
    mat_meth = ds["meth"]
    mat_expr = ds["expr"]
    meta = ds["row_meta"]
    enhancer = ds["enhancer"]

    sample_type = ["Tumor"] * 10 + ["Normal"] * 10
    type_col = {"Tumor": "#E41A1C", "Normal": "#377EB8"}
    col_dend = ch.hclust(ch.dist(mat_meth.to_numpy().T))

    direction_df = pd.DataFrame({"Direction": meta["direction"].astype(str).to_numpy()})
    neg_log_p = -np.log10(np.clip(meta["cor_pvalue"].to_numpy(dtype=float), 1e-6, 1.0))
    pval_df = pd.DataFrame({"-log10(P)": neg_log_p})
    gtype_df = pd.DataFrame({"Gene Type": meta["gene_type"].astype(str).to_numpy()})
    anno_df = pd.DataFrame({"Genomic Anno": meta["anno"].astype(str).to_numpy()})

    ht_dir = ch.Heatmap(
        direction_df,
        name="Direction",
        col={"hyper": "#E41A1C", "hypo": "#377EB8"},
        width=ch.unit(5, "mm"),
        cluster_columns=False,
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        use_raster=True,
        border=True,
    )
    ht_meth = ch.Heatmap(
        mat_meth,
        name="Methylation",
        col=ch.colorRamp2([0.0, 0.5, 1.0], ["#377EB8", "white", "#E41A1C"]),
        cluster_columns=col_dend,
        column_split=2,
        row_split=meta["direction"].astype(str).tolist(),
        row_title="%s",
        row_title_gp=ch.gpar(fontsize=9.5, fontface="bold"),
        top_annotation=ch.HeatmapAnnotation(
            Type=sample_type,
            col={"Type": type_col},
            annotation_name_side="left",
            annotation_name_gp=ch.gpar(fontsize=8.5),
            simple_anno_size=ch.unit(4.0, "mm"),
        ),
        column_title="Methylation",
        column_title_gp=ch.gpar(fontsize=10.5, fontface="bold"),
        show_row_names=False,
        show_column_names=False,
        width=ch.unit(48, "mm"),
        use_raster=True,
        border=True,
    )
    ht_expr = ch.Heatmap(
        mat_expr,
        name="Expression",
        col=ch.colorRamp2([-2.0, 0.0, 2.0], ["#1B9E77", "white", "#D95F02"]),
        cluster_columns=col_dend,
        column_split=2,
        top_annotation=ch.HeatmapAnnotation(
            Type=sample_type,
            col={"Type": type_col},
            show_annotation_name=False,
            show_legend=False,
            simple_anno_size=ch.unit(4.0, "mm"),
        ),
        column_title="Expression",
        column_title_gp=ch.gpar(fontsize=10.5, fontface="bold"),
        show_row_names=False,
        show_column_names=False,
        width=ch.unit(48, "mm"),
        use_raster=True,
        border=True,
    )
    ht_p = ch.Heatmap(
        pval_df,
        name="-log10(P)",
        col=ch.colorRamp2([0.0, 2.0, 4.0], ["white", "#FEB24C", "#BD0026"]),
        cluster_columns=False,
        width=ch.unit(6, "mm"),
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        use_raster=True,
        border=True,
    )
    gtype_levels = sorted(meta["gene_type"].astype(str).unique())
    gtype_pal = ch.brewer_pal(max(len(gtype_levels), 3), "Set2")
    ht_gtype = ch.Heatmap(
        gtype_df,
        name="Gene Type",
        col=dict(zip(gtype_levels, gtype_pal[:len(gtype_levels)])),
        cluster_columns=False,
        width=ch.unit(6, "mm"),
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        use_raster=True,
        border=True,
    )
    ht_anno = ch.Heatmap(
        anno_df,
        name="Genomic Anno",
        col={"TSS": "#E41A1C", "intragenic": "#377EB8", "intergenic": "#BDBDBD"},
        cluster_columns=False,
        width=ch.unit(6, "mm"),
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        use_raster=True,
        border=True,
    )
    ra_dist = ch.rowAnnotation(
        Dist_kb=ch.anno_points(
            meta["dist"].to_numpy(dtype=float) / 1000.0,
            size=ch.unit(0.9, "mm"),
            gp=ch.gpar(col="#00000040"),
            width=ch.unit(18, "mm"),
            axis_param={"gp": ch.gpar(fontsize=7)},
        ),
        annotation_name_rot=0,
        annotation_name_gp=ch.gpar(fontsize=8),
    )
    ht_enh = ch.Heatmap(
        enhancer,
        name="Enhancer",
        col=ch.colorRamp2([0.0, 1.0], ["white", "#6A51A3"]),
        cluster_columns=False,
        width=ch.unit(14, "mm"),
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=7.5),
        column_title="Enhancer",
        column_title_gp=ch.gpar(fontsize=9.5, fontface="bold"),
        use_raster=True,
        border=True,
    )

    hl = ht_dir + ht_meth + ht_expr + ht_p + ht_gtype + ht_anno + ra_dist + ht_enh
    ch.draw(
        hl,
        main_heatmap="Methylation",
        merge_legends=True,
        column_title="Correlations Between DNA Methylation, Gene Expression & Genomic Features",
        column_title_gp=ch.gpar(fontsize=12.5, fontface="bold"),
        save=str(OUT_DIR / "r_readme_02_meth_expr_corr.png"),
        width=10.2,
        height=8.5,
        dpi=140,
    )
    plt.close("all")


def render_r_readme_03_scrnaseq_heterogeneity() -> None:
    """3. Visualize Cell Heterogeneity from Single Cell RNASeq (Ch 14.3)."""
    ch.ht_opt(kmeans_seed=123)
    rng = np.random.default_rng(77)

    n_genes, n_cells = 150, 75
    phases = ["G1", "S", "G2/M"]
    cell_phase = np.repeat(phases, n_cells // 3)
    gene_cluster = np.repeat([0, 1, 2], n_genes // 3)

    # Simulate single-cell log-expression with state programs + dropouts, then quantile-scale [10%, 90%]
    base_mean = rng.uniform(2.0, 8.8, size=n_genes)
    raw_expr = np.empty((n_genes, n_cells), dtype=float)
    for g in range(n_genes):
        gc = gene_cluster[g]
        for c in range(n_cells):
            cc = c // (n_cells // 3)
            shift = 1.9 if gc == cc else (-0.8 if (gc + 1) % 3 == cc else -0.4)
            raw_expr[g, c] = base_mean[g] + shift + rng.normal(0, 0.75)

    # 10th-90th percentile scaling per gene (as in Buettner et al. / Ch 14.3)
    q10 = np.quantile(raw_expr, 0.10, axis=1, keepdims=True)
    q90 = np.quantile(raw_expr, 0.90, axis=1, keepdims=True)
    denom = np.where((q90 - q10) > 1e-6, q90 - q10, 1.0)
    mat_scaled = (raw_expr - 0.5 * (q10 + q90)) / denom * 3.0
    mat_scaled = np.clip(mat_scaled, -2.0, 2.0)

    rnp_flag = ["RNP" if (gene_cluster[g] == 0 and rng.random() < 0.55) or rng.random() < 0.08 else "Other" for g in range(n_genes)]
    cc_flag = ["CC" if (gene_cluster[g] == 1 and rng.random() < 0.65) or (gene_cluster[g] == 2 and rng.random() < 0.55) or rng.random() < 0.06 else "Other" for g in range(n_genes)]

    corr_mat = np.corrcoef(mat_scaled)
    hc_genes = ch.hclust(ch.dist(corr_mat))

    # Curated signature gene labels linked by anno_mark
    gene_labels = [f"Gene_{i}" for i in range(n_genes)]
    markers = [
        (4, "Rpl13a"), (14, "Rps6"), (26, "Npm1"), (38, "Eef1a1"),
        (54, "Mcm2"), (68, "Pcna"), (81, "Cdk2"), (92, "E2f1"),
        (106, "Cdk1"), (118, "Ccnb1"), (131, "Plk1"), (143, "Aurkb"),
    ]
    for idx, nm in markers:
        gene_labels[idx] = nm

    df_scaled = pd.DataFrame(mat_scaled, index=gene_labels, columns=[f"C{j+1}" for j in range(n_cells)])
    df_corr = pd.DataFrame(corr_mat, index=gene_labels, columns=gene_labels)
    df_base = pd.DataFrame({"Base Mean": base_mean}, index=gene_labels)
    df_rnp = pd.DataFrame({"RNP": rnp_flag}, index=gene_labels)
    df_cc = pd.DataFrame({"Cell Cycle": cc_flag}, index=gene_labels)

    ht_expr = ch.Heatmap(
        df_scaled,
        name="Scaled Expr",
        col=ch.colorRamp2([-1.5, 0.0, 1.5], ["#2166AC", "white", "#B2182B"]),
        cluster_rows=hc_genes,
        row_split=3,
        column_km=3,
        row_title="Gene Cluster %s",
        row_title_gp=ch.gpar(fontsize=9.5),
        top_annotation=ch.HeatmapAnnotation(
            Phase=cell_phase,
            col={"Phase": {"G1": "#66C2A5", "S": "#FC8D62", "G2/M": "#8DA0CB"}},
            annotation_name_side="left",
            annotation_name_gp=ch.gpar(fontsize=8.5),
            simple_anno_size=ch.unit(4.0, "mm"),
        ),
        show_row_names=False,
        show_column_names=False,
        column_title="Scaled Expression (75 T-cells)",
        column_title_gp=ch.gpar(fontsize=10.5, fontface="bold"),
        width=ch.unit(60, "mm"),
        use_raster=True,
        border=True,
    )
    ht_base = ch.Heatmap(
        df_base,
        name="Base Mean",
        col=ch.colorRamp2([2.0, 5.5, 9.0], ["white", "#FEB24C", "#BD0026"]),
        cluster_columns=False,
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        width=ch.unit(6, "mm"),
        border=True,
    )
    ht_rnp = ch.Heatmap(
        df_rnp,
        name="RNP",
        col={"RNP": "#7570B3", "Other": "#F2F2F2"},
        cluster_columns=False,
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        width=ch.unit(5.5, "mm"),
        show_heatmap_legend=False,
        border=True,
    )
    ht_cc = ch.Heatmap(
        df_cc,
        name="Cell Cycle",
        col={"CC": "#1B9E77", "Other": "#F2F2F2"},
        cluster_columns=False,
        show_row_names=False,
        column_names_gp=ch.gpar(fontsize=8),
        width=ch.unit(5.5, "mm"),
        show_heatmap_legend=False,
        border=True,
    )
    ht_corr = ch.Heatmap(
        df_corr,
        name="Gene Corr",
        col=ch.colorRamp2([-1.0, 0.0, 1.0], ["#2166AC", "white", "#B2182B"]),
        cluster_columns=hc_genes,
        column_split=3,
        show_column_dend=False,
        show_row_names=False,
        show_column_names=False,
        column_title="Gene-Gene Correlation",
        column_title_gp=ch.gpar(fontsize=10.5, fontface="bold"),
        width=ch.unit(60, "mm"),
        use_raster=True,
        border=True,
    )
    ra_marks = ch.rowAnnotation(
        mark=ch.anno_mark(
            at=[idx for idx, _ in markers],
            labels=[nm for _, nm in markers],
            labels_gp=ch.gpar(fontsize=8.5, fontface="italic"),
            link_width=ch.unit(6, "mm"),
        ),
        show_annotation_name=False,
    )

    hl = ht_expr + ht_base + ht_rnp + ht_cc + ht_corr + ra_marks
    ch.draw(
        hl,
        main_heatmap="Scaled Expr",
        merge_legends=True,
        column_title="Single-Cell RNA-Seq Heterogeneity, Gene Sets & Correlation Structure",
        column_title_gp=ch.gpar(fontsize=12.5, fontface="bold"),
        save=str(OUT_DIR / "r_readme_03_scrnaseq_heterogeneity.png"),
        width=9.2,
        height=7.8,
        dpi=140,
    )
    plt.close("all")


def render_r_readme_04_enhanced_oncoprint() -> None:
    """4. Making Enhanced OncoPrint (TCGA Lung Adenocarcinoma + Clinical + Pathway Matrix)."""
    rng = np.random.default_rng(42)
    df = ch.load_dataset("tcga_lung_adenocarcinoma")
    n_samples = df.shape[1]

    # Simulated patient clinical annotations paired to TCGA samples
    age = np.clip(rng.normal(64, 9, size=n_samples), 38, 86)
    smoking = rng.choice(["Current", "Former", "Never"], size=n_samples, p=[0.30, 0.52, 0.18])
    stage = rng.choice(["Stage I", "Stage II", "Stage III", "Stage IV"], size=n_samples, p=[0.46, 0.28, 0.18, 0.08])

    alt_col = {"HOMDEL": "#377EB8", "AMP": "#E41A1C", "MUT": "#4DAF4A"}
    alter_fun = {
        "background": ch.alter_graphic("rect", fill="#E8E8E8"),
        "HOMDEL": ch.alter_graphic("rect", fill=alt_col["HOMDEL"]),
        "AMP": ch.alter_graphic("rect", fill=alt_col["AMP"]),
        "MUT": ch.alter_graphic("rect", height=0.38, fill=alt_col["MUT"]),
    }

    ha_top = ch.HeatmapAnnotation(
        cbar=ch.anno_oncoprint_barplot(height=ch.unit(14, "mm")),
        Age=ch.anno_points(
            age,
            height=ch.unit(13, "mm"),
            size=ch.unit(1.1, "mm"),
            gp=ch.gpar(col="#444444"),
            axis_param={"gp": ch.gpar(fontsize=7)},
        ),
        Smoking=smoking,
        Stage=stage,
        col={
            "Smoking": {"Current": "#E41A1C", "Former": "#FF7F00", "Never": "#4DAF4A"},
            "Stage": {"Stage I": "#DEEBF7", "Stage II": "#9ECAE1", "Stage III": "#3182BD", "Stage IV": "#08519C"},
        },
        annotation_name_side="left",
        annotation_name_gp=ch.gpar(fontsize=8.5),
        simple_anno_size=ch.unit(3.6, "mm"),
    )
    ha_right = ch.rowAnnotation(
        rbar=ch.anno_oncoprint_barplot(which="row", width=ch.unit(20, "mm"), axis_param={"side": "top", "labels_rot": 0}),
    )

    op = ch.oncoPrint(
        df,
        alter_fun=alter_fun,
        col=alt_col,
        top_annotation=ha_top,
        right_annotation=ha_right,
        remove_empty_columns=True,
        remove_empty_rows=True,
        column_title="Enhanced OncoPrint: TCGA Lung Adenocarcinoma (Ras-Raf-MEK-JNK)",
        column_title_gp=ch.gpar(fontsize=12.5, fontface="bold"),
        row_names_gp=ch.gpar(fontsize=8.5),
        pct_gp=ch.gpar(fontsize=8.5),
        heatmap_legend_param={
            "title": "Alterations",
            "at": ["HOMDEL", "AMP", "MUT"],
            "labels": ["Deep deletion", "Amplification", "Mutation"],
        },
    )

    # Biological pathway membership heatmap aligned with retained OncoPrint genes
    rtk_ras = {"KRAS", "HRAS", "BRAF", "RAF1", "DAB2", "RAB25"}
    mapk_erk = {"KRAS", "HRAS", "BRAF", "RAF1", "MAP3K1", "MAP2K1", "MAP2K2", "MAPK3"}
    jnk_p38 = {"MAP3K1", "MAP3K2", "MAP3K3", "MAP3K4", "MAP3K5", "MAP2K3", "MAP2K4", "MAPK4", "MAPK7", "MAPK8", "MAPK9", "MAPK12", "MAPK14"}
    genes = op.row_names_param["labels"]
    pw_df = pd.DataFrame(
        {
            "RTK/RAS": ["Active" if g in rtk_ras else "None" for g in genes],
            "MAPK/ERK": ["Active" if g in mapk_erk else "None" for g in genes],
            "JNK/p38": ["Active" if g in jnk_p38 else "None" for g in genes],
        },
        index=genes,
    )
    ht_pw = ch.Heatmap(
        pw_df,
        name="Pathway",
        col={"Active": "#333333", "None": "#F2F2F2"},
        rect_gp=ch.gpar(col="white", lwd=0.8),
        cluster_columns=False,
        width=ch.unit(15, "mm"),
        column_names_gp=ch.gpar(fontsize=8),
        show_row_names=True,
        row_names_gp=ch.gpar(fontsize=8),
        show_heatmap_legend=False,
        border=True,
    )

    ch.draw(
        op + ht_pw,
        main_heatmap=op.name,
        merge_legends=True,
        save=str(OUT_DIR / "r_readme_04_enhanced_oncoprint.png"),
        width=13.0,
        height=8.2,
        dpi=140,
    )
    plt.close("all")


def render_r_readme_05_upset() -> None:
    """5. Multi-Set UpSet Plot Replicating jokergoo/ComplexHeatmap README Example."""
    rng = np.random.default_rng(123)
    universe = [f"loc_{i}" for i in range(400)]
    sets = {
        "H3K27ac": rng.choice(universe, 145, replace=False),
        "H3K4me1": rng.choice(universe, 160, replace=False),
        "H3K4me3": rng.choice(universe, 110, replace=False),
        "ATAC-seq": rng.choice(universe, 175, replace=False),
        "CTCF": rng.choice(universe, 95, replace=False),
        "RNA-seq DE": rng.choice(universe, 85, replace=False),
    }
    cm = ch.make_comb_mat(sets)
    cm = cm[:, ch.comb_size(cm) >= 3]
    order_idx = np.argsort(-ch.comb_size(cm), kind="stable")
    deg = ch.comb_degree(cm)
    deg_palette = {1: "#333333", 2: "#377EB8", 3: "#4DAF4A", 4: "#984EA3", 5: "#E41A1C", 6: "#FF7F00"}

    up = ch.UpSet(
        cm,
        comb_order=order_idx,
        pt_size=ch.unit(2.6, "mm"),
        lwd=1.8,
        comb_col=[deg_palette.get(int(d), "#E41A1C") for d in deg],
        top_annotation=ch.upset_top_annotation(
            cm,
            add_numbers=True,
            numbers_gp=ch.gpar(fontsize=7),
            height=ch.unit(38, "mm"),
            gp=ch.gpar(fill=[deg_palette.get(int(d), "#E41A1C") for d in deg], col=None),
        ),
        right_annotation=ch.upset_right_annotation(
            cm,
            add_numbers=True,
            width=ch.unit(30, "mm"),
            gp=ch.gpar(fill="#4D4D4D", col=None),
        ),
        row_names_gp=ch.gpar(fontsize=10, fontface="bold"),
        column_title="Multi-Omics Chromatin & Expression Overlap (UpSet Plot)",
        column_title_gp=ch.gpar(fontsize=12.5, fontface="bold"),
    )
    up.draw(
        save=str(OUT_DIR / "r_readme_05_upset.png"),
        width=12.0,
        height=6.0,
        dpi=140,
        padding=ch.unit([8, 4, 4, 6], "mm"),
    )
    plt.close("all")


def render_r_readme_06_measles_3d() -> None:
    """6. 3D Heatmap of US Measles Cases (Heatmap3D)."""
    mat = ch.load_dataset("measles")
    # Select top 28 states by cumulative cases so state labels and 3D bars stay crisp
    top_states = mat.fillna(0).sum(axis=1).sort_values(ascending=False).head(28).index
    sub = mat.loc[top_states]

    years = [int(c) for c in sub.columns]
    year_labels = [str(y) if y % 10 == 0 else "" for y in years]

    h3 = ch.Heatmap3D(
        sub,
        col=ch.colorRamp2([0.0, 4000.0, 20000.0, 80000.0], ["#FFFFFF", "#FFEDA0", "#FEB24C", "#E31A1C"]),
        name="Cases",
        cluster_rows=False,
        cluster_columns=False,
        column_labels=year_labels,
        column_names_gp=ch.gpar(fontsize=8.5, fontface="bold"),
        row_names_gp=ch.gpar(fontsize=7.5),
        bar_angle=55.0,
        bar_max_length=ch.unit(12, "mm"),
        rect_gp=ch.gpar(col="#444444", lwd=0.3),
        top_annotation=ch.HeatmapAnnotation(
            _spacer=ch.anno_empty(border=False, height=ch.unit(12, "mm")),
            show_annotation_name=False,
        ),
        column_title="3D Heatmap of US Measles Cases (1930–2001, Top 28 States)",
        column_title_gp=ch.gpar(fontsize=12.5, fontface="bold"),
    )
    h3.draw(
        save=str(OUT_DIR / "r_readme_06_measles_3d.png"),
        width=12.5,
        height=7.5,
        dpi=140,
        padding=ch.unit([6, 4, 6, 4], "mm"),
    )
    plt.close("all")


def main() -> None:
    for fn in (
        render_r_readme_01_methylation_profile,
        render_r_readme_02_meth_expr_corr,
        render_r_readme_03_scrnaseq_heterogeneity,
        render_r_readme_04_enhanced_oncoprint,
        render_r_readme_05_upset,
        render_r_readme_06_measles_3d,
    ):
        fn()
        ch.ht_opt(RESET=True)
        print(f"Rendered {fn.__name__}")


if __name__ == "__main__":
    main()
