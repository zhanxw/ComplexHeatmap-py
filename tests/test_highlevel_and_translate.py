"""Tests for high-level plots (density/frequency/3D), genomic binning, datasets, and pheatmap/heatmap translations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import complexheatmap as ch
from complexheatmap import (
    GRanges,
    Heatmap3D,
    IRanges,
    bar3D,
    bin_genome,
    compare_heatmap,
    compare_heatmap_2,
    compare_pheatmap,
    densityHeatmap,
    draw,
    findOverlaps,
    frequencyHeatmap,
    generateRandomBed,
    grid_newpage,
    heatmap,
    heatmap_2,
    ks_dist,
    load_dataset,
    normalize_genomic_signals_to_bins,
    pheatmap,
)


def test_density_and_frequency_heatmaps():
    rng = np.random.default_rng(10)
    mat = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"C{i}" for i in range(5)])
    d = ks_dist(mat)
    assert d.n == 5

    dh = densityHeatmap(mat, cluster_columns=True, title="Density")
    draw(dh)

    fh = frequencyHeatmap(mat, breaks=8, stat="proportion")
    draw(fh)

    fh3d = frequencyHeatmap(mat, breaks=6, use_3d=True)
    draw(fh3d)


def test_bar3d_and_heatmap3d():
    grid_newpage()
    bar3D([0.3, 0.7], 0.5, 0.15, 0.15, 0.25, fill=[2, 4])

    rng = np.random.default_rng(3)
    m = rng.uniform(0.0, 10.0, size=(4, 5))
    h3 = Heatmap3D(m, name="H3D")
    draw(h3)

    with pytest.raises(ValueError):
        Heatmap3D(np.array([[-1.0, 2.0], [3.0, 4.0]]))


def test_genomic_binning_and_normalization():
    bins = bin_genome("hg19", bin_size=50_000_000, chromosomes=["chr1", "chr2"])
    assert len(bins) >= 10
    bed = generateRandomBed(nr=40, nc=2, species="hg19", seed=1)
    gr = GRanges(bed["chr"].tolist(), IRanges(bed["start"].to_numpy(), bed["end"].to_numpy()), s1=bed["value1"].to_numpy(), s2=bed["value2"].to_numpy())
    qh, sh = findOverlaps(bins, gr)
    assert len(qh) == len(sh)

    res_gr = normalize_genomic_signals_to_bins(gr, ["s1", "s2"], method="weighted", bins=bins)
    assert len(res_gr) == len(bins)
    assert list(res_gr.mcols.columns) == ["s1", "s2"]


def test_translations_and_bundled_datasets():
    rng = np.random.default_rng(2)
    mat = pd.DataFrame(rng.standard_normal((8, 6)), index=[f"R{i}" for i in range(8)], columns=[f"C{j}" for j in range(6)])
    ann_c = pd.DataFrame({"Group": ["G1", "G2"] * 3}, index=mat.columns)
    ann_r = pd.DataFrame({"Type": ["T1", "T2"] * 4}, index=mat.index)

    ph = pheatmap(
        mat,
        scale="row",
        cutree_rows=2,
        annotation_col=ann_c,
        annotation_row=ann_r,
        display_numbers=True,
        main="Pheatmap Title",
    )
    draw(ph)

    draw(heatmap(mat.to_numpy(), RowSideColors=["red", "blue"] * 4, main="Base heatmap"))
    draw(heatmap_2(mat.to_numpy(), dendrogram="row", main="heatmap.2"))
    draw(compare_pheatmap(mat))
    draw(compare_heatmap(mat))
    draw(compare_heatmap_2(mat))

    # Bundled datasets check
    assert not load_dataset("measles").empty
    assert not load_dataset("rand_mat").empty
    assert not load_dataset("tcga_lung_adenocarcinoma").empty
    assert not load_dataset("gene_expression").empty
    assert not load_dataset("gr").empty
    assert len(load_dataset("sample_order")) > 0
    assert "meth" in load_dataset("meth")
    assert "mat_meth" in load_dataset("random_meth_expr_data")
    assert "mat_enrich_gf" in load_dataset("dmr_summary")
