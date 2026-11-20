"""Tests for CombMat/UpSet (sets, matrices, GRanges, all 3 modes) and OncoPrint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import complexheatmap as ch
from complexheatmap import (
    GRanges,
    IRanges,
    UpSet,
    alter_graphic,
    comb_degree,
    comb_name,
    comb_size,
    complement_size,
    draw,
    extract_comb,
    full_comb_code,
    list_to_matrix,
    make_comb_mat,
    normalize_comb_mat,
    oncoPrint,
    order_comb_mat,
    set_name,
    set_size,
    t,
    unify_mat_list,
    upset_left_annotation,
    upset_right_annotation,
    upset_top_annotation,
)


def test_comb_mat_modes_sets_and_matrix():
    lt = {
        "A": ["e1", "e2", "e3", "e4"],
        "B": ["e2", "e3", "e5"],
        "C": ["e3", "e6"],
    }
    bm = list_to_matrix(lt, universal_set=["e1", "e2", "e3", "e4", "e5", "e6", "e7"])
    assert bm.shape == (7, 3)

    m_dist = make_comb_mat(lt, mode="distinct")
    assert list(set_name(m_dist)) == ["A", "B", "C"]
    assert list(set_size(m_dist)) == [4.0, 3.0, 2.0]
    assert sorted(extract_comb(m_dist, "111")) == ["e3"]
    assert sorted(extract_comb(m_dist, "110")) == ["e2"]

    m_int = make_comb_mat(lt, mode="intersect")
    c_map = dict(zip(comb_name(m_int), comb_size(m_int)))
    assert c_map["110"] == 2.0  # e2 and e3 in A & B
    assert c_map["111"] == 1.0

    m_uni = make_comb_mat(lt, mode="union")
    u_map = dict(zip(comb_name(m_uni), comb_size(m_uni)))
    assert u_map["111"] == 6.0

    # Complement size with universal set
    m_comp = make_comb_mat(lt, universal_set=["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"])
    assert complement_size(m_comp) == 2.0

    # Transpose and subsetting and normalization
    mt = t(m_dist)
    assert mt.shape == (m_dist.ncol, m_dist.nrow)
    draw(UpSet(mt))
    draw(UpSet(m_dist, left_annotation=upset_left_annotation(m_dist), right_annotation=None))

    m_ord = order_comb_mat(m_dist)
    assert comb_size(m_ord)[0] >= comb_size(m_ord)[-1]

    m1 = make_comb_mat({"A": ["a", "b"], "B": ["b", "c"]})
    m2 = make_comb_mat({"B": ["b", "c", "d"], "C": ["d", "e"]})
    normed = normalize_comb_mat(m1, m2, full_comb_sets=True)
    assert set_name(normed[0]) == set_name(normed[1]) == ["A", "B", "C"]
    assert len(full_comb_code(3, complement=True)) == 8


def test_comb_mat_genomic_ranges():
    gr1 = GRanges(["chr1", "chr1"], IRanges([1, 20], [10, 30]))   # width 10 + 11 = 21
    gr2 = GRanges(["chr1"], IRanges([5], [25]))                  # width 21; overlap [5,10]=6 + [20,25]=6 -> 12
    m = make_comb_mat({"R1": gr1, "R2": gr2}, mode="distinct")
    cmap = dict(zip(comb_name(m), comb_size(m)))
    assert cmap["11"] == pytest.approx(12.0)
    assert cmap["10"] == pytest.approx(9.0)
    assert cmap["01"] == pytest.approx(9.0)
    assert isinstance(extract_comb(m, "11"), GRanges)
    draw(UpSet(m))


def test_oncoprint_variants_and_helpers():
    # Character matrix input
    df = pd.DataFrame(
        {"S1": ["MUT;AMP", "", "HOMDEL", "MUT"], "S2": ["AMP", "MUT", "", ""], "S3": ["", "HOMDEL;MUT", "AMP", "MUT"]},
        index=["TP53", "KRAS", "EGFR", "MYC"],
    )
    op = oncoPrint(df, remove_empty_rows=True, remove_empty_columns=True)
    draw(op)
    assert op._oncoprint_env["arr"].shape[:2] == (4, 3)

    # List/dict of binary matrices via unify_mat_list
    m_mut = pd.DataFrame([[1, 0], [0, 1]], index=["G1", "G2"], columns=["S1", "S2"])
    m_amp = pd.DataFrame([[0, 1], [1, 0]], index=["G2", "G3"], columns=["S2", "S3"])
    unified = unify_mat_list({"MUT": m_mut, "AMP": m_amp}, default=0)
    assert unified["MUT"].shape == (3, 3)
    op2 = oncoPrint(unified)
    draw(op2)

    af = {"MUT": alter_graphic("rect", fill="forestgreen"), "AMP": alter_graphic("point", fill="red", pch=17)}
    ch.test_alter_fun(af)
