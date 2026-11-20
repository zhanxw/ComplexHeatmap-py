"""Lightweight genomic interval classes (`IRanges`, `GRanges`) and binning helpers."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


_CHROM_SIZES: dict[str, dict[str, int]] = {
    "hg19": {
        "chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
        "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
        "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
        "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
        "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
        "chr21": 48129895, "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
    },
    "hg38": {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
        "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
        "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
        "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
    },
    "mm10": {
        "chr1": 195471971, "chr2": 182113224, "chr3": 160039680, "chr4": 156508116,
        "chr5": 151834684, "chr6": 149736546, "chr7": 145441459, "chr8": 129401213,
        "chr9": 124595110, "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
        "chr13": 120421639, "chr14": 124902244, "chr15": 104043685, "chr16": 98207768,
        "chr17": 94987271, "chr18": 90702639, "chr19": 61431566, "chrX": 171031299, "chrY": 91744698,
    },
    "mm9": {
        "chr1": 197195432, "chr2": 181748087, "chr3": 159599783, "chr4": 155630120,
        "chr5": 152537259, "chr6": 149517037, "chr7": 152524553, "chr8": 131738871,
        "chr9": 124076172, "chr10": 129993255, "chr11": 121843856, "chr12": 121257530,
        "chr13": 120284312, "chr14": 125194864, "chr15": 103494974, "chr16": 98319150,
        "chr17": 95272651, "chr18": 90772031, "chr19": 61342430, "chrX": 166650296, "chrY": 15902555,
    },
}


class IRanges:
    """1-based closed integer interval collection analogous to Bioconductor `IRanges`."""

    def __init__(self, start: Sequence[int], end: Sequence[int] | None = None, width: Sequence[int] | None = None, names: Sequence[str] | None = None) -> None:
        self.start = np.asarray(start, dtype=np.int64)
        if end is not None:
            self.end = np.asarray(end, dtype=np.int64)
        elif width is not None:
            self.end = self.start + np.asarray(width, dtype=np.int64) - 1
        else:
            raise ValueError("Either `end` or `width` must be supplied to `IRanges`.")
        self.names = list(names) if names is not None else None

    @property
    def width(self) -> np.ndarray:
        return np.maximum(0, self.end - self.start + 1)

    def __len__(self) -> int:
        return len(self.start)

    def __repr__(self) -> str:
        return f"IRanges(n={len(self)})"


class GRanges:
    """Genomic interval collection analogous to Bioconductor `GenomicRanges::GRanges`."""

    def __init__(
        self,
        seqnames: Sequence[str],
        ranges: IRanges | None = None,
        *,
        start: Sequence[int] | None = None,
        end: Sequence[int] | None = None,
        strand: Sequence[str] | str = "*",
        **mcols: Any,
    ) -> None:
        self.seqnames = [str(s) for s in seqnames]
        n = len(self.seqnames)
        if ranges is not None:
            self.ranges = ranges
        elif start is not None and end is not None:
            self.ranges = IRanges(start=start, end=end)
        else:
            raise ValueError("Provide either `ranges=IRanges(...)` or both `start` and `end`.")
        if len(self.ranges) != n:
            raise ValueError("Length of `seqnames` must match `ranges`.")
        self.strand = [strand] * n if isinstance(strand, str) else list(strand)
        self.mcols = pd.DataFrame({k: np.asarray(v) for k, v in mcols.items()}) if mcols else pd.DataFrame(index=range(n))

    @property
    def start(self) -> np.ndarray:
        return self.ranges.start

    @property
    def end(self) -> np.ndarray:
        return self.ranges.end

    @property
    def width(self) -> np.ndarray:
        return self.ranges.width

    def __len__(self) -> int:
        return len(self.seqnames)

    def __getitem__(self, idx: Any) -> "GRanges":
        if isinstance(idx, (int, np.integer)):
            sl = [int(idx)]
        elif isinstance(idx, slice):
            sl = list(range(*idx.indices(len(self))))
        else:
            arr_i = np.asarray(idx)
            sl = np.where(arr_i)[0].tolist() if arr_i.dtype == bool else arr_i.tolist()
        new_mcols = self.mcols.iloc[sl].reset_index(drop=True) if len(self.mcols.columns) else pd.DataFrame(index=range(len(sl)))
        return GRanges(
            [self.seqnames[i] for i in sl],
            IRanges(self.start[sl], self.end[sl]),
            strand=[self.strand[i] for i in sl],
            **{c: new_mcols[c].to_numpy() for c in new_mcols.columns},
        )

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"seqnames": self.seqnames, "start": self.start, "end": self.end, "strand": self.strand})
        for col in self.mcols.columns:
            df[col] = self.mcols[col].to_numpy()
        return df

    def __repr__(self) -> str:
        return f"GRanges(n={len(self)}, mcols={list(self.mcols.columns)})"


def _overlap_intervals(
    q_starts: np.ndarray,
    q_ends: np.ndarray,
    s_starts: np.ndarray,
    s_ends: np.ndarray,
    q_orig: np.ndarray | None = None,
    s_orig: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find overlapping index pairs using start-sort + prefix-max-end sweep pruning."""
    nq = len(q_starts)
    ns = len(s_starts)
    if nq == 0 or ns == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    order_s = np.argsort(s_starts, kind="mergesort")
    ss = s_starts[order_s]
    se = s_ends[order_s]
    s_mapped = s_orig[order_s] if s_orig is not None else order_s
    prefix_max_e = np.maximum.accumulate(se)

    q_chunks: list[np.ndarray] = []
    s_chunks: list[np.ndarray] = []
    for qi in range(nq):
        qs = int(q_starts[qi])
        qe = int(q_ends[qi])
        hi = int(np.searchsorted(ss, qe, side="right"))
        if hi == 0 or prefix_max_e[hi - 1] < qs:
            continue
        lo = int(np.searchsorted(prefix_max_e[:hi], qs, side="left"))
        rel = np.flatnonzero(se[lo:hi] >= qs) + lo
        if len(rel) > 0:
            q_val = int(q_orig[qi]) if q_orig is not None else qi
            q_chunks.append(np.full(len(rel), q_val, dtype=int))
            s_chunks.append(s_mapped[rel])

    if not q_chunks:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(q_chunks), np.concatenate(s_chunks)


def findOverlaps(query: GRanges | IRanges, subject: GRanges | IRanges) -> tuple[np.ndarray, np.ndarray]:
    """Find overlapping interval index pairs `(queryHits, subjectHits)`."""
    if isinstance(query, IRanges) and isinstance(subject, IRanges):
        return _overlap_intervals(query.start, query.end, subject.start, subject.end)

    if isinstance(query, GRanges) and isinstance(subject, GRanges):
        by_chr_s = _group_by_chr(subject.seqnames)
        q_all: list[np.ndarray] = []
        s_all: list[np.ndarray] = []
        for ch, q_list in _group_by_chr(query.seqnames).items():
            if ch not in by_chr_s:
                continue
            q_arr = np.asarray(q_list, dtype=int)
            s_arr = np.asarray(by_chr_s[ch], dtype=int)
            qh, sh = _overlap_intervals(
                query.start[q_arr], query.end[q_arr],
                subject.start[s_arr], subject.end[s_arr],
                q_orig=q_arr, s_orig=s_arr,
            )
            if len(qh) > 0:
                q_all.append(qh)
                s_all.append(sh)
        if not q_all:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        return np.concatenate(q_all), np.concatenate(s_all)

    raise TypeError("`query` and `subject` must both be GRanges or both be IRanges.")


def _group_by_chr(seqnames: Sequence[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, ch in enumerate(seqnames):
        out.setdefault(ch, []).append(i)
    return out


def generateRandomBed(
    nr: int = 10000,
    nc: int = 10,
    fun: Callable[[int], np.ndarray] | None = None,
    species: str = "hg19",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate random genomic intervals with random numeric value columns (`generateRandomBed`)."""
    rng = np.random.default_rng(seed)
    chrom_sizes = _CHROM_SIZES.get(species, _CHROM_SIZES["hg19"])
    chrs = list(chrom_sizes.keys())
    lengths = np.array([chrom_sizes[c] for c in chrs], dtype=float)
    probs = lengths / np.sum(lengths)
    chosen = rng.choice(len(chrs), size=nr, p=probs)
    seqnames = [chrs[i] for i in chosen]
    starts = np.array([int(rng.integers(1, max(2, chrom_sizes[c] - 1000))) for c in seqnames], dtype=np.int64)
    widths = rng.integers(100, 10000, size=nr, dtype=np.int64)
    ends = np.minimum(starts + widths, np.array([chrom_sizes[c] for c in seqnames], dtype=np.int64))
    df = pd.DataFrame({"chr": seqnames, "start": starts, "end": ends})
    if fun is None:
        fun = lambda k: rng.standard_normal(k)
    for ci in range(nc):
        df[f"value{ci + 1}"] = fun(nr)
    return df


def bin_genome(
    species: str = "hg19",
    bin_size: int = 1_000_000,
    bins: int | None = None,
    chromosomes: Sequence[str] | None = None,
    chr_len: dict[str, int] | None = None,
) -> GRanges:
    """Partition a reference genome into fixed-size chromosome bins (`bin_genome.Rd`)."""
    if chr_len is not None:
        sizes = dict(chr_len)
    elif species in _CHROM_SIZES:
        sizes = dict(_CHROM_SIZES[species])
    else:
        raise KeyError(f"Unknown species {species!r}; supported: {list(_CHROM_SIZES.keys())} or pass `chr_len`.")

    if chromosomes is not None:
        sizes = {k: sizes[k] for k in chromosomes if k in sizes}

    seq_out: list[str] = []
    st_out: list[int] = []
    en_out: list[int] = []

    if bins is not None:
        tot_len = sum(sizes.values())
        eff_bin_size = max(1, int(tot_len // max(1, int(bins))))
    else:
        eff_bin_size = max(1, int(bin_size))

    for ch, clen in sizes.items():
        pos = 1
        while pos <= clen:
            en = min(clen, pos + eff_bin_size - 1)
            seq_out.append(ch)
            st_out.append(pos)
            en_out.append(en)
            pos = en + 1

    return GRanges(seq_out, IRanges(st_out, en_out))


def normalize_genomic_signals_to_bins(
    gr: GRanges | pd.DataFrame,
    value_column: str | Sequence[str] | Sequence[int] | None = None,
    method: str = "weighted",
    empty_value: float = np.nan,
    bins: GRanges | None = None,
    species: str = "hg19",
    bin_size: int = 1_000_000,
) -> tuple[GRanges, np.ndarray] | GRanges:
    """Summarize genomic signals over uniform genomic bins (`normalize_genomic_signals_to_bins.Rd`)."""
    method = str(method).lower()
    if method not in ("weighted", "w0", "absolute", "number"):
        raise ValueError("`method` must be 'weighted', 'w0', 'absolute', or 'number'.")

    if isinstance(gr, pd.DataFrame):
        cols = list(gr.columns)
        ch_col = cols[0]
        st_col = cols[1]
        en_col = cols[2]
        mcols = {c: gr[c].to_numpy() for c in cols[3:]}
        gr_obj = GRanges(gr[ch_col].tolist(), IRanges(gr[st_col].to_numpy(), gr[en_col].to_numpy()), **mcols)
    else:
        gr_obj = gr

    target_bins = bins if bins is not None else bin_genome(species=species, bin_size=bin_size)

    if value_column is None:
        val_cols = list(gr_obj.mcols.columns)
    elif isinstance(value_column, str):
        val_cols = [value_column]
    elif isinstance(value_column, (list, tuple)):
        if val_is_int := (value_column and isinstance(value_column[0], (int, np.integer))):
            val_cols = [list(gr_obj.mcols.columns)[int(i)] for i in value_column]
        else:
            val_cols = [str(v) for v in value_column]
    else:
        val_cols = [list(gr_obj.mcols.columns)[int(value_column)]]

    n_bins = len(target_bins)
    n_vals = max(1, len(val_cols))
    out_mat = np.full((n_bins, n_vals), float(empty_value), dtype=float)

    qh, sh = findOverlaps(target_bins, gr_obj)
    if len(qh) > 0:
        ov_start = np.maximum(target_bins.start[qh], gr_obj.start[sh])
        ov_end = np.minimum(target_bins.end[qh], gr_obj.end[sh])
        ov_w = np.maximum(0, ov_end - ov_start + 1).astype(float)
        bin_w = target_bins.width.astype(float)

        if not val_cols:
            # Count or fraction covered
            for b in np.unique(qh):
                mask = qh == b
                if method == "number":
                    out_mat[b, 0] = float(np.sum(mask))
                else:
                    out_mat[b, 0] = float(np.sum(ov_w[mask])) / bin_w[b]
        else:
            val_arr = gr_obj.mcols[val_cols].to_numpy(dtype=float)
            # Group by bin id
            order_b = np.argsort(qh)
            qh_s = qh[order_b]
            sh_s = sh[order_b]
            ow_s = ov_w[order_b]
            uniq_b, starts_b = np.unique(qh_s, return_index=True)
            ends_b = np.append(starts_b[1:], len(qh_s))
            for idx_u, b in enumerate(uniq_b):
                s0, s1 = int(starts_b[idx_u]), int(ends_b[idx_u])
                s_hits = sh_s[s0:s1]
                w_hits = ow_s[s0:s1]
                vals_b = val_arr[s_hits, :]
                if method == "number":
                    out_mat[b, :] = float(len(s_hits))
                elif method == "absolute":
                    out_mat[b, :] = np.nanmean(vals_b, axis=0)
                elif method == "w0":
                    out_mat[b, :] = np.nansum(vals_b * w_hits[:, None], axis=0) / bin_w[b]
                else:  # weighted
                    tot_w = float(np.sum(w_hits))
                    if tot_w > 0:
                        out_mat[b, :] = np.nansum(vals_b * w_hits[:, None], axis=0) / tot_w

    res_df = target_bins.to_dataframe()
    if val_cols:
        for ci, cn in enumerate(val_cols):
            res_df[cn] = out_mat[:, ci]
    else:
        res_df["value"] = out_mat[:, 0]
    res_gr = GRanges(
        res_df["seqnames"].tolist(),
        IRanges(res_df["start"].to_numpy(), res_df["end"].to_numpy()),
        strand=res_df["strand"].tolist(),
        **{c: res_df[c].to_numpy() for c in res_df.columns[4:]},
    )
    return res_gr
