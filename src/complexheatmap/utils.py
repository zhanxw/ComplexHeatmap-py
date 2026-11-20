"""Utility functions, R-compatible helpers, and bundled dataset loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .annotations import HeatmapAnnotation, c_HeatmapAnnotation
from .colors import ColorMapping, c_ColorMapping
from .units import Unit, _resolve_vp, is_unit, unit_c


_DATA_DIR = Path(__file__).resolve().parent / "data"


def pindex(m: Any, i: Any, j: Any) -> np.ndarray:
    """Pairwise element indexing `m[i[k], j[k]]` for 2D or 3D arrays (`pindex.Rd`)."""
    arr = np.asarray(m)
    ii = np.asarray(i, dtype=int)
    jj = np.asarray(j, dtype=int)
    if len(ii) == 1 and len(jj) > 1:
        ii = np.repeat(ii, len(jj))
    elif len(jj) == 1 and len(ii) > 1:
        jj = np.repeat(jj, len(ii))
    if len(ii) != len(jj):
        raise ValueError("`i` and `j` must have equal length or length 1.")
    if arr.ndim == 2:
        return arr[ii, jj]
    if arr.ndim == 3:
        return arr[ii, jj, :]
    raise ValueError(f"`pindex` expects 2D or 3D array, got {arr.ndim}D.")


def restore_matrix(j: Any, i: Any, x: Any, y: Any) -> np.ndarray:
    """Reconstruct 2D slice position index matrix inside `layer_fun` (`restore_matrix.Rd`)."""
    vp = _resolve_vp(None)
    x_vals = x.to_mm("x", vp, pos_mode=True) if isinstance(x, Unit) else np.asarray(x, dtype=float)
    y_vals = y.to_mm("y", vp, pos_mode=True) if isinstance(y, Unit) else np.asarray(y, dtype=float)
    # Round slightly to group identical coordinates cleanly
    xu = np.sort(np.unique(np.round(x_vals, 6)))
    yu = np.sort(np.unique(np.round(y_vals, 6)))[::-1]  # top-to-bottom descending y
    x_map = {v: idx for idx, v in enumerate(xu)}
    y_map = {v: idx for idx, v in enumerate(yu)}
    out = np.full((len(yu), len(xu)), -1, dtype=int)
    for pos in range(len(x_vals)):
        r = y_map[float(np.round(y_vals[pos], 6))]
        c = x_map[float(np.round(x_vals[pos], 6))]
        out[r, c] = pos
    return out


def c(*args: Any, **kwargs: Any) -> Any:
    """Polymorphic R-like concatenation for Units, ColorMappings, HeatmapAnnotations, or sequences."""
    if not args:
        return []
    first = args[0]
    if isinstance(first, HeatmapAnnotation) or any(isinstance(a, HeatmapAnnotation) for a in args if a is not None):
        return c_HeatmapAnnotation(*[a for a in args if a is not None], **kwargs)
    if isinstance(first, ColorMapping) and all(isinstance(a, ColorMapping) for a in args):
        return c_ColorMapping(*args, **kwargs)
    if is_unit(first) or any(is_unit(a) for a in args):
        return unit_c(*args)
    out: list[Any] = []
    for a in args:
        if a is None:
            continue
        if isinstance(a, (str, bytes)):
            out.append(a)
        elif isinstance(a, np.ndarray):
            out.extend(a.ravel().tolist())
        elif isinstance(a, (list, tuple, pd.Series, pd.Index)):
            out.extend(list(a))
        else:
            out.append(a)
    return out


def t(x: Any) -> Any:
    """Transpose matrix, DataFrame, or CombMat (`UpSet` orientation switch)."""
    if hasattr(x, "t") and callable(x.t):
        return x.t()
    if isinstance(x, pd.DataFrame):
        return x.T
    return np.asarray(x).T


def names(x: Any) -> Any:
    if hasattr(x, "names"):
        nm = x.names
        return nm() if callable(nm) else nm
    if isinstance(x, pd.DataFrame):
        return list(x.columns)
    if isinstance(x, pd.Series):
        return list(x.index)
    if isinstance(x, dict):
        return list(x.keys())
    return None


def set_names(x: Any, value: Sequence[str]) -> Any:
    cp = x.copy_all() if hasattr(x, "copy_all") else x
    if hasattr(cp, "names"):
        cp.names = list(value)
    return cp


def width(x: Any) -> Any:
    if hasattr(x, "width"):
        return x.width
    return None


def height(x: Any) -> Any:
    if hasattr(x, "height"):
        return x.height
    return None


def size(x: Any) -> Any:
    if hasattr(x, "size"):
        return x.size
    return None


def nrow(x: Any) -> int:
    if hasattr(x, "nrow"):
        v = x.nrow
        return int(v() if callable(v) else v)
    return int(np.asarray(x).shape[0])


def ncol(x: Any) -> int:
    if hasattr(x, "ncol"):
        v = x.ncol
        return int(v() if callable(v) else v)
    arr = np.asarray(x)
    return int(arr.shape[1]) if arr.ndim >= 2 else 1


def dim(x: Any) -> tuple[int, ...]:
    if hasattr(x, "shape"):
        return tuple(x.shape)
    return tuple(np.asarray(x).shape)


def summary(x: Any) -> Any:
    if hasattr(x, "summary") and callable(x.summary):
        return x.summary()
    return repr(x)


def load_dataset(name: str) -> Any:
    """Load bundled reference datasets exported from ComplexHeatmap `inst/extdata/`."""
    nm = name.strip().lower()
    if nm == "measles":
        df = pd.read_csv(_DATA_DIR / "measles.csv", index_col=0)
        return df
    if nm == "rand_mat":
        return pd.read_csv(_DATA_DIR / "rand_mat.csv", index_col=0)
    if nm in ("tcga", "tcga_lung_adenocarcinoma"):
        return pd.read_csv(_DATA_DIR / "tcga_lung_adenocarcinoma.csv", index_col=0, keep_default_na=False)
    if nm in ("gene_expression", "expr"):
        return pd.read_csv(_DATA_DIR / "gene_expression.csv", index_col=0)
    if nm == "gr":
        return pd.read_csv(_DATA_DIR / "gr.csv")
    if nm == "sample_order":
        txt = (_DATA_DIR / "sample_order.txt").read_text().strip()
        return [line.strip() for line in txt.splitlines() if line.strip()]
    if nm == "meth":
        d = _DATA_DIR / "meth"
        return {p.stem: pd.read_csv(p, index_col=0) for p in sorted(d.glob("*.csv"))}
    if nm == "dmr_summary":
        d = _DATA_DIR / "dmr_summary"
        return {p.stem: pd.read_csv(p, index_col=0) for p in sorted(d.glob("*.csv"))}
    if nm == "random_meth_expr_data":
        d = _DATA_DIR / "random_meth_expr_data"
        return {p.stem: pd.read_csv(p, index_col=0) for p in sorted(d.glob("*.csv"))}
    raise KeyError(
        f"Unknown bundled dataset {name!r}. Available: 'measles', 'rand_mat', 'tcga_lung_adenocarcinoma', "
        "'gene_expression', 'gr', 'sample_order', 'meth', 'dmr_summary', 'random_meth_expr_data'."
    )
