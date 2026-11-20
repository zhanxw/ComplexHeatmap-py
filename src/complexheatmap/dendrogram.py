"""Distance metrics, hierarchical clustering, Dendrogram trees, and grid.dendrogram."""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage as scipy_linkage
from scipy.spatial.distance import pdist as scipy_pdist, squareform
from scipy.stats import rankdata

from .colors import brewer_pal, to_hex_rgba
from .grid import (
    GTree,
    Viewport,
    current_device,
    grid_newpage,
    grid_rect,
    grid_xaxis,
    grid_yaxis,
    pointsGrob,
    popViewport,
    pushViewport,
    segmentsGrob,
)
from .units import Unit, check_gp, gpar, unit, unit_c


class DistMatrix:
    """Condensed distance object mirroring R `stats::dist`."""

    def __init__(self, condensed: np.ndarray, n: int, labels: list[str] | None = None, method: str = "euclidean") -> None:
        self.condensed = np.asarray(condensed, dtype=float)
        self.n = int(n)
        self.labels = list(labels) if labels is not None else None
        self.method = method

    def to_square(self) -> np.ndarray:
        return squareform(self.condensed)

    def __sub__(self, other: "DistMatrix | np.ndarray") -> np.ndarray:
        if isinstance(other, DistMatrix):
            return self.condensed - other.condensed
        return self.condensed - np.asarray(other)

    def __len__(self) -> int:
        return len(self.condensed)

    def __repr__(self) -> str:
        return f"DistMatrix(n={self.n}, method={self.method!r})"


def as_dist(mat: Any, labels: list[str] | None = None, method: str = "custom") -> DistMatrix:
    if isinstance(mat, DistMatrix):
        return mat
    arr = np.asarray(mat, dtype=float)
    if arr.ndim == 1:
        n = int(round((1 + np.sqrt(1 + 8 * len(arr))) / 2))
        return DistMatrix(arr, n=n, labels=labels, method=method)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        n = arr.shape[0]
        # Read lower triangle column-wise/row-wise symmetrically via upper triangle indices
        arr_sym = arr.copy()
        np.fill_diagonal(arr_sym, 0.0)
        # If upper triangle is zero/NaN (lower-triangle matrix like R), mirror lower to upper
        iu = np.triu_indices(n, k=1)
        il = (iu[1], iu[0])
        if np.all(np.isnan(arr_sym[iu])) or np.all(arr_sym[iu] == 0):
            arr_sym[iu] = arr_sym[il]
        condensed = arr_sym[iu]
        return DistMatrix(condensed, n=n, labels=labels, method=method)
    raise ValueError("Cannot convert input to DistMatrix.")


def dist(x: Any, method: str = "euclidean") -> DistMatrix:
    """Compute row-wise pairwise distances analogous to R `stats::dist` plus correlation distances."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n < 2:
        return DistMatrix(np.array([], dtype=float), n=n, method=method)

    m = method.lower()
    if m in ("euclidean", "maximum", "chebyshev", "manhattan", "cityblock", "canberra", "binary", "jaccard", "minkowski"):
        scipy_m = {
            "maximum": "chebyshev",
            "manhattan": "cityblock",
            "binary": "jaccard",
        }.get(m, m)
        if m == "binary":
            arr_b = (arr != 0).astype(float)
            cond = scipy_pdist(arr_b, metric="jaccard")
            cond = np.nan_to_num(cond, nan=0.0)
        else:
            # Handle NAs pairwise if any NaN exists
            if np.isnan(arr).any():
                cond = _pairwise_dist_with_na(arr, m)
            else:
                cond = scipy_pdist(arr, metric=scipy_m)
        return DistMatrix(cond, n=n, method=m)

    if m in ("pearson", "spearman", "kendall"):
        # R: (1 - cor(t(mat), method=method, use="pairwise.complete.obs")) / 2
        if m in ("pearson", "spearman") and not np.isnan(arr).any():
            work = rankdata(arr, axis=1) if m == "spearman" else arr
            centered = work - np.mean(work, axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            unit_rows = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-15)
            corr = np.clip(unit_rows @ unit_rows.T, -1.0, 1.0)
            # Zero-norm (constant) rows produce 0 correlation -> 0.5 distance matching R
            iu = np.triu_indices(n, k=1)
            cond = 0.5 * (1.0 - corr[iu])
            return DistMatrix(np.asarray(cond, dtype=float), n=n, method=m)

        corr_mat = pd.DataFrame(arr.T).corr(method=m, min_periods=2).to_numpy()
        corr_mat = np.nan_to_num(np.clip(corr_mat, -1.0, 1.0), nan=0.0)
        iu = np.triu_indices(n, k=1)
        cond = 0.5 * (1.0 - corr_mat[iu])
        return DistMatrix(np.asarray(cond, dtype=float), n=n, method=m)

    raise ValueError(f"Unsupported distance method: {method!r}")


def _pairwise_dist_with_na(arr: np.ndarray, m: str) -> np.ndarray:
    n, p = arr.shape
    cond = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = arr[i], arr[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            k = int(np.sum(mask))
            if k == 0:
                cond.append(0.0)
                continue
            d = xi[mask] - xj[mask]
            if m == "euclidean":
                cond.append(float(np.sqrt(np.sum(d * d) * (p / k))))
            elif m in ("manhattan", "cityblock"):
                cond.append(float(np.sum(np.abs(d)) * (p / k)))
            elif m in ("maximum", "chebyshev"):
                cond.append(float(np.max(np.abs(d))))
            elif m == "canberra":
                denom = np.abs(xi[mask]) + np.abs(xj[mask])
                nz = denom > 0
                cond.append(float(np.sum(np.abs(d[nz]) / denom[nz]) * (p / k)) if np.any(nz) else 0.0)
            else:
                cond.append(float(np.sqrt(np.sum(d * d) * (p / k))))
    return np.asarray(cond, dtype=float)


def dist2(x: Any, pairwise_fun: Callable[[Any, Any], float] | None = None, **kwargs: Any) -> DistMatrix:
    """Pairwise distance matrix using custom 2-arg function on matrix rows or list elements."""
    if pairwise_fun is None:
        def pairwise_fun(a: Any, b: Any) -> float:
            aa = np.asarray(a, dtype=float)
            bb = np.asarray(b, dtype=float)
            return float(np.sqrt(np.sum((aa - bb) ** 2)))

    if isinstance(x, np.ndarray) or (hasattr(x, "values") and hasattr(x, "index")):
        labels = list(x.index) if hasattr(x, "index") else None
        arr = np.asarray(x)
        nr = arr.shape[0]
        if nr < 2:
            raise ValueError("`x` must have >= 2 rows.")
        cond = []
        for i in range(nr):
            for j in range(i + 1, nr):
                cond.append(float(pairwise_fun(arr[i], arr[j], **kwargs)))
        return DistMatrix(np.asarray(cond, dtype=float), n=nr, labels=labels, method="dist2")

    if isinstance(x, dict):
        labels = list(x.keys())
        items = list(x.values())
    elif isinstance(x, (list, tuple)):
        labels = None
        items = list(x)
    else:
        items = list(x)
        labels = None

    nr = len(items)
    if nr < 2:
        raise ValueError("`x` must have >= 2 elements.")
    cond = []
    for i in range(nr):
        for j in range(i + 1, nr):
            cond.append(float(pairwise_fun(items[i], items[j], **kwargs)))
    return DistMatrix(np.asarray(cond, dtype=float), n=nr, labels=labels, method="dist2")


class Hclust:
    """Hierarchical clustering result wrapping SciPy linkage matrix."""

    def __init__(self, linkage_mat: np.ndarray, n: int, labels: list[str] | None = None, method: str = "complete") -> None:
        self.linkage = np.asarray(linkage_mat, dtype=float)
        self.n = int(n)
        self.labels = labels
        self.method = method
        self.height = self.linkage[:, 2].copy() if len(self.linkage) else np.array([], dtype=float)
        self.order = _linkage_order(self.linkage, self.n)

    def __repr__(self) -> str:
        return f"Hclust(n={self.n}, method={self.method!r})"


def _linkage_order(Z: np.ndarray, n: int) -> list[int]:
    if n == 0:
        return []
    if n == 1:
        return [0]
    return leaves_list(Z).astype(int).tolist()


def hclust(d: DistMatrix | np.ndarray, method: str = "complete") -> Hclust:
    """Perform hierarchical clustering equivalent to R `stats::hclust`."""
    from ._options import ht_opt

    dm = as_dist(d)
    if dm.n <= 1:
        return Hclust(np.zeros((0, 4), dtype=float), n=dm.n, labels=dm.labels, method=method)

    link_fn = scipy_linkage
    if ht_opt.fast_hclust:
        try:
            import fastcluster
            link_fn = fastcluster.linkage
        except ImportError:
            link_fn = scipy_linkage

    ml = method.lower()
    cond = np.nan_to_num(dm.condensed, nan=0.0)
    if ml in ("ward.d", "ward"):
        # R `ward.D` applies the Ward recurrence to unsquared input distances
        Z = link_fn(np.sqrt(np.maximum(0.0, cond)), method="ward").copy()
        Z[:, 2] = Z[:, 2] ** 2
    else:
        m_map = {
            "ward.d2": "ward",
            "mcquitty": "weighted",
            "median": "median",
            "centroid": "centroid",
            "single": "single",
            "complete": "complete",
            "average": "average",
        }
        sp_m = m_map.get(ml, method)
        Z = link_fn(cond, method=sp_m)
    return Hclust(Z, n=dm.n, labels=dm.labels, method=method)


class Dendrogram:
    """Tree node representing a hierarchical dendrogram with R-compatible attributes."""

    def __init__(
        self,
        children: list["Dendrogram"] | None = None,
        height: float = 0.0,
        leaf: bool = False,
        value: int | None = None,
        label: Any = None,
        members: int = 1,
        x: Any = None,
        midpoint: float | None = None,
        edge_par: dict[str, Any] | None = None,
        node_par: dict[str, Any] | None = None,
        class_label: Any = None,
    ) -> None:
        self.children: list[Dendrogram] = list(children) if children is not None else []
        self.height = float(height)
        self.leaf = bool(leaf)
        self.value = value
        self.label = label
        self.members = int(members)
        self.x = x
        self.midpoint = midpoint
        self.edge_par: dict[str, Any] = dict(edge_par) if edge_par else {}
        self.node_par: dict[str, Any] = dict(node_par) if node_par else {}
        self.class_label = class_label

    @property
    def is_leaf(self) -> bool:
        return self.leaf or len(self.children) == 0

    def __len__(self) -> int:
        return len(self.children)

    def __getitem__(self, idx: int) -> "Dendrogram":
        return self.children[idx]

    def __setitem__(self, idx: int, val: "Dendrogram") -> None:
        self.children[idx] = val

    def copy(self) -> "Dendrogram":
        return copy.deepcopy(self)

    def nobs(self) -> int:
        if self.is_leaf:
            return 1
        return sum(ch.nobs() for ch in self.children)

    def order(self) -> list[int]:
        if self.is_leaf:
            return [int(self.value if self.value is not None else 0)]
        out: list[int] = []
        for ch in self.children:
            out.extend(ch.order())
        return out

    def set_order(self, values: Sequence[int]) -> None:
        it = iter(values)
        def _assign(node: Dendrogram) -> None:
            if node.is_leaf:
                node.value = int(next(it))
            else:
                for ch in node.children:
                    _assign(ch)
        _assign(self)

    def __repr__(self) -> str:
        return f"Dendrogram(members={self.nobs()}, height={self.height:.4g}, leaf={self.is_leaf})"


def order_dendrogram(dend: Dendrogram) -> list[int]:
    return dend.order()


def nobs(obj: Any) -> int:
    if hasattr(obj, "nobs") and callable(obj.nobs):
        return obj.nobs()
    if isinstance(obj, Dendrogram):
        return obj.nobs()
    raise TypeError(f"Cannot compute nobs for {type(obj).__name__}.")


def as_dendrogram(obj: Any) -> Dendrogram:
    """Convert Hclust, SciPy linkage array, or Dendrogram into a Dendrogram tree."""
    if isinstance(obj, Dendrogram):
        return obj
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 4:
        n = obj.shape[0] + 1
        obj = Hclust(obj, n=n)
    if isinstance(obj, Hclust):
        n = obj.n
        if n == 0:
            return Dendrogram(leaf=True, value=0, height=0.0, members=0)
        if n == 1:
            lbl = obj.labels[0] if obj.labels else 0
            return Dendrogram(leaf=True, value=0, label=lbl, height=0.0, members=1, midpoint=0.0)
        nodes: dict[int, Dendrogram] = {}
        for i in range(n):
            lbl = obj.labels[i] if obj.labels and i < len(obj.labels) else i
            nodes[i] = Dendrogram(leaf=True, value=i, label=lbl, height=0.0, members=1, midpoint=0.0)
        Z = obj.linkage
        for k, row in enumerate(Z):
            c1, c2, h, cnt = int(row[0]), int(row[1]), float(row[2]), int(row[3])
            n1, n2 = nodes[c1], nodes[c2]
            nodes[n + k] = Dendrogram(
                children=[n1, n2],
                height=h,
                leaf=False,
                members=cnt,
            )
        root = nodes[2 * n - 2]
        _recompute_midpoints(root)
        return root
    raise TypeError(f"Cannot convert {type(obj).__name__} to Dendrogram.")


def _recompute_midpoints(node: Dendrogram) -> int:
    if node.is_leaf:
        node.members = 1
        node.midpoint = 0.0
        return 1
    m_left = _recompute_midpoints(node.children[0])
    total = m_left
    for ch in node.children[1:]:
        total += _recompute_midpoints(ch)
    node.members = total
    # Match R midpoint calculation
    node.midpoint = 0.5 * (node.children[0].midpoint + (total - node.children[-1].members + node.children[-1].midpoint))
    return total


def reorder_dendrogram(dend: Dendrogram, wts: Sequence[float], agglo_fun: Callable[[Sequence[float]], float] = np.mean) -> Dendrogram:
    """Reorder dendrogram branches by leaf weights `wts` (indexed by 0-based leaf value)."""
    d = dend.copy()
    w = np.asarray(wts, dtype=float)

    def _rec(node: Dendrogram) -> float:
        if node.is_leaf:
            val = float(w[node.value]) if node.value is not None and node.value < len(w) else 0.0
            return val
        child_vals = [_rec(ch) for ch in node.children]
        if len(node.children) > 1:
            order_idx = sorted(range(len(node.children)), key=lambda i: child_vals[i])
            node.children = [node.children[i] for i in order_idx]
        return float(agglo_fun(child_vals))

    _rec(d)
    _recompute_midpoints(d)
    return d


def dend_branches_heights(dend: Dendrogram) -> list[float]:
    if dend.is_leaf:
        return []
    out = [dend.height]
    for ch in dend.children:
        out.extend(dend_branches_heights(ch))
    return out


def cut_dendrogram(dend: Dendrogram, k: int) -> dict[str, Any]:
    """Cut dendrogram into `k` subtrees, returning `{'upper': Dendrogram, 'lower': list[Dendrogram]}`."""
    h = sorted(dend_branches_heights(dend), reverse=True)
    if k <= 1 or len(h) == 0:
        upper = Dendrogram(leaf=True, value=0, height=dend.height, members=1)
        return {"upper": upper, "lower": [dend.copy()]}
    if k > len(h):
        cut_h = 0.0
    else:
        cut_h = 0.5 * (h[k - 2] + h[k - 1])

    lower: list[Dendrogram] = []

    def _cut(node: Dendrogram) -> Dendrogram:
        if node.is_leaf or node.height <= cut_h + 1e-12:
            leaf_idx = len(lower)
            sub = node.copy()
            lower.append(sub)
            return Dendrogram(leaf=True, value=leaf_idx, height=node.height, members=1)
        new_children = [_cut(ch) for ch in node.children]
        return Dendrogram(
            children=new_children,
            height=node.height,
            leaf=False,
            members=len(lower),
            edge_par=node.edge_par,
            node_par=node.node_par,
        )

    upper = _cut(dend)
    _recompute_midpoints(upper)
    return {"upper": upper, "lower": lower}


def cutree(hc: Hclust | Dendrogram, k: int) -> np.ndarray:
    """Cut clustering tree into `k` cluster labels `1..k` ordered by observation index."""
    dend = as_dendrogram(hc)
    ct = cut_dendrogram(dend, k)
    n = dend.nobs()
    res = np.zeros(n, dtype=int)
    for cl_id, sub in enumerate(ct["lower"], start=1):
        for leaf_v in sub.order():
            res[leaf_v] = cl_id
    return res


def adjust_dend_by_x(dend: Dendrogram, leaf_pos: Any = None) -> Dendrogram:
    """Assign `x` positions to every leaf and internal node (bottom-up)."""
    d = dend.copy()
    n = d.nobs()
    if leaf_pos is None:
        leaf_pos = np.arange(n, dtype=float) + 0.5
    if isinstance(leaf_pos, Unit):
        if len(leaf_pos) != n:
            raise ValueError("`leaf_pos` length must match number of leaves in `dend`.")
        pos_list = [leaf_pos[i] for i in range(n)]
        is_u = True
    else:
        arr = np.asarray(leaf_pos, dtype=float)
        if len(arr) != n:
            raise ValueError("`leaf_pos` length must match number of leaves in `dend`.")
        pos_list = [float(v) for v in arr]
        is_u = False

    od = d.order()
    # Map leaf value -> position index in traversal order
    val_to_idx = {val: i for i, val in enumerate(od)}
    counter = [0]

    def _walk(node: Dendrogram) -> None:
        if node.is_leaf:
            i = counter[0]
            counter[0] += 1
            node.x = pos_list[i]
        else:
            for ch in node.children:
                _walk(ch)
            x0 = node.children[0].x
            x1 = node.children[-1].x
            node.x = (x0 + x1) * 0.5

    _walk(d)
    return d


def dend_heights(x: Any) -> Any:
    if x is None:
        return 0.0
    if isinstance(x, Dendrogram):
        return float(x.height)
    return [float(d.height) for d in x]


def dend_xy(dend: Dendrogram) -> dict[str, Any]:
    """Return leaf positions `x` and vertical range `y = [0, height]`."""
    d = dend if dend.x is not None else adjust_dend_by_x(dend)
    xs: list[Any] = []

    def _collect(node: Dendrogram) -> None:
        if node.is_leaf:
            xs.append(node.x)
        else:
            for ch in node.children:
                _collect(ch)

    _collect(d)
    if xs and isinstance(xs[0], Unit):
        x_out: Any = unit_c(*xs)
    else:
        x_out = np.asarray(xs, dtype=float)
    return {"x": x_out, "y": np.array([0.0, float(dend_heights(d))], dtype=float)}


def merge_dendrogram(parent: Dendrogram, children: Sequence[Dendrogram], only_parent: bool = False) -> Dendrogram:
    """Connect parent dendrogram to child dendrograms at its leaves."""
    p = parent.copy()
    ch_list = [c.copy() for c in children]
    n = p.nobs()
    if n != len(ch_list):
        raise ValueError("Number of children dendrograms must equal number of leaves in `parent`.")

    children_height = [float(c.height) for c in ch_list]
    children_height_max = max(children_height) if children_height else 0.0
    parent_height_min = min(dend_branches_heights(p), default=0.0)
    h_line = children_height_max + parent_height_min * 0.1

    od_parent = p.order()
    od2index = {val: i for i, val in enumerate(od_parent)}

    def _update_heights(node: Dendrogram) -> None:
        if node.is_leaf:
            idx = node.value if node.value is not None else 0
            node.height = children_height[idx] if idx < len(children_height) else 0.0
        else:
            node.height = node.height + children_height_max
            for ch in node.children:
                _update_heights(ch)

    _update_heights(p)
    if only_parent:
        return p

    def _replace_leaves(node: Dendrogram) -> Dendrogram:
        if node.is_leaf:
            idx = node.value if node.value is not None else 0
            return ch_list[idx]
        node.children = [_replace_leaves(ch) for ch in node.children]
        return node

    merged = _replace_leaves(p)
    merged.members = sum(c.nobs() for c in ch_list)

    # Re-index leaf values consecutively across children in parent leaf order
    offset = 0
    new_orders: list[int] = []
    for p_idx in od_parent:
        c_ord = ch_list[p_idx].order()
        ranks = np.argsort(np.argsort(c_ord))
        shifted = (ranks + offset).tolist()
        ch_list[p_idx].set_order(shifted)
        new_orders.extend(shifted)
        offset += len(c_ord)
    merged.set_order(new_orders)
    merged._children_height = children_height  # type: ignore[attr-defined]
    merged._parent_height = parent.height      # type: ignore[attr-defined]
    merged._h_line = h_line                    # type: ignore[attr-defined]
    return merged


def color_branches(dend: Dendrogram, k: int | None = None, h: float | None = None, col: Any = None) -> Dendrogram:
    """Color dendrogram branches across `k` clusters or cut height `h` (dendextend::color_branches)."""
    d = dend.copy()
    if k is None and h is None:
        k = 1
    if k is None and h is not None:
        # Count clusters above h
        hs = sorted(dend_branches_heights(d), reverse=True)
        k = 1 + sum(1 for v in hs if v > h)
    assert k is not None

    if col is None:
        cols = brewer_pal(max(k, 3), "Set1")[:k] if k <= 9 else [to_hex_rgba(i + 1) for i in range(k)]
    elif isinstance(col, (str, int, np.integer)):
        cols = [to_hex_rgba(col)] * k
    else:
        cols = [to_hex_rgba(c) for c in col]
        if len(cols) < k:
            cols = [cols[i % len(cols)] for i in range(k)]

    if k == 1:
        c0 = cols[0]
        def _set_all(node: Dendrogram) -> None:
            node.edge_par["col"] = c0
            for ch in node.children:
                _set_all(ch)
        _set_all(d)
        return d

    hs = sorted(dend_branches_heights(d), reverse=True)
    cut_h = 0.5 * (hs[k - 2] + hs[k - 1]) if k <= len(hs) else 0.0
    cluster_idx = [0]

    def _color_rec(node: Dendrogram, active_col: str | None) -> None:
        if active_col is not None:
            node.edge_par["col"] = active_col
            for ch in node.children:
                _color_rec(ch, active_col)
            return
        if node.is_leaf or node.height <= cut_h + 1e-12:
            c = cols[cluster_idx[0] % len(cols)]
            cluster_idx[0] += 1
            node.edge_par["col"] = c
            for ch in node.children:
                _color_rec(ch, c)
        else:
            for ch in node.children:
                _color_rec(ch, None)

    _color_rec(d, None)
    return d


def cluster_within_group(mat: Any, factor: Sequence[Any]) -> Dendrogram:
    """Cluster columns of `mat` within each level of `factor`, then cluster group means."""
    arr = np.asarray(mat, dtype=float)
    fa = [str(v) for v in factor]
    levels = list(dict.fromkeys(fa))
    dend_list: list[Dendrogram] = []
    order_list: list[list[int]] = []
    group_means: list[np.ndarray] = []

    for le in levels:
        cols_idx = [i for i, v in enumerate(fa) if v == le]
        sub = arr[:, cols_idx]
        if len(cols_idx) == 1:
            d_sub = Dendrogram(leaf=True, value=cols_idx[0], height=0.0, members=1)
            ord_sub = cols_idx
        else:
            hc1 = hclust(dist(sub.T))
            d_sub = as_dendrogram(hc1)
            ord_sub = [cols_idx[i] for i in d_sub.order()]
            d_sub.set_order(ord_sub)
        # Flatten within-group heights to 0
        def _zero_h(node: Dendrogram) -> None:
            node.height = 0.0
            for ch in node.children:
                _zero_h(ch)
        _zero_h(d_sub)
        d_sub.class_label = le
        dend_list.append(d_sub)
        order_list.append(ord_sub)
        group_means.append(np.mean(sub, axis=1))

    if len(levels) == 1:
        return dend_list[0]
    parent = as_dendrogram(hclust(dist(np.vstack(group_means))))
    parent_order = parent.order()
    merged = merge_dendrogram(parent, dend_list)
    flat_order = [idx for p_i in parent_order for idx in order_list[p_i]]
    merged.set_order(flat_order)
    return merged


def cluster_between_groups(mat: Any, factor: Sequence[Any]) -> Dendrogram:
    """Keep column order within each factor level while clustering between group means."""
    arr = np.asarray(mat, dtype=float)
    fa = [str(v) for v in factor]
    levels = list(dict.fromkeys(fa))
    dend_list: list[Dendrogram] = []
    order_list: list[list[int]] = []
    group_means: list[np.ndarray] = []

    for le in levels:
        cols_idx = [i for i, v in enumerate(fa) if v == le]
        sub = arr[:, cols_idx]
        if len(cols_idx) == 1:
            d_sub = Dendrogram(leaf=True, value=cols_idx[0], height=0.0, members=1)
        else:
            hc1 = hclust(dist(np.arange(len(cols_idx), dtype=float)[:, None]))
            d_sub = reorder_dendrogram(as_dendrogram(hc1), np.arange(len(cols_idx), dtype=float), np.mean)
            d_sub.set_order(cols_idx)
        def _zero_h(node: Dendrogram) -> None:
            node.height = 0.0
            for ch in node.children:
                _zero_h(ch)
        _zero_h(d_sub)
        d_sub.class_label = le
        dend_list.append(d_sub)
        order_list.append(cols_idx)
        group_means.append(np.mean(sub, axis=1))

    if len(levels) == 1:
        return dend_list[0]
    parent = as_dendrogram(hclust(dist(np.vstack(group_means))))
    parent_order = parent.order()
    merged = merge_dendrogram(parent, dend_list)
    flat_order = [idx for p_i in parent_order for idx in order_list[p_i]]
    merged.set_order(flat_order)
    return merged


def _construct_dend_segments(dend: Dendrogram, gp: dict[str, Any]) -> dict[str, Any]:
    if dend.x is None:
        dend = adjust_dend_by_x(dend)
    if dend.is_leaf:
        return {}

    x_is_unit = isinstance(dend.x, Unit)
    x0_list: list[Any] = []
    x1_list: list[Any] = []
    y0_list: list[float] = []
    y1_list: list[float] = []
    cols: list[Any] = []
    lwds: list[float] = []
    ltys: list[Any] = []

    node_x: list[Any] = []
    node_y: list[float] = []
    node_col: list[Any] = []
    node_fill: list[Any] = []
    node_pch: list[Any] = []
    node_size: list[Any] = []
    node_flag: list[bool] = []

    def _add_node_par(node: Dendrogram, xval: Any, yval: float) -> None:
        np_d = node.node_par
        has_np = any(k in np_d for k in ("col", "fill", "pch", "cex", "size"))
        node_flag.append(has_np)
        node_x.append(xval)
        node_y.append(yval)
        node_col.append(np_d.get("col", "black"))
        node_fill.append(np_d.get("fill", "transparent"))
        node_pch.append(np_d.get("pch", 1))
        node_size.append(np_d.get("size", unit(1.0, "char")))

    _add_node_par(dend, dend.x, dend.height)

    queue: list[Dendrogram] = [dend]
    while queue:
        next_q: list[Dendrogram] = []
        for parent in queue:
            if parent.is_leaf:
                continue
            h = float(parent.height)
            h_zero = abs(h) < 1e-10
            xl = [ch.x for ch in parent.children]
            yl = [float(ch.height) for ch in parent.children]
            if x_is_unit:
                # For mid_x of unit children
                mid_x = (xl[0] + xl[-1]) * 0.5
            else:
                mid_x = 0.5 * (min(xl) + max(xl))
            nc = len(parent.children)
            idx_order = [i for i in range(1, nc - 1)] + ([0, nc - 1] if nc > 1 else [0])
            for i in idx_order:
                ch = parent.children[i]
                ep = ch.edge_par
                c_val = ep.get("col", "black")
                lw_val = ep.get("lwd", 1.0)
                lt_val = ep.get("lty", "solid")
                _add_node_par(ch, xl[i], yl[i])
                if h_zero:
                    x0_list.append(xl[i])
                    x1_list.append(mid_x)
                    y0_list.append(h)
                    y1_list.append(h)
                    cols.append(c_val)
                    lwds.append(lw_val)
                    ltys.append(lt_val)
                else:
                    x0_list.extend([xl[i], xl[i]])
                    x1_list.extend([xl[i], mid_x])
                    y0_list.extend([yl[i], h])
                    y1_list.extend([h, h])
                    cols.extend([c_val, c_val])
                    lwds.extend([lw_val, lw_val])
                    ltys.extend([lt_val, lt_val])
            for ch in parent.children:
                if not ch.is_leaf:
                    next_q.append(ch)
        queue = next_q

    gp_d = check_gp(gp)
    if "col" in gp_d:
        cols = gp_d["col"]
    if "lwd" in gp_d:
        lwds = gp_d["lwd"]
    if "lty" in gp_d:
        ltys = gp_d["lty"]

    return {
        "x0": unit_c(*x0_list) if x_is_unit else np.asarray(x0_list, dtype=float),
        "x1": unit_c(*x1_list) if x_is_unit else np.asarray(x1_list, dtype=float),
        "y0": np.asarray(y0_list, dtype=float),
        "y1": np.asarray(y1_list, dtype=float),
        "col": cols,
        "lwd": lwds,
        "lty": ltys,
        "node": np.asarray(node_flag, dtype=bool),
        "node_x": unit_c(*node_x) if x_is_unit else np.asarray(node_x, dtype=float),
        "node_y": np.asarray(node_y, dtype=float),
        "node_col": node_col,
        "node_fill": node_fill,
        "node_pch": node_pch,
        "node_size": unit_c(*node_size) if node_size else unit([], "char"),
    }


def dendrogramGrob(
    dend: Dendrogram,
    facing: str = "bottom",
    order: str = "normal",
    gp: dict[str, Any] | None = None,
) -> GTree:
    """Build a GTree representing a dendrogram with given `facing` and `order`."""
    facing = facing.lower()
    order = order.lower()
    if facing not in ("bottom", "top", "left", "right"):
        raise ValueError(f"Invalid `facing`: {facing!r}")
    if order not in ("normal", "reverse"):
        raise ValueError(f"Invalid `order`: {order!r}")

    lt = _construct_dend_segments(dend, gp or {})
    if not lt:
        gt = GTree(children=[])
        gt.facing = facing  # type: ignore[attr-defined]
        return gt

    x0, x1 = lt["x0"], lt["x1"]
    y0, y1 = lt["y0"], lt["y1"]
    nx, ny = lt["node_x"], lt["node_y"]

    if order == "reverse":
        ux0 = x0 if isinstance(x0, Unit) else unit(x0, "native")
        ux1 = x1 if isinstance(x1, Unit) else unit(x1, "native")
        unx = nx if isinstance(nx, Unit) else unit(nx, "native")
        x0 = unit(1.0, "npc") - ux0
        x1 = unit(1.0, "npc") - ux1
        nx = unit(1.0, "npc") - unx

    if facing in ("top", "right"):
        y0 = unit(1.0, "npc") - unit(y0, "native")
        y1 = unit(1.0, "npc") - unit(y1, "native")
        ny = unit(1.0, "npc") - unit(ny, "native")

    seg_gp = gpar(col=lt["col"], lwd=lt["lwd"], lty=lt["lty"])
    children: list[Any] = []
    if facing in ("bottom", "top"):
        children.append(segmentsGrob(x0, y0, x1, y1, gp=seg_gp, default_units="native"))
        if np.any(lt["node"]):
            mask = lt["node"]
            idxs = np.where(mask)[0]
            children.append(
                pointsGrob(
                    nx[mask] if isinstance(nx, Unit) else nx[mask],
                    ny[mask],
                    pch=[lt["node_pch"][i] for i in idxs],
                    size=lt["node_size"][mask],
                    gp=gpar(
                        col=[lt["node_col"][i] for i in idxs],
                        fill=[lt["node_fill"][i] for i in idxs],
                    ),
                    default_units="native",
                )
            )
    else:
        children.append(segmentsGrob(y0, x0, y1, x1, gp=seg_gp, default_units="native"))
        if np.any(lt["node"]):
            mask = lt["node"]
            idxs = np.where(mask)[0]
            children.append(
                pointsGrob(
                    ny[mask],
                    nx[mask],
                    pch=[lt["node_pch"][i] for i in idxs],
                    size=lt["node_size"][mask],
                    gp=gpar(
                        col=[lt["node_col"][i] for i in idxs],
                        fill=[lt["node_fill"][i] for i in idxs],
                    ),
                    default_units="native",
                )
            )

    gt = GTree(children=children)
    gt.facing = facing  # type: ignore[attr-defined]
    return gt


def grid_dendrogram(dend: Dendrogram, facing: str = "bottom", order: str = "normal", gp: dict[str, Any] | None = None, test: bool = False) -> GTree:
    """Draw a `Dendrogram` object into the current viewport (or a test viewport if `test=True`)."""
    gb = dendrogramGrob(dend, facing=facing, order=order, gp=gp)
    if test:
        h = float(dend_heights(dend))
        if h <= 0:
            h = 1.0
        n = dend.nobs()
        grid_newpage()
        xy = dend_xy(dend)
        is_u = isinstance(xy["x"], Unit)
        if is_u:
            max_x = max(float(v) for v in xy["x"].to_mm("x")) / 10.0
            xmin, xmax = -0.5, max(n + 0.5, max_x + 0.5)
        else:
            xmin, xmax = -0.5, n + 0.5
        if facing in ("top", "bottom"):
            pushViewport(
                Viewport(
                    xscale=(xmin, xmax),
                    yscale=(-h * 0.05, h * 1.05),
                    width=unit(1, "npc") - unit(4, "cm"),
                    height=unit(1, "npc") - unit(4, "cm"),
                )
            )
        else:
            pushViewport(
                Viewport(
                    yscale=(xmin, xmax),
                    xscale=(-h * 0.05, h * 1.05),
                    width=unit(1, "npc") - unit(4, "cm"),
                    height=unit(1, "npc") - unit(4, "cm"),
                )
            )
    gb.draw()
    if test:
        grid_xaxis()
        grid_yaxis()
        grid_rect()
        popViewport()
    return gb
