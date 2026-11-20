"""Color conversion, interpolation, palettes, and ColorMapping class."""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Iterable, Sequence

import matplotlib as mpl
import matplotlib.colors as mpl_colors
import numpy as np

# Default R4 categorical color palette (palette() in R >= 4.0)
R_PALETTE = [
    "#000000",  # 1: black
    "#DF536B",  # 2: red
    "#61D04F",  # 3: green3
    "#2297E6",  # 4: blue
    "#28E2E5",  # 5: cyan
    "#CD0BBC",  # 6: magenta
    "#F5C710",  # 7: yellow
    "#9E9E9E",  # 8: grey
]

# R named colors that differ from or are absent in matplotlib CSS4 colors
_R_EXTRA_COLORS: dict[str, str] = {
    "grey": "#BEBEBE",
    "gray": "#BEBEBE",
    "green": "#00FF00",
    "green3": "#00CD00",
    "green4": "#008B00",
    "red2": "#EE0000",
    "red3": "#CD0000",
    "blue2": "#0000EE",
    "blue3": "#0000CD",
    "cyan2": "#00EEEE",
    "cyan3": "#00CDCD",
    "magenta2": "#EE00EE",
    "magenta3": "#CD00CD",
    "yellow2": "#EEEE00",
    "yellow3": "#CDCD00",
    "orange2": "#EE9A00",
    "orange3": "#CD8500",
    "purple": "#A020F0",
    "brown": "#A52A2A",
    "darkred": "#8B0000",
    "darkblue": "#00008B",
    "darkgreen": "#006400",
    "darkorange": "#FF8C00",
    "lightgrey": "#D3D3D3",
    "lightgray": "#D3D3D3",
    "darkgrey": "#A9A9A9",
    "darkgray": "#A9A9A9",
    "transparent": "#00000000",
}
for _k in range(101):
    _hex_v = f"#{round(_k * 255 / 100):02X}" * 3
    _R_EXTRA_COLORS[f"grey{_k}"] = _hex_v
    _R_EXTRA_COLORS[f"gray{_k}"] = _hex_v


_COLOR_MAPPING_COUNTER = itertools.count(1)


def _is_missing_color(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    if isinstance(val, str):
        s = val.strip().lower()
        return s in ("", "na", "nan", "none")
    return False


def to_hex_rgba(col: Any) -> str:
    """Convert any R/Matplotlib/hex/RGB(A)/integer color specification to '#RRGGBBAA'."""
    if _is_missing_color(col):
        return "#00000000"
    if isinstance(col, (int, np.integer)):
        idx = int(col)
        if idx <= 0:
            return "#00000000"
        base = R_PALETTE[(idx - 1) % len(R_PALETTE)]
        return base + "FF"
    if isinstance(col, str):
        s = col.strip()
        low = s.lower()
        if low in _R_EXTRA_COLORS:
            res = _R_EXTRA_COLORS[low]
            return res if len(res) == 9 else res + "FF"
        if low.startswith("#"):
            if len(low) == 4:
                r, g, b = low[1], low[2], low[3]
                return f"#{r}{r}{g}{g}{b}{b}FF".upper()
            if len(low) == 5:
                r, g, b, a = low[1], low[2], low[3], low[4]
                return f"#{r}{r}{g}{g}{b}{b}{a}{a}".upper()
            if len(low) == 7:
                return (low + "FF").upper()
            if len(low) == 9:
                return low.upper()
        rgba = mpl_colors.to_rgba(s)
        return _tuple_to_hex_rgba(rgba)
    if isinstance(col, (tuple, list, np.ndarray)):
        arr = [float(v) for v in col]
        if len(arr) in (3, 4):
            if any(v > 1.0 for v in arr[:3]):
                arr = [v / 255.0 for v in arr]
            if len(arr) == 3:
                arr.append(1.0)
            return _tuple_to_hex_rgba(tuple(arr))
    rgba = mpl_colors.to_rgba(col)
    return _tuple_to_hex_rgba(rgba)


def _tuple_to_hex_rgba(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, a = (max(0, min(255, int(round(v * 255)))) for v in rgba)
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}"


def to_rgba_tuple(col: Any) -> tuple[float, float, float, float]:
    """Convert any color specification to normalized (r, g, b, a) floats in [0, 1]."""
    hx = to_hex_rgba(col)
    r = int(hx[1:3], 16) / 255.0
    g = int(hx[3:5], 16) / 255.0
    b = int(hx[5:7], 16) / 255.0
    a = int(hx[7:9], 16) / 255.0
    return (r, g, b, a)


def to_hex_rgba_list(cols: Any) -> list[str]:
    """Normalize a single color or sequence of colors to a list of '#RRGGBBAA' strings."""
    if isinstance(cols, (str, int, np.integer, float, np.floating)) or cols is None:
        return [to_hex_rgba(cols)]
    if isinstance(cols, tuple) and len(cols) in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in cols):
        return [to_hex_rgba(cols)]
    return [to_hex_rgba(c) for c in cols]


def add_transparency(col: Any, transparency: Any = 0.0) -> Any:
    """Apply transparency in [0, 1] (0 = opaque, 1 = transparent) to color(s)."""
    is_scalar = isinstance(col, (str, int, np.integer)) or col is None
    cols = to_hex_rgba_list(col)
    if np.ndim(transparency) == 0:
        trans = [float(transparency)] * len(cols)
    else:
        t_list = [float(v) for v in transparency]
        if len(t_list) == 1:
            trans = t_list * len(cols)
        elif len(t_list) == len(cols):
            trans = t_list
        else:
            if len(cols) == 1:
                cols = cols * len(t_list)
                trans = t_list
                is_scalar = False
            else:
                raise ValueError("`transparency` length must match `col` length or be 1.")
    out = []
    for c, tr in zip(cols, trans):
        tr = max(0.0, min(1.0, tr))
        alpha_byte = int(round((1.0 - tr) * 255))
        out.append(c[:7] + f"{alpha_byte:02X}")
    return out[0] if is_scalar and len(out) == 1 else out


# ---- Color space conversions (sRGB <-> XYZ <-> Lab <-> HSV) ----

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.maximum(c, 0.0) ** (1.0 / 2.4)) - 0.055)


# D65 reference white
_D65 = np.array([0.95047, 1.00000, 1.08883])
_M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_M_XYZ_TO_RGB = np.linalg.inv(_M_RGB_TO_XYZ)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = _srgb_to_linear(rgb)
    xyz = lin @ _M_RGB_TO_XYZ.T
    xyz_n = xyz / _D65
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz_n > eps, np.cbrt(np.maximum(xyz_n, 0.0)), (kappa * xyz_n + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr, yr, zr], axis=-1) * _D65
    lin = xyz @ _M_XYZ_TO_RGB.T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)


def _srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return _srgb_to_linear(rgb) @ _M_RGB_TO_XYZ.T


def _xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    return np.clip(_linear_to_srgb(xyz @ _M_XYZ_TO_RGB.T), 0.0, 1.0)


def _srgb_to_polar_luv(rgb: np.ndarray) -> tuple[float, float, float]:
    """Convert single sRGB (r, g, b) to polarLUV (H_deg, C, L)."""
    xyz = _srgb_to_xyz(rgb)
    X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    denom = X + 15.0 * Y + 3.0 * Z
    u_prime = (4.0 * X / denom) if denom > 1e-12 else 0.0
    v_prime = (9.0 * Y / denom) if denom > 1e-12 else 0.0
    Xn, Yn, Zn = _D65
    denom_n = Xn + 15.0 * Yn + 3.0 * Zn
    un = 4.0 * Xn / denom_n
    vn = 9.0 * Yn / denom_n
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    yr = Y / Yn
    L = 116.0 * (yr ** (1.0 / 3.0)) - 16.0 if yr > eps else kappa * yr
    u = 13.0 * L * (u_prime - un)
    v = 13.0 * L * (v_prime - vn)
    C = math.hypot(u, v)
    H = (math.degrees(math.atan2(v, u))) % 360.0
    return H, C, L


def _polar_luv_to_srgb(H_deg: float, C: float, L: float) -> tuple[float, float, float]:
    """Convert polarLUV (H_deg, C, L) to clipped sRGB tuple."""
    if L <= 0.0:
        return (0.0, 0.0, 0.0)
    h_rad = math.radians(H_deg)
    u = C * math.cos(h_rad)
    v = C * math.sin(h_rad)
    Xn, Yn, Zn = _D65
    denom_n = Xn + 15.0 * Yn + 3.0 * Zn
    un = 4.0 * Xn / denom_n
    vn = 9.0 * Yn / denom_n
    u_prime = u / (13.0 * L) + un
    v_prime = v / (13.0 * L) + vn
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    Y = Yn * (((L + 16.0) / 116.0) ** 3 if L > kappa * eps else L / kappa)
    if abs(v_prime) < 1e-12:
        return (0.0, 0.0, 0.0)
    X = Y * (9.0 * u_prime) / (4.0 * v_prime)
    Z = Y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime)
    rgb = _xyz_to_srgb(np.array([X, Y, Z]))
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def add_luminance(col: Any) -> list[str]:
    """Generate 9-step sequential HCL palette like colorspace::sequential_hcl for 3D shading."""
    r, g, b, _ = to_rgba_tuple(col)
    H, C, L = _srgb_to_polar_luv(np.array([r, g, b]))
    # Mimic colorspace::sequential_hcl(n = 9, h = H, c = C, l = L)
    # sequential_hcl ramps luminance from L (darkest/base) up to ~95 (lightest)
    # Note: R sequential_hcl(n=9, h=H, c=C, l=L) uses l=c(L, 90) or c(L, L)
    # Actually in colorspace::sequential_hcl(n=9, h=H, c=C, l=L), single l uses l=c(L, 90)
    l_end = max(L, 90.0)
    c_end = C * 0.1
    out = []
    for i in range(9):
        t = i / 8.0
        li = L + t * (l_end - L)
        ci = C + t * (c_end - C)
        rgb = _polar_luv_to_srgb(H, ci, li)
        out.append(_tuple_to_hex_rgba((rgb[0], rgb[1], rgb[2], 1.0)))
    return out


def colorRamp2(
    breaks: Sequence[float],
    colors: Sequence[Any],
    transparency: float = 0.0,
    space: str = "LAB",
) -> Callable[[Any], Any]:
    """Create a continuous color mapping function analogous to circlize::colorRamp2."""
    brk = np.asarray(breaks, dtype=float)
    if brk.ndim != 1 or len(brk) < 2:
        raise ValueError("`breaks` must be a 1D sequence of at least 2 numbers.")
    if len(colors) != len(brk):
        raise ValueError("`breaks` and `colors` must have the same length.")

    rgba_list = np.array([to_rgba_tuple(c) for c in colors], dtype=float)
    if transparency and transparency > 0:
        rgba_list[:, 3] *= max(0.0, min(1.0, 1.0 - float(transparency)))

    # Sort breaks if descending, preserving correspondence
    order = np.argsort(brk)
    brk_sorted = brk[order]
    rgba_sorted = rgba_list[order]

    sp = space.upper()
    if sp in ("RGB", "SRGB"):
        coords = rgba_sorted[:, :3]
        back = lambda c: np.clip(c, 0.0, 1.0)
    elif sp == "XYZ":
        coords = _srgb_to_xyz(rgba_sorted[:, :3])
        back = _xyz_to_srgb
    elif sp == "HSV":
        coords = mpl_colors.rgb_to_hsv(rgba_sorted[:, :3])
        back = lambda c: np.clip(mpl_colors.hsv_to_rgb(np.clip(c, 0.0, 1.0)), 0.0, 1.0)
    else:  # "LAB"
        coords = _srgb_to_lab(rgba_sorted[:, :3])
        back = _lab_to_srgb

    alphas = rgba_sorted[:, 3]

    def to_rgba_array(x: Any, na_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        orig_shape = arr.shape
        flat = arr.ravel()
        out = np.empty((len(flat), 4), dtype=float)

        nan_mask = np.isnan(flat)
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 0:
            v = flat[valid_idx]
            interp_coords = np.column_stack([
                np.interp(v, brk_sorted, coords[:, d]) for d in range(coords.shape[1])
            ])
            interp_alpha = np.interp(v, brk_sorted, alphas)
            rgb = back(interp_coords)
            out[valid_idx, :3] = np.clip(rgb, 0.0, 1.0)
            out[valid_idx, 3] = np.clip(interp_alpha, 0.0, 1.0)

        if np.any(nan_mask):
            out[nan_mask] = np.asarray(na_rgba, dtype=float)

        return out.reshape(*orig_shape, 4)

    def col_fun(x: Any) -> Any:
        is_scalar = np.ndim(x) == 0 and not isinstance(x, (list, tuple, np.ndarray))
        arr = np.asarray(x, dtype=float)
        orig_shape = arr.shape
        rgba_flat = to_rgba_array(arr.ravel())
        out = np.empty(len(rgba_flat), dtype=object)

        nan_mask = np.isnan(arr.ravel())
        if np.any(~nan_mask):
            rgba_bytes = np.clip(np.round(rgba_flat[~nan_mask] * 255), 0, 255).astype(int)
            out[~nan_mask] = [
                f"#{r:02X}{g:02X}{b:02X}{a:02X}"
                for r, g, b, a in rgba_bytes
            ]
        if np.any(nan_mask):
            out[nan_mask] = "#00000000"

        if is_scalar:
            return out[0]
        return out.reshape(orig_shape)

    col_fun.breaks = brk
    col_fun.colors = [to_hex_rgba(c) for c in colors]
    col_fun.space = space
    col_fun.to_rgba_array = to_rgba_array
    return col_fun


def rand_color(
    n: int = 1,
    hue: str | None = None,
    luminosity: str = "random",
    transparency: float = 0.0,
    friendly: bool = False,
) -> list[str] | str:
    """Generate random visually distinct colors similar to circlize::rand_color."""
    rng = np.random.default_rng()
    hue_ranges = {
        "red": (345, 375),
        "orange": (15, 45),
        "yellow": (45, 70),
        "green": (80, 155),
        "blue": (185, 255),
        "purple": (260, 305),
        "pink": (305, 345),
    }
    if hue is not None and hue.lower() in hue_ranges:
        h0, h1 = hue_ranges[hue.lower()]
        hues = rng.uniform(h0, h1, size=n) % 360.0
    elif friendly:
        hues = (np.arange(n) * 137.508 + rng.uniform(0, 360)) % 360.0
    else:
        hues = rng.uniform(0.0, 360.0, size=n)

    lum = luminosity.lower()
    if lum == "light":
        s = rng.uniform(0.35, 0.75, size=n)
        v = rng.uniform(0.80, 0.98, size=n)
    elif lum == "dark":
        s = rng.uniform(0.55, 0.95, size=n)
        v = rng.uniform(0.30, 0.60, size=n)
    elif lum == "bright":
        s = rng.uniform(0.75, 1.00, size=n)
        v = rng.uniform(0.75, 0.98, size=n)
    elif friendly:
        s = rng.uniform(0.55, 0.85, size=n)
        v = rng.uniform(0.65, 0.90, size=n)
    elif lum == "monochrome":
        s = np.zeros(n)
        v = rng.uniform(0.20, 0.85, size=n)
    else:
        s = rng.uniform(0.45, 0.95, size=n)
        v = rng.uniform(0.55, 0.95, size=n)

    hsv = np.stack([hues / 360.0, s, v], axis=-1)
    rgb = mpl_colors.hsv_to_rgb(hsv)
    alpha = max(0.0, min(1.0, 1.0 - float(transparency)))
    res = [_tuple_to_hex_rgba((float(r), float(g), float(b), alpha)) for r, g, b in rgb]
    return res


_HCL_ALIASES = {
    "zissou 1": "RdYlBu_r",
    "zissou1": "RdYlBu_r",
    "vik": "RdBu_r",
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "blues": "Blues",
    "reds": "Reds",
    "greens": "Greens",
    "ylorrd": "YlOrRd",
    "rdylbu": "RdYlBu",
    "rdbu": "RdBu",
    "spectral": "Spectral",
    "blue-red": "bwr",
    "blue-yellow 2": "cividis",
}


def _get_cmap(name: str) -> Any:
    if name in mpl.colormaps:
        return mpl.colormaps[name]
    alias = _HCL_ALIASES.get(name.strip().lower())
    if alias and alias in mpl.colormaps:
        return mpl.colormaps[alias]
    for k in mpl.colormaps:
        if k.lower() == name.strip().lower():
            return mpl.colormaps[k]
    return mpl.colormaps["YlOrRd"]


def brewer_pal(n: int, name: str) -> list[str]:
    """Return n hex colors from a ColorBrewer / Matplotlib colormap."""
    cmap = _get_cmap(name)
    if hasattr(cmap, "colors") and len(cmap.colors) <= 20:
        cols = [to_hex_rgba(c) for c in cmap.colors]
        if n <= len(cols):
            return cols[:n]
        return [cols[i % len(cols)] for i in range(n)]
    pts = np.linspace(0.0, 1.0, max(1, n))
    return [to_hex_rgba(cmap(p)) for p in pts]


def hcl_colors(n: int, palette: str = "YlOrRd", rev: bool = False) -> list[str]:
    """Approximate R's grDevices::hcl.colors."""
    cmap = _get_cmap(palette)
    pts = np.linspace(0.0, 1.0, max(1, n))
    if rev:
        pts = pts[::-1]
    return [to_hex_rgba(cmap(p)) for p in pts]


def pretty_breaks(rng: Sequence[float], n: int = 3) -> np.ndarray:
    """Compute human-friendly axis/legend break values similar to R's pretty(x, n)."""
    lo, hi = float(rng[0]), float(rng[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.array([0.0, 1.0])
    if lo == hi:
        return np.array([lo])
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    raw_step = span / max(1, n)
    mag = 10.0 ** math.floor(math.log10(raw_step))
    norm = raw_step / mag
    for cand in (1.0, 2.0, 2.5, 5.0, 10.0):
        if norm <= cand * 1.15:
            step = cand * mag
            break
    else:
        step = 10.0 * mag
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    vals = []
    cur = start
    while cur <= end + 1e-9 * step:
        vals.append(round(cur, 12))
        cur += step
    return np.asarray(vals, dtype=float)


def grid_pretty(rng: Sequence[float], n: int = 5) -> np.ndarray:
    """Compute interior pretty tick locations inside [lo, hi] like grid::grid.pretty."""
    lo, hi = float(min(rng)), float(max(rng))
    br = pretty_breaks((lo, hi), n=n)
    inside = br[(br >= lo - 1e-9 * (hi - lo)) & (br <= hi + 1e-9 * (hi - lo))]
    return inside if len(inside) > 0 else br


def default_col(x: Any, main_matrix: bool = False) -> Any:
    """Infer default color mapping (discrete dict or continuous colorRamp2) for matrix or vector."""
    from ._options import ht_opt

    arr = np.asarray(x)
    # Check numeric vs categorical
    is_num = np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_)
    if not is_num:
        # Check if object array containing only numbers + nan
        flat = arr.ravel()
        non_na = [v for v in flat if not _is_missing_color(v)]
        if len(non_na) > 0 and all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) for v in non_na):
            is_num = True
            arr = np.array([np.nan if _is_missing_color(v) else float(v) for v in flat], dtype=float)

    if is_num:
        vals = arr.astype(float).ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return {"NA": "#CCCCCCFF"}
        uniq = np.unique(vals)
        if len(uniq) == 1:
            return {str(uniq[0]): to_hex_rgba(rand_color(1)[0])}
        colors = list(ht_opt.COLOR)
        if main_matrix:
            nz = vals[vals != 0]
            p = float(np.sum(vals > 0)) / len(nz) if len(nz) > 0 else 0.0
            if 0.25 < p < 0.75:
                q1 = float(np.quantile(np.abs(vals), 0.99)) if len(uniq) >= 100 else float(np.max(np.abs(vals)))
                if q1 == 0:
                    q1 = float(np.max(np.abs(vals)))
                brks = np.linspace(-q1, q1, len(colors))
            else:
                if len(uniq) >= 100:
                    q1 = float(np.quantile(vals, 0.01))
                    q2 = float(np.quantile(vals, 0.99))
                    if q1 == q2 or len(np.unique(vals[(vals >= q1) & (vals <= q2)])) <= 1:
                        q1, q2 = float(np.min(vals)), float(np.max(vals))
                else:
                    q1, q2 = float(np.min(vals)), float(np.max(vals))
                brks = np.linspace(q1, q2, len(colors))
        else:
            q1, q2 = float(np.min(vals)), float(np.max(vals))
            brks = np.linspace(q1, q2, len(colors))
        return colorRamp2(brks, colors)
    else:
        flat = arr.ravel()
        levels = []
        seen = set()
        for v in flat:
            if _is_missing_color(v):
                continue
            sv = str(v)
            if sv not in seen:
                seen.add(sv)
                levels.append(sv)
        if len(levels) == 0:
            return {"NA": "#CCCCCCFF"}
        cols = rand_color(len(levels))
        return dict(zip(levels, cols))


class ColorMapping:
    """Mapping between discrete or continuous data values and RGBA hex colors."""

    def __init__(
        self,
        name: str | None = None,
        colors: dict[Any, Any] | Sequence[Any] | None = None,
        levels: Sequence[Any] | None = None,
        col_fun: Callable[[Any], Any] | None = None,
        breaks: Sequence[float] | None = None,
        na_col: Any = "#FFFFFF",
        full_col: dict[Any, Any] | None = None,
    ) -> None:
        if name is None:
            name = f"color_mapping_{next(_COLOR_MAPPING_COUNTER)}"
        self.name = str(name)
        self.na_col = to_hex_rgba(na_col)
        self.full_col: dict[str, str] = {}
        self.colors: dict[str, str] | list[str]
        self.levels: list[str] | np.ndarray
        self.col_fun: Callable[[Any], Any] | None = None

        if colors is not None:
            if isinstance(colors, dict):
                levs = list(colors.keys()) if levels is None else list(levels)
                col_vals = list(colors.values())
            else:
                col_vals = list(colors)
                if levels is None:
                    raise ValueError("Either provide `levels` or pass `colors` as a dict.")
                levs = list(levels)
            if len(col_vals) != len(levs):
                raise ValueError("Length of `colors` and `levels` must be the same.")
            norm_cols = [to_hex_rgba(c) for c in col_vals]
            str_levs = [str(lv) for lv in levs]
            self.levels = str_levs
            self.colors = dict(zip(str_levs, norm_cols))
            if full_col is not None:
                self.full_col = {str(k): to_hex_rgba(v) for k, v in full_col.items()}
            self.type = "discrete"
        elif col_fun is not None:
            if breaks is None:
                brk = getattr(col_fun, "breaks", None)
                if brk is None:
                    raise ValueError("You should provide `breaks` when `col_fun` has no `breaks` attribute.")
                rg = [float(np.min(brk)), float(np.max(brk))]
                diff = rg[1] - rg[0]
                rg[0] += diff * 0.05
                rg[1] -= diff * 0.05
                le = pretty_breaks(rg, n=3)
            else:
                le = np.asarray(breaks, dtype=float)
            self.levels = le
            mapped = col_fun(le)
            self.colors = [to_hex_rgba(c) for c in mapped]
            self.col_fun = col_fun
            self.type = "continuous"
        else:
            raise ValueError("Initialization failed: specify `colors` (+ `levels`) or `col_fun`.")

    def map_to_colors(self, x: Any) -> Any:
        """Map scalar, 1D vector, or 2D array of values to hex RGBA colors."""
        is_scalar = np.ndim(x) == 0 and not isinstance(x, (list, tuple, np.ndarray))
        arr = np.asarray(x, dtype=object)
        orig_shape = arr.shape
        flat = arr.ravel()
        out = np.empty(len(flat), dtype=object)

        if self.type == "discrete":
            lookup = self.full_col if self.full_col else self.colors
            assert isinstance(lookup, dict)
            unmapped = []
            for idx, val in enumerate(flat):
                if _is_missing_color(val) or (isinstance(val, str) and val.strip() == ""):
                    out[idx] = self.na_col
                else:
                    sv = str(val)
                    # Handle float like 1.0 vs "1" if integer key was provided
                    if sv not in lookup and isinstance(val, (float, np.floating)) and float(val).is_integer():
                        sv_int = str(int(val))
                        if sv_int in lookup:
                            sv = sv_int
                    if sv not in lookup:
                        unmapped.append(sv)
                    else:
                        out[idx] = lookup[sv]
            if unmapped:
                uniq_bad = list(dict.fromkeys(unmapped))
                raise ValueError(f"{self.name}: cannot map colors to some levels: {', '.join(uniq_bad)}")
        else:
            assert self.col_fun is not None
            num_arr = np.empty(len(flat), dtype=float)
            for idx, val in enumerate(flat):
                if _is_missing_color(val):
                    num_arr[idx] = np.nan
                else:
                    num_arr[idx] = float(val)
            lna = np.isnan(num_arr)
            if np.any(lna):
                out[lna] = self.na_col
            if np.any(~lna):
                mapped = self.col_fun(num_arr[~lna])
                out[~lna] = [to_hex_rgba(c) for c in np.atleast_1d(mapped)]

        if is_scalar:
            return out[0]
        return out.reshape(orig_shape)

    def map_to_rgba(self, x: Any) -> np.ndarray:
        """Vectorized float RGBA array conversion in [0, 1] of shape `(*x.shape, 4)`."""
        arr = np.asarray(x)
        if self.type == "continuous" and self.col_fun is not None and hasattr(self.col_fun, "to_rgba_array"):
            if np.issubdtype(arr.dtype, np.number):
                num_arr = arr.astype(float)
            else:
                flat = arr.ravel()
                num_flat = np.empty(len(flat), dtype=float)
                for idx, val in enumerate(flat):
                    num_flat[idx] = np.nan if _is_missing_color(val) else float(val)
                num_arr = num_flat.reshape(arr.shape)
            return self.col_fun.to_rgba_array(num_arr, na_rgba=to_rgba_tuple(self.na_col))

        hex_arr = np.asarray(self.map_to_colors(x), dtype=object)
        orig_shape = hex_arr.shape
        uniq, inv = np.unique(hex_arr.ravel(), return_inverse=True)
        lut = np.array([to_rgba_tuple(hx) for hx in uniq], dtype=float)
        return lut[inv].reshape(*orig_shape, 4)

    def color_mapping_legend(
        self,
        plot: bool = True,
        color_bar: str | None = None,
        title: str | None = None,
        title_gp: Any = None,
        title_position: str = "topleft",
        grid_height: Any = None,
        grid_width: Any = None,
        tick_length: Any = None,
        border: Any = None,
        at: Sequence[Any] | None = None,
        labels: Sequence[Any] | None = None,
        labels_gp: Any = None,
        labels_rot: float = 0.0,
        nrow: int | None = None,
        ncol: int = 1,
        by_row: bool = False,
        legend_gp: Any = None,
        legend_height: Any = None,
        legend_width: Any = None,
        legend_direction: str = "vertical",
        break_dist: Sequence[float] | None = None,
        graphics: Any = None,
        param: dict[str, Any] | None = None,
        **draw_kwargs: Any,
    ) -> Any:
        from .legend import Legend
        from .units import gpar, unit

        local_params: dict[str, Any] = {
            "color_bar": self.type if color_bar is None else color_bar,
            "title": self.name if title is None else title,
            "title_gp": gpar(fontsize=10, fontface="bold") if title_gp is None else title_gp,
            "title_position": title_position,
            "grid_height": unit(4, "mm") if grid_height is None else grid_height,
            "grid_width": unit(4, "mm") if grid_width is None else grid_width,
            "tick_length": unit(0.8, "mm") if tick_length is None else tick_length,
            "border": border,
            "at": list(self.levels) if at is None else list(at),
            "labels": None if labels is None else list(labels),
            "labels_gp": gpar(fontsize=10) if labels_gp is None else labels_gp,
            "labels_rot": labels_rot,
            "nrow": nrow,
            "ncol": ncol,
            "by_row": by_row,
            "legend_gp": gpar() if legend_gp is None else gpar(**legend_gp),
            "legend_height": legend_height,
            "legend_width": legend_width,
            "legend_direction": legend_direction,
            "break_dist": break_dist,
            "graphics": graphics,
        }
        if param:
            for k, v in param.items():
                if k == "direction":
                    local_params["legend_direction"] = v
                else:
                    local_params[k] = v

        if local_params["labels"] is None:
            local_params["labels"] = list(local_params["at"])

        cb = local_params["color_bar"]
        if self.type == "discrete" and cb == "continuous":
            raise ValueError("`color_bar` cannot be 'continuous' when ColorMapping is discrete.")

        at_vals = list(local_params["at"])
        lbl_vals = list(local_params["labels"])
        if len(at_vals) != len(lbl_vals):
            raise ValueError("Length of `at` must equal length of `labels`.")

        if cb == "discrete" and self.type == "discrete":
            valid = set(self.full_col.keys()) if self.full_col else set(self.levels)
            keep = [i for i, v in enumerate(at_vals) if str(v) in valid]
            at_vals = [at_vals[i] for i in keep]
            lbl_vals = [lbl_vals[i] for i in keep]

        if cb == "discrete":
            if self.type == "continuous":
                at_vals = list(reversed(at_vals))
                lbl_vals = list(reversed(lbl_vals))
            lgp = gpar(**local_params["legend_gp"])
            if len(at_vals) == 0:
                lgp["fill"] = self.na_col
                at_vals = ["NA"]
                lbl_vals = ["NA"]
            else:
                lgp["fill"] = list(self.map_to_colors(at_vals))
            lgd = Legend(
                at=at_vals,
                labels=lbl_vals,
                name=self.name,
                title=local_params["title"],
                title_gp=local_params["title_gp"],
                grid_height=local_params["grid_height"],
                grid_width=local_params["grid_width"],
                tick_length=local_params["tick_length"],
                border=local_params["border"],
                labels_gp=local_params["labels_gp"],
                direction=local_params["legend_direction"],
                nrow=local_params["nrow"],
                ncol=local_params["ncol"],
                legend_gp=lgp,
                title_position=local_params["title_position"],
                by_row=local_params["by_row"],
                graphics=local_params["graphics"],
                break_dist=local_params["break_dist"],
            )
        else:
            lgd = Legend(
                at=at_vals,
                labels=lbl_vals,
                name=self.name,
                col_fun=self.col_fun,
                title=local_params["title"],
                title_gp=local_params["title_gp"],
                grid_height=local_params["grid_height"],
                grid_width=local_params["grid_width"],
                tick_length=local_params["tick_length"],
                border=local_params["border"],
                labels_gp=local_params["labels_gp"],
                labels_rot=local_params["labels_rot"],
                direction=local_params["legend_direction"],
                legend_gp=local_params["legend_gp"],
                legend_width=local_params["legend_width"],
                legend_height=local_params["legend_height"],
                title_position=local_params["title_position"],
                by_row=local_params["by_row"],
                break_dist=local_params["break_dist"],
            )

        if plot:
            lgd.draw(**draw_kwargs)
        return lgd

    def __repr__(self) -> str:
        if self.type == "discrete":
            return (
                f"Discrete color mapping:\nname: {self.name}\n"
                f"levels: {self.levels}\ncolors: {list(self.colors.values())}"
            )
        return (
            f"Continuous color mapping:\nname: {self.name}\n"
            f"default breaks: {list(self.levels)}\ncolors: {self.colors}"
        )


def map_to_colors(cm: ColorMapping, x: Any) -> Any:
    return cm.map_to_colors(x)


def color_mapping_legend(cm: ColorMapping, plot: bool = True, **kwargs: Any) -> Any:
    return cm.color_mapping_legend(plot=plot, **kwargs)


def c_ColorMapping(*cms: ColorMapping, name: str | None = None) -> ColorMapping:
    """Concatenate discrete ColorMapping objects, preserving first occurrence of duplicates."""
    if not cms:
        raise ValueError("At least one ColorMapping is required.")
    if not all(cm.type == "discrete" for cm in cms):
        raise ValueError("Only discrete color mappings can be concatenated.")
    merged: dict[str, str] = {}
    for cm in cms:
        assert isinstance(cm.colors, dict)
        for k, v in cm.colors.items():
            if k not in merged:
                merged[k] = v
    return ColorMapping(name=name, colors=merged, na_col=cms[0].na_col)
