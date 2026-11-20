"""Global configuration options (`ht_opt`) and shared internal runtime state."""

from __future__ import annotations

from contextlib import contextmanager
import itertools
from typing import Any, Iterator

from .units import Gpar, Unit, check_gp, unit


_DEFAULTS: dict[str, Any] = {
    "heatmap_row_names_gp": None,
    "heatmap_column_names_gp": None,
    "heatmap_row_title_gp": None,
    "heatmap_column_title_gp": None,
    "legend_title_gp": None,
    "legend_title_position": None,
    "legend_labels_gp": None,
    "legend_grid_height": None,
    "legend_grid_width": None,
    "legend_border": None,
    "legend_gap": unit([4.0, 4.0], "mm"),
    "merge_legends": False,
    "heatmap_border": None,
    "annotation_border": None,
    "fast_hclust": False,
    "show_parent_dend_line": True,
    "verbose": False,
    "message": True,
    "show_vp": False,
    "simple_anno_size": unit(5.0, "mm"),
    "DENDROGRAM_PADDING": unit(0.5, "mm"),
    "DIMNAME_PADDING": unit(1.0, "mm"),
    "TITLE_PADDING": None,
    "COLUMN_ANNO_PADDING": unit(1.0, "mm"),
    "ROW_ANNO_PADDING": unit(1.0, "mm"),
    "HEATMAP_LEGEND_PADDING": unit(2.0, "mm"),
    "ANNOTATION_LEGEND_PADDING": unit(2.0, "mm"),
    "save_last": False,
    "validate_names": True,
    "raster_temp_image_max_width": 30000,
    "raster_temp_image_max_height": 30000,
    "COLOR": ["blue", "#EEEEEE", "red"],
}

GLOBAL_PADDING = unit([5.5, 5.5, 5.5, 5.5], "points")


def _filter_option(key: str, val: Any) -> Any:
    if key not in _DEFAULTS:
        raise KeyError(f"Unknown ComplexHeatmap option: {key!r}")
    if val is None:
        return None
    if key == "legend_gap":
        u = val if isinstance(val, Unit) else unit(val, "mm")
        if len(u) == 1:
            return unit([u.to_mm_scalar("x"), u.to_mm_scalar("x")], "mm")
        return u[:2]
    if key == "TITLE_PADDING":
        u = val if isinstance(val, Unit) else unit(val, "mm")
        if len(u) == 1:
            return unit([u.to_mm_scalar("y"), u.to_mm_scalar("y")], "mm")
        return u[:2]
    if key.endswith("_gp"):
        return check_gp(val)
    if key in ("simple_anno_size", "legend_grid_height", "legend_grid_width") or key.endswith("_PADDING"):
        return val if isinstance(val, Unit) else unit(val, "mm")
    if key in ("verbose", "message", "fast_hclust", "merge_legends", "show_parent_dend_line", "show_vp", "save_last", "validate_names"):
        return bool(val)
    return val


class _OptionsManager:
    """Global options container supporting attribute, dict, and call syntax like R `ht_opt`."""

    def __init__(self) -> None:
        object.__setattr__(self, "_opts", dict(_DEFAULTS))

    def reset(self) -> None:
        self._opts.clear()
        self._opts.update(_DEFAULTS)

    def __call__(self, *args: str, RESET: bool = False, **kwargs: Any) -> Any:
        if RESET:
            self.reset()
            return None
        if args and not kwargs:
            if len(args) == 1:
                return self._opts[args[0]]
            return {k: self._opts[k] for k in args}
        if kwargs:
            old = {k: self._opts[k] for k in kwargs if k in self._opts}
            for k, v in kwargs.items():
                self._opts[k] = _filter_option(k, v)
            return old
        return dict(self._opts)

    def __getattr__(self, name: str) -> Any:
        opts = object.__getattribute__(self, "_opts")
        if name in opts:
            return opts[name]
        raise AttributeError(f"No option named {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        self._opts[name] = _filter_option(name, value)

    def __getitem__(self, name: str) -> Any:
        return self._opts[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._opts[name] = _filter_option(name, value)

    def get(self, name: str, default: Any = None) -> Any:
        return self._opts.get(name, default)

    def keys(self):
        return self._opts.keys()

    def items(self):
        return self._opts.items()

    @contextmanager
    def local(self, **overrides: Any) -> Iterator[None]:
        saved: dict[str, Any] = {}
        for k, v in overrides.items():
            if v is not None and k in self._opts:
                saved[k] = self._opts[k]
                self._opts[k] = _filter_option(k, v)
        try:
            yield
        finally:
            for k, v in saved.items():
                self._opts[k] = v

    def __repr__(self) -> str:
        lines = [" Option                        Value", " -----------------------------:--------------------------------"]
        for k, v in self._opts.items():
            lines.append(f" {k:<29} {v!r}")
        return "\n".join(lines)


ht_opt = _OptionsManager()
ht_global_opt = ht_opt


class _RuntimeEnv:
    def __init__(self) -> None:
        self.current_annotation_which: str | None = None
        self.current_heatmap: Any = None
        self.last: Any = None
        self.heatmap_counter = itertools.count(1)
        self.anno_counter = itertools.count(1)
        self.oncoprint_counter = itertools.count(1)
        self.chr_window: Any = None

    def next_heatmap_index(self) -> int:
        return next(self.heatmap_counter)


_ENV = _RuntimeEnv()


def get_last_ht() -> Any:
    return _ENV.last
