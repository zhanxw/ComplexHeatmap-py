"""Base AdditiveUnit class and horizontal (`+`) / vertical (`%v%`, `/`) concatenation hooks."""

from __future__ import annotations

from typing import Any


class _VerticalInfix:
    """Infix sentinel enabling R-like `ht1 %v% ht2` vertical heatmap concatenation."""

    def __rmod__(self, left: Any) -> "_PendingVertical":
        return _PendingVertical(left)

    def __repr__(self) -> str:
        return "<ComplexHeatmap vertical operator %v%>"


class _PendingVertical:
    def __init__(self, left: Any) -> None:
        self.left = left

    def __mod__(self, right: Any) -> Any:
        from .heatmap_list import add_heatmap
        return add_heatmap(self.left, right, direction="vertical")


v = _VerticalInfix()


class AdditiveUnit:
    """Superclass for Heatmap, HeatmapAnnotation, and HeatmapList supporting `+`, `%v%`, `/`."""

    def __add__(self, other: Any) -> Any:
        from .heatmap_list import add_heatmap
        return add_heatmap(self, other, direction="horizontal")

    def __radd__(self, other: Any) -> Any:
        if other is None:
            from .heatmap_list import add_heatmap
            return add_heatmap(None, self, direction="horizontal")
        return NotImplemented

    def __truediv__(self, other: Any) -> Any:
        from .heatmap_list import add_heatmap
        return add_heatmap(self, other, direction="vertical")

    def __rtruediv__(self, other: Any) -> Any:
        if other is None:
            from .heatmap_list import add_heatmap
            return add_heatmap(None, self, direction="vertical")
        return NotImplemented

    def __mod__(self, other: Any) -> Any:
        if isinstance(other, _VerticalInfix):
            return _PendingVertical(self)
        return NotImplemented
