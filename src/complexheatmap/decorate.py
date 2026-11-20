"""Decoration functions matching R ComplexHeatmap::decorate_*."""

from __future__ import annotations

from typing import Any, Callable

from .grid import current_device, seekViewport


class _DecoratorContext:
    """Context manager, decorator, or eager runner for named viewport decoration."""

    def __init__(self, vp_name: str) -> None:
        self.vp_name = vp_name
        self._saved_stack: list[Any] | None = None

    def __enter__(self) -> "_DecoratorContext":
        dev = current_device(create=False)
        self._saved_stack = list(dev.vp_stack)
        seekViewport(self.vp_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._saved_stack is not None:
            dev = current_device(create=False)
            dev.vp_stack = self._saved_stack
            self._saved_stack = None

    def __call__(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        with self:
            fn()
        return fn


def _run_or_context(vp_name: str, code: Callable[[], Any] | None) -> _DecoratorContext | None:
    ctx = _DecoratorContext(vp_name)
    if code is not None:
        if not callable(code):
            raise TypeError("`code` must be a callable function/lambda or None (for `with` / `@` usage).")
        with ctx:
            code()
        return None
    return ctx


def decorate_heatmap_body(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
    row_slice: int | None = None,
    column_slice: int = 1,
) -> _DecoratorContext | None:
    """Seek a heatmap body slice viewport `{heatmap}_heatmap_body_{row_slice}_{column_slice}`."""
    rs = slice if row_slice is None else row_slice
    vp_name = f"{heatmap}_heatmap_body_{rs}_{column_slice}"
    return _run_or_context(vp_name, code)


def decorate_dend(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
    which: str = "column",
) -> _DecoratorContext | None:
    """Seek a dendrogram slice viewport `{heatmap}_dend_{which}_{slice}`."""
    if which not in ("column", "row"):
        raise ValueError("`which` must be 'column' or 'row'.")
    vp_name = f"{heatmap}_dend_{which}_{slice}"
    return _run_or_context(vp_name, code)


def decorate_column_dend(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
) -> _DecoratorContext | None:
    """Seek a column dendrogram slice viewport."""
    return decorate_dend(heatmap=heatmap, code=code, slice=slice, which="column")


def decorate_row_dend(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
) -> _DecoratorContext | None:
    """Seek a row dendrogram slice viewport."""
    return decorate_dend(heatmap=heatmap, code=code, slice=slice, which="row")


def decorate_dimnames(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
    which: str = "column",
) -> _DecoratorContext | None:
    """Seek a dimension-names slice viewport `{heatmap}_{which}_names_{slice}`."""
    if which not in ("column", "row"):
        raise ValueError("`which` must be 'column' or 'row'.")
    vp_name = f"{heatmap}_{which}_names_{slice}"
    return _run_or_context(vp_name, code)


def decorate_row_names(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
) -> _DecoratorContext | None:
    """Seek a row-names slice viewport."""
    return decorate_dimnames(heatmap=heatmap, code=code, slice=slice, which="row")


def decorate_column_names(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
) -> _DecoratorContext | None:
    """Seek a column-names slice viewport."""
    return decorate_dimnames(heatmap=heatmap, code=code, slice=slice, which="column")


def decorate_title(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
    which: str = "column",
) -> _DecoratorContext | None:
    """Seek a title slice viewport `{heatmap}_{which}_title_{slice}`."""
    if which not in ("column", "row"):
        raise ValueError("`which` must be 'column' or 'row'.")
    vp_name = f"{heatmap}_{which}_title_{slice}"
    return _run_or_context(vp_name, code)


def decorate_row_title(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
) -> _DecoratorContext | None:
    """Seek a row-title slice viewport."""
    return decorate_title(heatmap=heatmap, code=code, slice=slice, which="row")


def decorate_column_title(
    heatmap: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
) -> _DecoratorContext | None:
    """Seek a column-title slice viewport."""
    return decorate_title(heatmap=heatmap, code=code, slice=slice, which="column")


def decorate_annotation(
    annotation: str,
    code: Callable[[], Any] | None = None,
    slice: int = 1,
) -> _DecoratorContext | None:
    """Seek an annotation slice viewport `annotation_{annotation}_{slice}`."""
    vp_name = f"annotation_{annotation}_{slice}"
    return _run_or_context(vp_name, code)
