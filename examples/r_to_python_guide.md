# R `ComplexHeatmap` to Python `complexheatmap` Migration Guide

`ComplexHeatmap-py` mirrors the R `ComplexHeatmap` v2.25.3 API closely, enabling near-1:1 line-by-line translation.

## 1. Imports

| R | Python |
|---|---|
| `library(ComplexHeatmap)` | `import complexheatmap as ch` (or `import ComplexHeatmap as ch`) |
| `library(circlize)` | `from complexheatmap import colorRamp2, rand_color` |
| `library(grid)` | `from complexheatmap import grid, unit, gpar, viewport, pushViewport, popViewport` |

## 2. Syntax Differences

1. **0-based Indexing**: Python uses `0..n-1` indices everywhere (`row_order`, `column_order`, `cell_fun(j, i, ...)`, `anno_mark(at=...)`, `pindex`, slicing `ht[0:5, 0:5]`).
2. **Dots Become Underscores**:
   - `grid.rect(...)` -> `grid.rect(...)` OR `grid_rect(...)` (the `grid` namespace preserves `grid.rect`, `grid.text`, `grid.newpage`, etc. directly!).
   - `default.units = "native"` -> `default_units="native"` (also accepts `**{"default.units": "native"}` on grid primitives).
   - `unit.c(...)` -> `unit_c(...)` or `c(...)`.
   - `heatmap.2(...)` -> `heatmap_2(...)` / `compare_heatmap_2(...)`.
3. **Concatenation Operators**:
   - Horizontal concatenation: `ht1 + ht2 + rowAnnotation(...)` (exact match!).
   - Vertical concatenation: `ht1 %v% ht2 %v% columnAnnotation(...)` (`%v%` works literally via `from complexheatmap import v`!) OR Pythonic `ht1 / ht2`.
4. **Legends, Splitting & Booleans**:
   - `TRUE` / `FALSE` / `NULL` / `NA` -> `True` / `False` / `None` / `np.nan` (or `None` in `col`/`pch`).
   - Named vectors `c(A = "red", B = "blue")` -> Python dicts `{"A": "red", "B": "blue"}`.
   - `list(...)` -> `{...}` for named lists (`heatmap_legend_param={...}`) or `[...]` for positional lists.
   - R `@` S4 slot access (e.g., `lgd@grob`) -> dot attribute (`lgd.grob`).
5. **Post-Draw Decorations (`decorate_*`)**:
   Use either a `with` context manager (cleanest match to R `{ ... }` blocks) or a lambda:
   ```python
   # R: decorate_heatmap_body("mat", { grid.circle(gp = gpar(fill = "#FF000080")) })
   with decorate_heatmap_body("mat"):
       grid.circle(gp=gpar(fill="#FF000080"))

   # Or callable form:
   decorate_heatmap_body("mat", lambda: grid.circle(gp=gpar(fill="#FF000080")))
   ```
6. **Subsetting**:
   R 1-based `ht[1:5, ]` becomes 0-based `ht[:5, :]`, `ha[:, ["foo", "bar"]]`, `ht_list[:5, "rnorm"]`.
7. **Saving Figures**:
   `draw(ht, save="out.png", width=8, height=6, dpi=150)` or `ht.draw(save="out.png")`.
