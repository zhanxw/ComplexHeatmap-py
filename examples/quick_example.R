# R ComplexHeatmap counterpart to ComplexHeatmap-py's Quick Example (README.md)
# Upstream ComplexHeatmap source ported in this repo: v2.25.3 (DESCRIPTION)
# Tested locally with R ComplexHeatmap v2.24.1+
#
# Note: In upstream R ComplexHeatmap, `adjust_heatmap_list()` only accounts for
# outer edge label extensions (first/last heatmap outer margins) and keeps the
# interior `ht_gap` at 2 mm by default. As a result, `draw(ht1 + ht2)` in R
# also draws "Group" and "Score" across `ht2`'s column dendrogram unless
# `ht_gap = unit(16, "mm")` (or `annotation_name_side = "left"`) is passed.
# ComplexHeatmap-py automatically widens the interior `ht_gap` when
# `auto_adjust = TRUE` (unless `ht_gap` / `gap` is explicitly specified).

suppressPackageStartupMessages({
    library(ComplexHeatmap)
    library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else tempdir()
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("ComplexHeatmap R package version: %s\n", as.character(packageVersion("ComplexHeatmap"))))

set.seed(123)
mat <- matrix(
    rnorm(12 * 10),
    nrow = 12,
    ncol = 10,
    dimnames = list(paste0("gene_", 0:11), paste0("sample_", 0:9))
)

ha_top <- HeatmapAnnotation(
    Group = c(rep("Ctrl", 5), rep("Treat", 5)),
    Score = anno_barplot(runif(10, 0, 1)),
    col = list(Group = c("Ctrl" = "#4DAF4A", "Treat" = "#E41A1C"))
)

ht1 <- Heatmap(mat, name = "Expr", row_km = 2, top_annotation = ha_top, border = TRUE)
ht2 <- Heatmap(mat[, 1:4] * 2, name = "Delta")

# 1. Default R draw(ht1 + ht2): uses default ht_gap = unit(2, "mm")
f_default <- file.path(out_dir, "r_quick_example_default.png")
png(f_default, width = 9.0, height = 6.5, units = "in", res = 150)
draw(ht1 + ht2)
decorate_heatmap_body("Expr", row_slice = 1, column_slice = 1, {
    grid.rect(gp = gpar(fill = "transparent", col = "gold", lwd = 2))
})
dev.off()

# 2. Explicit ht_gap = unit(16, "mm") clears "Group" and "Score" from Delta
f_gap <- file.path(out_dir, "r_quick_example_ht_gap.png")
png(f_gap, width = 9.0, height = 6.5, units = "in", res = 150)
draw(ht1 + ht2, ht_gap = unit(16, "mm"))
decorate_heatmap_body("Expr", row_slice = 1, column_slice = 1, {
    grid.rect(gp = gpar(fill = "transparent", col = "gold", lwd = 2))
})
dev.off()

cat(sprintf("Wrote:\n  %s\n  %s\n", f_default, f_gap))
