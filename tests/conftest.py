"""Pytest configuration for ComplexHeatmap-py."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import complexheatmap as ch


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global ComplexHeatmap options and close Matplotlib figures between tests."""
    ch.ht_opt(RESET=True)
    yield
    plt.close("all")
    ch.ht_opt(RESET=True)
