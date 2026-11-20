"""Parametrized verification executing all 102 ported ComplexHeatmap/man/*.Rd examples."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from run_all_rd_examples import RD_EXAMPLES


def _discover_executable_rd_names() -> list[str]:
    man_dir = Path(__file__).resolve().parents[2] / "man"
    rd_files = sorted(glob.glob(str(man_dir / "*.Rd")))
    names = []
    for p in rd_files:
        txt = open(p, encoding="utf-8", errors="replace").read()
        idx = txt.find("\\examples{")
        if idx < 0:
            continue
        i = idx + len("\\examples{")
        depth = 1
        out = []
        while i < len(txt) and depth > 0:
            c = txt[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(c)
            i += 1
        body = "".join(out).strip()
        if not body or "# There is no example" in body:
            continue
        lines = [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith("#")]
        if lines and lines != ["NULL"]:
            names.append(os.path.basename(p))
    return names


def test_all_102_rd_examples_covered():
    expected = _discover_executable_rd_names()
    assert len(expected) == 102
    assert set(RD_EXAMPLES.keys()) == set(expected)


@pytest.mark.parametrize("rd_name", sorted(RD_EXAMPLES.keys()))
def test_rd_example(rd_name: str):
    RD_EXAMPLES[rd_name]()
