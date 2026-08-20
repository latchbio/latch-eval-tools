from __future__ import annotations

import importlib

import pytest

pytest.importorskip("rdkit", reason="chemistry extra is not installed")


@pytest.mark.parametrize(
    "module_name",
    [
        "matplotlib",
        "nmrglue",
        "numpy",
        "openpyxl",
        "pandas",
        "PIL",
        "pyarrow",
        "pymzml",
        "pynumpress",
        "pyteomics",
        "rdkit",
        "sklearn",
        "scipy",
        "seaborn",
        "statsmodels",
    ],
)
def test_chemistry_extra_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
