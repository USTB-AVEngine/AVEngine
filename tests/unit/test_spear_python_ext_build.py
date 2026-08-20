"""Static checks for the optional AVEngine-local SPEAR host extension."""

from __future__ import annotations

import re
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = REPOSITORY / "native/spear/python_ext"


def test_python_extension_uses_selected_supported_minor() -> None:
    cmake = (EXTENSION_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    python_find = re.search(
        r"find_package\(\s*Python\s+3\.11\s+REQUIRED\s+"
        r"COMPONENTS\s+Interpreter\s+Development\.Module\s*\)",
        cmake,
    )
    assert python_find is not None
    assert "3.11\n  EXACT\n" not in cmake


def test_python_extension_docs_keep_explicit_minor_specific_boundary() -> None:
    readme = (EXTENSION_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Python 3.11-or-newer" in readme
    assert '-DPython_EXECUTABLE="$AVENGINE_SPEAR_PYTHON"' in readme
    assert "cpython-311-x86_64-linux-gnu.so" in readme
    assert "cpython-312-x86_64-linux-gnu.so" in readme
    assert "does not claim stable-ABI or cross-minor binary compatibility" in readme
