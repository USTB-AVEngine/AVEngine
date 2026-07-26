"""Linux atomic no-replace publication helpers for formal artifact bundles."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path


AT_FDCWD = -100
RENAME_NOREPLACE = 1


def atomic_publish_directory(staging: Path, output: Path) -> Path:
    """Atomically rename a complete directory while refusing replacement."""

    if not staging.is_dir():
        raise FileNotFoundError(staging)
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "atomic no-replace directory publication requires renameat2"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(staging),
        AT_FDCWD,
        os.fsencode(output),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                f"refusing to replace output: {output}",
                str(output),
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(output),
        )
    return output


__all__ = ["atomic_publish_directory"]
