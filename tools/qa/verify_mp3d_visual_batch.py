#!/usr/bin/env python3
"""CLI wrapper for AVEngine's native MP3D visual batch verifier."""

from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.mp3d_visual_verification import (  # noqa: E402,F401
    MP3DVisualVerificationError,
    main,
    verify_batch,
    verify_point,
)


if __name__ == "__main__":
    raise SystemExit(main())
