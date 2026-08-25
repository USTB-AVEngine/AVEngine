#!/usr/bin/env python3
"""CLI entry point for the exact 48-point QA held-out selector."""

from __future__ import annotations

import sys

from avengine.qa.heldout_selector import main


if __name__ == "__main__":
    sys.exit(main())
