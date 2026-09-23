#!/usr/bin/env python3
"""Run stdlib-only historical return analysis without importing torch."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2]/'src/topoformer/return_diagnostics.py'),run_name='__main__')
