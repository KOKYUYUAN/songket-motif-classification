"""Songket Motif Classifier comparison entrypoint.

Run:  streamlit run songket_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("songket_comparison")), run_name="__main__")
