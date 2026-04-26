"""Launcher for the DealSight Intelligence Gradio app.

Run with no args to start the UI; pass `--once` to run a single planning
cycle and print the resulting opportunities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealsight_intelligence.app.gradio_app import main


if __name__ == "__main__":
    main()
