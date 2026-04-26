"""Hugging Face Spaces entrypoint for DealSight Intelligence."""

import os
import sys
from pathlib import Path

# Keep the public demo safe by default. Live alerts and paid API calls should
# only be enabled through Hugging Face Space secrets.
os.environ.setdefault("DEALSIGHT_INTELLIGENCE_DRY_RUN", "true")
os.environ.setdefault("DEALSIGHT_INTELLIGENCE_DO_PUSH", "false")
os.environ.setdefault("DEALSIGHT_INTELLIGENCE_ENABLE_MODAL", "false")
os.environ.setdefault("DEALSIGHT_INTELLIGENCE_ENABLE_DNN", "false")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dealsight_intelligence.app.gradio_app import App

space_app = App()
demo = space_app.build()

if __name__ == "__main__":
    space_app.launch(demo)
