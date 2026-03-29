"""
NEXUS Trading Terminal - Streamlit Cloud Entry Point
=====================================================
This file serves as the entry point for Streamlit Cloud deployment.
It properly sets up paths and runs the main engine app.
"""

import sys
import os
from pathlib import Path

# Add engine directory to Python path
engine_dir = Path(__file__).parent / "engine"
sys.path.insert(0, str(engine_dir))

# Change to engine directory so relative imports work
os.chdir(engine_dir)

# Now import and run the app by executing it
exec(open(engine_dir / "app.py", encoding="utf-8").read())
