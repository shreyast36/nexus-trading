"""
NEXUS Trading Terminal - Streamlit Cloud Entry Point
=====================================================
This file serves as the entry point for Streamlit Cloud deployment.
It properly sets up paths and imports the main engine app.
"""

import sys
import os
from pathlib import Path

# Get the directory where this script is located
ROOT_DIR = Path(__file__).parent.resolve()
ENGINE_DIR = ROOT_DIR / "engine"

# Add engine directory to Python path
sys.path.insert(0, str(ENGINE_DIR))

# Change working directory to engine so relative paths work (e.g., Data/)
os.chdir(ENGINE_DIR)

# Import and run the app module
import runpy
runpy.run_path(str(ENGINE_DIR / "app.py"), run_name="__main__")
