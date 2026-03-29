"""
Research Agent Pro (HITL Edition)
==================================
Entry point — run this file to start the agent.

Usage
-----
    python main.py

Or via uv:
    uv run main.py
"""

import sys
import os

# Ensure the project root is on the path regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.interface import main

if __name__ == "__main__":
    main()
