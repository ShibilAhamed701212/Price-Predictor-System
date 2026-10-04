import os
import sys

# Make the repository root importable so tests can use `src.*` like the pipeline steps do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
