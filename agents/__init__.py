"""
agents/__init__.py
------------------
Makes `agents/` importable as a package from the project root.

The modules inside agents/ (agent.py, scorer.py, etc.) were originally designed
to be run stand-alone from inside the agents/ directory and therefore use bare
imports (e.g. `from medical_dataset import …`).  Adding the directory to
sys.path here ensures those imports work when the package is imported from the
project root (e.g. by `backend/medicalDiagnosisAgent.py`).
"""
import sys
import os

_agents_dir = os.path.dirname(__file__)
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)
