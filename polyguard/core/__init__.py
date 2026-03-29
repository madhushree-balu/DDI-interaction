"""
polyguard.core — public surface of the analysis engine.

File layout
-----------
The three original engine files live at the **project root** (next to the
``polyguard/`` package directory), not inside the package:

    project_root/
        nlp_engine.py
        polyguard_engine_evidence_based.py
        evidence_based_weights.py
        polyguard/
            core/          ← you are here
                __init__.py
                analyser.py
                xai_explainer.py
                ...
        datasets/
        scripts/

This ``__init__.py`` adds the project root to ``sys.path`` once, at package
import time, so that every submodule (analyser.py, xai_explainer.py, …) can
simply write ``import nlp_engine`` and it will resolve regardless of the
working directory Python was launched from.

Typical usage::

    from polyguard.core import PolyGuardAnalyser
    from polyguard.core.data_loader import DataLoader

    loader   = DataLoader("./datasets").load()
    analyser = PolyGuardAnalyser(loader)
    result   = analyser.analyse(["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"])
"""

import sys
from pathlib import Path

# __file__ = project_root/polyguard/core/__init__.py
# .parent  = project_root/polyguard/core/
# .parent  = project_root/polyguard/
# .parent  = project_root/
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from .analyser import PolyGuardAnalyser
from .data_loader import DataLoader
from .models import AnalysisResult, InteractionResult, BrandSearchResult
from . import xai_explainer

__all__ = [
    "PolyGuardAnalyser",
    "DataLoader",
    "AnalysisResult",
    "InteractionResult",
    "BrandSearchResult",
    "xai_explainer",
]