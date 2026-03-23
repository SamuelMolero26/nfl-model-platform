"""ML package.

Provides a lowercase module alias for the Windows-cased team diagnostic file:
    import ml.team_diagnostic
"""

import sys
from importlib import import_module

team_diagnostic = import_module(".Team_diagnostic", __name__)
sys.modules[f"{__name__}.team_diagnostic"] = team_diagnostic

__all__ = ["team_diagnostic"]
