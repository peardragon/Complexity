from __future__ import annotations

import sys
from pathlib import Path


if __package__:
    from .utils.pm_sais_core import *  # noqa: F401,F403
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.pm_sais_core import *  # type: ignore  # noqa: F401,F403
