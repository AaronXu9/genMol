"""Phase 1a #2: Flow Matching on QM9 — minimal subclass of BaseGenLitModule.

NOTE: this class is structurally identical to `EDMOnQM9`. The only difference
between EDM and FM is the `Process` instance passed in via Hydra config.
This is the contract that lets the EDM↔FM swap be one Hydra line.
"""
from __future__ import annotations

from genmol.models.base_lit import BaseGenLitModule


class FlowOnQM9(BaseGenLitModule):
    pass
