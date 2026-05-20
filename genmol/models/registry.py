"""Optional registry — Hydra `_target_` handles this for us in practice."""
from __future__ import annotations

from genmol.models.base_lit import BaseGenLitModule
from genmol.models.edm_qm9 import EDMOnQM9
from genmol.models.flow_qm9 import FlowOnQM9

REGISTRY = {
    "edm_qm9": EDMOnQM9,
    "flow_qm9": FlowOnQM9,
    "base": BaseGenLitModule,
}
