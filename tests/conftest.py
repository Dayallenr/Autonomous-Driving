"""Shared fixtures.

The CIL checkpoint is session-scoped because serialising a ResNet-18 state
dict is seconds and ~45 MB of disk; one untrained checkpoint serves every test
that needs real loadable weights, and untrained is the point — issue #21
requires the whole comparison mechanism to be provable before any training
has happened.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def cil_checkpoint(tmp_path_factory):
    """An untrained control-output CIL checkpoint in the DAgger CLI's format."""
    # Imported inside the fixture so collecting the suite stays torch-free —
    # the same isolation the registry itself keeps.
    import torch

    from pathfinder.planning.cil_model import CILModel

    torch.manual_seed(0)
    model = CILModel(pretrained=False, output_mode="control")
    path = tmp_path_factory.mktemp("cil") / "iteration_000.pt"
    torch.save({"model": model.state_dict(), "optimizer": {}, "row": {}}, path)
    return path
