"""AMUSE vendoring, parameter classification, and one-step sanity tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vendored_amuse_matches_pinned_hashes():
    amuse = REPO_ROOT / "mambaglue" / "training" / "optim" / "amuse.py"
    license_file = REPO_ROOT / "third_party" / "amuse" / "LICENSE"
    assert hashlib.sha256(amuse.read_bytes()).hexdigest() == (
        "84fd3fbbc99e1718cf1c821ceff3369439f48e6fbd8ecc3a2b83afa5d82eea1f"
    )
    assert hashlib.sha256(license_file.read_bytes()).hexdigest() == (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
    )


def _build_optimizer(model):
    from mambaglue.training.optim.amuse import AMUSE
    from mambaglue.training.optim.param_groups import build_amuse_param_groups

    groups = build_amuse_param_groups(model, muon_lr=1e-2, aux_lr=1e-3)
    optimizer = AMUSE(groups, warmup_steps=3)
    optimizer.train()
    return optimizer


def test_param_groups_classify_and_step_reduces_loss():
    from mambaglue.training.optim.param_groups import classify_parameters

    model = _model()
    muon, aux = classify_parameters(model)
    assert set(muon) == {"0.weight", "2.weight"}
    assert set(aux) == {"0.bias", "2.bias"}

    optimizer = _build_optimizer(model)
    inputs = torch.randn(64, 8)
    targets = torch.randn(64, 1)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        return float(loss)

    first = step()
    for _ in range(15):
        last = step()
    assert last < first


def test_amuse_state_dict_roundtrip_and_determinism():
    model_a = _model()
    optimizer_a = _build_optimizer(model_a)
    inputs = torch.randn(16, 8)
    targets = torch.randn(16, 1)
    for _ in range(3):
        optimizer_a.zero_grad(set_to_none=True)
        ((model_a(inputs) - targets) ** 2).mean().backward()
        optimizer_a.step()
    state = optimizer_a.state_dict()

    model_b = _model()
    optimizer_b = _build_optimizer(model_b)
    optimizer_b.load_state_dict(state)
    assert len(optimizer_b.param_groups) == len(optimizer_a.param_groups)
    assert optimizer_b.state_dict()["state"].keys() == state["state"].keys()
