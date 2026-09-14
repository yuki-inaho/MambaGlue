"""AMUSE parameter grouping for MambaGlue.

Matrix/convolution weights (``ndim >= 2``) go to the Muon path; scalars, biases
and normalization parameters go to the AdamW fallback. Mirrors
``vggt_omega/training/optimizer_factory.py``'s classification principle without
the VGGT-specific name prefixes.
"""

from __future__ import annotations

import torch


def classify_parameters(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Return ``(muon_names, aux_names)`` for the model's trainable parameters."""
    muon: list[str] = []
    aux: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            muon.append(name)
        else:
            aux.append(name)
    return muon, aux


def build_amuse_param_groups(
    model: torch.nn.Module,
    *,
    muon_lr: float,
    aux_lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    aux_update_type: str = "adamw",
) -> list[dict]:
    """Build the AMUSE parameter groups (no implicit defaults for the LRs)."""
    if aux_update_type not in ("adamw", "sgd"):
        raise ValueError(
            f"aux_update_type must be 'adamw' or 'sgd', got {aux_update_type!r}"
        )

    muon_names, aux_names = classify_parameters(model)
    named = dict(model.named_parameters())
    groups: list[dict] = []
    if muon_names:
        groups.append(
            {
                "params": [named[name] for name in muon_names],
                "lr": muon_lr,
                "use_muon": True,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "aux_update_type": aux_update_type,
            }
        )
    if aux_names:
        groups.append(
            {
                "params": [named[name] for name in aux_names],
                "lr": aux_lr,
                "use_muon": False,
                "update_type": aux_update_type,
                "weight_decay": weight_decay,
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups
