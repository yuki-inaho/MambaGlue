"""Portable and optional CUDA selective-scan backends for MambaGlue."""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _mamba_selective_scan_fn,
    )
except (ImportError, ModuleNotFoundError):
    _mamba_selective_scan_fn = None


@torch.jit.script
def selective_scan_portable(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    delta_bias: torch.Tensor,
) -> torch.Tensor:
    """Reference selective scan that exports as standard ONNX operators."""
    input_dtype = u.dtype
    u_float = u.float()
    delta_float = F.softplus(delta.float() + delta_bias.float().unsqueeze(1))
    a_float = a.float()
    b_float = b.float()
    c_float = c.float()
    d_float = d.float()

    state = torch.zeros(
        (u.size(0), u.size(1), a.size(1)),
        dtype=torch.float32,
        device=u.device,
    )
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for index in range(u.size(2)):
        delta_step = delta_float[:, :, index]
        transition = torch.exp(delta_step.unsqueeze(2) * a_float.unsqueeze(0))
        input_update = (
            delta_step.unsqueeze(2)
            * b_float[:, :, index].unsqueeze(1)
            * u_float[:, :, index].unsqueeze(2)
        )
        state = transition * state + input_update
        output = (state * c_float[:, :, index].unsqueeze(1)).sum(dim=2)
        outputs.append(output)

    scanned = torch.stack(outputs, dim=2)
    scanned = scanned + u_float * d_float.view(1, -1, 1)
    return scanned.to(dtype=input_dtype)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    delta_bias: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    """Select the CUDA extension when available, otherwise use portable scan."""
    if backend not in {"auto", "portable", "mamba"}:
        raise ValueError(f"Unknown selective-scan backend: {backend!r}")

    exporting = torch.onnx.is_in_onnx_export()
    # Released mamba-ssm wheels generally target up to Ampere/Hopper and may
    # import successfully while lacking an sm_120 kernel.  On Blackwell,
    # prefer the numerically equivalent PyTorch implementation unless the
    # user explicitly opts in after building a matching extension.
    blackwell = False
    if u.device.type == "cuda":
        try:
            capability = torch.cuda.get_device_capability(u.device)
            blackwell = capability >= (12, 0)
        except RuntimeError:
            blackwell = False
    allow_blackwell_mamba = os.environ.get("MAMBAGLUE_ALLOW_BLACKWELL_MAMBA", "") in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }
    can_use_mamba = (
        _mamba_selective_scan_fn is not None
        and u.device.type == "cuda"
        and not exporting
        and (not blackwell or allow_blackwell_mamba)
    )
    if backend == "mamba" and not can_use_mamba:
        if exporting:
            reason = "ONNX export is active"
        elif blackwell and not allow_blackwell_mamba:
            reason = (
                "Blackwell sm_120 uses the portable backend by default; set "
                "MAMBAGLUE_ALLOW_BLACKWELL_MAMBA=1 only for a matching build"
            )
        else:
            reason = "CUDA mamba_ssm is unavailable"
        raise RuntimeError(f"The mamba selective-scan backend cannot be used: {reason}.")
    if backend in {"auto", "mamba"} and can_use_mamba:
        assert _mamba_selective_scan_fn is not None
        return _mamba_selective_scan_fn(
            u,
            delta,
            a,
            b,
            c,
            d.float(),
            z=None,
            delta_bias=delta_bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
    return selective_scan_portable(u, delta, a, b, c, d, delta_bias)
