"""Fixed-keypoint ONNX export utilities for MambaGlue.

The exported model contains the MambaGlue matcher only.  Feature extraction
(SuperPoint, DISK, ALIKED, SIFT, or DoGHardNet) remains outside the graph.  Both
keypoint axes are fixed at export time, while the batch axis can remain dynamic.
"""

from __future__ import annotations

import argparse
import inspect
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .mambaglue import MambaGlue, filter_matches, normalize_keypoints


def _masked_log_assignment(
    assignment: nn.Module,
    desc0: torch.Tensor,
    desc1: torch.Tensor,
    valid0: torch.Tensor,
    valid1: torch.Tensor,
) -> torch.Tensor:
    """Build a log-assignment matrix while excluding padded keypoints."""

    mdesc0 = assignment.final_proj(desc0)
    mdesc1 = assignment.final_proj(desc1)
    dim = mdesc0.shape[2]
    mdesc0 = mdesc0 / dim**0.25
    mdesc1 = mdesc1 / dim**0.25
    sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)

    z0 = assignment.matchability(desc0)
    z1 = assignment.matchability(desc1)
    valid_pair = valid0.unsqueeze(2) & valid1.unsqueeze(1)
    negative = sim.new_tensor(-1.0e4)

    row_sim = torch.where(valid1.unsqueeze(1), sim, negative)
    column_sim = torch.where(valid0.unsqueeze(2), sim, negative)
    scores0 = F.log_softmax(row_sim, dim=2)
    scores1 = F.log_softmax(
        column_sim.transpose(1, 2).contiguous(), dim=2
    ).transpose(1, 2)
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    core = torch.where(valid_pair, scores0 + scores1 + certainties, negative)

    dustbin0 = F.logsigmoid(-z0.squeeze(2))
    dustbin1 = F.logsigmoid(-z1.squeeze(2))
    dustbin0 = torch.where(valid0, dustbin0, torch.zeros_like(dustbin0))
    dustbin1 = torch.where(valid1, dustbin1, torch.zeros_like(dustbin1))

    top = torch.cat((core, dustbin0.unsqueeze(2)), dim=2)
    corner = torch.zeros_like(dustbin1[:, :1]).unsqueeze(1)
    bottom = torch.cat((dustbin1.unsqueeze(1), corner), dim=2)
    return torch.cat((top, bottom), dim=1)


class _FixedKeypointMambaGlueBase(nn.Module):
    """Shared implementation for fixed-keypoint ONNX wrappers."""

    def __init__(self, matcher: MambaGlue) -> None:
        super().__init__()
        self.matcher = matcher

    @staticmethod
    def _mask_inputs(
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = valid.to(dtype=torch.bool)
        valid_float = valid.unsqueeze(2).to(dtype=descriptors.dtype)
        keypoints = keypoints * valid_float.to(dtype=keypoints.dtype)
        descriptors = descriptors * valid_float
        return keypoints, descriptors, valid

    def _masked_assignment(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> torch.Tensor:
        """Build the assignment matrix without padded points in softmax terms."""

        return _masked_log_assignment(
            self.matcher.log_assignment[-1], desc0, desc1, valid0, valid1
        )

    def _forward_impl(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        valid0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
        valid1: torch.Tensor,
        positional0: torch.Tensor,
        positional1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keypoints0, descriptors0, valid0 = self._mask_inputs(
            keypoints0, descriptors0, valid0
        )
        keypoints1, descriptors1, valid1 = self._mask_inputs(
            keypoints1, descriptors1, valid1
        )

        normalized0 = normalize_keypoints(keypoints0, image_size0)
        normalized1 = normalize_keypoints(keypoints1, image_size1)
        normalized0 = torch.cat((normalized0, positional0), dim=2)
        normalized1 = torch.cat((normalized1, positional1), dim=2)

        desc0 = self.matcher.input_proj(descriptors0)
        desc1 = self.matcher.input_proj(descriptors1)
        encoding0 = self.matcher.posenc(normalized0)
        encoding1 = self.matcher.posenc(normalized1)
        mask0 = valid0.unsqueeze(2)
        mask1 = valid1.unsqueeze(2)
        valid0_float = valid0.unsqueeze(2).to(dtype=desc0.dtype)
        valid1_float = valid1.unsqueeze(2).to(dtype=desc1.dtype)

        for layer in self.matcher.transformermambas:
            desc0, desc1 = layer(
                desc0,
                desc1,
                encoding0,
                encoding1,
                mask0=mask0,
                mask1=mask1,
            )
            # Padded tokens must stay zero because the Mamba convolution and
            # recurrence operate on the full fixed sequence.
            desc0 = desc0 * valid0_float
            desc1 = desc1 * valid1_float

        scores = self._masked_assignment(desc0, desc1, valid0, valid1)
        matches0, matches1, matching_scores0, matching_scores1 = filter_matches(
            scores, self.matcher.conf.filter_threshold
        )
        matches0 = torch.where(valid0, matches0, -torch.ones_like(matches0))
        matches1 = torch.where(valid1, matches1, -torch.ones_like(matches1))
        matching_scores0 = torch.where(
            valid0, matching_scores0, torch.zeros_like(matching_scores0)
        )
        matching_scores1 = torch.where(
            valid1, matching_scores1, torch.zeros_like(matching_scores1)
        )
        return matches0, matching_scores0, matches1, matching_scores1


class FixedKeypointMambaGlue(_FixedKeypointMambaGlueBase):
    """ONNX wrapper for feature types without scale/orientation inputs."""

    def __init__(self, matcher: MambaGlue) -> None:
        if matcher.conf.add_scale_ori:
            raise ValueError(
                "This matcher expects scale/orientation inputs; use "
                "FixedKeypointMambaGlueScaleOrientation instead."
            )
        super().__init__(matcher)

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        valid0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        empty0 = keypoints0.new_zeros((*keypoints0.shape[:-1], 0))
        empty1 = keypoints1.new_zeros((*keypoints1.shape[:-1], 0))
        return self._forward_impl(
            keypoints0,
            descriptors0,
            image_size0,
            valid0,
            keypoints1,
            descriptors1,
            image_size1,
            valid1,
            empty0,
            empty1,
        )


class FixedKeypointMambaGlueScaleOrientation(_FixedKeypointMambaGlueBase):
    """ONNX wrapper for SIFT/DoGHardNet scale and orientation inputs."""

    def __init__(self, matcher: MambaGlue) -> None:
        if not matcher.conf.add_scale_ori:
            raise ValueError("The matcher was not configured with add_scale_ori=True.")
        super().__init__(matcher)

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        valid0: torch.Tensor,
        scales0: torch.Tensor,
        orientations0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
        valid1: torch.Tensor,
        scales1: torch.Tensor,
        orientations1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid0_float = valid0.unsqueeze(2).to(dtype=keypoints0.dtype)
        valid1_float = valid1.unsqueeze(2).to(dtype=keypoints1.dtype)
        positional0 = torch.cat(
            (scales0.unsqueeze(2), orientations0.unsqueeze(2)), dim=2
        ) * valid0_float
        positional1 = torch.cat(
            (scales1.unsqueeze(2), orientations1.unsqueeze(2)), dim=2
        ) * valid1_float
        return self._forward_impl(
            keypoints0,
            descriptors0,
            image_size0,
            valid0,
            keypoints1,
            descriptors1,
            image_size1,
            valid1,
            positional0,
            positional1,
        )


class _FixedKeypointPreprocessorBase(nn.Module):
    """Matcher input projection and positional encoding for modular export."""

    def __init__(self, matcher: MambaGlue) -> None:
        super().__init__()
        self.input_proj = matcher.input_proj
        self.posenc = matcher.posenc

    @staticmethod
    def _mask_inputs(
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = valid.to(dtype=torch.bool)
        valid_float = valid.unsqueeze(2).to(dtype=descriptors.dtype)
        return (
            keypoints * valid_float.to(dtype=keypoints.dtype),
            descriptors * valid_float,
            valid,
        )

    def _forward_impl(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        valid0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
        valid1: torch.Tensor,
        positional0: torch.Tensor,
        positional1: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        keypoints0, descriptors0, valid0 = self._mask_inputs(
            keypoints0, descriptors0, valid0
        )
        keypoints1, descriptors1, valid1 = self._mask_inputs(
            keypoints1, descriptors1, valid1
        )
        normalized0 = torch.cat(
            (normalize_keypoints(keypoints0, image_size0), positional0), dim=2
        )
        normalized1 = torch.cat(
            (normalize_keypoints(keypoints1, image_size1), positional1), dim=2
        )
        hidden0 = self.input_proj(descriptors0)
        hidden1 = self.input_proj(descriptors1)
        return (
            hidden0,
            hidden1,
            self.posenc(normalized0),
            self.posenc(normalized1),
            valid0,
            valid1,
        )


class _FixedKeypointPreprocessor(_FixedKeypointPreprocessorBase):
    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        valid0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        positional0 = keypoints0.new_zeros((*keypoints0.shape[:-1], 0))
        positional1 = keypoints1.new_zeros((*keypoints1.shape[:-1], 0))
        return self._forward_impl(
            keypoints0,
            descriptors0,
            image_size0,
            valid0,
            keypoints1,
            descriptors1,
            image_size1,
            valid1,
            positional0,
            positional1,
        )


class _FixedKeypointPreprocessorScaleOrientation(
    _FixedKeypointPreprocessorBase
):
    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        valid0: torch.Tensor,
        scales0: torch.Tensor,
        orientations0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
        valid1: torch.Tensor,
        scales1: torch.Tensor,
        orientations1: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        valid0_float = valid0.unsqueeze(2).to(dtype=keypoints0.dtype)
        valid1_float = valid1.unsqueeze(2).to(dtype=keypoints1.dtype)
        positional0 = torch.cat(
            (scales0.unsqueeze(2), orientations0.unsqueeze(2)), dim=2
        ) * valid0_float
        positional1 = torch.cat(
            (scales1.unsqueeze(2), orientations1.unsqueeze(2)), dim=2
        ) * valid1_float
        return self._forward_impl(
            keypoints0,
            descriptors0,
            image_size0,
            valid0,
            keypoints1,
            descriptors1,
            image_size1,
            valid1,
            positional0,
            positional1,
        )


class _FixedKeypointLayer(nn.Module):
    """One matcher layer with fixed-length padding reset and pass-through state."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden0: torch.Tensor,
        hidden1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        updated0, updated1 = self.layer(
            hidden0,
            hidden1,
            encoding0,
            encoding1,
            mask0=valid0.unsqueeze(2),
            mask1=valid1.unsqueeze(2),
        )
        updated0 = updated0 * valid0.unsqueeze(2).to(dtype=updated0.dtype)
        updated1 = updated1 * valid1.unsqueeze(2).to(dtype=updated1.dtype)
        return updated0, updated1, encoding0, encoding1, valid0, valid1


class _FixedKeypointPostprocessor(nn.Module):
    """Final assignment and dense match filtering for modular export."""

    def __init__(self, matcher: MambaGlue) -> None:
        super().__init__()
        self.assignment = matcher.log_assignment[-1]
        self.filter_threshold = float(matcher.conf.filter_threshold)

    def forward(
        self,
        hidden0: torch.Tensor,
        hidden1: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = _masked_log_assignment(
            self.assignment, hidden0, hidden1, valid0, valid1
        )
        matches0, matches1, matching_scores0, matching_scores1 = filter_matches(
            scores, self.filter_threshold
        )
        matches0 = torch.where(valid0, matches0, -torch.ones_like(matches0))
        matches1 = torch.where(valid1, matches1, -torch.ones_like(matches1))
        matching_scores0 = torch.where(
            valid0, matching_scores0, torch.zeros_like(matching_scores0)
        )
        matching_scores1 = torch.where(
            valid1, matching_scores1, torch.zeros_like(matching_scores1)
        )
        return matches0, matching_scores0, matches1, matching_scores1


def pad_features_to_fixed(
    features: Mapping[str, torch.Tensor], num_keypoints: int
) -> dict[str, torch.Tensor]:
    """Truncate or zero-pad extracted features to an ONNX model's fixed length.

    Valid entries are packed at the beginning of the sequence.  The returned
    ``valid`` mask must be passed to the ONNX model.  Keypoints beyond
    ``num_keypoints`` are truncated in their existing order, which is normally
    score order for the bundled extractors.
    """

    if num_keypoints <= 0:
        raise ValueError("num_keypoints must be positive.")
    for key in ("keypoints", "descriptors", "image_size"):
        if key not in features:
            raise KeyError(f"Missing feature tensor: {key}")

    keypoints = features["keypoints"]
    descriptors = features["descriptors"]
    if keypoints.ndim != 3 or descriptors.ndim != 3:
        raise ValueError("keypoints and descriptors must be batched rank-3 tensors.")
    if keypoints.shape[:2] != descriptors.shape[:2]:
        raise ValueError("keypoints and descriptors must share batch/keypoint axes.")

    count = min(keypoints.shape[1], num_keypoints)
    pad = num_keypoints - count
    fixed: dict[str, torch.Tensor] = {
        "keypoints": F.pad(keypoints[:, :count], (0, 0, 0, pad)),
        "descriptors": F.pad(descriptors[:, :count], (0, 0, 0, pad)),
        "image_size": features["image_size"],
        "valid": torch.arange(num_keypoints, device=keypoints.device)
        .unsqueeze(0)
        .expand(keypoints.shape[0], -1)
        < count,
    }
    for source, target in (("scales", "scales"), ("oris", "orientations")):
        if source in features:
            fixed[target] = F.pad(features[source][:, :count], (0, pad))
    return fixed


def _input_spec(
    matcher: MambaGlue,
) -> tuple[list[str], list[str]]:
    common0 = ["keypoints0", "descriptors0", "image_size0", "valid0"]
    common1 = ["keypoints1", "descriptors1", "image_size1", "valid1"]
    if matcher.conf.add_scale_ori:
        input_names = common0 + ["scales0", "orientations0"] + common1 + [
            "scales1",
            "orientations1",
        ]
    else:
        input_names = common0 + common1
    output_names = [
        "matches0",
        "matching_scores0",
        "matches1",
        "matching_scores1",
    ]
    return input_names, output_names


def _dummy_inputs(
    matcher: MambaGlue,
    batch_size: int,
    num_keypoints0: int,
    num_keypoints1: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(7)
    size0 = torch.tensor([[640.0, 480.0]]).repeat(batch_size, 1)
    size1 = torch.tensor([[800.0, 600.0]]).repeat(batch_size, 1)
    keypoints0 = torch.rand(
        batch_size, num_keypoints0, 2, generator=generator
    ) * size0.unsqueeze(1)
    keypoints1 = torch.rand(
        batch_size, num_keypoints1, 2, generator=generator
    ) * size1.unsqueeze(1)
    descriptors0 = F.normalize(
        torch.randn(
            batch_size,
            num_keypoints0,
            matcher.conf.input_dim,
            generator=generator,
        ),
        dim=-1,
    )
    descriptors1 = F.normalize(
        torch.randn(
            batch_size,
            num_keypoints1,
            matcher.conf.input_dim,
            generator=generator,
        ),
        dim=-1,
    )
    valid0 = torch.ones(batch_size, num_keypoints0, dtype=torch.bool)
    valid1 = torch.ones(batch_size, num_keypoints1, dtype=torch.bool)

    common0: list[torch.Tensor] = [keypoints0, descriptors0, size0, valid0]
    common1: list[torch.Tensor] = [keypoints1, descriptors1, size1, valid1]
    if matcher.conf.add_scale_ori:
        scales0 = torch.ones(batch_size, num_keypoints0)
        orientations0 = torch.zeros(batch_size, num_keypoints0)
        scales1 = torch.ones(batch_size, num_keypoints1)
        orientations1 = torch.zeros(batch_size, num_keypoints1)
        return tuple(
            common0
            + [scales0, orientations0]
            + common1
            + [scales1, orientations1]
        )
    return tuple(common0 + common1)


_PREPROCESSOR_OUTPUT_NAMES = [
    "pre_hidden0",
    "pre_hidden1",
    "pre_encoding0",
    "pre_encoding1",
    "pre_valid0",
    "pre_valid1",
]
_LAYER_INPUT_NAMES = [
    "layer_hidden0",
    "layer_hidden1",
    "layer_encoding0",
    "layer_encoding1",
    "layer_valid0",
    "layer_valid1",
]
_LAYER_OUTPUT_NAMES = [
    "layer_hidden0_out",
    "layer_hidden1_out",
    "layer_encoding0_out",
    "layer_encoding1_out",
    "layer_valid0_out",
    "layer_valid1_out",
]
_POSTPROCESSOR_INPUT_NAMES = [
    "post_hidden0",
    "post_hidden1",
    "post_valid0",
    "post_valid1",
]


def _dynamic_axes_for_names(
    names: Sequence[str],
    *,
    encoding_names: Sequence[str] = (),
) -> dict[str, dict[int, str]]:
    encoding_set = set(encoding_names)
    return {
        name: {1 if name in encoding_set else 0: "batch"}
        for name in names
    }


def _torchscript_export(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int,
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> None:
    export_kwargs: dict[str, object] = {
        "export_params": True,
        "opset_version": opset_version,
        # Constant folding is intentionally disabled. Repeated matcher layers
        # otherwise make the legacy exporter's optimization cost grow sharply.
        "do_constant_folding": False,
        "input_names": list(input_names),
        "output_names": list(output_names),
        "dynamic_axes": dynamic_axes,
    }
    export_signature = inspect.signature(torch.onnx.export)
    if "autograd_inlining" in export_signature.parameters:
        export_kwargs["autograd_inlining"] = False
    if "dynamo" in export_signature.parameters:
        # TorchScript lowers selective_scan_portable to a standard ONNX Loop.
        export_kwargs["dynamo"] = False

    with torch.inference_mode():
        torch.onnx.export(module.eval(), inputs, str(output_path), **export_kwargs)


def _remap_graph_values(graph: object, mapping: Mapping[str, str]) -> None:
    """Rename value edges recursively, including names captured by Loop bodies."""

    if not mapping:
        return

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        value_info.name = mapping.get(value_info.name, value_info.name)
    for initializer in graph.initializer:
        initializer.name = mapping.get(initializer.name, initializer.name)
    for sparse in graph.sparse_initializer:
        sparse.values.name = mapping.get(sparse.values.name, sparse.values.name)
        sparse.indices.name = mapping.get(sparse.indices.name, sparse.indices.name)

    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = mapping.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = mapping.get(name, name)
        for attribute in node.attribute:
            # Importing onnx here would make the base package require it. The
            # protobuf fields are safe to inspect directly by presence.
            if attribute.HasField("g"):
                _remap_graph_values(attribute.g, mapping)
            for child_graph in attribute.graphs:
                _remap_graph_values(child_graph, mapping)


def _clone_onnx_model(model: object) -> object:
    """Create an independent protobuf clone without relying on deepcopy."""

    cloned = type(model)()
    cloned.ParseFromString(model.SerializeToString())
    return cloned


def _materialize_module_state_initializers(
    model: object, module: nn.Module
) -> object:
    """Ensure every module state tensor has its own ONNX initializer.

    The TorchScript exporter may deduplicate equal parameters and reconnect the
    duplicate through ``Identity`` (LayerNorm's initial one/zero tensors are a
    common example).  A reusable layer template must retain those parameters as
    independent initializers so each trained matcher layer can receive its own
    checkpoint values.
    """

    from onnx import numpy_helper

    state = module.state_dict()
    initializer_names = {item.name for item in model.graph.initializer}
    producer_by_output = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    identity_outputs_to_remove: set[str] = set()
    unresolved: list[str] = []

    for name, tensor in state.items():
        if name in initializer_names:
            continue
        producer = producer_by_output.get(name)
        if (
            producer is None
            or producer.op_type != "Identity"
            or len(producer.input) != 1
            or len(producer.output) != 1
        ):
            unresolved.append(name)
            continue
        model.graph.initializer.append(
            numpy_helper.from_array(tensor.detach().cpu().numpy(), name=name)
        )
        initializer_names.add(name)
        identity_outputs_to_remove.add(name)

    if unresolved:
        names = ", ".join(unresolved)
        raise RuntimeError(
            "Unable to convert the exported layer into a reusable ONNX "
            f"template; missing state initializers: {names}"
        )

    if identity_outputs_to_remove:
        retained_nodes = [
            node
            for node in model.graph.node
            if not (
                node.op_type == "Identity"
                and len(node.output) == 1
                and node.output[0] in identity_outputs_to_remove
            )
        ]
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)
    return model


def _clone_component_with_module_state(
    template: object, module: nn.Module
) -> object:
    """Clone an ONNX layer template and replace all trainable state tensors."""

    from onnx import numpy_helper

    model = _clone_onnx_model(template)
    initializers = {item.name: item for item in model.graph.initializer}
    missing = [name for name in module.state_dict() if name not in initializers]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"ONNX layer template is missing initializers: {names}")

    for name, tensor in module.state_dict().items():
        replacement = numpy_helper.from_array(
            tensor.detach().cpu().numpy(), name=name
        )
        initializers[name].CopyFrom(replacement)
    return model


def _combine_component_models(
    component_models: Sequence[object],
    final_outputs: Sequence[object],
) -> object:
    import onnx

    if not component_models:
        raise ValueError("At least one ONNX component model is required.")
    reference = component_models[0]
    graph = onnx.GraphProto()
    graph.name = "FixedKeypointMambaGlue"
    graph.doc_string = (
        "MambaGlue matcher assembled from a preprocessor, fixed-depth matcher "
        "layers, and a dense assignment postprocessor."
    )
    graph.input.extend(reference.graph.input)
    graph.output.extend(final_outputs)
    for component in component_models:
        graph.node.extend(component.graph.node)
        graph.initializer.extend(component.graph.initializer)
        graph.sparse_initializer.extend(component.graph.sparse_initializer)

    combined = onnx.ModelProto()
    combined.ir_version = reference.ir_version
    combined.producer_name = "MambaGlue"
    combined.producer_version = "0.1.0"
    combined.model_version = 1
    combined.domain = ""
    combined.graph.CopyFrom(graph)
    combined.opset_import.extend(reference.opset_import)
    for component in component_models:
        combined.functions.extend(component.functions)
    return combined


def _export_modular_onnx(
    matcher: MambaGlue,
    inputs: tuple[torch.Tensor, ...],
    output: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int,
    dynamic_batch: bool,
) -> None:
    """Export each repeated matcher layer separately, then compose the graphs.

    PyTorch's legacy ONNX exporter becomes disproportionately slow when all nine
    large matcher layers are traced as one module. Component export keeps that
    cost linear while producing one ordinary ONNX model at the end.
    """

    import onnx
    from onnx import compose

    preprocessor: nn.Module
    if matcher.conf.add_scale_ori:
        preprocessor = _FixedKeypointPreprocessorScaleOrientation(matcher)
    else:
        preprocessor = _FixedKeypointPreprocessor(matcher)
    preprocessor.eval()

    with tempfile.TemporaryDirectory(prefix="mambaglue_onnx_") as temp_dir:
        temp_root = Path(temp_dir)
        pre_path = temp_root / "preprocessor.onnx"
        pre_dynamic_axes = None
        if dynamic_batch:
            pre_dynamic_axes = _dynamic_axes_for_names(
                list(input_names) + _PREPROCESSOR_OUTPUT_NAMES,
                encoding_names=("pre_encoding0", "pre_encoding1"),
            )
        _torchscript_export(
            preprocessor,
            inputs,
            pre_path,
            input_names=input_names,
            output_names=_PREPROCESSOR_OUTPUT_NAMES,
            opset_version=opset_version,
            dynamic_axes=pre_dynamic_axes,
        )
        pre_model = onnx.load(str(pre_path), load_external_data=True)
        component_models: list[object] = [pre_model]
        current_names = [item.name for item in pre_model.graph.output]
        with torch.inference_mode():
            current_values = tuple(preprocessor(*inputs))

        layer_dynamic_axes = None
        if dynamic_batch:
            layer_dynamic_axes = _dynamic_axes_for_names(
                _LAYER_INPUT_NAMES + _LAYER_OUTPUT_NAMES,
                encoding_names=(
                    "layer_encoding0",
                    "layer_encoding1",
                    "layer_encoding0_out",
                    "layer_encoding1_out",
                ),
            )

        # All matcher layers have the same architecture. Export the expensive
        # Loop-heavy graph once, materialize any deduplicated parameters, then
        # clone the graph and replace its state for every trained layer.
        template_wrapper = _FixedKeypointLayer(
            matcher.transformermambas[0]
        ).eval()
        layer_template_path = temp_root / "layer_template.onnx"
        _torchscript_export(
            template_wrapper,
            current_values,
            layer_template_path,
            input_names=_LAYER_INPUT_NAMES,
            output_names=_LAYER_OUTPUT_NAMES,
            opset_version=opset_version,
            dynamic_axes=layer_dynamic_axes,
        )
        layer_template = onnx.load(
            str(layer_template_path), load_external_data=True
        )
        layer_template = _materialize_module_state_initializers(
            layer_template, template_wrapper
        )

        for layer_index, layer in enumerate(matcher.transformermambas):
            layer_wrapper = _FixedKeypointLayer(layer).eval()
            layer_model = _clone_component_with_module_state(
                layer_template, layer_wrapper
            )
            prefix = f"layer_{layer_index}/"
            layer_model = compose.add_prefix(layer_model, prefix)
            layer_inputs = [item.name for item in layer_model.graph.input]
            _remap_graph_values(
                layer_model.graph,
                dict(zip(layer_inputs, current_names, strict=True)),
            )
            current_names = [item.name for item in layer_model.graph.output]
            component_models.append(layer_model)

        # Postprocessing is shape-driven. Reusing the preprocessor sample avoids
        # executing all portable selective-scan layers during export, which is
        # especially important for realistic fixed lengths such as 512 points.
        postprocessor = _FixedKeypointPostprocessor(matcher).eval()
        post_inputs = (
            current_values[0],
            current_values[1],
            current_values[4],
            current_values[5],
        )
        post_source_names = (
            current_names[0],
            current_names[1],
            current_names[4],
            current_names[5],
        )
        post_path = temp_root / "postprocessor.onnx"
        post_dynamic_axes = None
        if dynamic_batch:
            post_dynamic_axes = _dynamic_axes_for_names(
                _POSTPROCESSOR_INPUT_NAMES + list(output_names)
            )
        _torchscript_export(
            postprocessor,
            post_inputs,
            post_path,
            input_names=_POSTPROCESSOR_INPUT_NAMES,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=post_dynamic_axes,
        )
        post_model = onnx.load(str(post_path), load_external_data=True)
        post_model = compose.add_prefix(post_model, "post/")
        post_input_names = [item.name for item in post_model.graph.input]
        post_output_names = [item.name for item in post_model.graph.output]
        mapping = dict(zip(post_input_names, post_source_names, strict=True))
        mapping.update(dict(zip(post_output_names, output_names, strict=True)))
        _remap_graph_values(post_model.graph, mapping)
        component_models.append(post_model)

        combined = _combine_component_models(
            component_models, list(post_model.graph.output)
        )
        onnx.checker.check_model(combined)
        onnx.save_model(combined, str(output))


def export_matcher_onnx(
    matcher: MambaGlue,
    output_path: str | Path,
    *,
    num_keypoints0: int = 512,
    num_keypoints1: int | None = None,
    batch_size: int = 1,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    export_strategy: str = "auto",
) -> tuple[Path, tuple[torch.Tensor, ...]]:
    """Export a MambaGlue matcher with fixed keypoint axes to ONNX.

    ``export_strategy="auto"`` uses a single graph for one or two matcher
    layers and component-wise composition for deeper models. The resulting file
    is one self-contained ONNX model in either case.
    """

    if num_keypoints0 <= 0:
        raise ValueError("num_keypoints0 must be positive.")
    num_keypoints1 = num_keypoints0 if num_keypoints1 is None else num_keypoints1
    if num_keypoints1 <= 0:
        raise ValueError("num_keypoints1 must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if opset_version < 17:
        raise ValueError("ONNX opset 17 or newer is required.")
    if matcher.conf.n_layers <= 0:
        raise ValueError("The matcher must contain at least one layer.")
    if export_strategy not in {"auto", "monolithic", "modular"}:
        raise ValueError(
            "export_strategy must be 'auto', 'monolithic', or 'modular'."
        )

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The ONNX package is required; run `uv sync --extra onnx`."
        ) from exc

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    matcher = matcher.eval().cpu()
    inputs = _dummy_inputs(matcher, batch_size, num_keypoints0, num_keypoints1)
    input_names, output_names = _input_spec(matcher)

    selected_strategy = export_strategy
    if selected_strategy == "auto":
        selected_strategy = "modular" if matcher.conf.n_layers >= 3 else "monolithic"

    if selected_strategy == "modular":
        _export_modular_onnx(
            matcher,
            inputs,
            output,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
        )
    else:
        wrapper: nn.Module
        if matcher.conf.add_scale_ori:
            wrapper = FixedKeypointMambaGlueScaleOrientation(matcher)
        else:
            wrapper = FixedKeypointMambaGlue(matcher)
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = _dynamic_axes_for_names(input_names + output_names)
        _torchscript_export(
            wrapper,
            inputs,
            output,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )

    model = onnx.load(str(output), load_external_data=True)
    onnx.checker.check_model(model)
    metadata = {
        "model": "MambaGlue matcher",
        "fixed_keypoints0": str(num_keypoints0),
        "fixed_keypoints1": str(num_keypoints1),
        "descriptor_input_dim": str(matcher.conf.input_dim),
        "descriptor_model_dim": str(matcher.conf.descriptor_dim),
        "matcher_layers": str(matcher.conf.n_layers),
        "opset": str(opset_version),
        "export_strategy": selected_strategy,
        "valid_mask_contract": "valid keypoints must be packed before padding",
    }
    existing = {item.key: item for item in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            item = model.metadata_props.add()
            item.key = key
            item.value = value
    onnx.save_model(model, str(output))
    return output, inputs


def verify_matcher_onnx(
    matcher: MambaGlue,
    onnx_path: str | Path,
    inputs: Sequence[torch.Tensor],
    *,
    rtol: float = 1.0e-1,
    atol: float = 1.0e-2,
) -> dict[str, float]:
    """Compare PyTorch and CUDA ONNX Runtime outputs for the export sample.

    Match indices must agree exactly.  The portable recurrent scan and CUDA
    layer-normalization kernels can accumulate small float32 differences across
    nine layers, so score comparison uses a GPU-oriented tolerance.
    """

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required for verification; run `uv sync --extra onnx`."
        ) from exc

    matcher = matcher.eval().cpu()
    wrapper: nn.Module
    if matcher.conf.add_scale_ori:
        wrapper = FixedKeypointMambaGlueScaleOrientation(matcher)
    else:
        wrapper = FixedKeypointMambaGlue(matcher)
    wrapper.eval()
    input_names, output_names = _input_spec(matcher)

    with torch.inference_mode():
        torch_outputs = wrapper(*inputs)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime-gpu did not expose CUDAExecutionProvider.")
    session_options = ort.SessionOptions()
    # Loop-heavy graphs can spend substantially longer in global graph
    # optimization than in inference. Verification only needs semantic parity.
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        str(Path(onnx_path).resolve()),
        sess_options=session_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "ONNX Runtime did not select CUDAExecutionProvider: "
            f"{session.get_providers()}"
        )
    ort_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, inputs)
    }
    ort_outputs = session.run(output_names, ort_inputs)

    report: dict[str, float] = {}
    for name, expected, actual in zip(output_names, torch_outputs, ort_outputs):
        expected_array = expected.detach().cpu().numpy()
        if np.issubdtype(expected_array.dtype, np.integer):
            if not np.array_equal(expected_array, actual):
                mismatch = int(np.count_nonzero(expected_array != actual))
                raise AssertionError(f"{name} differs at {mismatch} elements.")
            report[name] = 0.0
        else:
            np.testing.assert_allclose(actual, expected_array, rtol=rtol, atol=atol)
            report[name] = float(np.max(np.abs(actual - expected_array)))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the MambaGlue matcher to fixed-keypoint ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Matcher checkpoint path (defaults to the released SuperPoint weight).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("mambaglue_fixed.onnx")
    )
    parser.add_argument(
        "--features",
        choices=["superpoint", "disk", "aliked", "sift", "doghardnet", "custom"],
        default="superpoint",
    )
    parser.add_argument("--num-keypoints", type=int, default=512)
    parser.add_argument("--num-keypoints0", type=int)
    parser.add_argument("--num-keypoints1", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--descriptor-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--filter-threshold", type=float, default=0.01)
    parser.add_argument(
        "--export-strategy",
        choices=["auto", "monolithic", "modular"],
        default="auto",
        help=(
            "Use one graph, component composition, or select automatically. "
            "The modular strategy is recommended for three or more layers."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="Allow export without a checkpoint for structural testing only.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run an ONNX Runtime numerical comparison after export.",
    )
    parser.add_argument(
        "--fixed-batch",
        action="store_true",
        help="Keep the batch axis fixed instead of dynamic.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.descriptor_dim % args.num_heads != 0:
        parser.error("--descriptor-dim must be divisible by --num-heads.")

    features = None if args.features == "custom" else args.features
    matcher_kwargs = {
        "checkpoint": args.checkpoint,
        "strict": args.strict,
        "n_layers": args.n_layers,
        "descriptor_dim": args.descriptor_dim,
        "num_heads": args.num_heads,
        "filter_threshold": args.filter_threshold,
        "flash": False,
        "mp": False,
        "depth_confidence": -1,
        "width_confidence": -1,
        "scan_backend": "portable",
    }
    if features is None:
        matcher_kwargs["input_dim"] = args.input_dim
    matcher = MambaGlue(features=features, **matcher_kwargs).eval()

    num_keypoints0 = args.num_keypoints0 or args.num_keypoints
    num_keypoints1 = args.num_keypoints1 or args.num_keypoints
    output, inputs = export_matcher_onnx(
        matcher,
        args.output,
        num_keypoints0=num_keypoints0,
        num_keypoints1=num_keypoints1,
        batch_size=args.batch_size,
        opset_version=args.opset,
        dynamic_batch=not args.fixed_batch,
        export_strategy=args.export_strategy,
    )
    print(f"Exported: {output}")
    print(
        "Fixed keypoints: "
        f"image0={num_keypoints0}, image1={num_keypoints1}; "
        f"input descriptor dim={matcher.conf.input_dim}"
    )
    if args.verify:
        report = verify_matcher_onnx(matcher, output, inputs)
        formatted = ", ".join(f"{key}={value:.3e}" for key, value in report.items())
        print(f"ONNX Runtime CUDA verification passed: {formatted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
