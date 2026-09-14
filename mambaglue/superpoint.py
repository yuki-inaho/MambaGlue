# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger

import torch
from kornia.color import rgb_to_grayscale
from torch import nn

from .utils import Extractor


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


SUPERPOINT_WEIGHTS_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"  # noqa


def build_superpoint_convs(
    module: nn.Module, in_channels: int, descriptor_dim: int = 256
) -> nn.Module:
    """Attach the SuperPoint backbone convolutions to ``module`` in place."""
    module.relu = nn.ReLU(inplace=True)
    module.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

    module.conv1a = nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1)
    module.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    module.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    module.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    module.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    module.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    module.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    module.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

    module.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    module.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

    module.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    module.convDb = nn.Conv2d(c5, descriptor_dim, kernel_size=1, stride=1, padding=0)
    return module


def expand_superpoint_state_dict(state_dict):
    """Duplicate the 1-channel SuperPoint stem into 2 channels (depth = copy).

    No implicit ``strict=False``: the returned dict is loaded strictly, so any
    surprise key/shape change is surfaced instead of silently ignored.
    """
    if "conv1a.weight" not in state_dict:
        raise KeyError("conv1a.weight is missing from the SuperPoint state dict")
    weight = state_dict["conv1a.weight"]
    if weight.dim() != 4 or weight.shape[1] != 1:
        raise ValueError(
            f"conv1a.weight must have shape (out, 1, k, k), got {tuple(weight.shape)}"
        )
    expanded = dict(state_dict)
    expanded["conv1a.weight"] = torch.cat([weight, weight], dim=1).contiguous()
    return expanded


def init_rgbd_from_superpoint(module: "SuperPointRGBD") -> None:
    """Replace a SuperPoint stem with a 2-channel stem (ch0 original, ch1 copy)."""
    old = module.conv1a
    new = nn.Conv2d(
        2,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(torch.cat([old.weight, old.weight], dim=1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    module.conv1a = new


class SuperPoint(Extractor):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }

    preprocess_conf = {
        "resize": 1024,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf):
        super().__init__(**conf)  # Update with default configuration.
        build_superpoint_convs(self, 1, self.conf.descriptor_dim)
        self.load_state_dict(torch.hub.load_state_dict_from_url(SUPERPOINT_WEIGHTS_URL))

        if self.conf.max_num_keypoints is not None and self.conf.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")

    def forward(self, data: dict) -> dict:
        """Compute keypoints, scores, descriptors for image"""
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)

        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        best_kp = torch.where(scores > self.conf.detection_threshold)
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [
            torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)
        ]
        scores = [scores[best_kp[0] == i] for i in range(b)]

        # Keep the k keypoints with highest score
        if self.conf.max_num_keypoints is not None:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.conf.max_num_keypoints)
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        return {
            "keypoints": torch.stack(keypoints, 0),
            "keypoint_scores": torch.stack(scores, 0),
            "descriptors": torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }


class SuperPointRGBD(SuperPoint):
    """SuperPoint whose stem takes 2 channels: grayscale RGB and depth.

    ``forward`` is inherited unchanged: it only converts 3-channel input to
    grayscale, so a (B, 2, H, W) RGB-D tensor passes straight into the 2-channel
    stem. Weights come from the released 1-channel SuperPoint checkpoint with
    ``conv1a`` expanded (ch0 = original, ch1 = copy).
    """

    def __init__(self, *, source_state_dict=None, **conf):
        # Skip ``SuperPoint.__init__``: it would build a 1-channel stem and load
        # the 1-channel checkpoint before we could expand it.
        Extractor.__init__(self, **conf)
        build_superpoint_convs(self, 2, self.conf.descriptor_dim)

        if self.conf.max_num_keypoints is not None and self.conf.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")

        if source_state_dict is None:
            source_state_dict = torch.hub.load_state_dict_from_url(
                SUPERPOINT_WEIGHTS_URL
            )
        self.load_state_dict(expand_superpoint_state_dict(source_state_dict))
