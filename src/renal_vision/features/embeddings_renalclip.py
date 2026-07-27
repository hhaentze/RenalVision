"""
Feature extractor implementation using the RenalCLIP image encoder.

The 3D ResNet-18 backbone below is vendored from the official RenalCLIP release
(https://github.com/dt-yuhui/RenalCLIP, ``models/resnet.py`` / ``models/RenalModel.py``)
with module names preserved so the published checkpoint loads directly. We keep
only the image backbone and return its 512-dim global-average-pooled feature
vector -- i.e. ``RenalModel.forward3D`` -- which is what the authors feed to
their downstream classification/prognosis heads (the 4096-dim ``global_embedding``
projection is used only for cross-modal / zero-shot tasks and is intentionally
omitted here).

Weights: https://huggingface.co/taoyh/RenalCLIP (RenalCLIP-image-encoder-model-best-acc.pt)
"""

import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base_extractor import BaseFeatureExtractor
from .base_preprocessor import BasePreprocessor
from .preprocessing import RenalCLIPPreprocessor

RENALCLIP_WEIGHTS_FILENAME = "RenalCLIP-image-encoder-model-best-acc.pt"
RENALCLIP_WEIGHTS_URL = (
    "https://huggingface.co/taoyh/RenalCLIP/resolve/main/" + RENALCLIP_WEIGHTS_FILENAME
)


# ======================= Vendored 3D ResNet-18 backbone =======================


def _conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def _downsample_basic_block(x: torch.Tensor, planes: int, stride: int) -> torch.Tensor:
    """Shortcut type 'A': strided avg-pool followed by zero-padding of channels."""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class RenalCLIPBackbone(nn.Module):
    """
    3D ResNet-18 image backbone (shortcut type 'A') from RenalCLIP.

    ``forward`` returns the 512-dim global-average-pooled feature vector, matching
    ``RenalModel.forward3D``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=False)
        # Equivalent to the authors' MaxPool3dDeterministic here: the input is
        # post-ReLU (>= 0), so zero-padding and -inf-padding max-pooling agree.
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = partial(_downsample_basic_block, planes=planes, stride=stride)
        layers = [_BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.avgpool(x).flatten(1)


# ============================= Weight loading =================================


def _get_weights(url: str, filename: str) -> Path:
    """Resolve/download the checkpoint, caching under RENALCLIP_CACHE_DIR or ~/.cache/renalclip."""
    cache_dir = Path(os.environ.get("RENALCLIP_CACHE_DIR", Path.home() / ".cache" / "renalclip"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if not destination.exists():
        print(f"Downloading RenalCLIP weights to {destination} ...")
        torch.hub.download_url_to_file(url, str(destination))
    return destination


def _load_backbone_weights(backbone: RenalCLIPBackbone, checkpoint_path: str | Path) -> int:
    """
    Load the RenalCLIP image-encoder weights into ``backbone``.

    Mirrors the authors' ``load_state_with_same_shape`` (shape-matched, non-strict)
    but is prefix-agnostic: it unwraps ``checkpoint['model']``, strips a ``module.``
    DDP prefix, then tries the encoder prefixes used by the authors and keeps the
    one that matches the most tensors.
    """
    # weights_only=False: the official checkpoint is a full training checkpoint
    # (more than plain tensors); we only read its state dict below.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {
            k.partition("module.")[2] if k.startswith("module.") else k: v for k, v in state.items()
        }

    model_state = backbone.state_dict()
    # 'student' is the authors' default model_type for the image encoder.
    prefixes = ["image_encoder_q_student.", "image_encoder_q_teacher.", "image_encoder.", ""]
    best: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        stripped = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        matched = {
            k: v
            for k, v in stripped.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        if len(matched) > len(best):
            best = matched

    backbone.load_state_dict(best, strict=False)
    if len(best) < len(model_state):
        warnings.warn(
            f"RenalCLIP: loaded {len(best)}/{len(model_state)} backbone tensors from "
            f"{checkpoint_path}. Check the checkpoint file and its key layout."
        )
    return len(best)


def renalclip_backbone(
    eval_mode: bool = True, weights_path: str | Path | None = None
) -> RenalCLIPBackbone:
    backbone = RenalCLIPBackbone()
    if weights_path is None:
        weights_path = _get_weights(RENALCLIP_WEIGHTS_URL, RENALCLIP_WEIGHTS_FILENAME)
    _load_backbone_weights(backbone, weights_path)
    if eval_mode:
        backbone.eval()
    return backbone


# ============================== Extractor ====================================


class RenalCLIPExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: BasePreprocessor | None = None,
        feature_names: list[str] | None = None,
        min_volume: int = 400,
        weights_path: str | Path | None = None,
    ) -> None:
        if preprocessor is None:
            preprocessor = RenalCLIPPreprocessor()

        super().__init__(preprocessor, min_volume)

        if feature_names is None:
            self._active_features = [f"F{f}" for f in range(512)]
        else:
            self._active_features = feature_names

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = renalclip_backbone(weights_path=weights_path).to(self.device)
        self.model.eval()

    @property
    def feature_names(self) -> list[str]:
        return self._active_features

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "renalclip",
            "feature_names": self.feature_names,
            "min_volume": self.min_volume,
            "preprocessor": self.preprocessor.get_config(),
        }

    def _extract_single_lesion(
        self, image: np.ndarray, lesion_mask: np.ndarray
    ) -> dict[str, float]:
        """Extract the 512-dim RenalCLIP backbone embedding for one lesion crop."""
        if np.sum(lesion_mask) == 0:
            raise ValueError("Lesion mask is empty during feature extraction.")

        # Reorder axes to match the authors' network input convention. The
        # preprocessed crop is RAS-ordered ``(R, A, S)`` == ``(128, 128, 32)``,
        # with the anisotropic 5 mm axis (S, 32 voxels) last. RenalCLIP instead
        # feeds ``[N, C, D, W, H]`` with that depth axis *first* -- their
        # data loader transposes ``(C, W, H, D) -> (C, D, W, H)`` right before
        # the ResNet (utils/data_util.py, ``Transposed(indices=(0, 3, 1, 2))``).
        # Because the 3D convolutions are not permutation-invariant, feeding the
        # depth axis in the wrong slot yields a different (invalid) embedding, so
        # we move S to the front: ``(R, A, S) -> (S, R, A)`` == ``(D, W, H)``.
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))

        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(image_tensor)

        return {
            fname: float(val)
            for fname, val in zip(self.feature_names, prediction.squeeze().tolist())
        }
