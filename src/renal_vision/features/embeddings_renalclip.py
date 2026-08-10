"""
Feature extractor implementation using the RenalCLIP image encoder.

The 3D ResNet-18 backbone below is vendored from the official RenalCLIP release
(https://github.com/dt-yuhui/RenalCLIP, ``models/resnet.py`` / ``models/RenalModel.py``)
with module names preserved so the published checkpoint loads directly.

We reproduce the authors' downstream image-embedding recipe: the 512-dim
global-average-pooled backbone vector (``RenalModel.forward3D``) is passed through
the trained ``global_embedding`` projection head and L2-normalized. This is
exactly what the authors' "offline save image embeddings" notebook does
(``global_embedding(feats)`` followed by ``feats / feats.norm(...)``), and it is
the CLIP-aligned representation the model was optimized to expose -- i.e. the
space aligned to the text encoder and validated on their downstream tasks. The
``global_embedding`` head is a real MLP (``Linear -> BN -> ReLU -> Linear -> BN``)
that lives inside the resnet module and whose weights ship in the checkpoint
(``image_encoder_q_student.global_embedding.*``); using the raw pre-projection
512-dim vector instead would feed the classifier a representation the authors
never use for classification. If the checkpoint happens to lack the head, we fall
back to the raw 512-dim features and warn.

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


class GlobalEmbedding(nn.Module):
    """
    RenalCLIP projection head (``models/resnet.py``, ``GlobalEmbedding``).

    ``Linear -> BatchNorm1d -> ReLU -> Linear -> BatchNorm1d(affine=False)`` when a
    hidden dim is given, else a single ``Linear``. Module/parameter names are kept
    identical to the release so the checkpoint tensors load by key.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None, output_dim: int) -> None:
        super().__init__()
        if hidden_dim is not None:
            self.head: nn.Module = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim, affine=False),
            )
        else:
            self.head = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


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

    ``forward`` returns the RenalCLIP downstream image embedding: the 512-dim
    global-average-pooled backbone vector (``RenalModel.forward3D``) passed through
    the ``global_embedding`` projection head and L2-normalized, matching the
    authors' offline embedding script. The projection head is attached from the
    checkpoint by :func:`_load_backbone_weights`; until then (or if the checkpoint
    lacks it, or ``use_projection=False``) ``forward`` returns the raw 512-dim
    vector.
    """

    def __init__(self, use_projection: bool = True) -> None:
        super().__init__()
        self.use_projection = use_projection
        # Attached from the checkpoint once its dims are known (see loader below).
        self.global_embedding: GlobalEmbedding | None = None
        self._projection_out_dim: int | None = None
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

    @property
    def output_dim(self) -> int:
        """Dimensionality of the vector returned by :meth:`forward`."""
        if self._projection_active():
            assert self._projection_out_dim is not None
            return self._projection_out_dim
        return 512

    def _projection_active(self) -> bool:
        return self.use_projection and self.global_embedding is not None

    def attach_projection_head(
        self, input_dim: int, hidden_dim: int | None, output_dim: int
    ) -> None:
        """Create the ``global_embedding`` head so its checkpoint weights can load."""
        self.global_embedding = GlobalEmbedding(input_dim, hidden_dim, output_dim)
        self._projection_out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.avgpool(self.forward_features(x)).flatten(1)  # (N, 512)
        if self._projection_active():
            assert self.global_embedding is not None
            feats = self.global_embedding(feats)
            feats = F.normalize(feats, dim=-1)  # L2-normalize, as the authors do
        return feats


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


def _infer_projection_dims(
    stripped: dict[str, torch.Tensor],
) -> tuple | None:
    """
    Recover ``(input_dim, hidden_dim, output_dim)`` for the ``global_embedding``
    head from checkpoint tensors, or ``None`` if the head is absent/incomplete.

    Two layouts, matching :class:`GlobalEmbedding`:
      * MLP:    ``head.0`` (Linear in) and ``head.3`` (Linear out) present.
      * Linear: ``head.weight`` present.
    """
    w0 = stripped.get("global_embedding.head.0.weight")
    w3 = stripped.get("global_embedding.head.3.weight")
    if w0 is not None and w3 is not None:
        hidden_dim, input_dim = int(w0.shape[0]), int(w0.shape[1])
        return input_dim, hidden_dim, int(w3.shape[0])
    w = stripped.get("global_embedding.head.weight")
    if w is not None:
        return int(w.shape[1]), None, int(w.shape[0])
    return None


def _load_backbone_weights(backbone: RenalCLIPBackbone, checkpoint_path: str | Path) -> int:
    """
    Load the RenalCLIP image-encoder weights into ``backbone``.

    Mirrors the authors' ``load_state_with_same_shape`` (shape-matched, non-strict)
    but is prefix-agnostic: it unwraps ``checkpoint['model']``, strips a ``module.``
    DDP prefix, then tries the encoder prefixes used by the authors and keeps the
    one that matches the most tensors. The ``global_embedding`` projection head is
    attached (with dims read from the checkpoint) and loaded alongside the backbone
    so that :meth:`RenalCLIPBackbone.forward` can reproduce the authors' downstream
    embedding. If the checkpoint has no head, projection is disabled and the raw
    512-dim features are returned.
    """
    # weights_only=False: the official checkpoint is a full training checkpoint
    # (more than plain tensors); we only read its state dict below.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {
            k.partition("module.")[2] if k.startswith("module.") else k: v for k, v in state.items()
        }

    # Pick the encoder prefix that matches the most *backbone* (resnet) tensors.
    # 'student' is the authors' default model_type for the image encoder.
    resnet_state = backbone.state_dict()  # head not attached yet -> resnet only
    prefixes = ["image_encoder_q_student.", "image_encoder_q_teacher.", "image_encoder.", ""]
    best_stripped: dict[str, torch.Tensor] = {}
    best_n = -1
    for prefix in prefixes:
        stripped = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        n = sum(
            1 for k, v in stripped.items() if k in resnet_state and v.shape == resnet_state[k].shape
        )
        if n > best_n:
            best_stripped, best_n = stripped, n

    # Attach the projection head (if present) so its weights load and forward()
    # reproduces the authors' downstream embedding.
    dims = _infer_projection_dims(best_stripped)
    if dims is not None:
        backbone.attach_projection_head(*dims)
    else:
        backbone.use_projection = False
        warnings.warn(
            "RenalCLIP: no GlobalEmbedding projection head found in the checkpoint; "
            "falling back to the raw 512-dim backbone features. The authors' downstream "
            "embedding uses the projected + L2-normalized vector -- check the checkpoint."
        )

    model_state = backbone.state_dict()  # now includes the head, if attached
    matched = {
        k: v
        for k, v in best_stripped.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    backbone.load_state_dict(matched, strict=False)
    if len(matched) < len(model_state):
        warnings.warn(
            f"RenalCLIP: loaded {len(matched)}/{len(model_state)} tensors from "
            f"{checkpoint_path}. Check the checkpoint file and its key layout."
        )
    return len(matched)


def renalclip_backbone(
    eval_mode: bool = True,
    weights_path: str | Path | None = None,
    use_projection: bool = True,
) -> RenalCLIPBackbone:
    backbone = RenalCLIPBackbone(use_projection=use_projection)
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
        use_projection: bool = True,
    ) -> None:
        if preprocessor is None:
            preprocessor = RenalCLIPPreprocessor()

        super().__init__(preprocessor, min_volume)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = renalclip_backbone(
            weights_path=weights_path, use_projection=use_projection
        ).to(self.device)
        self.model.eval()

        # Derived from the model so it tracks the projected (4096-d) vs raw (512-d)
        # embedding rather than being hard-coded.
        if feature_names is None:
            self._active_features = [f"F{f}" for f in range(self.model.output_dim)]
        else:
            self._active_features = feature_names

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
