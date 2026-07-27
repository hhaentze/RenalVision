"""
Mevis UNICORN embeddings extractor.

Same idea as ``embeddings_mevis`` (2D encoder over sliding 3-slice CT slabs,
max + mask-weighted-mean pooled over depth), but uses Fraunhofer MEVIS's
UNICORN encoder -- the first-place UNICORN challenge model -- loaded through the
newer ``mmm.api.M3Model`` API.

This is a separate module (not a flag on MevisExtractor) because UNICORN requires
medicalmultitaskmodeling >= 1.6.1, which replaced the ``mmm.labelstudio_ext.NativeBlocks``
API that MevisExtractor imports. The two package versions cannot coexist, so the
extraction logic below is intentionally duplicated from MevisExtractor rather
than shared via import.

The UNICORN encoder is a PyramidEncoder that accepts variable input sizes, so the
existing MevisPreprocessor is reused unchanged. Weights download automatically.
"""

import os
from typing import Any

import numpy as np
import torch
from torch import nn

from .base_extractor import BaseFeatureExtractor
from .base_preprocessor import BasePreprocessor
from .preprocessing import MevisPreprocessor

os.environ["MMM_LICENSE_ACCEPTED"] = "i accept"
from mmm.api.M3Model import M3_MODELS, UNICORN_ENCODER, M3Model


class MevisUnicornExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: BasePreprocessor | None = None,
        feature_names: list[str] | None = None,
        min_volume: int = 400,
    ) -> None:
        if preprocessor is None:
            preprocessor = MevisPreprocessor()

        super().__init__(preprocessor, min_volume)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # M3Model.__init__ calls torch.load without map_location and then .to(device).
        # The UNICORN encoder was saved on a CUDA device, so that load fails whenever
        # the current process cannot place it there (CPU-only job, or a GPU-index
        # mismatch). M3Model swallows the failure via logfire, so the block silently
        # goes missing (-> KeyError: 'encoder'). Force CPU-mapped loading during
        # construction, then relocate the blocks we use to the target device.
        original_load = torch.load

        def _cpu_mapped_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("map_location", "cpu")
            return original_load(*args, **kwargs)

        torch.load = _cpu_mapped_load  # type: ignore[assignment]
        try:
            self.model = M3Model(M3_MODELS[UNICORN_ENCODER], device_identifier="cpu")
        finally:
            torch.load = original_load  # type: ignore[assignment]

        # Guard against a block still failing to load (surfaced instead of a KeyError).
        available = list(self.model.keys())
        missing = [key for key in ("encoder", "squeezer") if key not in available]
        if missing:
            raise KeyError(
                f"UNICORN model is missing block(s) {missing}; loaded blocks: {available}. "
                "A block likely failed to load silently. Re-run with logfire enabled "
                '(e.g. `python -c "import logfire; logfire.configure(send_to_logfire=False)"` '
                "before loading) to see the underlying error."
            )

        self.model["encoder"].to(device).eval()
        self.model["squeezer"].to(device).eval()
        self.model.device = device

        if feature_names is None:
            dim = self._feature_dim()
            self._active_features = [f"F{f}_max" for f in range(dim)]
            self._active_features += [f"F{f}_mean" for f in range(dim)]
        else:
            self._active_features = feature_names

    def _feature_dim(self) -> int:
        """Per-slab embedding dimension (UNICORN's squeezer output may differ from 512)."""
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 64, 64, device=self.model.device)
            feature_pyramid = self.model["encoder"](dummy)
            vector = nn.Flatten(1)(self.model["squeezer"](feature_pyramid)[1])
        return int(vector.shape[1])

    @property
    def feature_names(self) -> list[str]:
        return self._active_features

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "MevisUnicornEmbeddings",
            "feature_names": self.feature_names,
            "min_volume": self.min_volume,
            "preprocessor": self.preprocessor.get_config(),
        }

    def _extract_single_lesion(
        self, image: np.ndarray, lesion_mask: np.ndarray
    ) -> dict[str, float]:
        if np.sum(lesion_mask) == 0:
            raise ValueError("Lesion mask is empty during feature extraction.")

        prediction = self._extract_lesion_features(image, lesion_mask)
        return {
            fname: float(val)
            for fname, val in zip(self.feature_names, prediction.squeeze().tolist())
        }

    def _extract_lesion_features(
        self, img: np.ndarray, seg: np.ndarray, step_size: int = 3
    ) -> np.ndarray:
        """
        Slide a 3-slice (pseudo-RGB) window along depth, encode each slab, and pool
        with max + mask-weighted mean. Mirrors MevisExtractor._extract_lesion_features.

        img, seg: numpy arrays of shape (H, W, D).
        """
        depth = img.shape[-1]
        embeddings = []
        weights = []

        with torch.inference_mode():
            for z in range(1, depth - 1, step_size):
                slab = torch.tensor(img[..., z - 1 : z + 2].transpose(2, 0, 1)[None])

                feature_pyramid: list[torch.Tensor] = self.model["encoder"](
                    slab.to(self.model.device)
                )
                feature_vector = nn.Flatten(1)(self.model["squeezer"](feature_pyramid)[1])
                embeddings.append(feature_vector)

                mask_sum = seg[..., z - 1 : z + 2].sum() + 10
                weights.append(mask_sum)

        embeddings_tensor = torch.stack(embeddings)
        weights_tensor = torch.tensor(weights).view(-1, 1, 1)

        max_pooled = torch.max(embeddings_tensor, dim=0).values

        total_weight = weights_tensor.sum() + 1e-6
        weighted_avg = (embeddings_tensor * weights_tensor).sum(dim=0) / total_weight
        combined = torch.cat([max_pooled, weighted_avg], dim=-1)

        return combined.squeeze().cpu().numpy()
