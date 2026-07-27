"""
Feature extractor implementation using the SPECTRE foundation model.

SPECTRE (Claessens et al., CVPR 2026; https://github.com/cclaess/SPECTRE) is a
3D CT vision transformer distributed as the ``spectre-fm`` package. It tiles a
volume into 128 x 128 x 64 windows, embeds each with a local ViT backbone, and
aggregates the windows into a single scan-level embedding with a feature
combiner (a global ViT). We hand it a raw-HU lesion crop (a whole number of
windows, produced by ``SpectrePreprocessor``); SPECTRE performs the HU windowing
and tiling internally, and we take the scan-level CLS token as the per-lesion
embedding -- the same vector the authors use for their frozen-embedding
biomarker/linear-probe evaluation (``outputs[:, 0]``).

Weights are downloaded automatically by ``from_pretrained`` from
https://huggingface.co/cclaess/SPECTRE.
"""

from typing import Any

import numpy as np
import torch

from .base_extractor import BaseFeatureExtractor
from .base_preprocessor import BasePreprocessor
from .preprocessing import SpectrePreprocessor


class SpectreExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: BasePreprocessor | None = None,
        feature_names: list[str] | None = None,
        min_volume: int = 400,
        model_name: str = "spectre-large",
        include_feature_combiner: bool = True,
    ) -> None:
        if preprocessor is None:
            preprocessor = SpectrePreprocessor()

        super().__init__(preprocessor, min_volume)

        # Imported lazily so the package is only required when SPECTRE is used.
        try:
            from spectre import SpectreImageFeatureExtractor
        except ImportError as exc:
            raise ImportError(
                "SPECTRE requires the 'spectre-fm' package with timm>=1.0.0 "
                "(spectre-fm imports timm.layers, which older timm versions lack). "
                "Install with: pip install 'renal-vision[spectre]' "
                "(or: pip install -U spectre-fm 'timm>=1.0.0'). "
                f"Original error: {exc}"
            ) from exc

        self.model_name = model_name
        self.include_feature_combiner = include_feature_combiner
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SpectreImageFeatureExtractor.from_pretrained(
            model_name,
            include_feature_combiner=include_feature_combiner,
            device=self.device,
        )
        self.model.eval()

        if self.model.has_feature_combiner:
            embed_dim = self.model.feature_combiner.embed_dim
        else:
            embed_dim = self.model.backbone.embed_dim

        if feature_names is None:
            self._active_features = [f"F{f}" for f in range(embed_dim)]
        else:
            self._active_features = feature_names

    @property
    def feature_names(self) -> list[str]:
        return self._active_features

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "spectre",
            "model_name": self.model_name,
            "include_feature_combiner": self.include_feature_combiner,
            "feature_names": self.feature_names,
            "min_volume": self.min_volume,
            "preprocessor": self.preprocessor.get_config(),
        }

    def _extract_single_lesion(
        self, image: np.ndarray, lesion_mask: np.ndarray
    ) -> dict[str, float]:
        """Extract the scan-level SPECTRE CLS embedding for one lesion crop."""
        if np.sum(lesion_mask) == 0:
            raise ValueError("Lesion mask is empty during feature extraction.")

        # (H, W, D) raw HU -> (C=1, H, W, D). SPECTRE windows and tiles internally.
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        with torch.no_grad():
            output = self.model(image_tensor)  # (T', F'): CLS token + one per window

        cls_token = output[0].detach().cpu()  # scan-level CLS -> (F',)

        return {fname: float(val) for fname, val in zip(self.feature_names, cls_token.tolist())}
