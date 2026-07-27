import numpy as np
import torch

from renal_vision.features.embeddings_renalclip import (
    RenalCLIPBackbone,
    RenalCLIPExtractor,
    _load_backbone_weights,
)


def _randomize(v: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(v) if v.is_floating_point() else v.clone()


class _ShapeCaptureModel:
    """Stand-in backbone that records the shape of the tensor it is fed."""

    def __init__(self) -> None:
        self.seen_shape = None

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(x.shape)
        return torch.zeros(1, 512)


def test_extractor_feeds_depth_first_axis_order():
    """
    RenalCLIP expects ``[N, C, D=32, W, H]`` (the authors transpose the
    RAS-ordered crop so the anisotropic 5 mm/32-voxel axis is first). The
    preprocessed crop arrives RAS-ordered as ``(R, A, S) = (128, 128, 32)``;
    the extractor must move S to the front before calling the network.
    """
    extractor = object.__new__(RenalCLIPExtractor)
    extractor.device = "cpu"
    extractor._active_features = [f"F{f}" for f in range(512)]
    extractor.model = _ShapeCaptureModel()

    image = np.zeros((128, 128, 32), dtype=np.float32)  # (R, A, S), depth last
    mask = np.ones((128, 128, 32), dtype=np.int16)

    extractor._extract_single_lesion(image, mask)

    # [N, C, D, W, H] with depth (32) first, not last.
    assert extractor.model.seen_shape == (1, 1, 32, 128, 128)


class TestRenalCLIPBackbone:
    def test_forward_returns_512d_vector(self):
        """The backbone maps a 128x128x32 volume to a single 512-d feature vector."""
        backbone = RenalCLIPBackbone().eval()
        x = torch.rand(1, 1, 128, 128, 32)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (1, 512)

    def test_loader_matches_authors_checkpoint_layout(self, tmp_path):
        """
        The weight loader should recover every backbone tensor from a checkpoint
        laid out like the official release: real backbone weights under the
        'image_encoder_q_student.' prefix, wrapped in {'model': ...}, alongside
        unrelated (projection head / text encoder) keys that must be ignored.
        """
        backbone = RenalCLIPBackbone()
        state = backbone.state_dict()

        fake = {"image_encoder_q_student." + k: _randomize(v) for k, v in state.items()}
        fake["image_encoder_q_student.global_embedding.head.0.weight"] = torch.randn(2048, 512)
        fake["text_encoder_q.model.foo"] = torch.randn(3, 3)
        ckpt_path = tmp_path / "fake_renalclip.pt"
        torch.save({"model": fake, "epoch": 1}, ckpt_path)

        fresh = RenalCLIPBackbone()
        n_loaded = _load_backbone_weights(fresh, ckpt_path)

        assert n_loaded == len(state), "Loader should recover every backbone tensor"
        key = "layer3.0.conv1.weight"
        assert torch.allclose(fresh.state_dict()[key], fake["image_encoder_q_student." + key])

    def test_loader_handles_ddp_prefix(self, tmp_path):
        """A 'module.' DDP prefix on checkpoint keys should be stripped transparently."""
        backbone = RenalCLIPBackbone()
        state = backbone.state_dict()
        fake = {"module.image_encoder_q_student." + k: _randomize(v) for k, v in state.items()}
        ckpt_path = tmp_path / "fake_ddp.pt"
        torch.save({"model": fake}, ckpt_path)

        fresh = RenalCLIPBackbone()
        n_loaded = _load_backbone_weights(fresh, ckpt_path)
        assert n_loaded == len(state)
