import torch

from renal_vision.features.embeddings_renalclip import (
    RenalCLIPBackbone,
    _load_backbone_weights,
)


def _randomize(v: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(v) if v.is_floating_point() else v.clone()


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
