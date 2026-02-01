from pathlib import Path

import torch.nn as nn
from monai.networks.nets import resnet50

from renal_vision.features.embeddings_fmcib import LoadModel, get_model_weights


def build_model(num_classes: int, fmcib_pretrained: bool = False) -> nn.Module:
    if fmcib_pretrained:
        trunk = resnet50(
            pretrained=False,
            n_input_channels=1,
            widen_factor=2,
            conv1_t_stride=2,
            feed_forward=False,
            bias_downsample=True,
        )
        weights_url = "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
        weights_path = get_model_weights(weights_url)
        model = LoadModel(
            trunk=trunk, weights_path=Path(weights_path), heads=[4096, 2048, num_classes]
        )

    else:
        trunk = resnet50(
            pretrained=True,
            n_input_channels=1,
            feed_forward=False,
            bias_downsample=False,
            num_classes=num_classes,
        )
        model = LoadModel(trunk=trunk, weights_path=None, heads=[2048, 512, num_classes])

    return model
