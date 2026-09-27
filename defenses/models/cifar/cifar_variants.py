"""32x32 ("CIFAR-style") variants of ResNet-50, DenseNet-121 and MobileNetV2.

The multi-dataset benchmark follows FDINet: VGG19 on CIFAR10, MobileNetV2 on
GTSRB, DenseNet121 on CelebA and ResNet50 on Skin Cancer, all at 32x32 with the
stem adjusted for the small input.

These wrap the torchvision architectures (so the layer counts are the real
ones) and apply the standard small-input adaptation:

- ResNet-50: 3x3 stride-1 first convolution, no max-pool (32 -> 4 spatial).
- DenseNet-121: 3x3 stride-1 first convolution, no max-pool (32 -> 4).
- MobileNetV2: stride-1 first convolution and stride 1 in the 24-channel stage
  (the usual CIFAR MobileNetV2), leaving 4x4 before pooling.

Note: ``defenses.models.cifar.resnet50`` is *not* a ResNet-50 -- it is the
BasicBlock ResNet-56 of the bearpaw CIFAR code (~0.85M parameters) -- and
``densenet`` is a small DenseNet-BC. Use the ``*_cifar`` names below for the
FDINet architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models

__all__ = ["resnet50_cifar", "densenet121_cifar", "mobilenetv2_cifar"]


class _BackboneClassifier(nn.Module):
    """Feature extractor + global average pool + linear head.

    ``rot_semi``/``rot_forward`` mirror the interface of the repository's VGG
    and ResNet models, which the semi-supervised S4L attack mode calls.
    """

    def __init__(
        self,
        features: nn.Module,
        feature_dim: int,
        num_classes: int,
        *,
        rot_semi: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.rot_semi = bool(rot_semi)
        if self.rot_semi:
            self.rot_classifier = nn.Linear(feature_dim, 4)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))

    def rot_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.rot_semi, "Model was built without rot_semi=True."
        return self.rot_classifier(self.embed(x))


def resnet50_cifar(num_classes: int = 10, rot_semi: bool = False, **kwargs) -> nn.Module:
    net = tv_models.resnet50(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    features = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3, net.layer4,
    )
    return _BackboneClassifier(features, 2048, num_classes, rot_semi=rot_semi)


def densenet121_cifar(num_classes: int = 10, rot_semi: bool = False, **kwargs) -> nn.Module:
    net = tv_models.densenet121(weights=None)
    net.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.features.pool0 = nn.Identity()
    # torchvision applies the final ReLU in DenseNet.forward, not in features.
    features = nn.Sequential(net.features, nn.ReLU(inplace=True))
    return _BackboneClassifier(features, 1024, num_classes, rot_semi=rot_semi)


_MOBILENETV2_CIFAR_SETTING = [
    # t, c, n, s  -- torchvision's setting with the 24-channel stage at stride 1.
    [1, 16, 1, 1],
    [6, 24, 2, 1],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


def mobilenetv2_cifar(num_classes: int = 10, rot_semi: bool = False, **kwargs) -> nn.Module:
    net = tv_models.mobilenet_v2(weights=None, inverted_residual_setting=_MOBILENETV2_CIFAR_SETTING)
    first = net.features[0][0]
    net.features[0][0] = nn.Conv2d(
        first.in_channels, first.out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )
    return _BackboneClassifier(net.features, net.last_channel, num_classes, rot_semi=rot_semi, dropout=0.2)
