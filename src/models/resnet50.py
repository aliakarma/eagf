"""ResNet-50 classifier for CS1 biometric case study.

Paper Section 5.1: ResNet-50 with ImageNet pretraining, Dropout 0.3,
GroupNorm (Opacus-compatible replacement for BatchNorm).
"""
import torch
import torch.nn as nn

try:
    import torchvision.models as tv_models
    _HAS_TORCHVISION = True
except ImportError:
    tv_models = None
    _HAS_TORCHVISION = False


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """Replace all BatchNorm layers with GroupNorm for Opacus DP-SGD compatibility."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                num_groups=min(num_groups, num_channels),
                num_channels=num_channels,
                affine=True,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)
    return module


class ResNet50Classifier(nn.Module):
    """ResNet-50 with GroupNorm head for biometric classification.

    Architecture (paper Section 5.1, Table 7):
        - Backbone: ResNet-50, ImageNet-pretrained
        - BatchNorm → GroupNorm (Opacus compatibility)
        - Head: Dropout(0.3) → Linear(2048, num_classes)
    """

    def __init__(self, n_classes: int = 158, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for ResNet-50. "
                              "Install: pip install torchvision")

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        _replace_bn_with_gn(backbone)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Image tensor (B, 3, 224, 224) or flattened features (B, D).
               If flattened and D != 3*224*224, treats as feature vector
               and projects through a linear layer first.
        """
        if x.ndim == 2:
            if x.shape[1] == 3 * 224 * 224:
                x = x.view(-1, 3, 224, 224)
            else:
                raise ValueError(
                    f"ResNet-50 expects image input (B, 3, 224, 224) or "
                    f"flattened (B, {3*224*224}), got (B, {x.shape[1]})")
        features = self.backbone(x)
        return self.head(features)
