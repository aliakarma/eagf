"""Model factory for EAGF framework.

Dispatches to the appropriate architecture based on config.
"""
import torch.nn as nn


def build_model(architecture: str, in_dim: int, n_classes: int, **kwargs) -> nn.Module:
    """Instantiate a model by architecture name.

    Args:
        architecture: One of "tabular_mlp", "resnet50", "lstm".
        in_dim: Input feature dimension (ignored for resnet50 image input).
        n_classes: Number of output classes.
        **kwargs: Architecture-specific options (hidden_dims, dropout, etc.).
    """
    arch = architecture.lower().replace("-", "_")

    if arch == "tabular_mlp":
        from src.models.tabular_mlp import TabularMLP
        hidden = kwargs.get("hidden_dims", (256, 128, 64))
        dropout = kwargs.get("dropout", 0.3)
        return TabularMLP(in_dim=in_dim, hidden=tuple(hidden),
                          n_classes=n_classes, dropout=dropout)

    elif arch == "resnet50":
        from src.models.resnet50 import ResNet50Classifier
        pretrained = kwargs.get("pretrained", True)
        dropout = kwargs.get("dropout", 0.3)
        return ResNet50Classifier(n_classes=n_classes, pretrained=pretrained,
                                   dropout=dropout)

    elif arch == "lstm":
        from src.models.lstm import BiLSTMClassifier
        hidden_size = kwargs.get("hidden_size", 128)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.3)
        bidirectional = kwargs.get("bidirectional", True)
        return BiLSTMClassifier(input_dim=in_dim, hidden_size=hidden_size,
                                 num_layers=num_layers, n_classes=n_classes,
                                 dropout=dropout, bidirectional=bidirectional)

    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                         f"Supported: tabular_mlp, resnet50, lstm")
