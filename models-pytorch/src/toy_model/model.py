import torch
from torch import nn


class LinearNet(nn.Module):  # type: ignore[misc]
    """Toy MLP, trainable on CPU."""

    def __init__(self, layer_dims: list[int]) -> None:
        """Initialise LinearNet.

        Args:
            layer_dims: dimensions for each layer including input and output
        """
        super().__init__()
        layers = []

        for idx in range(len(layer_dims) - 2):
            layers.append(
                nn.Linear(in_features=layer_dims[idx], out_features=layer_dims[idx + 1])
            )
            layers.append(nn.ReLU())

        layers.append(
            nn.Linear(in_features=layer_dims[-2], out_features=layer_dims[-1])
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x
