import torch
from torch import nn


class MLP(nn.Module):
  """Multi-layer perceptron for MNIST classification."""

  def __init__(
    self,
    input_size: int = 784,
    hidden_sizes: tuple[int, ...] = (128, 64),
    num_classes: int = 10,
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    prev_size = input_size

    for hidden_size in hidden_sizes:
      layers.extend(
        [
          nn.Linear(prev_size, hidden_size),
          nn.ReLU(),
        ]
      )
      prev_size = hidden_size

    layers.append(nn.Linear(prev_size, num_classes))
    self.network = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    return self.network(x)
