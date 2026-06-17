import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mlp import MLP


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train an MLP on MNIST.")
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--data-dir", type=Path, default=Path("data"))
  parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
  parser.add_argument("--device", type=str, default="auto")
  return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
  if device_arg == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(device_arg)


def create_dataloaders(
  data_dir: Path,
  batch_size: int,
) -> tuple[DataLoader, DataLoader]:
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
    ]
  )

  train_dataset = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
  )
  test_dataset = datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
  )

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, test_loader


def train_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> tuple[float, float]:
  model.train()
  total_loss = 0.0
  correct = 0
  total = 0

  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * images.size(0)
    predictions = outputs.argmax(dim=1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

  return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  device: torch.device,
) -> tuple[float, float]:
  model.eval()
  total_loss = 0.0
  correct = 0
  total = 0

  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    total_loss += loss.item() * images.size(0)
    predictions = outputs.argmax(dim=1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

  return total_loss / total, correct / total


def main() -> None:
  args = parse_args()
  device = resolve_device(args.device)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  train_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)
  model = MLP().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  print(f"Training on {device}")

  for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_epoch(
      model, train_loader, criterion, optimizer, device
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(
      f"Epoch {epoch}/{args.epochs} "
      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
      f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
    )

  checkpoint_path = args.output_dir / "mlp_mnist.pt"
  torch.save(
    {
      "model_state_dict": model.state_dict(),
      "hidden_sizes": (128, 64),
      "num_classes": 10,
    },
    checkpoint_path,
  )
  print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
  main()
