# this-is-a-test

PyTorch MLP trained on the MNIST handwritten digit dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Optional flags:

- `--epochs` (default: 5)
- `--batch-size` (default: 64)
- `--learning-rate` (default: 0.001)
- `--data-dir` (default: `data`)
- `--output-dir` (default: `checkpoints`)

The trained model checkpoint is saved to `checkpoints/mlp_mnist.pt`.
