"""Shared code: MNIST data splits, the MLP model, and the train / evaluate loops."""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# Mean / std of the MNIST training pixels (after scaling to [0, 1]).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

DATA_DIR = "data"
SPLIT_SEED = 42
VAL_SIZE = 10_000


def normalize(x):
    """x: float tensor in [0, 1] -> standardized with MNIST statistics."""
    return (x - MNIST_MEAN) / MNIST_STD


def load_mnist_splits():
    """Return (train, val, test) as (images, labels) tensor pairs.

    - The official 60,000 MNIST training images are split into
      50,000 train / 10,000 validation with a fixed seed.
    - The official 10,000 MNIST test images are kept completely separate and
      are only used once, after all training and model selection is done.
    Images are float tensors of shape (N, 1, 28, 28), normalized.
    """
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)

    x_full = normalize(train_full.data.float().div(255).unsqueeze(1))
    y_full = train_full.targets
    x_test = normalize(test.data.float().div(255).unsqueeze(1))
    y_test = test.targets

    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(x_full), generator=g)
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]

    return (
        (x_full[train_idx], y_full[train_idx]),
        (x_full[val_idx], y_full[val_idx]),
        (x_test, y_test),
    )


class MLP(nn.Module):
    """Fully connected network: 784 -> hidden layers (ReLU) -> 10 logits."""

    def __init__(self, hidden=(256, 128), dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        in_features = 28 * 28
        for h in hidden:
            layers += [nn.Linear(in_features, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def random_affine(x, max_rotation=12, max_shift=0.12, scale_range=(0.85, 1.15)):
    """Per-image random rotation / shift / zoom (data augmentation).

    Works on the whole batch at once with affine_grid, so it stays fast.
    The background value of a normalized MNIST image is the normalized 0,
    so we pad with that.
    """
    n = x.size(0)
    angle = (torch.rand(n) * 2 - 1) * math.radians(max_rotation)
    scale = torch.empty(n).uniform_(*scale_range)
    tx = (torch.rand(n) * 2 - 1) * max_shift * 2  # grid coords span [-1, 1]
    ty = (torch.rand(n) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1
    )
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    background = normalize(torch.tensor(0.0))
    out = F.grid_sample(x - background, grid, align_corners=False, padding_mode="zeros")
    return out + background


def batches(x, y, batch_size, shuffle):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i : i + batch_size]
        yield x[b], y[b]


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    """Return (mean cross-entropy loss, accuracy) on the given data."""
    model.eval()
    total_loss, correct = 0.0, 0
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)


def make_optimizer(model, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.0))
    raise ValueError(f"unknown optimizer {cfg['optimizer']}")


def train(cfg, train_data, val_data, log=print):
    """Train one configuration. Only train + validation data are seen here."""
    torch.manual_seed(cfg.get("seed", 0))
    model = MLP(cfg["hidden"], cfg.get("dropout", 0.0))
    optimizer = make_optimizer(model, cfg)
    loss_fn = nn.CrossEntropyLoss()
    (x_train, y_train), (x_val, y_val) = train_data, val_data

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss, correct = 0.0, 0
        for xb, yb in batches(x_train, y_train, cfg["batch_size"], shuffle=True):
            if cfg.get("augment"):
                xb = random_affine(xb)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        train_loss = running_loss / len(x_train)
        train_acc = correct / len(x_train)
        val_loss, val_acc = evaluate(model, x_val, y_val)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(time.perf_counter() - t0)
        log(
            f"  epoch {epoch:2d}/{cfg['epochs']}  "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    history["training_time_sec"] = time.perf_counter() - start
    return model, history
