"""Train the MLP configurations on MNIST and compare them.

Usage:
    python train.py                      # run every experiment
    python train.py --only exp1_baseline # run a single experiment

Protocol (no test data during development):
  1. Every configuration is trained on the 50k train split and monitored on
     the 10k validation split.
  2. The best configuration is chosen by validation accuracy only.
  3. After all training is finished, the 10k MNIST test set is evaluated once
     per model, purely for reporting.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mlp import count_parameters, evaluate, load_mnist_splits, train

torch.set_num_threads(max(1, os.cpu_count() or 1))

BASE = dict(
    hidden=[256, 128],
    dropout=0.0,
    optimizer="adam",
    lr=1e-3,
    batch_size=64,
    epochs=15,
    augment=False,
    seed=0,
)

# Experiment 1 is the baseline; the rest change one thing at a time.
EXPERIMENTS = {
    "exp1_baseline": dict(),
    "lr_0.01": dict(lr=1e-2),
    "lr_0.0001": dict(lr=1e-4),
    "sgd_momentum": dict(optimizer="sgd", lr=0.01, momentum=0.9),
    "small_64_32": dict(hidden=[64, 32]),
    "deep_512_256_128": dict(hidden=[512, 256, 128]),
    "dropout_0.2": dict(dropout=0.2),
    "batch_256": dict(batch_size=256),
    "augmented_dropout": dict(hidden=[512, 256, 128], dropout=0.2, augment=True, epochs=25),
}

RESULTS_DIR = "results"
MODELS_DIR = "models"


def plot_history(name, h):
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, h["train_loss"], label="train")
    ax1.plot(epochs, h["val_loss"], label="validation")
    ax1.set(title=f"{name} – loss", xlabel="epoch", ylabel="cross-entropy loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [a * 100 for a in h["train_acc"]], label="train")
    ax2.plot(epochs, [a * 100 for a in h["val_acc"]], label="validation")
    ax2.set(title=f"{name} – accuracy", xlabel="epoch", ylabel="accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{name}_curves.png"), dpi=120)
    plt.close(fig)


def plot_comparison(histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        epochs = range(1, len(h["val_loss"]) + 1)
        ax1.plot(epochs, h["val_loss"], label=name)
        ax2.plot(epochs, [a * 100 for a in h["val_acc"]], label=name)
    ax1.set(title="Validation loss", xlabel="epoch", ylabel="loss")
    ax2.set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy (%)", ylim=(94, 100))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "comparison_validation.png"), dpi=120)
    plt.close(fig)


def write_summary(rows, best_name):
    lines = [
        "| Experiment | Hidden layers | Optimizer | LR | Batch | Dropout | Aug. | Epochs | Params "
        "| Train loss | Val loss | Train acc | Val acc | Test acc | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["config"]
        name = f"**{r['name']}**" if r["name"] == best_name else r["name"]
        lines.append(
            f"| {name} | {'-'.join(map(str, c['hidden']))} | {c['optimizer']} | {c['lr']} "
            f"| {c['batch_size']} | {c['dropout']} | {'yes' if c['augment'] else 'no'} | {c['epochs']} "
            f"| {r['params']:,} | {r['train_loss']:.4f} | {r['val_loss']:.4f} "
            f"| {r['train_acc'] * 100:.2f}% | {r['val_acc'] * 100:.2f}% | {r['test_acc'] * 100:.2f}% "
            f"| {r['training_time_sec']:.1f} |"
        )
    lines.append("")
    lines.append(f"Best configuration by **validation** accuracy: `{best_name}`")
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="names of experiments to run")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_data, val_data, test_data = load_mnist_splits()
    print(
        f"train: {len(train_data[0])}  validation: {len(val_data[0])}  "
        f"test (held out): {len(test_data[0])}"
    )

    names = args.only or list(EXPERIMENTS)
    histories, models, configs = {}, {}, {}
    for name in names:
        cfg = {**BASE, **EXPERIMENTS[name]}
        print(f"\n=== {name} === {cfg}")
        model, history = train(cfg, train_data, val_data)
        print(f"  training time: {history['training_time_sec']:.1f}s")
        histories[name], models[name], configs[name] = history, model, cfg
        plot_history(name, history)
        torch.save({"config": cfg, "state_dict": model.state_dict()}, os.path.join(MODELS_DIR, f"{name}.pt"))

    # Model selection uses validation accuracy only.
    best_name = max(histories, key=lambda n: histories[n]["val_acc"][-1])

    # Training is over: now (and only now) touch the test set, once per model.
    rows = []
    for name in names:
        h = histories[name]
        test_loss, test_acc = evaluate(models[name], *test_data)
        row = dict(
            name=name,
            config=configs[name],
            params=count_parameters(models[name]),
            train_loss=h["train_loss"][-1],
            val_loss=h["val_loss"][-1],
            train_acc=h["train_acc"][-1],
            val_acc=h["val_acc"][-1],
            test_loss=test_loss,
            test_acc=test_acc,
            training_time_sec=h["training_time_sec"],
        )
        rows.append(row)
        with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
            json.dump({**row, "history": h}, f, indent=2)

    if len(names) > 1:
        plot_comparison(histories)
    print("\n" + write_summary(rows, best_name))


if __name__ == "__main__":
    main()
