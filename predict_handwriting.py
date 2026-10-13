"""Run the trained MLPs on my own handwritten student-ID digits.

Put photos/scans in handwriting/raw/. Two kinds of input are supported:
  * one image per digit, e.g.  01_2.jpg, 02_1.jpg, ... (the digit after the
    underscore is the true label)
  * one image of the whole student number, e.g. student_id.jpg — it is split
    into digits automatically and compared against STUDENT_ID.

Every digit goes through the same MNIST-style preprocessing:
  grayscale -> white digit on black background -> crop to the digit ->
  resize so the longest side is 20 px -> paste into a 28x28 canvas centred on
  the centre of mass -> scale to [0, 1] -> normalize with MNIST mean/std.

Usage:
    python predict_handwriting.py
    python predict_handwriting.py --models exp1_baseline augmented_dropout
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

from mlp import MLP, normalize

STUDENT_ID = "21120205709"
RAW_DIR = "handwriting/raw"
PROCESSED_DIR = "handwriting/processed"
RESULTS_DIR = "results"


# ---------------------------------------------------------------- preprocessing

def to_ink(img):
    """Grayscale array in [0, 1] where ink is bright and paper is dark (MNIST style)."""
    gray = ImageOps.exif_transpose(img).convert("L")
    # Remove paper texture / sensor noise and even out uneven lighting.
    gray = gray.filter(ImageFilter.GaussianBlur(1))
    a = np.asarray(gray, dtype=np.float32) / 255.0
    # Photos are usually dark ink on light paper: invert if the border is bright.
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if border.mean() > 0.5:
        a = 1.0 - a
    # Subtract the background level and stretch the ink to full brightness.
    background = np.median(a)
    a = np.clip(a - background, 0, None)
    if a.max() > 0:
        a = a / np.percentile(a[a > 0], 99)
    a = np.clip(a, 0, 1)
    # Suppress faint leftovers (shadows, paper grain).
    a[a < 0.25] = 0
    return a


def mnist_style(ink):
    """Crop, resize to fit 20x20, centre by centre of mass in 28x28."""
    ys, xs = np.nonzero(ink > 0)
    if len(xs) == 0:
        return np.zeros((28, 28), np.float32)
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(small, dtype=np.float32) / 255.0

    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = small

    # Shift so the centre of mass sits at the image centre, like MNIST.
    total = canvas.sum()
    cy = (canvas.sum(1) * np.arange(28)).sum() / total
    cx = (canvas.sum(0) * np.arange(28)).sum() / total
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    canvas = np.roll(canvas, (dy, dx), axis=(0, 1))
    return np.clip(canvas, 0, 1)


def split_digits(ink):
    """Split an image of a written number into digit images (left to right).

    Finds connected blobs of ink. Blobs touching the image border (page edges,
    the table behind the paper) and tiny specks are ignored. Blobs that overlap
    horizontally are merged, so a digit drawn with two strokes stays one digit.
    """
    labels, _ = ndimage.label(ink > 0, structure=np.ones((3, 3)))
    h, w = ink.shape
    blobs = []
    for i, (ys, xs) in enumerate(ndimage.find_objects(labels), start=1):
        if ys.start == 0 or xs.start == 0 or ys.stop == h or xs.stop == w:
            continue
        blobs.append(dict(x0=xs.start, x1=xs.stop, ids=[i], size=int((labels[ys, xs] == i).sum())))
    if not blobs:
        return []
    biggest = max(b["size"] for b in blobs)
    blobs = sorted((b for b in blobs if b["size"] >= 0.05 * biggest), key=lambda b: b["x0"])

    digits = []
    for b in blobs:
        if digits:
            last = digits[-1]
            overlap = min(last["x1"], b["x1"]) - max(last["x0"], b["x0"])
            if overlap > 0.5 * min(last["x1"] - last["x0"], b["x1"] - b["x0"]):
                last["x1"] = max(last["x1"], b["x1"])
                last["ids"] += b["ids"]
                continue
        digits.append(dict(b))

    # Keep only this digit's own ink, so parts of neighbouring digits are not included.
    return [np.where(np.isin(labels, d["ids"]), ink, 0) for d in digits]


def load_digits(raw_dir=RAW_DIR):
    """Return a list of (source_name, true_label, 28x28 array in [0, 1])."""
    files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, "*"))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    )
    if not files:
        raise SystemExit(f"No images found in {raw_dir}/")

    digits = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        ink = to_ink(Image.open(path))
        m = re.search(r"_(\d)$", stem)
        if m:  # single digit, label in the file name
            digits.append((stem, int(m.group(1)), mnist_style(ink)))
            continue
        parts = split_digits(ink)
        if len(parts) != len(STUDENT_ID):
            print(
                f"warning: {stem}: found {len(parts)} digits, expected {len(STUDENT_ID)}; "
                "labels may be misaligned (try more spacing between digits)"
            )
        for i, part in enumerate(parts):
            label = int(STUDENT_ID[i]) if i < len(STUDENT_ID) else -1
            digits.append((f"{stem}[{i}]", label, mnist_style(part)))
    return digits


# ---------------------------------------------------------------- prediction

def load_model(name):
    ckpt = torch.load(os.path.join("models", f"{name}.pt"), weights_only=False)
    cfg = ckpt["config"]
    model = MLP(cfg["hidden"], cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def save_processed(digits):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, label, arr in digits:
        safe = re.sub(r"[^\w-]", "_", name)
        Image.fromarray((arr * 255).astype(np.uint8)).save(
            os.path.join(PROCESSED_DIR, f"{safe}_label{label}.png")
        )


def plot_predictions(digits, results, out_path):
    n = len(digits)
    fig, axes = plt.subplots(1, n, figsize=(1.3 * n, 2.6))
    axes = np.atleast_1d(axes)
    for i, (ax, (name, label, arr)) in enumerate(zip(axes, digits)):
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        text = [f"true {label}"]
        color = "green"
        for model_name, (preds, confs) in results.items():
            short = {"exp1_baseline": "base", "augmented_dropout": "aug"}.get(model_name, model_name[:8])
            text.append(f"{short}: {preds[i]} ({confs[i] * 100:.0f}%)")
            if preds[i] != label:
                color = "red"
        ax.set_title("\n".join(text), fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--models", nargs="*", default=["exp1_baseline", "augmented_dropout"])
    args = parser.parse_args()

    digits = load_digits(args.raw_dir)
    save_processed(digits)
    x = normalize(torch.tensor(np.stack([d[2] for d in digits])).unsqueeze(1))
    labels = [d[1] for d in digits]

    results, lines = {}, []
    with torch.no_grad():
        for name in args.models:
            probs = torch.softmax(load_model(name)(x), 1)
            conf, pred = probs.max(1)
            results[name] = (pred.tolist(), conf.tolist())
            correct = sum(p == l for p, l in zip(pred.tolist(), labels))
            predicted_id = "".join(map(str, pred.tolist()))
            lines.append(
                f"{name}: predicted {predicted_id}  true {''.join(map(str, labels))}  "
                f"-> {correct}/{len(labels)} correct ({correct / len(labels) * 100:.1f}%)"
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_predictions(digits, results, os.path.join(RESULTS_DIR, "handwriting_predictions.png"))

    lines.append("")
    lines.append("| Digit | True | " + " | ".join(args.models) + " |")
    lines.append("|---|---|" + "---|" * len(args.models))
    for i, (name, label, _) in enumerate(digits):
        cells = [f"{results[m][0][i]} ({results[m][1][i] * 100:.0f}%)" for m in args.models]
        lines.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "handwriting_results.md"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
