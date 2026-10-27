import argparse
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .data import make_loaders
from .engine import train_one_epoch, evaluate
from .evaluate import save_test_report
from .model import MODEL_NAMES, build_model, n_params

EXPERIMENTS = {
    "lenet_avg": (64, True, 1e-3),
    "lenet_max": (64, True, 1e-3),
    "googlenet": (64, True, 1e-3),
    "resnet18_noaug": (96, False, 1e-3),
    "resnet18_aug": (96, True, 1e-3),
    "resnet50_fe": (224, True, 1e-3),
    "resnet50_ft": (224, True, 1e-4),
}

def run_experiment(name, data_root, output_root, epochs=20, batch_size=32,
                   num_workers=2, seed=42, val_ratio=0.10):
    img_size, augment, lr = EXPERIMENTS[name]
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dl, classes = make_loaders(
        data_root, img_size, augment, batch_size, val_ratio,
        num_workers, seed)
    folder = Path(output_root) / name
    folder.mkdir(parents=True, exist_ok=True)

    model = build_model(name, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    print(name, "| device:", device, "| parameters:", n_params(model),
          "| trainable:", n_params(model, trainable_only=True))

    history = []
    best_f1, best_epoch = -1.0, 0
    for ep in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, truth, pred = evaluate(
            model, val_dl, criterion, device)
        va_f1 = f1_score(
            truth, pred, labels=list(range(len(classes))),
            average="macro", zero_division=0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        history.append({
            "epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
            "train_acc": tr_acc, "val_acc": va_acc,
            "val_macro_f1": va_f1, "seconds": seconds,
        })
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, ep
            torch.save({
                "model_name": name, "img_size": img_size,
                "classes": classes, "state_dict": model.state_dict(),
                "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
            }, folder / "best.pt")
        print(f"{name} | epoch {ep:02d}/{epochs} | "
              f"train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f}, acc {va_acc:.4f}, F1 {va_f1:.4f}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(folder / "history.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "loss_curve.png")
    plt.close(fig)

    checkpoint = torch.load(
        folder / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = save_test_report(model, test_dl, classes, folder, device)
    metrics.update({
        "name": name, "img_size": img_size,
        "parameters": n_params(model),
        "trainable_parameters": n_params(model, trainable_only=True),
        "seconds_per_epoch": history_df["seconds"].mean(),
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "epochs": epochs, "n_train": len(train_dl.dataset),
        "n_val": len(val_dl.dataset),
    })
    pd.DataFrame([metrics]).to_csv(folder / "metrics.csv", index=False)
    return metrics

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Huấn luyện CNN trên VinaFood21")
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs phải >= 1")

    names = MODEL_NAMES if args.model == "all" else (args.model,)
    results = [run_experiment(
        name, args.data_root, args.output_root, args.epochs,
        args.batch_size, args.num_workers, args.seed, args.val_ratio,
    ) for name in names]
    pd.DataFrame(results).to_csv(
        args.output_root / "comparison_runs.csv", index=False)
    print(pd.DataFrame(results)[["name", "accuracy", "macro_f1"]].round(4))


if __name__ == "__main__":
    main()
