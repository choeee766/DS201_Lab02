import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from .data import make_test_loader
from .engine import evaluate, four_metrics
from .model import build_model

def save_test_report(model, test_dl, classes, folder, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    test_loss, _, y_true, y_pred = evaluate(
        model, test_dl, criterion, device)
    labels = list(range(len(classes)))
    metrics = four_metrics(y_true, y_pred, len(classes))
    metrics["test_loss"] = test_loss
    metrics["n_test"] = len(y_true)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        digits=4, zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    print(report)
    (folder / "classification_report.txt").write_text(report, encoding="utf-8")
    per_class = pd.DataFrame(report_dict).T.loc[
        classes, ["precision", "recall", "f1-score", "support"]]
    per_class.to_csv(folder / "per_class_metrics.csv", encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        folder / "confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax)
    ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thật",
           title="Confusion matrix trên tập test")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png")
    plt.close(fig)
    pd.DataFrame([metrics]).to_csv(folder / "test_metrics.csv", index=False)
    return metrics

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint Lab 2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True)
    name = checkpoint["model_name"]
    classes = checkpoint["classes"]
    img_size = checkpoint["img_size"]
    test_dl, test_classes = make_test_loader(
        args.data_root, img_size, args.batch_size, args.num_workers)
    if test_classes != classes:
        raise ValueError("Tên/thứ tự lớp test khác checkpoint")
    model = build_model(name, num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = save_test_report(
        model, test_dl, classes, args.checkpoint.parent, device)
    print(f"{name}: test macro-F1 = {metrics['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
