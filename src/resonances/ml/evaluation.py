import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from resonances.ml.training import model_prediction


def get_predictions(model, dataloader, model_type, device):
    labels = []
    probabilities = []
    model.eval()

    with torch.no_grad():
        for series, scalars, batch_labels in dataloader:
            series = series.to(device)
            scalars = scalars.to(device)

            logits = model_prediction(
                model,
                series,
                scalars,
                model_type
            ).squeeze(dim=1)
            batch_probabilities = torch.sigmoid(logits)

            labels.extend(batch_labels.numpy())
            probabilities.extend(batch_probabilities.cpu().numpy())

    return np.asarray(labels), np.asarray(probabilities)


def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "brier_score": brier_score_loss(labels, probabilities)
    }


def plot_loss(history, test_loss, model_name, output_file):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss"], label="Train")
    axis.plot(history["epoch"], history["test_loss"], label="Test")
    axis.axhline(
        test_loss,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Selected test loss"
    )
    axis.set_title(f"Loss function | {model_name}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("BCE loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_roc(
    train_labels,
    train_probabilities,
    test_labels,
    test_probabilities,
    model_name,
    output_file
):
    train_fpr, train_tpr, _ = roc_curve(train_labels, train_probabilities)
    test_fpr, test_tpr, _ = roc_curve(test_labels, test_probabilities)
    train_auc = roc_auc_score(train_labels, train_probabilities)
    test_auc = roc_auc_score(test_labels, test_probabilities)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(train_fpr, train_tpr, label=f"Train AUC = {train_auc:.3f}")
    axis.plot(test_fpr, test_tpr, label=f"Test AUC = {test_auc:.3f}")
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5)
    axis.set_title(f"ROC curve | {model_name}")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(labels, probabilities, model_name, output_file):
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    axis.set_title(f"Test confusion matrix | {model_name}")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_probability_distribution(labels, probabilities, model_name, output_file):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        probabilities[labels == 0],
        bins=20,
        alpha=0.7,
        label="Label 0"
    )
    axis.hist(
        probabilities[labels == 1],
        bins=20,
        alpha=0.7,
        label="Label 1"
    )
    axis.axvline(0.5, color="black", linestyle="--", label="Threshold")
    axis.set_title(f"Test probability distribution | {model_name}")
    axis.set_xlabel("Predicted probability of libration")
    axis.set_ylabel("Objects")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(metrics, output_file):
    columns = ["roc_auc", "balanced_accuracy", "f1"]
    positions = np.arange(len(metrics))
    width = 0.25

    figure, axis = plt.subplots(figsize=(9, 5))

    for index, column in enumerate(columns):
        axis.bar(
            positions + (index - 1) * width,
            metrics[column],
            width,
            label=column.replace("_", " ")
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(metrics["model"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test score")
    axis.set_title("Model comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_combined_architecture(scalar_input_size, output_file):
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")

    boxes = [
        (0.5, 4.2, 1.6, 0.8, "Series\n1 × 128", "#d6eaf8"),
        (2.7, 4.2, 1.8, 0.8, "Conv1d\n8 channels", "#aed6f1"),
        (5.1, 4.2, 1.8, 0.8, "Conv1d\n16 channels", "#85c1e9"),
        (7.5, 4.2, 1.6, 0.8, "Dense\n32", "#5dade2"),
        (0.5, 1.2, 1.6, 0.8, f"Scalars\n{scalar_input_size}", "#d5f5e3"),
        (3.2, 1.2, 1.6, 0.8, "Dense\n64", "#a9dfbf"),
        (6.0, 1.2, 1.6, 0.8, "Dense\n32", "#7dcea0"),
        (9.5, 2.7, 1.8, 0.8, "Classifier\n64 → 32 → 1", "#f9e79f")
    ]

    for x, y, width, height, text, color in boxes:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="black"
        )
        axis.add_patch(box)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")

    arrows = [
        ((2.1, 4.6), (2.7, 4.6)),
        ((4.5, 4.6), (5.1, 4.6)),
        ((6.9, 4.6), (7.5, 4.6)),
        ((2.1, 1.6), (3.2, 1.6)),
        ((4.8, 1.6), (6.0, 1.6)),
        ((9.1, 4.6), (9.8, 3.5)),
        ((7.6, 1.6), (9.8, 2.7))
    ]

    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "linewidth": 1.5}
        )

    axis.set_title("Combined network architecture", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)
