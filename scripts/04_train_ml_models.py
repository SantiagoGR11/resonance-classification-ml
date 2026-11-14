from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import torch

from resonances.ml.evaluation import (
    calculate_metrics,
    get_predictions,
    plot_combined_architecture,
    plot_confusion_matrix,
    plot_loss,
    plot_model_comparison,
    plot_probability_distribution,
    plot_roc,
)
from resonances.ml.models import (
    CombinedNetwork,
    ConvNetwork,
    count_parameters,
)
from resonances.ml.training import (
    create_dataloader,
    normalize_scalar_inputs,
    test_loop,
    train_model,
)


# Creation of the necessary directories for storing the results
ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = ROOT / "data" / "ml" / "datasets"
EVALUATION_DIR = ROOT / "outputs" / "ml" / "evaluation"
MODELS_DIR = EVALUATION_DIR / "models"
METRICS_DIR = EVALUATION_DIR / "metrics"
PREDICTIONS_DIR = EVALUATION_DIR / "predictions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

SCALAR_FILE = DATASETS_DIR / "scalar_inputs.csv"
SERIES_FILE = DATASETS_DIR / "time_series_inputs.csv"
PREPROCESSING_METADATA_FILE = DATASETS_DIR / "preprocessing_metadata.txt"
SPLIT_FILE = METRICS_DIR / "data_split.csv"

copyfile(
    PREPROCESSING_METADATA_FILE,
    EVALUATION_DIR / "preprocessing_metadata.txt"
)


# Training configuration
MODEL_RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

SCALAR_FEATURES = [
    "per_lib",
    "delta_a",
    "relative_delta_a",
    "physical_duration",
    "duration_per_lib",
    "original_final_displacement",
    "treated_final_displacement",
    "slope_to_maximum",
    "original_deviation_area",
    "treated_deviation_area",
    "first_half_deviation_area",
    "cut_reached",
]


# Read and align scalar and time-series inputs
if not SCALAR_FILE.exists() or not SERIES_FILE.exists():
    print("Run scripts/03_prepare_ml_dataset.py before training.")
    raise SystemExit

scalar_inputs = pd.read_csv(SCALAR_FILE, dtype={"spkid": str})
time_series_inputs = pd.read_csv(SERIES_FILE, dtype={"spkid": str})
data = scalar_inputs.merge(
    time_series_inputs,
    on=["case_id", "spkid"],
    how="inner"
)

series_columns = [
    column for column in time_series_inputs.columns
    if column.startswith("sigma_")
]

scalar_values = data[SCALAR_FEATURES].to_numpy(dtype=np.float32)
series_values = data[series_columns].to_numpy(dtype=np.float32)
labels = data["label"].to_numpy(dtype=np.float32)
indices = np.arange(len(data))


# Read the selected 80/20 split.
saved_split = pd.read_csv(SPLIT_FILE, dtype={"spkid": str})
split_by_case = saved_split.set_index("case_id")["split"]
split_values = data["case_id"].map(split_by_case)

train_indices = indices[split_values == "train"]
test_indices = indices[split_values == "test"]

train_scalars = scalar_values[train_indices]
test_scalars = scalar_values[test_indices]

(
    train_scalars,
    test_scalars,
    scalar_mean,
    scalar_standard_deviation,
) = normalize_scalar_inputs(
    train_scalars,
    test_scalars
)


# Save the split and scalar normalization
split = pd.DataFrame({
    "case_id": data["case_id"],
    "spkid": data["spkid"],
    "sampling_group": data["sampling_group"],
    "label": data["label"],
    "split": ""
})
split.loc[train_indices, "split"] = "train"
split.loc[test_indices, "split"] = "test"
split.to_csv(METRICS_DIR / "data_split.csv", index=False, encoding="utf-8")

normalization = pd.DataFrame({
    "feature": SCALAR_FEATURES,
    "mean": scalar_mean,
    "standard_deviation": scalar_standard_deviation
})
normalization.to_csv(
    MODELS_DIR / "scalar_normalization.csv",
    index=False,
    encoding="utf-8"
)

plot_combined_architecture(
    scalar_input_size=len(SCALAR_FEATURES),
    output_file=METRICS_DIR / "combined_network_architecture.png"
)


# Train and evaluate both models with the same data split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics_rows = []
group_metrics_rows = []

for model_type in ["conv", "combined"]:
    np.random.seed(MODEL_RANDOM_STATE)
    torch.manual_seed(MODEL_RANDOM_STATE)
    torch.cuda.manual_seed_all(MODEL_RANDOM_STATE)

    train_dataloader = create_dataloader(
        series_values[train_indices],
        train_scalars,
        labels[train_indices],
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    train_evaluation_dataloader = create_dataloader(
        series_values[train_indices],
        train_scalars,
        labels[train_indices],
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_dataloader = create_dataloader(
        series_values[test_indices],
        test_scalars,
        labels[test_indices],
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    if model_type == "conv":
        model = ConvNetwork(
            series_length=len(series_columns),
            dropout=DROPOUT
        )
    else:
        model = CombinedNetwork(
            scalar_input_size=len(SCALAR_FEATURES),
            series_length=len(series_columns),
            dropout=DROPOUT
        )

    model, history, best_epoch = train_model(
        model,
        train_dataloader,
        test_dataloader,
        model_type,
        device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn, model_type, device)
    train_labels, train_probabilities = get_predictions(model, train_evaluation_dataloader, model_type, device)
    test_labels, test_probabilities = get_predictions(model, test_dataloader, model_type, device)

    model_metrics = calculate_metrics(test_labels, test_probabilities)
    model_metrics.update({
        "model": model_type,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_accuracy_from_loop": test_accuracy
    })
    metrics_rows.append(model_metrics)

    history.to_csv(
        METRICS_DIR / f"{model_type}_history.csv",
        index=False,
        encoding="utf-8"
    )

    test_information = data.iloc[test_indices][
        ["case_id", "spkid", "sampling_group", "label"]
    ].reset_index(drop=True)
    test_information["probability"] = test_probabilities
    test_information["prediction"] = (test_probabilities >= 0.5).astype(int)
    test_information.to_csv(
        PREDICTIONS_DIR / f"{model_type}_test_predictions.csv",
        index=False,
        encoding="utf-8"
    )

    for sampling_group, group in test_information.groupby("sampling_group"):
        group_metrics_rows.append({
            "model": model_type,
            "sampling_group": sampling_group,
            "objects": len(group),
            "positive_labels": int(group["label"].sum()),
            "accuracy": (group["label"] == group["prediction"]).mean(),
            "mean_probability": group["probability"].mean()
        })

    model_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": model_state,
            "scalar_features": SCALAR_FEATURES,
            "series_length": len(series_columns),
            "dropout": DROPOUT,
            "best_epoch": best_epoch
        },
        MODELS_DIR / f"{model_type}_model.pt"
    )

    plot_loss(history, test_loss, model_type, METRICS_DIR / f"{model_type}_loss.png")
    plot_roc(train_labels, train_probabilities, test_labels, test_probabilities, model_type, METRICS_DIR / f"{model_type}_roc.png")
    plot_confusion_matrix(test_labels, test_probabilities, model_type, METRICS_DIR / f"{model_type}_confusion_matrix.png")
    plot_probability_distribution(test_labels, test_probabilities, model_type, METRICS_DIR / f"{model_type}_probabilities.png")


# Save the final comparison
metrics = pd.DataFrame(metrics_rows)
metrics = metrics[[
        "model", "parameters", "best_epoch", "test_loss", "accuracy",
        "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
        "brier_score"]]
metrics.to_csv(METRICS_DIR / "model_metrics.csv", index=False, encoding="utf-8")

pd.DataFrame(group_metrics_rows).to_csv(METRICS_DIR / "metrics_by_sampling_group.csv", index=False, encoding="utf-8")

plot_model_comparison(metrics, METRICS_DIR / "model_comparison.png")

with open(METRICS_DIR / "training_configuration.txt", "w", encoding="utf-8") as file:
    file.write(f"Device: {device}\n")
    file.write("Checkpoint selection: minimum test loss (exploratory)\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Weight decay: {WEIGHT_DECAY}\n")
    file.write(f"Dropout: {DROPOUT}\n")
    file.write(f"Train objects: {len(train_indices)}\n")
    file.write(f"Test objects: {len(test_indices)}\n")

print(f"Evaluation outputs saved in {EVALUATION_DIR}.")
