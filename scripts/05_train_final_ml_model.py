from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from resonances.ml.models import ConvNetwork, count_parameters
from resonances.ml.training import ResonanceDataset, train_loop


# Creation of the necessary directories for storing the final model
ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = ROOT / "data" / "ml" / "datasets"
FINAL_DIR = ROOT / "outputs" / "ml" / "final"
MODELS_DIR = FINAL_DIR / "models"
METRICS_DIR = FINAL_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SCALAR_FILE = DATASETS_DIR / "scalar_inputs.csv"
SERIES_FILE = DATASETS_DIR / "time_series_inputs.csv"
PREPROCESSING_METADATA_FILE = DATASETS_DIR / "preprocessing_metadata.txt"

copyfile(
    PREPROCESSING_METADATA_FILE,
    FINAL_DIR / "preprocessing_metadata.txt"
)


# Final training configuration
MODEL_RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
CHECKPOINT_EPOCHS = [80, 85, 90, 95, 100]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2


# Read and align all training inputs
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

series_values = data[series_columns].to_numpy(dtype=np.float32)
labels = data["label"].to_numpy(dtype=np.float32)
dummy_scalars = np.zeros((len(data), 1), dtype=np.float32)


# Train the convolutional model with the complete dataset
np.random.seed(MODEL_RANDOM_STATE)
torch.manual_seed(MODEL_RANDOM_STATE)
torch.cuda.manual_seed_all(MODEL_RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ResonanceDataset(
    series_values,
    dummy_scalars,
    labels
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

model = ConvNetwork(
    series_length=len(series_columns),
    dropout=DROPOUT
).to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

history = []
checkpoint_states = {}

for epoch in range(1, EPOCHS + 1):
    train_loss, train_accuracy = train_loop(
        dataloader,
        model,
        loss_fn,
        optimizer,
        "conv",
        device
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy
    })

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train loss: {train_loss:.4f}"
    )

    if epoch in CHECKPOINT_EPOCHS:
        checkpoint_states[epoch] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }


# Save the five members of the final probability ensemble
for epoch, model_state in checkpoint_states.items():
    torch.save(
        {
            "model_type": "conv",
            "model_state_dict": model_state,
            "series_length": len(series_columns),
            "dropout": DROPOUT,
            "epoch": epoch
        },
        MODELS_DIR / f"conv_epoch_{epoch}.pt"
    )

ensemble = pd.DataFrame({
    "model_file": [
        f"conv_epoch_{epoch}.pt"
        for epoch in CHECKPOINT_EPOCHS
    ],
    "epoch": CHECKPOINT_EPOCHS,
    "probability_weight": [1 / len(CHECKPOINT_EPOCHS)] * len(CHECKPOINT_EPOCHS)
})
ensemble.to_csv(
    MODELS_DIR / "ensemble.csv",
    index=False,
    encoding="utf-8"
)

history = pd.DataFrame(history)
history.to_csv(
    METRICS_DIR / "conv_history.csv",
    index=False,
    encoding="utf-8"
)

figure, axis = plt.subplots(figsize=(8, 5))
axis.plot(history["epoch"], history["train_loss"])
for epoch in CHECKPOINT_EPOCHS:
    axis.axvline(epoch, color="black", linestyle="--", alpha=0.3)
axis.set_title("Final training loss | conv")
axis.set_xlabel("Epoch")
axis.set_ylabel("BCE loss")
axis.grid(alpha=0.3)
figure.tight_layout()
figure.savefig(
    METRICS_DIR / "conv_loss.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close(figure)

with open(
    FINAL_DIR / "training_configuration.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Device: {device}\n")
    file.write(f"Objects: {len(data)}\n")
    file.write("Model: conv\n")
    file.write(f"Parameters: {count_parameters(model)}\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Checkpoint epochs: {CHECKPOINT_EPOCHS}\n")
    file.write("Ensemble: equal mean of checkpoint probabilities\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Weight decay: {WEIGHT_DECAY}\n")
    file.write(f"Dropout: {DROPOUT}\n")

print(f"Final convolutional ensemble saved in {FINAL_DIR}.")
