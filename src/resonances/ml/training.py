import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ResonanceDataset(Dataset):
    def __init__(self, series, scalars, labels):
        self.series = torch.tensor(
            np.asarray(series),
            dtype=torch.float32
        ).unsqueeze(1)
        self.scalars = torch.tensor(
            np.asarray(scalars),
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            np.asarray(labels),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        series = self.series[index]
        scalars = self.scalars[index]
        label = self.labels[index]

        return series, scalars, label


def create_dataloader(
    series,
    scalars,
    labels,
    batch_size=32,
    shuffle=False
):
    dataset = ResonanceDataset(series, scalars, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def normalize_scalar_inputs(train, test):
    mean = train.mean(axis=0)
    standard_deviation = train.std(axis=0)
    standard_deviation[standard_deviation == 0] = 1

    train = (train - mean) / standard_deviation
    test = (test - mean) / standard_deviation

    return train, test, mean, standard_deviation


def model_prediction(model, series, scalars, model_type):
    if model_type == "conv":
        return model(series)

    return model(series, scalars)


def train_loop(dataloader, model, loss_fn, optimizer, model_type, device):
    losses = []
    accuracies = []
    model.train()

    for series, scalars, labels in dataloader:
        series = series.to(device)
        scalars = scalars.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        prediction = model_prediction(
            model,
            series,
            scalars,
            model_type
        ).squeeze(dim=1)
        loss = loss_fn(prediction, labels)

        losses.append(loss.detach().cpu().item())
        accuracy = ((prediction >= 0) == labels.bool()).float().mean()
        accuracies.append(accuracy.detach().cpu().item())

        loss.backward()
        optimizer.step()

    return np.mean(losses), np.mean(accuracies)


def test_loop(dataloader, model, loss_fn, model_type, device):
    losses = []
    accuracies = []
    model.eval()

    with torch.no_grad():
        for series, scalars, labels in dataloader:
            series = series.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device)

            prediction = model_prediction(
                model,
                series,
                scalars,
                model_type
            ).squeeze(dim=1)
            loss = loss_fn(prediction, labels)

            losses.append(loss.detach().cpu().item())
            accuracy = ((prediction >= 0) == labels.bool()).float().mean()
            accuracies.append(accuracy.detach().cpu().item())

    return np.mean(losses), np.mean(accuracies)


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    model_type,
    device,
    epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4):
    model = model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = []
    best_test_loss = float("inf")
    best_state = None
    best_epoch = None

    for epoch in range(epochs):
        train_loss, train_accuracy = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            model_type,
            device
        )
        test_loss, test_accuracy = test_loop(
            test_dataloader,
            model,
            loss_fn,
            model_type,
            device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"test loss: {test_loss:.4f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    model.load_state_dict(best_state)
    history = pd.DataFrame(history)

    return model, history, best_epoch
