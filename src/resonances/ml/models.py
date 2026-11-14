import torch
import torch.nn as nn


class ConvNetwork(nn.Module):
    def __init__(self, series_length=128, dropout=0.2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16 * (series_length // 4), 1)
        )

    def forward(self, x):
        return self.conv(x)


class CombinedNetwork(nn.Module):
    def __init__(self, scalar_input_size, series_length=128, dropout=0.2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16 * (series_length // 4), 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dense = nn.Sequential(
            nn.Linear(scalar_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, series, scalars):
        series = self.conv(series)
        scalars = self.dense(scalars)
        combined = torch.cat([series, scalars], dim=1)

        return self.classifier(combined)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
