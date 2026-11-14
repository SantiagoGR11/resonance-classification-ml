from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from resonances.ml.models import ConvNetwork


# Paths for the candidate integrations, final model and predictions
ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_FILE = ROOT / "data" / "processed" / "resonance_candidates.csv"
RESULTS_DIR = ROOT / "data" / "resonance_results"
TIMESERIES_FILE = RESULTS_DIR / "timeseries.h5"
RESULTS_FILE = RESULTS_DIR / "resonance_results.csv"
MODE_PREDICTIONS_FILE = RESULTS_DIR / "mode_predictions.csv"
METADATA_FILE = RESULTS_DIR / "classification_metadata.txt"

MODELS_DIR = ROOT / "outputs" / "ml" / "final" / "models"
ENSEMBLE_FILE = MODELS_DIR / "ensemble.csv"

N_SAMPLES = 128
N_SMOOTHING_WINDOWS = 50
N_LIBRATION_PERIODS = 2
THRESHOLD = 0.5


def preprocess_series(time, sigma, per_lib):
    maximum_time = time[0] + N_LIBRATION_PERIODS * per_lib
    selected = time <= maximum_time
    time = time[selected]
    sigma = sigma[selected]

    treated_time = time - time[0]
    treated_sigma = sigma - sigma[0]

    window_length = len(treated_sigma) // N_SMOOTHING_WINDOWS

    if window_length % 2 == 0:
        window_length -= 1

    if window_length >= 3:
        treated_sigma = savgol_filter(
            treated_sigma,
            window_length=window_length,
            polyorder=2
        )

    treated_sigma = treated_sigma - treated_sigma[0]
    treated_time = treated_time / treated_time[-1]
    treated_turns = treated_sigma / (2 * np.pi)

    interpolation_time = np.linspace(0, 1, N_SAMPLES)
    return np.interp(
        interpolation_time,
        treated_time,
        treated_turns
    )


def ordered_modes(k_p, k, modes):
    priority_modes = [int(k - k_p), int(k + k_p), 0]
    ordered = []

    for mode in priority_modes:
        if mode in modes and mode not in ordered:
            ordered.append(mode)

    for mode in modes:
        mode = int(mode)
        if mode not in ordered:
            ordered.append(mode)

    return ordered


# Load the final convolutional ensemble once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensemble = pd.read_csv(ENSEMBLE_FILE)
weights = ensemble["probability_weight"].to_numpy(dtype=float)
weights = weights / weights.sum()
models = []

for _, member in ensemble.iterrows():
    checkpoint = torch.load(
        MODELS_DIR / member["model_file"],
        map_location=device
    )
    model = ConvNetwork(
        series_length=checkpoint["series_length"],
        dropout=checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)


def predict_series(series):
    series = torch.tensor(
        series,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    probabilities = []

    with torch.no_grad():
        for model in models:
            probability = torch.sigmoid(
                model(series).squeeze()
            ).cpu().item()
            probabilities.append(probability)

    probability = np.average(probabilities, weights=weights)
    ensemble_std = np.std(probabilities)

    return probability, ensemble_std


# Read the candidates and prepare their case identifiers
candidates = pd.read_csv(CANDIDATES_FILE, dtype={"spkid": str})
candidates["case_id"] = candidates.apply(
    lambda row: (
        f"{int(row['spkid'])}_"
        f"{row['planet']}_"
        f"{int(row['k_p'])}-{int(row['k'])}"
    ),
    axis=1
)

prediction_rows = []
mode_prediction_rows = []
missing_cases = []


# Stop at the first resonant mode in the physical priority order
with h5py.File(TIMESERIES_FILE, "r") as h5:
    cases = h5["cases"]

    for _, row in candidates.iterrows():
        case_id = row["case_id"]

        if case_id not in cases:
            missing_cases.append(case_id)
            continue

        group = cases[case_id]
        time = group["time"][:]
        modes = group["modes"][:].astype(int)
        mode_order = ordered_modes(
            int(row["k_p"]),
            int(row["k"]),
            modes
        )

        tested_predictions = []
        resonant_mode = np.nan

        for priority, mode in enumerate(mode_order, start=1):
            mode_index = np.flatnonzero(modes == mode)[0]
            sigma = group["sigma_unwrapped"][mode_index]
            series = preprocess_series(time, sigma, row["per_lib"])
            probability, ensemble_std = predict_series(series)
            prediction = int(probability >= THRESHOLD)

            mode_prediction = {
                "case_id": case_id,
                "mode": mode,
                "priority": priority,
                "libration_probability": probability,
                "ensemble_std": ensemble_std,
                "prediction": prediction
            }
            mode_prediction_rows.append(mode_prediction)
            tested_predictions.append(mode_prediction)

            if prediction == 1:
                resonant_mode = mode
                break

        if np.isnan(resonant_mode):
            selected = max(
                tested_predictions,
                key=lambda item: item["libration_probability"]
            )
            result = "Circulation"
            prediction = 0
        else:
            selected = tested_predictions[-1]
            result = "Libration"
            prediction = 1

        prediction_rows.append({
            "case_id": case_id,
            "result": result,
            "prediction": prediction,
            "resonant_mode": resonant_mode,
            "libration_probability": selected["libration_probability"],
            "ensemble_std": selected["ensemble_std"],
            "modes_tested": len(tested_predictions)
        })


if missing_cases:
    print(f"Missing integrations: {len(missing_cases)}")
    print("The candidates have not been classified.")
    raise SystemExit


# Save mode-level traceability and update the website summary
predictions = pd.DataFrame(prediction_rows)
mode_predictions = pd.DataFrame(mode_prediction_rows)
mode_predictions.to_csv(
    MODE_PREDICTIONS_FILE,
    index=False,
    encoding="utf-8"
)

results = pd.read_csv(
    RESULTS_FILE,
    dtype={"case_id": str, "spkid": str}
)
results = results.drop(
    columns=[
        "result", "prediction", "resonant_mode",
        "libration_probability", "ensemble_std", "modes_tested"
    ],
    errors="ignore"
)
results = results.merge(predictions, on="case_id", how="left")
results.to_csv(RESULTS_FILE, index=False, encoding="utf-8")


with open(METADATA_FILE, "w", encoding="utf-8") as file:
    file.write(f"# Date: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
    file.write(f"# Candidates: {CANDIDATES_FILE}\n")
    file.write(f"# Integrations: {TIMESERIES_FILE}\n")
    file.write(f"# Ensemble: {ENSEMBLE_FILE}\n")
    file.write(f"# Classified cases: {len(predictions)}\n")
    file.write("# Mode priority: prograde, retrograde, zero, remaining modes\n")
    file.write("# Prograde mode: m = k - k_p\n")
    file.write("# Retrograde mode: m = k + k_p\n")
    file.write(f"# Interpolated samples: {N_SAMPLES}\n")
    file.write(f"# Libration periods used: {N_LIBRATION_PERIODS}\n")
    file.write(f"# Probability threshold: {THRESHOLD}\n")
    file.write("# The search stops at the first mode classified as libration\n")
    file.write("# Result 1: Libration\n")
    file.write("# Result 0: Circulation\n")

print(f"Classified {len(predictions)} candidates in {RESULTS_FILE}.")
