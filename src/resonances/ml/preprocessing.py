from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def build_preprocessed_datasets(n_samples=128, n_smoothing_windows= 50):
    ROOT = Path(__file__).resolve().parents[3]

    SAMPLE_DIR = ROOT / "data" / "ml" / "sample"
    INTEGRATIONS_DIR = ROOT / "data" / "ml" / "integrations"
    DATASETS_DIR = ROOT / "data" / "ml" / "datasets"

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    SAMPLE_FILE = SAMPLE_DIR / "training_sample.csv"
    TIMESERIES_FILE = INTEGRATIONS_DIR / "training_timeseries.h5"
    SCALAR_FILE = DATASETS_DIR / "scalar_inputs.csv"
    SERIES_FILE = DATASETS_DIR / "time_series_inputs.csv"
    METADATA_FILE = DATASETS_DIR / "preprocessing_metadata.txt"

    sample = pd.read_csv(SAMPLE_FILE, dtype={"spkid": str})

    jfc_sample = sample[sample["sampling_group"] == "JFc"]
    volk_sample = sample[sample["sampling_group"] != "JFc"]
    volk_resonant = volk_sample[volk_sample["label"] == 1].sample(
        n=100,
        random_state=42
    )
    volk_non_resonant = volk_sample[volk_sample["label"] == 0].sample(
        n=100,
        random_state=42
    )

    sample = pd.concat([
        jfc_sample,
        volk_resonant,
        volk_non_resonant
    ], ignore_index=True)

    scalar_rows = []
    series_rows = []
    missing_cases = []
    invalid_cases = []

    with h5py.File(TIMESERIES_FILE, "r") as h5:
        cases = h5["cases"]

        for _, row in sample.iterrows():
            case_id = (
                f"{int(row['spkid'])}_"
                f"{row['planet']}_"
                f"{int(row['k_p'])}-{int(row['k'])}"
            )

            if case_id not in cases:
                missing_cases.append(case_id)
                continue

            time = cases[case_id]["time"][:]
            sigma = cases[case_id]["sigma_unwrapped"][0]

            # Original signal and its scalar inputs.
            dsigma0 = sigma - sigma[0]
            original_time = (time - time[0]) / (time[-1] - time[0])
            original_turns = dsigma0 / (2 * np.pi)

            original_final_displacement = np.abs(original_turns[-1])
            original_deviation_area = np.trapezoid(original_turns, original_time)

            maximum_index = np.argmax(np.abs(original_turns))
            time_to_maximum = time[maximum_index] - time[0]

            if time_to_maximum != 0:
                slope_to_maximum = (
                    np.abs(original_turns[maximum_index]) / time_to_maximum
                )
            else:
                slope_to_maximum = 0.0

            # Use the complete integrated signal.
            treated_time = time - time[0]
            treated_dsigma0 = dsigma0.copy()

            # Light Savitzky-Golay smoothing without changing the number of points.
            window_length = len(treated_dsigma0) // n_smoothing_windows

            if window_length % 2 == 0:
                window_length -= 1

            if window_length >= 3:
                treated_dsigma0 = savgol_filter(
                    treated_dsigma0,
                    window_length=window_length,
                    polyorder=2
                )

            treated_dsigma0 = treated_dsigma0 - treated_dsigma0[0]
            physical_duration = treated_time[-1]

            if physical_duration <= 0:
                invalid_cases.append(case_id)
                continue

            treated_time = treated_time / physical_duration
            treated_turns = treated_dsigma0 / (2 * np.pi)

            treated_final_displacement = np.abs(treated_turns[-1])
            treated_deviation_area = np.trapezoid(
                np.abs(treated_turns),
                treated_time
            )

            first_half = max(2, (len(treated_time) + 1) // 2)
            first_half_time = treated_time[:first_half]
            first_half_time = first_half_time / first_half_time[-1]
            first_half_deviation_area = np.trapezoid(
                treated_turns[:first_half],
                first_half_time
            )

            # Final interpolation to a common length.
            interpolation_time = np.linspace(0, 1, n_samples)
            interpolated_series = np.interp(
                interpolation_time,
                treated_time,
                treated_turns
            )

            n_res = np.sqrt(4 * np.pi**2 / row["a_res"]**3)
            delta_a = np.sqrt(8 / 3) / n_res * np.sqrt(row["Delta_R"])

            scalar_rows.append({
                "case_id": case_id,
                "spkid": str(row["spkid"]),
                "sampling_group": row["sampling_group"],
                "per_lib": row["per_lib"],
                "delta_a": delta_a,
                "relative_delta_a": delta_a / row["a_res"],
                "physical_duration": physical_duration,
                "duration_per_lib": physical_duration / row["per_lib"],
                "original_final_displacement": original_final_displacement,
                "treated_final_displacement": treated_final_displacement,
                "slope_to_maximum": slope_to_maximum,
                "original_deviation_area": original_deviation_area,
                "treated_deviation_area": treated_deviation_area,
                "first_half_deviation_area": first_half_deviation_area,
                "cut_reached": 0,
                "label": int(row["label"])
            })

            series_row = {
                "case_id": case_id,
                "spkid": str(row["spkid"])
            }

            for index, value in enumerate(interpolated_series):
                series_row[f"sigma_{index:03d}"] = value

            series_rows.append(series_row)

    if missing_cases or invalid_cases:
        print(f"Missing integrations: {len(missing_cases)}")
        print(f"Invalid integrations: {len(invalid_cases)}")
        print("The datasets have not been saved.")
        return None, None

    scalar_inputs = pd.DataFrame(scalar_rows)
    time_series_inputs = pd.DataFrame(series_rows)

    scalar_inputs.to_csv(SCALAR_FILE, index=False, encoding="utf-8")
    time_series_inputs.to_csv(SERIES_FILE, index=False, encoding="utf-8")

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        file.write(f"# Date: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        file.write(f"# Training sample: {SAMPLE_FILE}\n")
        file.write(f"# Integrations: {TIMESERIES_FILE}\n")
        file.write(f"# Objects: {len(sample)}\n")
        file.write(f"# Interpolated samples: {n_samples}\n")
        file.write(f"# Savitzky-Golay windows: signal length/{n_smoothing_windows}\n")
        file.write("# Savitzky-Golay polynomial order: 2\n")
        file.write("# Temporal cut: none\n")
        file.write("# Time normalization: [0, 1]\n")
        file.write("# Angular units: turns (displacement/(2*pi))\n")
        file.write("# Labels are stored in scalar_inputs.csv\n")

    print(f"Saved {len(scalar_inputs)} rows in {SCALAR_FILE}.")
    print(f"Saved {len(time_series_inputs)} rows in {SERIES_FILE}.")

    return scalar_inputs, time_series_inputs
