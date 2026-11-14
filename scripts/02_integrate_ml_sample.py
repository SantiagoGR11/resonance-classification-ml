import pandas as pd
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from resonances.integration import build_initial_simulations, integrate_candidate


# Creation of the necessary directories for storing the results
ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "ml" / "sample"
RESULTS_DIR = ROOT / "data" / "ml" / "integrations"
PLOTS_DIR = ROOT / "outputs" / "ml" / "integration_plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_FILE = PROCESSED_DIR / "training_sample.csv"
TIMESERIES_FILE = RESULTS_DIR / "training_timeseries.h5"
SUMMARY_FILE = RESULTS_DIR / "training_results.csv"


# Read the resonance candidates data
df = pd.read_csv(CANDIDATES_FILE)
df["case_id"] = df.apply(
    lambda row: (
        f"{int(row['spkid'])}_"
        f"{row['planet']}_"
        f"{int(row['k_p'])}-{int(row['k'])}"
    ),
    axis=1
)
valid_case_ids = set(df["case_id"])
current_per_lib = dict(zip(df["case_id"], df["per_lib"]))

# Columns for the summary DataFrame
SUMMARY_COLUMNS = [
    "case_id",
    "spkid",
    "object",
    "class",
    "resonance",
    "planet",
    "result",
    "plot",
]

# Read the existing summary so completed cases can be resumed
if SUMMARY_FILE.exists():
    summary_df = pd.read_csv(
        SUMMARY_FILE,
        dtype={"case_id": str, "spkid": str}
    )
else:
    summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)


# Remove results that do not belong to the current training sample
with h5py.File(TIMESERIES_FILE, "a") as h5:

    cases = h5.require_group("cases")

    for case_id in list(cases.keys()):
        stored_per_lib = cases[case_id].attrs.get("per_lib", np.nan)

        if (
            case_id not in valid_case_ids
            or not np.isclose(stored_per_lib, current_per_lib[case_id])
        ):
            del cases[case_id]

    completed_cases = set(cases.keys())
    print(f"Completed current cases: {len(completed_cases)}/{len(df)}")
    summary_df = summary_df[
        summary_df["case_id"].isin(completed_cases)
    ].drop_duplicates(subset="case_id")
    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")

    for plot_file in PLOTS_DIR.glob("*.png"):
        if plot_file.stem not in completed_cases:
            plot_file.unlink()

    pending_df = df[~df["case_id"].isin(completed_cases)]
    initial_sims = build_initial_simulations(pending_df["epoch"])

# Integrate each candidate and store the results
    for _, row in pending_df.iterrows():

        case_id = row["case_id"]

        result = integrate_candidate(row, initial_sims[row['epoch']], all_modes=False)

        # Write (or overwrite) the results in a file for each quantity
        if case_id in cases:
            del cases[case_id]

        group = cases.create_group(case_id)
        group.create_dataset("time",data=result["time"],compression="gzip")
        group.create_dataset("modes",data=result["modes"])
        group.create_dataset("sigma_unwrapped",data=result["sigma"],compression="gzip")
        group.create_dataset("a",data=result["a"],compression="gzip")
        group.create_dataset("e",data=result["e"],compression="gzip")
        group.create_dataset("i",data=result["i"],compression="gzip")

        # Add othe useful attributes or metadara
        group.attrs["spkid"] = str(row["spkid"])
        group.attrs["object"] = str(row["full_name"])
        group.attrs["class"] = str(row["class"])
        group.attrs["planet"] = str(row["planet"])
        group.attrs["resonance"] = f"{int(row['k_p'])}:{int(row['k'])}"
        group.attrs["epoch"] = float(row["epoch"])
        group.attrs["per_lib"] = float(row["per_lib"])

        # Generate and save the plots for the libration angles as a function of time
        plot_name = f"{case_id}.png"
        plot_file = PLOTS_DIR / plot_name

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, m in enumerate(result["modes"]):
            sigma_turns = result["sigma"][idx] / (2*np.pi)
            ax.plot(result["time"], sigma_turns, label=f"m = {m}")

        ax.set_title(f"{row['full_name']} | "f"{int(row['k_p'])}:{int(row['k'])} "f"{row['planet']}")
        ax.set_xlabel("Time [yr]")
        ax.set_ylabel(r"$\sigma_m / 360^\circ$ [turns]")

        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(plot_file,dpi=150,bbox_inches="tight")
        plt.close(fig)
        
        # Update the summary with the new results
        new_row = {
            "case_id": case_id,
            "spkid": str(row["spkid"]),
            "object": row["full_name"],
            "class": row["class"],
            "resonance": f"{int(row['k_p'])}:{int(row['k'])}",
            "planet": row["planet"],
            "result": "-",
            "plot": plot_name,
        }
        summary_df = summary_df[summary_df["case_id"] != case_id]
        summary_df = pd.concat([summary_df,pd.DataFrame([new_row])],ignore_index=True)

        # Save the summary DataFrame 
        summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")
