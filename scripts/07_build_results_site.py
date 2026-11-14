from pathlib import Path
import pandas as pd
import shutil

# Creation of the necessary directories for the website
ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = ROOT / "data" / "resonance_results" / "resonance_results.csv"
SOURCE_PLOTS = ROOT / "outputs" / "resonance_plots"

DOCS_DIR = ROOT / "docs"
WEB_DATA_DIR = DOCS_DIR / "data"
WEB_PLOTS_DIR = DOCS_DIR / "plots"

WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
WEB_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Convert the resonance results from a CSV file to a JSON file for the website
df = pd.read_csv(RESULTS_FILE)
df.to_json(WEB_DATA_DIR / "results.json",orient="records",indent=2,force_ascii=False)

# Copy the plots to the website directory
if WEB_PLOTS_DIR.exists():
    shutil.rmtree(WEB_PLOTS_DIR)
WEB_PLOTS_DIR.mkdir(parents=True)

for plot_name in df["plot"].dropna().unique():
    source = SOURCE_PLOTS / plot_name
    destination = WEB_PLOTS_DIR / plot_name
    if source.exists():
        shutil.copy2(source, destination)

print(f"Website built at: {DOCS_DIR}")
