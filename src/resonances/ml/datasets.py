import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from resonances.candidates import build_resonant_candidates
from resonances.resonance_strengths import (
    a_N,
    a_resonance_validation,
    m_N,
    resonance_strength,
    resonances,
)


def unpack_designation(designation):
    designation = str(designation).strip()

    if designation.isdigit():
        return designation

    centuries = {"I": "18", "J": "19", "K": "20", "L": "21"}
    codes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    year = centuries[designation[0]] + designation[1:3]
    cycle = codes.index(designation[4]) * 10 + int(designation[5])
    cycle = "" if cycle == 0 else str(cycle)

    return f"{year} {designation[3]}{designation[6]}{cycle}"


def build_training_dataset():
    QUERY_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    FIELDS = [
        "spkid", "full_name", "pdes", "name", "class", "epoch",
        "a", "e", "q", "i", "om", "w", "ma", "per",
        "sigma_a", "sigma_e", "sigma_q", "sigma_i", "sigma_om",
        "sigma_w", "sigma_ma", "sigma_per"
    ]
    OUTPUT_COLUMNS = [
        "spkid", "full_name", "pdes", "name", "class", "epoch",
        "a", "e_x", "q", "i_x", "om", "w", "ma", "per",
        "sigma_a", "sigma_e", "sigma_q", "sigma_i", "sigma_om",
        "sigma_w", "sigma_ma", "sigma_per", "a_sigma_ratio",
        "orbit_status", "e_idx", "i_idx", "e_y", "i_y",
        "resonance", "planet", "a_res", "SR", "Delta_R", "per_lib",
        "d_min/R_H", "close_encounter_fraction", "status", "k_p", "k",
        "sampling_group", "label"
    ]
    VOLK_COLUMNS = [
        "designation", "particle_id", "real_or_articial_TNO", "G08_class",
        "res_character", "res_p", "res_q", "res_m", "res_n",
        "e_mean", "i_mean"
    ]

    sample_size = 1000
    jfc_sample_size = 100
    random_state = 42

    volk_file = Path("./data/ml/sample/resonances_database_VolkMalhotra2024.csv")
    strengths = pd.read_csv("./data/processed/resonance_strengths_results.csv")
    candidates = pd.read_csv("./data/processed/resonance_candidates.csv", dtype={"spkid": str})
    original_catalog = pd.read_csv("./data/raw/centaurs_sbdb_Jewitt.csv", dtype={"spkid": str})
    volk = pd.read_csv(volk_file, usecols=VOLK_COLUMNS, dtype={"designation": str, "particle_id": str}, skipinitialspace=True)

    original_spkids = set(original_catalog["spkid"].dropna().astype(str))
    original_spkids.update(candidates["spkid"].dropna().astype(str))

    # Keep real objects in the physical domain when all clones agree.
    volk = volk[volk["real_or_articial_TNO"] == "r"].copy()
    volk["designation"] = volk["designation"].str.strip()
    volk["res_character"] = volk["res_character"].astype(str).str.strip()
    volk_objects = []

    for designation, clones in volk.groupby("designation", sort=False):
        clone_classes = clones[
            ["G08_class", "res_character", "res_p", "res_q", "res_m", "res_n"]
        ].drop_duplicates()

        if len(clone_classes) != 1:
            continue

        e_mean = clones["e_mean"].mean()
        i_mean = np.rad2deg(clones["i_mean"].mean())

        if e_mean <= 0.05 or i_mean >= 15:
            continue

        row = clones.iloc[0]

        if (
            row["G08_class"] == "Nresonant"
            and row["res_character"] == "r"
            and row["res_n"] == 0
            and row["res_m"] == row["res_p"] - row["res_q"]
        ):
            label = 1
        elif (
            row["G08_class"] in ["classical", "detached"]
            and row["res_character"] == "0"
        ):
            label = 0
        else:
            continue

        volk_objects.append({
            "designation": designation,
            "jpl_pdes": unpack_designation(designation),
            "res_p": int(row["res_p"]),
            "res_q": int(row["res_q"]),
            "label": label
        })

    volk_objects = pd.DataFrame(volk_objects)

    print(f"\nVolk-Malhotra objects after source filters: {len(volk_objects)}")
    print(volk_objects["label"].value_counts().rename({0: "non-resonant", 1: "resonant"}))

    # Recover the JFc sample with the original resonance-window query.
    selected_resonances = candidates[["planet", "resonance"]].drop_duplicates()
    selected_strengths = strengths.merge(
        selected_resonances,
        on=["planet", "resonance"],
        how="inner"
    )
    selected_strengths = selected_strengths[
        selected_strengths["status"] == "Valid"
    ].copy()
    selected_strengths["a_res"] = pd.to_numeric(
        selected_strengths["a_res"], errors="coerce"
    )
    selected_strengths["Delta_R"] = pd.to_numeric(
        selected_strengths["Delta_R"], errors="coerce"
    )
    selected_strengths = selected_strengths.dropna(subset=["a_res", "Delta_R"])

    n_res = np.sqrt(4 * np.pi**2 / selected_strengths["a_res"]**3)
    selected_strengths["Delta_a"] = (
        np.sqrt(8 / 3) / n_res * np.sqrt(selected_strengths["Delta_R"])
    )
    selected_strengths["a_min"] = (
        selected_strengths["a_res"] - selected_strengths["Delta_a"]
    )
    selected_strengths["a_max"] = (
        selected_strengths["a_res"] + selected_strengths["Delta_a"]
    )
    resonance_windows = (
        selected_strengths
        .groupby(["planet", "resonance"])
        .agg(a_min=("a_min", "min"), a_max=("a_max", "max"))
        .reset_index()
    )

    a_constraints = []

    for _, row in resonance_windows.iterrows():
        a_constraints.append(f"a|RG|{row['a_min']}|{row['a_max']}")

    constraints = {
        "AND": [
            {"OR": a_constraints},
            "epoch|DF", "e|DF", "i|DF", "om|DF", "w|DF", "ma|DF",
            "sigma_a|DF"
        ]
    }
    jfc_params = {
        "sb-cdata": json.dumps(constraints),
        "fields": ",".join(FIELDS),
        "sort": "a"
    }
    jfc_response = requests.get(QUERY_API_URL, params=jfc_params, timeout=60)
    jfc_response.raise_for_status()
    jfc_data = jfc_response.json()
    jfc_catalog = pd.DataFrame(jfc_data["data"], columns=jfc_data["fields"])

    numeric_columns = [
        "epoch", "a", "e", "q", "i", "om", "w", "ma", "per",
        "sigma_a", "sigma_e", "sigma_q", "sigma_i", "sigma_om",
        "sigma_w", "sigma_ma", "sigma_per"
    ]

    for column in numeric_columns:
        jfc_catalog[column] = pd.to_numeric(jfc_catalog[column], errors="coerce")

    jfc_catalog["spkid"] = jfc_catalog["spkid"].astype(str)
    jfc_catalog["a_sigma_ratio"] = jfc_catalog["sigma_a"] / jfc_catalog["a"]
    jfc_catalog["orbit_status"] = "unknown"
    jfc_catalog.loc[jfc_catalog["a_sigma_ratio"] <= 1, "orbit_status"] = "ok"
    jfc_catalog.loc[jfc_catalog["a_sigma_ratio"] > 1, "orbit_status"] = "poor orbit"

    jfc_candidates = build_resonant_candidates(jfc_catalog.copy(), strengths)
    jfc_candidates["spkid"] = jfc_candidates["spkid"].astype(str)
    jfc_candidates = jfc_candidates[
        (~jfc_candidates["spkid"].isin(original_spkids))
        & (jfc_candidates["class"] == "JFc")
    ].drop_duplicates(subset="spkid")

    number_jfc = min(jfc_sample_size, len(jfc_candidates))
    jfc_sample = jfc_candidates.sample(n=number_jfc, random_state=random_state).copy()
    jfc_sample["sampling_group"] = "JFc"
    jfc_sample["label"] = 0

    print(f"JFc objects selected: {number_jfc}/{jfc_sample_size}")

    # Download the complete JPL TNO table once and merge by designation.
    tno_params = {
        "sb-class": "TNO",
        "fields": ",".join(FIELDS),
        "sort": "pdes"
    }
    tno_response = requests.get(QUERY_API_URL, params=tno_params, timeout=120)
    tno_response.raise_for_status()
    tno_data = tno_response.json()
    tno_catalog = pd.DataFrame(tno_data["data"], columns=tno_data["fields"])

    for column in numeric_columns:
        tno_catalog[column] = pd.to_numeric(tno_catalog[column], errors="coerce")

    tno_catalog["spkid"] = tno_catalog["spkid"].astype(str)
    tno_catalog["pdes"] = tno_catalog["pdes"].astype(str).str.strip()

    volk_jpl = volk_objects.merge(
        tno_catalog,
        left_on="jpl_pdes",
        right_on="pdes",
        how="left",
        indicator=True
    )
    missing_jpl = (volk_jpl["_merge"] == "left_only").sum()
    volk_jpl = volk_jpl[volk_jpl["_merge"] == "both"].drop(columns="_merge")
    volk_jpl["a_sigma_ratio"] = volk_jpl["sigma_a"] / volk_jpl["a"]
    volk_jpl["orbit_status"] = "ok"

    original_matches = volk_jpl["spkid"].isin(original_spkids).sum()
    poor_orbits = (
        volk_jpl["a_sigma_ratio"].isna()
        | (volk_jpl["a_sigma_ratio"] > 1)
    ).sum()

    used_spkids = original_spkids | set(jfc_sample["spkid"].astype(str))
    volk_jpl = volk_jpl[
        (~volk_jpl["spkid"].isin(used_spkids))
        & volk_jpl["a_sigma_ratio"].notna()
        & (volk_jpl["a_sigma_ratio"] <= 1)
    ].drop_duplicates(subset="spkid")

    # Select the balanced sample before calculating resonance properties.
    volk_sample_size = sample_size - number_jfc
    resonant_target = volk_sample_size // 2
    non_resonant_target = volk_sample_size - resonant_target
    resonant_pool = volk_jpl[volk_jpl["label"] == 1]
    non_resonant_pool = volk_jpl[volk_jpl["label"] == 0]

    resonant_sample = resonant_pool.sample(
        n=min(resonant_target, len(resonant_pool)),
        random_state=random_state
    )
    non_resonant_sample = non_resonant_pool.sample(
        n=min(non_resonant_target, len(non_resonant_pool)),
        random_state=random_state
    )
    selected_spkids = set(resonant_sample["spkid"]) | set(non_resonant_sample["spkid"])
    remaining = volk_sample_size - len(resonant_sample) - len(non_resonant_sample)

    if remaining > 0:
        extra_objects = volk_jpl[~volk_jpl["spkid"].isin(selected_spkids)]
        extra_sample = extra_objects.sample(
            n=min(remaining, len(extra_objects)),
            random_state=random_state + 1
        )
    else:
        extra_sample = volk_jpl.iloc[0:0]

    volk_sample = pd.concat(
        [resonant_sample, non_resonant_sample, extra_sample],
        ignore_index=True
    )

    # Keep the article resonance or assign the closest nominal Neptune MMR.
    neptune_resonances = []

    for k_p, k in resonances:
        if k > k_p:
            a_res = a_resonance_validation(
                k, k_p, a_N, m_N, a_min=a_N, a_max=np.inf
            )[0]
            neptune_resonances.append((k_p, k, a_res))

    excentricities = np.linspace(0, 0.9, 10)
    inclinations = np.linspace(0, 180, 10)
    strength_cache = {}
    resonance_rows = []

    for _, row in volk_sample.iterrows():
        if row["label"] == 1:
            k_p = int(row["res_q"])
            k = int(row["res_p"])
            a_res = a_resonance_validation(
                k, k_p, a_N, m_N, a_min=0, a_max=np.inf
            )[0]
            sampling_group = "Volk-Malhotra resonant"
        else:
            k_p, k, a_res = min(
                neptune_resonances,
                key=lambda resonance: abs(row["a"] - resonance[2])
            )
            sampling_group = "Volk-Malhotra non-resonant"

        e_idx = int(np.clip(np.rint(row["e"] / 0.1), 1, 9))
        i_idx = int(np.clip(np.rint(row["i"] / 20), 0, 9))
        e_grid = excentricities[e_idx]
        i_grid = inclinations[i_idx]
        cache_key = (k_p, k, e_idx, i_idx)

        if cache_key not in strength_cache:
            strength_cache[cache_key] = resonance_strength(
                k, k_p, a_res, e_grid, np.deg2rad(i_grid), a_N, m_N
            )[:5]

        SR, Delta_R, per_lib, d_min, close_fraction = strength_cache[cache_key]
        resonance_rows.append({
            "e_idx": e_idx,
            "i_idx": i_idx,
            "e_y": e_grid,
            "i_y": i_grid,
            "resonance": f"{k_p}:{k}",
            "planet": "Neptune",
            "a_res": a_res,
            "SR": SR,
            "Delta_R": Delta_R,
            "per_lib": per_lib,
            "d_min/R_H": d_min,
            "close_encounter_fraction": close_fraction,
            "status": "Valid",
            "k_p": k_p,
            "k": k,
            "sampling_group": sampling_group
        })

    resonance_rows = pd.DataFrame(resonance_rows)
    volk_sample = pd.concat(
        [volk_sample.reset_index(drop=True), resonance_rows],
        axis=1
    )
    volk_sample = volk_sample.rename(columns={"e": "e_x", "i": "i_x"})

    training_sample = pd.concat(
        [jfc_sample[OUTPUT_COLUMNS], volk_sample[OUTPUT_COLUMNS]],
        ignore_index=True
    ).drop_duplicates(subset="spkid")

    output_dir = Path("./data/ml/sample")
    output_file = output_dir / "training_sample.csv"
    metadata_file = output_dir / "training_sample_metadata.txt"
    training_sample.to_csv(output_file, index=False, encoding="utf-8")

    with open(metadata_file, "w", encoding="utf-8") as file:
        file.write(f"# Date: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        file.write(f"# Volk-Malhotra source: {volk_file}\n")
        file.write(f"# JFC query: {jfc_response.url}\n")
        file.write(f"# TNO query: {tno_response.url}\n")
        file.write(f"# Random state: {random_state}\n")
        file.write(f"# Requested objects: {sample_size}\n")
        file.write(f"# Final objects: {len(training_sample)}\n")
        file.write(f"# JFC objects: {len(jfc_sample)}\n")
        file.write(f"# Volk-Malhotra resonant objects: {(volk_sample['label'] == 1).sum()}\n")
        file.write(f"# Volk-Malhotra non-resonant objects: {(volk_sample['label'] == 0).sum()}\n")
        file.write(f"# Volk-Malhotra objects without a JPL TNO match: {missing_jpl}\n")
        file.write(f"# Original-catalog matches excluded: {original_matches}\n")
        file.write(f"# Poor JPL orbits excluded: {poor_orbits}\n")
        file.write("# JFC objects are not restricted in eccentricity or inclination\n")
        file.write("# JFC labels are provisional label=0 and require visual review\n")
        file.write("# Volk-Malhotra e_mean and i_mean were used only for the physical-domain filter\n")
        file.write("# Resonant convention: k_p=res_q, k=res_p, principal mode m=k-k_p\n")
        file.write("# Non-resonant objects use the closest nominal Neptune MMR without rejection\n")

        if len(training_sample) < sample_size:
            file.write(f"# Missing objects: {sample_size - len(training_sample)}\n")

    print(f"Saved {len(training_sample)} objects in {output_file}.")

    return training_sample
