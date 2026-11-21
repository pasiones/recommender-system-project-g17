"""Fill the unchanged course workbook schema and export analysis CSV files."""

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


PARADIGM = {"Item-based kNN": "Neighbourhood CF", "Most popular": "Popularity baseline",
            "User-based kNN": "Neighbourhood CF", "Biased MF": "Latent-factor CF",
            "Content profile": "Content-based"}


def save_part_2(trials, models, disaggregated, sparsity, selected, out, track="A"):
    out = Path(out)
    if track not in {"A", "B", "C"}:
        raise ValueError("Part 3 track must be A, B, or C")
    path = out / "results.xlsx"
    workbook = load_workbook(path)
    workbook["Instructions"]["B6"] = track
    sheets = {name: workbook[name] for name in ("Part 2 - Models", "Part 2 - Disaggregated", "Part 2 - Sparsity")}
    records = {name: [] for name in sheets}
    for row in models.to_dict("records"):
        ci = row["ndcg_ci"]
        diff = row["paired_difference_ci"]
        note = f"NDCG 95% bootstrap CI [{ci[0]:.4f},{ci[1]:.4f}]; paired delta vs item-kNN [{diff[0]:.4f},{diff[1]:.4f}]"
        records["Part 2 - Models"].append([
            row["model"], PARADIGM[row["model"]],
            "Yes" if row["model"] in {"User-based kNN", "Biased MF", "Content profile"} else "No",
            row["settings"], row["precision@10"], row["recall@10"], row["ndcg@10"],
            row["catalogue_coverage"], note, 42,
            f"{int(row['evaluated_users'])} positive-test users; full catalogue; 1000 paired user bootstrap samples"])
    for row in disaggregated.to_dict("records"):
        records["Part 2 - Disaggregated"].append([
            row["model"], row["dimension"], row["group"], row["definition"],
            row["precision@10"], row["recall@10"], row["ndcg@10"],
            row["catalogue_coverage"], 42, f"{int(row['evaluated_users'])} positive-test users"])
    for row in sparsity.to_dict("records"):
        records["Part 2 - Sparsity"].append([
            row["model"], int(row["training_fraction"] * 100), row["precision@10"],
            row["recall@10"], row["ndcg@10"], row["catalogue_coverage"], 42,
            f"{row['fit_rows']} fit rows; validation-selected settings fixed"])
    for name, rows in records.items():
        sheet = sheets[name]
        for number, values in enumerate(rows, 7):
            for column, value in enumerate(values, 1):
                sheet.cell(number, column, value)
    workbook.save(path)
    trials.to_csv(out / "part_2_validation_grid.csv", index=False)
    models.to_csv(out / "part_2_models.csv", index=False)
    disaggregated.to_csv(out / "part_2_disaggregated.csv", index=False)
    sparsity.to_csv(out / "part_2_sparsity.csv", index=False)
    (out / "part_2_protocol.json").write_text(json.dumps({
        "seed": 42, "part_3_track": track, "validation_trials_per_family": 4,
        "selection": "max validation NDCG@10, then Recall@10, then Precision@10",
        "selected_settings": selected,
        "uncertainty": "1000 paired bootstrap resamples of test users; 95% percentile CI",
        "sparsity": "nested uniform row sample of final fit interactions, seed 42; validation/test unchanged; settings fixed",
        "part_2_self_implemented": "Content profile (genre and release-year profile matching)",
    }, indent=2), encoding="utf-8")
