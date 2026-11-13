"""
Reads test_accuracy_by_recording.csv (produced by segmenter_eval/evaluate_segmenter_all_datasets.py) and
surfaces the worst-performing recordings both overall and broken out per
dataset, so you can go eyeball the actual audio/tsv for the worst offenders.

The main csv's "dataset/recording" column is split back into its two parts
here so per-dataset worst-N lists are possible even though everything was
written to one combined file.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import csv

IN_CSV = paths.TEST_ACCURACY_BY_RECORDING_CSV
N_WORST_OVERALL = 20
N_WORST_PER_DATASET = 10

OUT_OVERALL_CSV = paths.SEGMENTER_RESULTS_DIR / "worst_overall.csv"
OUT_PER_DATASET_CSV = paths.SEGMENTER_RESULTS_DIR / "worst_per_dataset.csv"


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for dataset_recording, acc in reader:
            dataset, _, recording = dataset_recording.partition("/")
            rows.append({
                "dataset": dataset,
                "recording": recording,
                "dataset_recording": dataset_recording,
                "accuracy": float(acc),
            })
    return rows


def main():
    rows = load_rows(IN_CSV)
    print(f"Loaded {len(rows)} recordings from {IN_CSV}")

    rows_sorted = sorted(rows, key=lambda r: r["accuracy"])

    print(f"\n=== {N_WORST_OVERALL} worst recordings overall ===")
    worst_overall = rows_sorted[:N_WORST_OVERALL]
    for r in worst_overall:
        print(f"  {r['accuracy']:.4f}  {r['dataset_recording']}")

    with open(OUT_OVERALL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "recording", "accuracy"])
        for r in worst_overall:
            writer.writerow([r["dataset"], r["recording"], r["accuracy"]])
    print(f"\nWrote {OUT_OVERALL_CSV}")

    datasets = sorted(set(r["dataset"] for r in rows))
    per_dataset_rows = []

    print(f"\n=== worst {N_WORST_PER_DATASET} per dataset (or fewer if the dataset is smaller) ===")
    for dataset in datasets:
        group = sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: r["accuracy"])
        take = group[:N_WORST_PER_DATASET]
        print(f"\n{dataset} ({len(group)} test recordings total):")
        for r in take:
            print(f"  {r['accuracy']:.4f}  {r['recording']}")
        per_dataset_rows.extend(take)

    with open(OUT_PER_DATASET_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "recording", "accuracy"])
        for r in per_dataset_rows:
            writer.writerow([r["dataset"], r["recording"], r["accuracy"]])
    print(f"\nWrote {OUT_PER_DATASET_CSV}")


if __name__ == "__main__":
    main()