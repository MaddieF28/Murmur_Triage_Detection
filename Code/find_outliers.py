import os
import csv

# ============================================================
# SETTINGS
# ============================================================

CODE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(
    CODE_DIR,
    "analysis_results",
    "frequency_analysis_results.csv"
)

# How many files to show from each class
TOP_N = 10


# ============================================================
# LOAD CSV
# ============================================================

results = []

with open(CSV_PATH, "r", newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:
        row["energy_above_400Hz_percent"] = float(
            row["energy_above_400Hz_percent"]
        )

        results.append(row)


# ============================================================
# FIND TOP OUTLIERS
# ============================================================

classes = ["N", "AS", "MS", "MR", "MVP"]

print("=" * 80)
print("HIGH-FREQUENCY OUTLIERS")
print("=" * 80)

for class_name in classes:

    class_results = [
        r for r in results
        if r["class"] == class_name
    ]

    # Sort highest energy above 400 Hz first
    class_results.sort(
        key=lambda x: x["energy_above_400Hz_percent"],
        reverse=True
    )

    print()
    print(f"{class_name}")
    print("-" * 80)

    for result in class_results[:TOP_N]:

        print(
            f"{result['energy_above_400Hz_percent']:8.3f}%   "
            f"{result['filename']}"
        )


# ============================================================
# ALL FILES ABOVE THRESHOLD
# ============================================================

THRESHOLD = 5.0

print()
print("=" * 80)
print(f"FILES WITH MORE THAN {THRESHOLD}% OF ENERGY ABOVE 400 Hz")
print("=" * 80)

for class_name in classes:

    class_results = [
        r for r in results
        if r["class"] == class_name
        and r["energy_above_400Hz_percent"] > THRESHOLD
    ]

    class_results.sort(
        key=lambda x: x["energy_above_400Hz_percent"],
        reverse=True
    )

    print()
    print(f"{class_name}: {len(class_results)} files")

    for result in class_results:

        print(
            f"  {result['energy_above_400Hz_percent']:8.3f}%   "
            f"{result['filename']}"
        )