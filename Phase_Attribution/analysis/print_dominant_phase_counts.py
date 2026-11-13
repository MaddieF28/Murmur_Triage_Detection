import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import pandas as pd

df = pd.read_csv(paths.PHASE_DIAGNOSTICS_PER_RECORDING_CSV)
scoped = df[df["has_expected_phase"] == True]

for arm in ("ground_truth", "predicted"):
    arm_df = scoped[scoped["boundary_source"] == arm]
    print(f"\n--- {arm} arm (n={len(arm_df)}) ---")
    counts = arm_df["dominant_phase_raw"].value_counts()
    rates = arm_df["dominant_phase_raw"].value_counts(normalize=True)
    for phase in counts.index:
        print(f"  {phase:>10}: {counts[phase]} ({rates[phase]*100:.1f}%)")