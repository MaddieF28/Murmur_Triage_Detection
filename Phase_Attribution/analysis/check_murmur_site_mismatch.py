"""
analysis/check_murmur_site_mismatch.py

Murmur is a PATIENT-level label, but CirCor's "Murmur locations" field
names which specific valve site(s) (AV/PV/TV/MV) the murmur is actually
audible at. A recording from a Present patient whose murmur is NOT
listed at that recording's own valve site may be acoustically normal
at that site -- a real "label doesn't apply to this recording" case,
independent of segmentation, GMM scale, or feature quality.

This checks: does fidelity_match rate / margin quality differ between
recordings where the site IS in Murmur locations vs NOT?

Prints small aggregate tables only (the earlier full per-row CSV was
too large to share -- this avoids that by summarizing before output).

Input: phase_diagnostics_per_recording.csv (already generated).
Reads Murmur locations fresh from the CirCor metadata CSV.

ASSUMPTION FLAGGED: valve site is parsed from the recording_id suffix
after the last underscore (e.g. "13918_AV" -> "AV"), and "Murmur
locations" is assumed "+"-delimited (e.g. "AV+MV+PV+TV"). Tell me if
either doesn't match your actual data.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import pandas as pd

from experiments.arm_a_gmm import CIRCOR_METADATA_PATH

PER_RECORDING_CSV = paths.PHASE_DIAGNOSTICS_PER_RECORDING_CSV
MURMUR_LOCATIONS_COLUMN = "Murmur locations"


def get_site(recording_id):
    return recording_id.rsplit("_", 1)[-1]


def main():
    df = pd.read_csv(PER_RECORDING_CSV)
    meta = pd.read_csv(CIRCOR_METADATA_PATH)
    meta["patient_id"] = meta["Patient ID"].astype(str).str.strip()
    meta = meta.set_index("patient_id")

    df["site"] = df["recording_id"].apply(get_site)
    df["murmur_locations"] = df["patient_id"].astype(str).map(meta[MURMUR_LOCATIONS_COLUMN])

    def site_matches(row):
        locs = row["murmur_locations"]
        if pd.isna(locs):
            return None  # no location info (e.g. Absent/Unknown patients)
        return row["site"] in str(locs).split("+")

    df["site_in_murmur_locations"] = df.apply(site_matches, axis=1)

    scoped = df[df["has_expected_phase"] == True].copy()  # Present recordings w/ known timing
    print("=" * 70)
    print("Fidelity match rate: site IS vs IS NOT in Murmur locations")
    print("=" * 70)
    for arm in ("ground_truth", "predicted"):
        arm_df = scoped[scoped["boundary_source"] == arm]
        print(f"\n--- {arm} arm ---")
        summary = arm_df.groupby("site_in_murmur_locations").agg(
            n=("fidelity_match_raw", "size"),
            fidelity_match_raw_rate=("fidelity_match_raw", "mean"),
            fidelity_match_zscored_rate=("fidelity_match_zscored", "mean"),
            mean_abs_margin_systole=("margin_Systole", lambda s: s.abs().mean()),
            mean_correct=("correct", "mean"),
        )
        print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("Same breakdown, restricted to dominant_phase_raw in {S1, S2}")
    print("(does the site mismatch explain WHY S1/S2 wins specifically?)")
    print("=" * 70)
    s1s2_only = scoped[scoped["dominant_phase_raw"].isin(["S1", "S2"])]
    for arm in ("ground_truth", "predicted"):
        arm_df = s1s2_only[s1s2_only["boundary_source"] == arm]
        if len(arm_df) == 0:
            continue
        print(f"\n--- {arm} arm (n={len(arm_df)} S1/S2-dominant recordings) ---")
        counts = arm_df["site_in_murmur_locations"].value_counts(dropna=False)
        rates = arm_df["site_in_murmur_locations"].value_counts(normalize=True, dropna=False)
        for val in counts.index:
            print(f"  site_in_murmur_locations={val}: {counts[val]} ({rates[val]*100:.1f}%)")

    # Compare against the overall rate of site-in-locations among ALL
    # has_expected_phase recordings, as a baseline for whether S1/S2-dominant
    # recordings are disproportionately off-site.
    print("\nBaseline: site_in_murmur_locations rate across ALL has_expected_phase recordings:")
    for arm in ("ground_truth", "predicted"):
        arm_df = scoped[scoped["boundary_source"] == arm]
        rate = arm_df["site_in_murmur_locations"].mean()
        print(f"  {arm}: {rate*100:.1f}% (n={len(arm_df)})")


if __name__ == "__main__":
    main()