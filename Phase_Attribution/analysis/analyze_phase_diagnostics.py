"""
analysis/analyze_phase_diagnostics.py

Consumes phase_diagnostics_per_recording.csv (output of
analysis/gather_phase_diagnostics.py) and runs two INDEPENDENT checks on the
"raw margin conflates scale with real separation" hypothesis. It does
not propose a fix. It reports what each check says, including if they
disagree with each other -- that disagreement would itself be
informative, not something to average away.

PREREQUISITE: run analysis/gather_phase_diagnostics.py first. This script does
not regenerate that CSV.

TEST A -- does the pipeline's OWN z-scored attribution already do
better than raw, on the fidelity metric?
    Compares fidelity_match_raw.mean() vs fidelity_match_zscored.mean(),
    split by boundary_source (ground_truth / predicted).
    CAVEAT (carried over from analysis/gather_phase_diagnostics.py's own
    docstring): that z-score is computed from test-set mean/std per
    phase, which was flagged earlier in this project as vulnerable to
    being dominated by whichever phase happens to have the smallest
    test-set variance (e.g. Diastole). So:
      - if zscored beats raw convincingly -> suggestive, not proof.
        Check it's not 2-3 recordings' worth of gain (see per-recording
        breakdown printed below).
      - if zscored does NOT beat raw -> weak evidence against
        calibration in general (could just mean this particular
        z-scoring is the wrong one), not a full falsification.

TEST B -- for recordings where Systole was expected but didn't win,
was the RAW margin actually close to whichever phase did win?
    This does not depend on the z-scoring at all, so it isn't subject
    to Test A's caveat. If most flips are near-ties in raw margin
    space, that's real, scale-independent support for "a properly
    calibrated rescaling could plausibly change these outcomes." If
    most flips are blowouts (Systole's margin nowhere close), no
    rescaling scheme is going to fix that -- something else is
    happening and calibration work would be solving the wrong problem.

Prints both, plus where they agree/disagree, and does NOT collapse
them into a single verdict -- that call is yours to make once you see
both numbers.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = paths.PHASE_DIAGNOSTICS_PER_RECORDING_CSV
PHASES = ["S1", "Systole", "S2", "Diastole"]  # must match run_GMM.PHASES; edit if that list differs


def load(path):
    p = Path(path)
    if not p.exists():
        print(f"Could not find {path}.")
        print("Run analysis/gather_phase_diagnostics.py first, or pass the correct path:")
        print("  python analysis/analyze_phase_diagnostics.py path\\to\\phase_diagnostics_per_recording.csv")
        sys.exit(1)
    df = pd.read_csv(p)
    missing = [c for c in ["expected_phases", "dominant_phase_raw", "dominant_phase_zscored",
                            "fidelity_match_raw", "fidelity_match_zscored", "boundary_source"]
               if c not in df.columns]
    if missing:
        print(f"CSV is missing expected columns: {missing}")
        print(f"Actual columns: {list(df.columns)}")
        print("This means analysis/gather_phase_diagnostics.py's schema doesn't match what")
        print("this script assumes -- paste the column list and I'll fix it.")
        sys.exit(1)
    margin_cols_found = {p_: f"margin_{p_}" for p_ in PHASES if f"margin_{p_}" in df.columns}
    if len(margin_cols_found) < len(PHASES):
        missing_margins = [p_ for p_ in PHASES if p_ not in margin_cols_found]
        print(f"WARNING: missing margin_{{phase}} columns for: {missing_margins}")
        print(f"Continuing with only: {list(margin_cols_found.keys())}")
    return df, margin_cols_found


def test_a(df):
    print("=" * 70)
    print("TEST A: raw vs z-scored fidelity match rate (from existing pipeline output)")
    print("=" * 70)
    eligible = df[df["has_expected_phase"] == True].copy()  # noqa: E712
    for arm in eligible["boundary_source"].unique():
        sub = eligible[eligible["boundary_source"] == arm]
        raw_rate = sub["fidelity_match_raw"].mean()
        z_rate = sub["fidelity_match_zscored"].mean()
        n = len(sub)
        print(f"[{arm}] n={n}  raw={raw_rate:.1%}  zscored={z_rate:.1%}  delta={z_rate - raw_rate:+.1%}")

        # how many recordings actually flip fidelity outcome between raw and zscored?
        flips_to_correct = sub[(sub["fidelity_match_raw"] == False) & (sub["fidelity_match_zscored"] == True)]  # noqa: E712
        flips_to_wrong = sub[(sub["fidelity_match_raw"] == True) & (sub["fidelity_match_zscored"] == False)]  # noqa: E712
        print(f"    z-scoring fixed {len(flips_to_correct)} recordings, broke {len(flips_to_wrong)} recordings")
        if 0 < len(flips_to_correct) <= 5:
            print(f"    (fixed ids: {flips_to_correct['recording_id'].tolist()})")
        if 0 < len(flips_to_wrong) <= 5:
            print(f"    (broke ids: {flips_to_wrong['recording_id'].tolist()})")
    print()
    print("Read this as: is the gain broad (many recordings, roughly symmetric small")
    print("improvement) or is it a few recordings swinging the average? A net gain")
    print("driven by fixing 8 and breaking 1 is a different finding than fixing 8 and")
    print("breaking 7 net-positive-by-1.")
    print()


def test_b(df, margin_cols_found):
    print("=" * 70)
    print("TEST B: for Systole-expected flips, is the RAW margin actually close?")
    print("=" * 70)
    if "Systole" not in margin_cols_found:
        print("No margin_Systole column found -- cannot run this test.")
        return

    def expects_systole(val):
        return isinstance(val, str) and "Systole" in val

    flips = df[
        df["expected_phases"].apply(expects_systole)
        & (df["dominant_phase_raw"] != "Systole")
    ].copy()

    print(f"Flipped recordings (expected Systole, raw attribution said otherwise): {len(flips)}")
    if flips.empty:
        print("No flips found -- either the fidelity metric is already fine on this")
        print("axis, or expected_phases/dominant_phase_raw values don't look like")
        print("what this script assumes (check a few rows manually).")
        return

    rows = []
    for _, row in flips.iterrows():
        winner = row["dominant_phase_raw"]
        if winner not in margin_cols_found:
            continue
        systole_margin = row[margin_cols_found["Systole"]]
        winner_margin = row[margin_cols_found[winner]]
        winner_scale = df[margin_cols_found[winner]].std()
        gap = abs(winner_margin) - abs(systole_margin)
        gap_units = gap / winner_scale if winner_scale else np.nan
        rows.append({
            "recording_id": row["recording_id"],
            "boundary_source": row["boundary_source"],
            "winner": winner,
            "margin_Systole": systole_margin,
            "margin_winner": winner_margin,
            "gap_in_winner_std_units": gap_units,
        })

    gap_df = pd.DataFrame(rows)
    print(gap_df.to_string(index=False))
    print()

    near_tie = (gap_df["gap_in_winner_std_units"].abs() < 1.0).mean()
    print(f"Fraction of flips that are near-ties (gap < 1 std of winner's margin spread): {near_tie:.1%}")
    print()
    if near_tie > 0.6:
        print("-> Most flips are close calls in raw-margin space. This supports the")
        print("   scale-conflation story on its own terms (independent of the shaky")
        print("   test-set z-scoring flagged in Test A).")
    elif near_tie < 0.3:
        print("-> Most flips are NOT close calls -- Systole's raw margin is far from")
        print("   the winner's. Rescaling is unlikely to fix these specific cases.")
        print("   Whatever Test A showed, this says the flips themselves aren't")
        print("   simple near-ties that calibration would flip.")
    else:
        print("-> Mixed. Worth looking at whether near-tie vs blowout flips cluster")
        print("   by boundary_source, murmur_grade, or segmenter_accuracy.")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df, margin_cols_found = load(path)
    test_a(df)
    test_b(df, margin_cols_found)

    print("=" * 70)
    print("WHERE TEST A AND TEST B AGREE / DISAGREE")
    print("=" * 70)
    print("If Test A shows z-scoring net-improves fidelity AND Test B shows most")
    print("flips are near-ties: both point the same way -- calibration is plausibly")
    print("worth pursuing properly (with training-derived scale, not this test-set one).")
    print()
    print("If Test A shows little/no improvement but Test B shows near-ties: the")
    print("hypothesis may still be right, but this particular z-scoring implementation")
    print("is the wrong one -- don't conclude calibration is dead, conclude THIS")
    print("z-score is dead.")
    print()
    print("If Test B shows blowouts (not near-ties): stop pursuing calibration/rescaling")
    print("regardless of what Test A says -- the flips aren't close calls, so no scale")
    print("fix explains them. Something else is driving the fidelity gap.")


if __name__ == "__main__":
    main()