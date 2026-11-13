"""
Computes per-clip best shift (same as find_global_shift.py) but joins the
result against the CirCor patient metadata CSV to look for correlations with
clinical/recording factors, not just filename patterns.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import random
import re
import numpy as np
import pandas as pd
from collections import Counter

from segmenter.segmentation_model import SegmentationModel
from segmenter.duration_distributions import DataDistribution
from segmenter.data_loading import create_segmentation_array, create_train_test_split

DATA_DIR = paths.CIRCOR_DIR
# Adjust this path if your metadata CSV lives elsewhere -
# in the standard CirCor release it's usually a sibling folder/file, e.g.:
METADATA_PATH = paths.CIRCOR_METADATA_PATH

RECORDING_FREQUENCY = 4000
FEATURE_FREQUENCY = 50
MAX_SHIFT = 30

# --- load metadata ---
meta = pd.read_csv(METADATA_PATH)
print("Metadata columns found:")
print(list(meta.columns))
print(f"\n{len(meta)} patient rows loaded\n")

# figure out the patient-id column name (varies by release version)
id_col_candidates = [c for c in meta.columns if "patient" in c.lower() and "id" in c.lower()]
if not id_col_candidates:
    id_col_candidates = [meta.columns[0]]
ID_COL = id_col_candidates[0]
print(f"Using '{ID_COL}' as patient ID column")
meta[ID_COL] = meta[ID_COL].astype(str)

random.seed(0)
train_recordings, train_segmentations, test_recordings, test_segmentations, \
    train_names, test_names = create_train_test_split(directory=DATA_DIR, frac_train=0.8, return_names=True)

train_clips, train_annotations = [], []
for rec, seg in zip(train_recordings, train_segmentations):
    c, g = create_segmentation_array(rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY)
    train_clips.extend(c)
    train_annotations.extend(g)

model = SegmentationModel()
data_distribution = DataDistribution(train_segmentations)
model.fit(train_clips, train_annotations, data_distribution=data_distribution)

rows = []
for rec, seg, name in zip(test_recordings, test_segmentations, test_names):
    clips, annotations = create_segmentation_array(rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY)
    patient_id = re.match(r"(\d+)", name).group(1)  # leading digits = patient id
    valve = name[len(patient_id):].lstrip("_")       # remainder = valve/location suffix

    for clip, true_states in zip(clips, annotations):
        pred_states = model.predict(clip)
        n = min(len(true_states), len(pred_states))
        t_arr = np.array(true_states[:n])
        p_arr = np.array(pred_states[:n])

        best_shift, best_score = None, -1
        for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
            if shift >= 0:
                t2 = t_arr[shift:]
                p2 = p_arr[:len(p_arr) - shift] if shift > 0 else p_arr
            else:
                t2 = t_arr[:len(t_arr) + shift]
                p2 = p_arr[-shift:]
            m = min(len(t2), len(p2))
            if m == 0:
                continue
            score = (t2[:m] == p2[:m]).mean()
            if score > best_score:
                best_score = score
                best_shift = shift

        if best_shift is not None:
            rows.append({"patient_id": patient_id, "recording_name": name, "valve": valve,
                         "shift": best_shift, "abs_shift": abs(best_shift), "score": best_score})

df = pd.DataFrame(rows)
merged = df.merge(meta, left_on="patient_id", right_on=ID_COL, how="left")
print(f"\nMatched {merged[ID_COL].notna().sum()} / {len(df)} clips to metadata rows")

# --- flag "bad shift" clips ---
merged["large_shift"] = merged["abs_shift"] > 5

print(f"\n{merged['large_shift'].sum()} / {len(merged)} clips have |shift| > 5 frames")

# --- break down by every likely categorical metadata column ---
likely_cols = [c for c in meta.columns if any(k in c.lower() for k in
              ["murmur", "outcome", "diagnosis", "pregnan", "campaign", "sex", "location", "grad", "timing", "pitch", "quality", "shape"])]

print("\n=== Large-shift rate by categorical metadata field ===")
for col in likely_cols:
    if col not in merged.columns:
        continue
    grp = merged.groupby(col)["large_shift"].agg(["mean", "count"])
    grp = grp[grp["count"] >= 3].sort_values("mean", ascending=False)
    if len(grp) > 1:
        print(f"\n--- {col} ---")
        print(grp)

# --- numeric fields: age, height, weight ---
numeric_cols = [c for c in meta.columns if any(k in c.lower() for k in ["age", "height", "weight"])]
print("\n=== Correlation of |shift| with numeric metadata ===")
for col in numeric_cols:
    if col not in merged.columns:
        continue
    sub = merged[[col, "abs_shift"]].dropna()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) > 10:
        corr = sub[col].corr(sub["abs_shift"])
        print(f"{col}: corr={corr:.3f}  (n={len(sub)})")

# --- valve/recording location breakdown (still worth checking alongside clinical factors) ---
print("\n=== Large-shift rate by valve/recording location ===")
print(merged.groupby("valve")["large_shift"].agg(["mean", "count"]).sort_values("mean", ascending=False))

merged.to_csv(paths.DIAGNOSTICS_RESULTS_DIR / "shift_metadata_breakdown.csv", index=False)
print("\nWrote full merged table to shift_metadata_breakdown.csv for further inspection")

# =========================================================================
# Debug plot: predicted vs true states for one specific recording (49983_PV)
# =========================================================================
import matplotlib.pyplot as plt

STATE_LABELS = {0: "Unclassified", 1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}

# =========================================================================
# Debug plot: predicted vs true states for the worst-scoring clip found
# =========================================================================
import matplotlib.pyplot as plt

STATE_LABELS = {0: "Unclassified", 1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}

# find the actual worst-performing clip from the data you just computed
worst_row = df.loc[df["score"].idxmin()]
DEBUG_NAME = worst_row["recording_name"]
print(f"\n[debug plot] Worst-scoring clip: {DEBUG_NAME} (score={worst_row['score']:.4f}, shift={worst_row['shift']})")

debug_idx = None
for i, name in enumerate(test_names):
    if name == DEBUG_NAME:
        debug_idx = i
        break

if debug_idx is None:
    print(f"[debug plot] Couldn't re-locate {DEBUG_NAME} in test_names — naming mismatch between df and test_names")
else:
    rec = test_recordings[debug_idx]
    seg = test_segmentations[debug_idx]

    clips, annotations = create_segmentation_array(
        rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY
    )

    all_true, all_pred = [], []
    for clip, truth in zip(clips, annotations):
        pred = model.predict(clip)
        all_true.append(truth)
        all_pred.append(pred)

    true_frames = np.concatenate(all_true)
    pred_frames = np.concatenate(all_pred)

    n = min(len(true_frames), len(pred_frames))
    true_frames = true_frames[:n]
    pred_frames = pred_frames[:n]
    t = np.arange(n) / FEATURE_FREQUENCY

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].step(t, true_frames, where="post", color="tab:blue")
    axes[0].set_yticks(list(STATE_LABELS.keys()))
    axes[0].set_yticklabels(list(STATE_LABELS.values()))
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{DEBUG_NAME} — predicted vs true")

    axes[1].step(t, pred_frames, where="post", color="tab:red")
    axes[1].set_yticks(list(STATE_LABELS.keys()))
    axes[1].set_yticklabels(list(STATE_LABELS.values()))
    axes[1].set_ylabel("Predicted")
    axes[1].set_xlabel("Time (frames @ 50Hz)")

    plt.tight_layout()
    plt.show()

    overlap = np.mean(true_frames == pred_frames)
    print(f"[debug plot] {DEBUG_NAME} frame agreement: {overlap:.4f}")
    print(f"[debug plot] True unique labels: {np.unique(true_frames)}")
    print(f"[debug plot] Pred unique labels: {np.unique(pred_frames)}")
    print(f"[debug plot] True length: {len(true_frames)}, Pred length: {len(pred_frames)}")

    # =========================================================================
# Heart-rate correlation: does true heart rate predict |shift| directly?
# =========================================================================
from segmenter.heart_rate import get_heart_rate

hr_rows = []
for rec, seg, name in zip(test_recordings, test_segmentations, test_names):
    try:
        heart_rates, _ = get_heart_rate(rec, RECORDING_FREQUENCY)
        hr = heart_rates[0]  # take the top candidate (strongest autocorrelation peak)
    except Exception as e:
        print(f"[heart rate] Skipping {name}: {e}")
        continue
    hr_rows.append({"recording_name": name, "heart_rate": hr})

hr_df = pd.DataFrame(hr_rows)

# join heart rate onto the shift results already computed in df
hr_merged = df.merge(hr_df, on="recording_name", how="left")

sub = hr_merged[["heart_rate", "abs_shift"]].dropna()
if len(sub) > 10:
    corr = sub["heart_rate"].corr(sub["abs_shift"])
    print(f"\n[heart rate] Correlation of heart_rate with abs_shift: corr={corr:.3f} (n={len(sub)})")
else:
    print(f"\n[heart rate] Not enough matched rows to compute correlation (n={len(sub)})")

# where does 49568_AV's heart rate sit relative to the rest of the test set?
target_row = hr_merged[hr_merged["recording_name"] == "49568_AV"]
if not target_row.empty:
    target_hr = target_row["heart_rate"].values[0]
    percentile = (hr_merged["heart_rate"] < target_hr).mean() * 100
    print(f"\n[heart rate] 49568_AV heart rate: {target_hr:.1f} bpm "
          f"({percentile:.1f}th percentile of test set, n={hr_merged['heart_rate'].notna().sum()})")
    print(f"[heart rate] Test set heart rate stats: "
          f"mean={hr_merged['heart_rate'].mean():.1f}, "
          f"median={hr_merged['heart_rate'].median():.1f}, "
          f"max={hr_merged['heart_rate'].max():.1f}")
else:
    print("\n[heart rate] Couldn't find 49568_AV in merged heart-rate table")

hr_merged.to_csv(paths.DIAGNOSTICS_RESULTS_DIR / "shift_heartrate_breakdown.csv", index=False)
print("\n[heart rate] Wrote shift_heartrate_breakdown.csv for further inspection")


# =========================================================================
# Check whether 85338_AV's best shift is a real optimum or a search cutoff
# =========================================================================
WIDE_MAX_SHIFT = 90  # 3x the original range

target_idx = None
for i, name in enumerate(test_names):
    if name == "85338_AV":
        target_idx = i
        break

if target_idx is None:
    print("\n[wide shift check] Couldn't find 85338_AV in test_names — it may have landed in train this run")
else:
    rec = test_recordings[target_idx]
    seg = test_segmentations[target_idx]
    clips, annotations = create_segmentation_array(
        rec, seg, recording_frequency=RECORDING_FREQUENCY, feature_frequency=FEATURE_FREQUENCY
    )

    all_true, all_pred = [], []
    for clip, truth in zip(clips, annotations):
        pred = model.predict(clip)
        all_true.append(truth)
        all_pred.append(pred)
    t_arr = np.concatenate(all_true)
    p_arr = np.concatenate(all_pred)

    scores_by_shift = {}
    for shift in range(-WIDE_MAX_SHIFT, WIDE_MAX_SHIFT + 1):
        if shift >= 0:
            t2 = t_arr[shift:]
            p2 = p_arr[:len(p_arr) - shift] if shift > 0 else p_arr
        else:
            t2 = t_arr[:len(t_arr) + shift]
            p2 = p_arr[-shift:]
        m = min(len(t2), len(p2))
        if m == 0:
            continue
        scores_by_shift[shift] = (t2[:m] == p2[:m]).mean()
    best_shift = max(scores_by_shift, key=scores_by_shift.get)
    print(f"\n[wide shift check] 85338_AV best shift over ±{WIDE_MAX_SHIFT}: "
          f"shift={best_shift}, score={scores_by_shift[best_shift]:.4f}")
    print(f"[wide shift check] score at shift=-30 (original boundary): {scores_by_shift.get(-30, 'N/A'):.4f}")
    print(f"[wide shift check] score at shift=+30: {scores_by_shift.get(30, 'N/A'):.4f}")

    # print the trend near the old boundary to see if it was still climbing
    for s in range(-40, -19):
        if s in scores_by_shift:
            print(f"  shift={s:4d}  score={scores_by_shift[s]:.4f}")

    # =========================================================================
# Sanity check: does create_train_test_split give the same split every run?
# =========================================================================
random.seed(0)
check_train_rec, check_train_seg, check_test_rec, check_test_seg, \
    check_train_names, check_test_names = create_train_test_split(
        directory=DATA_DIR, frac_train=0.8, return_names=True
    )

print(f"\n[split check] Test set size this call: {len(check_test_names)}")
print(f"[split check] Original test set size (this run): {len(test_names)}")
print(f"[split check] Same test set as the one already computed above? "
      f"{set(check_test_names) == set(test_names)}")

if set(check_test_names) != set(test_names):
    only_in_first = set(test_names) - set(check_test_names)
    only_in_second = set(check_test_names) - set(test_names)
    print(f"[split check] In original but not repeat: {list(only_in_first)[:10]}")
    print(f"[split check] In repeat but not original: {list(only_in_second)[:10]}")