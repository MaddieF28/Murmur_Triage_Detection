"""
paths.py

Single source of truth for every on-disk location the scripts in this repo
read or write. Everything is anchored to the repo root (the folder holding
this file), so scripts work regardless of the current working directory.

Datasets live under data/, outputs under results/<kind>/. Nothing else in
the repo should hardcode a data or results path -- import from here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- datasets -----------------------------------------------------------------
DATA_DIR = ROOT / "data"

CIRCOR_ROOT = DATA_DIR / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
CIRCOR_DIR = CIRCOR_ROOT / "training_data"            # .wav + .tsv per recording
CIRCOR_METADATA_PATH = CIRCOR_ROOT / "training_data.csv"

PHYSIONET2016_RAW_DIR = DATA_DIR / (
    "classification-of-heart-sound-recordings-the-physionetcomputing-in-cardiology-challenge-2016-1.0.0"
)
# PhysioNet 2016 training-a..f after convert_physionet2016_annotations_to_tsv.py
# and resample_physionet2016_to_4000hz.py: CirCor-style .tsv + 4000 Hz .wav.
PHYSIONET2016_CONVERTED_DIR = DATA_DIR / "physionet2016_converted"
TRAINING_LETTERS = "abcdef"
TRAINING_DIRS = {
    f"training-{letter}": PHYSIONET2016_CONVERTED_DIR / f"training-{letter}"
    for letter in TRAINING_LETTERS
}

# Yaseen 2018: 1000 wavs, 200 per class, native 8000 Hz mono, no phase annotations.
# Raw copy under YASEEN_RAW_DIR/<class>/New_<class>_NNN.wav; 4000 Hz copies written
# by data_prep/resample_yaseen_to_4000hz.py under YASEEN_DIR/<class>/.
YASEEN_CLASSES = ("AS", "MR", "MS", "MVP", "N")
YASEEN_RAW_DIR = DATA_DIR / "yaseen2018"
YASEEN_DIR = DATA_DIR / "yaseen2018_converted"

TINY_TEST_DIR = DATA_DIR / "tiny_test"                # 13-recording smoke test set
EXAMPLES_DIR = DATA_DIR / "examples"                  # a0001.tsv, example_data.mat

# --- results ------------------------------------------------------------------
RESULTS_DIR = ROOT / "results"
SEGMENTER_RESULTS_DIR = RESULTS_DIR / "segmenter"     # segmenter accuracy reports, checkpoints
FEATURES_RESULTS_DIR = RESULTS_DIR / "features"       # per-phase feature tables
ARMS_RESULTS_DIR = RESULTS_DIR / "arms"               # Arm A / Arm B classifier outputs
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"       # fidelity follow-up analyses
DIAGNOSTICS_RESULTS_DIR = RESULTS_DIR / "diagnostics" # segmenter debugging dumps
LOGS_DIR = RESULTS_DIR / "logs"

for _d in (SEGMENTER_RESULTS_DIR, FEATURES_RESULTS_DIR, ARMS_RESULTS_DIR,
           ANALYSIS_RESULTS_DIR, DIAGNOSTICS_RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- frequently shared result files ------------------------------------------
# Segmenter
TEST_ACCURACY_BY_RECORDING_CSV = SEGMENTER_RESULTS_DIR / "test_accuracy_by_recording.csv"
PER_RECORDING_ACCURACY_CSV = SEGMENTER_RESULTS_DIR / "per_recording_accuracy.csv"
CIRCOR_SEGMENTER_ACCURACY_CSV = SEGMENTER_RESULTS_DIR / "circor_segmenter_accuracy.csv"
CATASTROPHIC_AND_POOR_CSV = SEGMENTER_RESULTS_DIR / "catastrophic_and_poor_recordings.csv"
CIRCOR_PREDICTED_FRAMES_CHECKPOINT = SEGMENTER_RESULTS_DIR / "circor_predicted_frames_checkpoint.pkl"

# Features
CIRCOR_FEATURE_TABLE = FEATURES_RESULTS_DIR / "circor_binary_features.tsv"

# Yaseen (Phase 2)
YASEEN_FEATURE_TABLE = FEATURES_RESULTS_DIR / "yaseen_phase_features.tsv"
YASEEN_PREDICTED_FRAMES_CHECKPOINT = SEGMENTER_RESULTS_DIR / "yaseen_predicted_frames_checkpoint.pkl"
YASEEN_SEGMENTATION_QUALITY_CSV = SEGMENTER_RESULTS_DIR / "yaseen_segmentation_quality.csv"
YASEEN_DUPLICATE_GROUPS_CSV = SEGMENTER_RESULTS_DIR / "yaseen_duplicate_groups.csv"
YASEEN_ARMS_RESULTS_PKL = ARMS_RESULTS_DIR / "yaseen_three_arms_results.pkl"
YASEEN_PER_RECORDING_TSV = ARMS_RESULTS_DIR / "yaseen_phase_attribution_per_recording.tsv"
YASEEN_CLASS_FIDELITY_CSV = ARMS_RESULTS_DIR / "yaseen_class_level_fidelity.csv"
# Group-level split (StratifiedGroupKFold over the ~324 crop groups); the numbers to publish.
YASEEN_CLASS_FIDELITY_GROUPED_CSV = ARMS_RESULTS_DIR / "yaseen_class_level_fidelity_grouped.csv"

# --- paper -------------------------------------------------------------------
PAPER_DIR = ROOT / "paper"                            # main.tex, references.bib
PAPER_FIGURES_DIR = PAPER_DIR / "figures"             # PNGs made by analysis/make_paper_figures.py
CAMERA_READY_FIGURES_DIR = PAPER_DIR / "camera_ready" / "figures_final"  # camera-ready versions (--camera-ready)
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Arms
GMM_RESULTS_PKL = ARMS_RESULTS_DIR / "gmm_murmur_comparison_results.pkl"
GMM_PER_RECORDING_CSV = ARMS_RESULTS_DIR / "gmm_murmur_per_recording_comparison.csv"
GMM_PHASE_FIDELITY_TSV = ARMS_RESULTS_DIR / "phase_fidelity_per_recording.tsv"
NN_RESULTS_PKL = ARMS_RESULTS_DIR / "nn_murmur_comparison_results.pkl"
NN_PER_RECORDING_CSV = ARMS_RESULTS_DIR / "nn_murmur_per_recording_comparison.csv"
NN_PHASE_FIDELITY_TSV = ARMS_RESULTS_DIR / "phase_fidelity_per_recording_nn.tsv"
NN_SHAP_SAMPLE_TSV = ARMS_RESULTS_DIR / "shap_top_features_sample.tsv"
# Arm C: whole-clip NN + post-hoc SHAP (experiments/arm_c_wholeclip_nn_shap.py)
WHOLECLIP_RESULTS_PKL = ARMS_RESULTS_DIR / "wholeclip_nn_shap_comparison_results.pkl"
WHOLECLIP_PER_RECORDING_CSV = ARMS_RESULTS_DIR / "wholeclip_nn_shap_per_recording_comparison.csv"
WHOLECLIP_PHASE_FIDELITY_TSV = ARMS_RESULTS_DIR / "phase_fidelity_per_recording_wholeclip.tsv"
WHOLECLIP_SHAP_STABILITY_CSV = ARMS_RESULTS_DIR / "wholeclip_shap_background_stability.csv"

# Analysis
PHASE_DIAGNOSTICS_PER_RECORDING_CSV = ANALYSIS_RESULTS_DIR / "phase_diagnostics_per_recording.csv"
PHASE_DIAGNOSTICS_GMM_FIT_CSV = ANALYSIS_RESULTS_DIR / "phase_diagnostics_gmm_fit.csv"
PATIENT_CLUSTERED_TESTS_CSV = ANALYSIS_RESULTS_DIR / "patient_clustered_paired_tests.csv"
SIGNED_PHASE_CONTRIBUTIONS_CSV = ANALYSIS_RESULTS_DIR / "signed_phase_contributions.csv"
MARGIN_SCALING_SENSITIVITY_CSV = ANALYSIS_RESULTS_DIR / "margin_scaling_sensitivity.csv"
MARGIN_SCALE_DIAGNOSTICS_CSV = ANALYSIS_RESULTS_DIR / "margin_scale_diagnostics.csv"
MARGIN_SCALING_CROSS_CLASSIFIER_CSV = ANALYSIS_RESULTS_DIR / "margin_scaling_cross_classifier.csv"

# --- upstream comparison ------------------------------------------------------
# Verbatim snapshot of EchoStatements/Springer-Segmentation-Python at 276b9cc,
# used by segmenter_eval/compare_upstream_vs_local.py.
UPSTREAM_SPRINGER_DIR = ROOT / "third_party" / "springer_upstream"
UPSTREAM_VS_LOCAL_PER_RECORDING_CSV = SEGMENTER_RESULTS_DIR / "upstream_vs_local_per_recording.csv"
UPSTREAM_VS_LOCAL_SUMMARY_MD = SEGMENTER_RESULTS_DIR / "upstream_vs_local_summary.md"
