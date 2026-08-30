"""
arm_b.py

Arm B: phase-aware discriminative NN classifier (PyTorch), the
counterpart to Arm A's generative GMM.

One small MLP per phase (S1, Systole, S2, Diastole), each a standalone
5-class classifier trained on that phase's standardized features.
Predictions are combined by summing per-phase log-probabilities per
class (same combination logic as Arm A) and picking the highest total.

Uses the UNPRUNED (full 40-feature) set to start, and the SAME
train/test split as arm_a_pruned.py (recording IDs reused from its
saved results) so the comparison to Arm A is on identical data splits.

SHAP (DeepExplainer) is used per phase for interpretability, saved
alongside predictions so it's directly comparable to Arm A's per-phase
log-likelihood decomposition.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import shap

from extract_features import standardize_features

# --- config -----------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
FEATURE_TABLE_PATH = THIS_DIR / "pcg_feature_table.csv"
ARM_A_PRUNED_RESULTS_PATH = THIS_DIR / "pruned_GMM_results.pkl"  # for split reuse only
RESULTS_PATH = THIS_DIR / "arm_b_results.pkl"

CLASSES = ("AS", "MR", "MS", "MVP", "N")
PHASES = ("S1", "Systole", "S2", "Diastole")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_FRACTION_OF_TRAIN = 0.15  # carved out of train for early stopping

HIDDEN_SIZES = (32, 16)
DROPOUT = 0.3
LEARNING_RATE = 1e-3
MAX_EPOCHS = 300
PATIENCE = 20  # early stopping patience, in epochs without val improvement
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- data prep --------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(FEATURE_TABLE_PATH)

    # Reuse the exact same split as arm_a_pruned.py for a fair comparison,
    # even though this run uses the unpruned (full) feature set.
    with open(ARM_A_PRUNED_RESULTS_PATH, "rb") as f:
        arm_a_results = pickle.load(f)
    train_ids = set(arm_a_results["train_recording_ids"])
    test_ids = set(arm_a_results["test_recording_ids"])

    train_df = df[df["recording_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["recording_id"].isin(test_ids)].reset_index(drop=True)

    assert len(train_df) == len(train_ids), "Train split size mismatch -- check recording_id overlap."
    assert len(test_df) == len(test_ids), "Test split size mismatch -- check recording_id overlap."

    train_scaled, test_scaled, scaler = standardize_features(train_df, test_df)
    return train_scaled, test_scaled, scaler


def phase_columns(df: pd.DataFrame, phase: str) -> list:
    prefix = f"{phase}_"
    return [c for c in df.columns if c.startswith(prefix)]


def class_to_idx():
    return {cls: i for i, cls in enumerate(CLASSES)}


# --- model --------------------------------------------------------------

class PhaseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=HIDDEN_SIZES, n_classes=len(CLASSES), dropout=DROPOUT):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits; CrossEntropyLoss applies softmax internally


# --- training per phase -----------------------------------------------------

def prepare_phase_data(train_df, phase, cls_to_idx):
    """Splits train_df into train/val for this phase, dropping NaN rows
    (missing-phase recordings) for this phase's training only."""
    cols = phase_columns(train_df, phase)
    X = train_df[cols].values.astype(np.float32)
    y = train_df["label"].map(cls_to_idx).values.astype(np.int64)

    mask = ~np.isnan(X).any(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  WARNING: {phase} training data has {n_dropped} NaN rows -- dropping for this phase.")
    X, y = X[mask], y[mask]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION_OF_TRAIN, stratify=y, random_state=RANDOM_STATE
    )
    return X_tr, y_tr, X_val, y_val, cols


def train_phase_mlp(X_tr, y_tr, X_val, y_val, input_dim):
    model = PhaseMLP(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"    early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def train_all_phase_models(train_df, cls_to_idx):
    models = {}
    phase_cols = {}
    for phase in PHASES:
        print(f"  training {phase}...")
        X_tr, y_tr, X_val, y_val, cols = prepare_phase_data(train_df, phase, cls_to_idx)
        model, val_loss = train_phase_mlp(X_tr, y_tr, X_val, y_val, input_dim=len(cols))
        models[phase] = model
        phase_cols[phase] = cols
        print(f"    {phase}: trained on {len(X_tr)}, val on {len(X_val)}, final val loss {val_loss:.4f}")
    return models, phase_cols


# --- classification (combine phases like Arm A) -----------------------------

def classify_recording(row: pd.Series, models: dict, phase_cols: dict, cls_to_idx: dict) -> dict:
    idx_to_cls = {i: c for c, i in cls_to_idx.items()}
    per_class_per_phase = {cls: {} for cls in CLASSES}
    per_class_total = {cls: 0.0 for cls in CLASSES}
    missing_phases = []
    any_phase_used_at_all = False

    for phase in PHASES:
        cols = phase_cols[phase]
        x = np.asarray(row[cols].values, dtype=np.float32).reshape(1, -1)

        if np.isnan(x).any():
            for cls in CLASSES:
                per_class_per_phase[cls][phase] = None
            missing_phases.append(phase)
            continue

        with torch.no_grad():
            x_t = torch.from_numpy(x).to(DEVICE)
            log_probs = torch.log_softmax(models[phase](x_t), dim=1).cpu().numpy()[0]

        any_phase_used_at_all = True
        for cls in CLASSES:
            lp = float(log_probs[cls_to_idx[cls]])
            per_class_per_phase[cls][phase] = lp
            per_class_total[cls] += lp

    if not any_phase_used_at_all:
        per_class_total = {cls: float("-inf") for cls in CLASSES}

    predicted_class = max(per_class_total, key=lambda k: per_class_total[k])

    return {
        "recording_id": row["recording_id"],
        "true_label": row["label"],
        "predicted_label": predicted_class,
        "per_class_total_logprob": per_class_total,
        "per_class_per_phase_logprob": per_class_per_phase,
        "missing_phases": missing_phases,
    }


# --- SHAP interpretability ---------------------------------------------------

def compute_shap_values(models: dict, phase_cols: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    One DeepExplainer per phase. Background set: a sample of training
    data (SHAP needs a background distribution to compute expected
    values against). Explains the test set.
    """
    shap_results = {}

    for phase in PHASES:
        print(f"  computing SHAP for {phase}...")
        cols = phase_cols[phase]

        train_X = train_df[cols].values.astype(np.float32)
        train_X = train_X[~np.isnan(train_X).any(axis=1)]
        background = train_X[np.random.RandomState(RANDOM_STATE).choice(
            len(train_X), size=min(100, len(train_X)), replace=False
        )]
        background_t = torch.from_numpy(background).to(DEVICE)

        test_X = test_df[cols].values.astype(np.float32)
        valid_mask = ~np.isnan(test_X).any(axis=1)
        test_X_valid = test_X[valid_mask]
        test_ids_valid = test_df["recording_id"].values[valid_mask]

        explainer = shap.DeepExplainer(models[phase], background_t)
        test_X_t = torch.from_numpy(test_X_valid).to(DEVICE)
        shap_values = explainer.shap_values(test_X_t)
        # shap_values shape: (n_samples, n_features, n_classes) for multi-class DeepExplainer

        shap_results[phase] = {
            "shap_values": shap_values,
            "feature_names": cols,
            "recording_ids": test_ids_valid.tolist(),
        }

    return shap_results


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print(f"Using device: {DEVICE}")
    cls_to_idx = class_to_idx()

    print(f"\nLoading {FEATURE_TABLE_PATH} and reusing Arm A's pruned split...")
    train_df, test_df, scaler = load_and_split()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    print("\nTraining per-phase MLPs (unpruned, 40 features/phase)...")
    models, phase_cols = train_all_phase_models(train_df, cls_to_idx)

    print("\nClassifying test recordings...")
    per_recording_results = []
    for _, row in test_df.iterrows():
        result = classify_recording(row, models, phase_cols, cls_to_idx)
        per_recording_results.append(result)

    y_true = [r["true_label"] for r in per_recording_results]
    y_pred = [r["predicted_label"] for r in per_recording_results]
    accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)])

    per_class_accuracy = {}
    for cls in CLASSES:
        idx = [i for i, t in enumerate(y_true) if t == cls]
        correct = sum(1 for i in idx if y_pred[i] == y_true[i])
        per_class_accuracy[cls] = correct / len(idx) if idx else float("nan")

    confusion = pd.crosstab(
        pd.Series(y_true, name="true"), pd.Series(y_pred, name="predicted")
    )

    print("\n" + "=" * 60)
    print("ARM B (PyTorch MLP, unpruned features) RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {accuracy * 100:.1f}%")
    print("\nPer-class accuracy:")
    for cls, acc in per_class_accuracy.items():
        print(f"  {cls}: {acc * 100:.1f}%")
    print("\nConfusion matrix:")
    print(confusion)

    print("\nComputing SHAP values (this may take a while)...")
    shap_results = compute_shap_values(models, phase_cols, train_df, test_df)

    # Save model state dicts (not full model objects, for portability)
    model_state_dicts = {phase: models[phase].state_dict() for phase in PHASES}

    output = {
        "per_recording_results": per_recording_results,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "model_state_dicts": model_state_dicts,
        "shap_results": shap_results,
        "config": {
            "hidden_sizes": HIDDEN_SIZES,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "classes": CLASSES,
            "phases": PHASES,
            "pruned": False,
        },
        "train_recording_ids": train_df["recording_id"].tolist(),
        "test_recording_ids": test_df["recording_id"].tolist(),
        "phase_cols": phase_cols,
    }

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(output, f)

    print(f"\nFull results saved to {RESULTS_PATH}")