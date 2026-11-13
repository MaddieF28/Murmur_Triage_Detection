"""
diagnostics/diagnose_flips.py

Standalone diagnostic for cycle-rotation errors (half-flip and both
quarter-flips) AND for search-internals bugs (duration/emission-span
mismatches in the padded tail region) in the HSMM heart-sound segmenter.

This does NOT modify segmenter/viterbi.py or segmenter/segmentation_model.py.

Two layers, both scored/computed under the model's real math:

  1. Rotation-scoring audit (diagnose_clip): reimplements just the
     probability-setup half of viterbi_segment (_compute_probs) so it can
     score arbitrary candidate label sequences -- ground truth, the model's
     real output, or a rotation of either -- under the exact same
     emission/duration numbers the real decode uses, for each of the top-k
     heart-rate candidates. Tells you WHETHER a different rotation or a
     different heart-rate candidate would have scored higher.

  2. Search-internals instrumentation (viterbi_segment_debug /
     print_search_internals): a copy of viterbi_segment with every line of
     actual delta/psi/psi_duration arithmetic left untouched, but every
     accepted update recorded -- tagged by clamp region, with nominal
     duration vs. actual (start,end) span -- plus the full delta[seq_len:,:]
     grid and a step-by-step backtrack trace. Tells you WHERE/HOW the search
     picked what it picked, independent of whether that pick was "correct."

  diagnose_clip_deep() runs both, in one call, for one clip, printing
  everything. verify_against_real() is run first and the internals dump is
  skipped (loudly) if it doesn't match segmenter/viterbi.py's real output exactly --
  don't trust search-internals output if that check fails.

  summarize_clip() / print_clip_summary_table() run the same two layers but
  return/print a compact, fixed-width row per clip instead of the full dump
  -- use these when scanning many clips at once (that's what
  diagnostics/summarize_flipped_clips.py's main() calls by default now). Use
  diagnose_clip_deep() directly when you want the full internals for one
  specific clip you've already narrowed in on.

If you ever change how segmenter/viterbi.py builds observation_probs or duration_probs,
mirror the change in BOTH _compute_probs and viterbi_segment_debug here, or
they will silently drift out of sync with the real decode.

KNOWN BUG IN segmenter/viterbi.py, REPRODUCED HERE ON PURPOSE: when building
duration_probs, the real viterbi_segment uses
cov=d_distributions[state_j, 1] for state_j == 1 (S1) -- i.e. Systole's
variance -- instead of d_distributions[state_j - 1, 1] (S1's own variance).
Every other state indexes correctly. This looks like an unintentional
copy-paste bug, not a design choice. Both _compute_probs and
viterbi_segment_debug below reproduce it deliberately (search for
"BUG-FOR-BUG PARITY"), so that scores/internals computed here match what
the real decode actually does. If segmenter/viterbi.py's indexing is ever fixed,
update both spots here too, or verify_against_real() will start failing
again for the same reason it did before this was found.

Nothing here uses ground truth at decode time. model.predict() and the
heart_rate/systolic_time actually used are exactly what normal usage would
produce (heart_rates[0]). Ground truth is only used afterward, offline, to
check whether the model's own math would have scored a different rotation
higher -- i.e. this is a post-hoc audit of the search, not a change to it.

Usage:
    from diagnose_flips import diagnose_clip_deep

    for clip_name, recording, ground_truth in your_flipped_clips:
        diagnose_clip_deep(clip_name, recording, ground_truth, model)

    # also run it on clips that did NOT flip, for comparison -- if the
    # scores/internals look qualitatively similar on good clips too, the
    # theory being tested isn't what's driving the errors.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import numpy as np
from scipy.stats import multivariate_normal

from segmenter.heart_rate import get_heart_rate

# Single source of truth for the heart-rate search window used throughout
# this file. Must match whatever model.predict() is actually called with
# wherever it's called (in diagnose_bad_clips.py or anywhere else) -- if
# these values and the min/max_heart_rate passed to model.predict() ever
# diverge, the autocorrelation search window differs and the candidate
# list / "USED BY predict()" label can silently be wrong.
DEFAULT_MIN_HEART_RATE = 40
DEFAULT_MAX_HEART_RATE = 200


# ---------------------------------------------------------------------------
# Layer 1: rotation-scoring audit
# ---------------------------------------------------------------------------

def _compute_probs(observation_sequence, models, total_obs_distribution,
                    distribution, heart_rate, systolic_time, recording_frequency):
    """Mirrors the setup half of viterbi_segment: builds observation_probs,
    duration_probs, duration_sum for a given heart-rate/systolic-time pair."""
    pi_vector = np.array([0.25, 0.25, 0.25, 0.25])
    seq_len = observation_sequence.shape[0]
    num_states = 4
    max_duration_D = int(np.round((60. / heart_rate) * recording_frequency))

    observation_probs = np.zeros((seq_len, num_states))
    for state_i in range(num_states):
        pihat = models[state_i].predict_proba(observation_sequence)
        pihat = pihat[:, ::-1]
        for time_ in range(seq_len):
            Po_correction = multivariate_normal.pdf(
                observation_sequence[time_, :].reshape(1, -1),
                mean=total_obs_distribution[0],
                cov=total_obs_distribution[1])
            observation_probs[time_, state_i] = (pihat[time_, 1] * Po_correction) / pi_vector[state_i]
    observation_probs = np.where(observation_probs == 0, np.finfo(float).tiny, observation_probs)

    d_distributions, max_S1, min_S1, max_S2, min_S2, \
        max_systole, min_systole, max_diastole, min_diastole = \
        distribution.get_distributions(heart_rate, systolic_time)

    # kept for printing/reference only -- no longer used to hard-zero
    # duration_probs, matching the fix in segmenter/viterbi.py: the Gaussian pdf
    # already expresses "unlikely" continuously, so an extra hard wall
    # made some true durations literally unreachable instead of just
    # improbable.
    bounds = {1: (min_S1, max_S1), 2: (min_systole, max_systole),
              3: (min_S2, max_S2), 4: (min_diastole, max_diastole)}

    duration_probs = np.zeros((num_states, 3 * recording_frequency))
    duration_sum = np.zeros(num_states)
    for state_j in [1, 2, 3, 4]:
        for duration in range(1, max_duration_D + 1):
            # BUG-FOR-BUG PARITY with segmenter/viterbi.py's real viterbi_segment: for
            # state_j == 1 (S1) the real code uses cov=d_distributions[state_j, 1]
            # (index 1 -- Systole's variance) instead of d_distributions[state_j - 1, 1]
            # (index 0 -- S1's own variance). This is almost certainly an
            # unintentional bug in segmenter/viterbi.py, not a design choice -- every
            # other state indexes correctly with state_j - 1. It's reproduced
            # here on purpose so this audit scores paths under the SAME
            # (buggy) duration priors the real decode actually uses. If
            # segmenter/viterbi.py's indexing is ever fixed, this must be updated too,
            # or the two will silently diverge again.
            cov_index = state_j if state_j == 1 else state_j - 1
            duration_probs[state_j - 1, duration] = multivariate_normal.pdf(
                duration, mean=d_distributions[state_j - 1, 0],
                cov=d_distributions[cov_index, 1])
        duration_sum[state_j - 1] = np.sum(duration_probs[state_j - 1, :])

    return observation_probs, duration_probs, duration_sum, max_duration_D, bounds


def _segments_from_labels(labels):
    """Per-frame label array (values 1-4) -> list of (state, start, end)."""
    segments = []
    start = 0
    n = len(labels)
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            segments.append((int(labels[start]), start, i))
            start = i
    return segments


def score_path(labels, observation_probs, duration_probs, duration_sum):
    """Total log-likelihood of a per-frame label sequence under the given
    emission/duration tables, plus a per-segment breakdown. Disallowed
    transitions (state isn't the fixed cycle's successor of the previous
    segment's state) are penalized the same way segmenter/viterbi.py penalizes them:
    log(np.finfo(float).tiny)."""
    segments = _segments_from_labels(labels)
    total = 0.0
    breakdown = []
    prev_state = None
    for state, start, end in segments:
        duration = end - start
        emission = float(np.sum(np.log(observation_probs[start:end, state - 1])))
        dcap = min(duration, duration_probs.shape[1] - 1)
        if dcap < 1 or duration_sum[state - 1] == 0:
            dur_prob = np.finfo(float).tiny
        else:
            dur_prob = duration_probs[state - 1, dcap] / duration_sum[state - 1]
            if dur_prob == 0:
                dur_prob = np.finfo(float).tiny
        duration_term = float(np.log(dur_prob))

        if prev_state is None or state == (prev_state % 4) + 1:
            transition_term = 0.0
        else:
            transition_term = float(np.log(np.finfo(float).tiny))

        seg_score = emission + duration_term + transition_term
        total += seg_score
        breakdown.append({
            "state": state, "start": start, "end": end, "duration": duration,
            "emission": emission, "duration_term": duration_term,
            "transition_term": transition_term, "segment_score": seg_score,
        })
        prev_state = state
    return total, breakdown


def rotate_labels(labels, k):
    """Cyclic relabel, boundaries unchanged: k=2 is the half-flip
    (S1<->S2, Systole<->Diastole); k=1 and k=3 are the two quarter-flips."""
    labels = np.asarray(labels)
    return ((labels - 1 + k) % 4) + 1


def diagnose_clip(clip_name, recording, ground_truth, model, top_k_heart_rates=3, verbose_segments=False,
                  min_heart_rate=DEFAULT_MIN_HEART_RATE, max_heart_rate=DEFAULT_MAX_HEART_RATE):
    """Runs the model's real decode, then scores ground truth and its
    half-flip / two quarter-flip rotations under each of the top-k
    heart-rate candidates the model considered (not just the one it used),
    all under the model's own emission+duration math.

    Read the printed table as:
      - If "model's actual output" is NOT the highest score at heart_rate
        candidate #0 (the one predict() actually used) -> search bug in
        segmenter/viterbi.py, not a data/prior problem.
      - If "ground truth" scores lower than a flipped rotation of itself,
        under the SAME heart-rate candidate the model used -> the duration
        priors (i.e. the heart-rate estimate) are misleading the model,
        even though the search itself is working correctly.
      - If a *quarter*-flip ever beats everything -> the model is
        confusing beat-type (S1/S2) and interval-type (Systole/Diastole)
        segments, not just which beat is which -- a different and bigger
        problem than the half-flips you've seen so far.
      - If a heart-rate candidate other than #0 makes ground truth score
        much better than candidate #0 does -> the top autocorrelation
        peak is the wrong one for this clip.
    """
    features = model.feature_extractor(recording, model.sampling_frequency)

    # min/max_heart_rate MUST match whatever model.predict() actually used for
    # this clip, or the autocorrelation search window differs and the
    # candidate list (and "USED BY predict()" label below) can be wrong.
    # NOTE: SegmentationModel.predict()'s own defaults are 60/200, not these
    # 40/200 -- if you call model.predict(recording) elsewhere without
    # overriding its bounds, that call and this diagnostic will disagree.
    # Pass min_heart_rate/max_heart_rate explicitly to model.predict() too.
    heart_rates, systolic_times = get_heart_rate(recording, model.sampling_frequency,
                                                  min_heart_rate=min_heart_rate,
                                                  max_heart_rate=max_heart_rate,
                                                  multiple_rates=True)

    print(f"\n{'=' * 70}\n{clip_name}\n{'=' * 70}")
    print(f"heart_rate candidates (loudest autocorrelation peak first): "
          f"{[round(h, 1) for h in heart_rates]}")
    print(f"corresponding systolic_time candidates: "
          f"{[round(s, 3) for s in systolic_times]}")

    # same min/max as above, so this is guaranteed to be the same call
    # predict() makes internally rather than two calls that could drift apart
    predicted = model.predict(recording, min_heart_rate=min_heart_rate, max_heart_rate=max_heart_rate)

    for i, (hr, st) in enumerate(zip(heart_rates[:top_k_heart_rates],
                                      systolic_times[:top_k_heart_rates])):
        obs_probs, dur_probs, dur_sum, max_dur, bounds = _compute_probs(
            features, model.models, model.total_obs_distribution,
            model.data_distribution, hr, st, model.feature_frequency)

        print(f"\n--- heart_rate candidate #{i} = {hr:.1f} bpm "
              f"({'USED BY predict()' if i == 0 else 'not used by default'}), "
              f"systolic_time = {st:.3f}s ---")
        for state, (lo, hi) in bounds.items():
            print(f"    state {state} duration bounds (frames @ {model.feature_frequency}Hz): "
                  f"[{lo:.1f}, {hi:.1f}]")

        min_len = min(len(predicted), len(ground_truth), obs_probs.shape[0])
        gt = np.asarray(ground_truth)[:min_len]
        pred = np.asarray(predicted)[:min_len]
        op = obs_probs[:min_len]

        candidates = {
            "model's actual output": pred,
            "ground truth": gt,
            "ground truth, half-flip (k=2)": rotate_labels(gt, 2),
            "ground truth, quarter-flip (k=1)": rotate_labels(gt, 1),
            "ground truth, quarter-flip (k=3)": rotate_labels(gt, 3),
        }

        scores = {}
        for name, labels in candidates.items():
            total, breakdown = score_path(labels, op, dur_probs, dur_sum)
            scores[name] = total
            print(f"    log-likelihood of [{name}]: {total:.2f}")
            if verbose_segments:
                for seg in breakdown:
                    print(f"        state={seg['state']} [{seg['start']}:{seg['end']}] "
                          f"dur={seg['duration']} emission={seg['emission']:.2f} "
                          f"duration_term={seg['duration_term']:.2f} "
                          f"transition_term={seg['transition_term']:.2f} "
                          f"-> {seg['segment_score']:.2f}")

        best = max(scores, key=scores.get)
        print(f"    -> highest-scoring candidate at this heart rate: '{best}'")


# ---------------------------------------------------------------------------
# Layer 2: DP search-internals instrumentation (formerly diagnostics/viterbi_instrumented.py)
# ---------------------------------------------------------------------------

def viterbi_segment_debug(observation_sequence,
                          models,
                          total_obs_distribution,
                          distribution,
                          heart_rate,
                          systolic_time,
                          recording_frequency):
    """An instrumented copy of viterbi_segment (matching the fixed version,
    with the hard duration-bound zeroing removed, AND the seq_len-1 final-
    argmax fix). Every line that computes delta/psi/psi_duration/
    assigned_states is identical to segmenter/viterbi.py -- the only additions are
    bookkeeping (the diagnostics dict) and appends, so this should be
    behaviorally indistinguishable from the real decode. That claim is
    checked, not assumed -- see verify_against_real() below.

    NEW (backtrack instrumentation -- tests THREE independent hypotheses in
    one pass, not just one):

      H1: "onset==2 at loop exit leaves frame 0 unassigned" -- when the
          backtrack's final onset lands exactly on 2, `if onset < 2:
          onset = 1` never fires, the loop's last assignment is
          assigned_states[1:offset] (never touching index 0), and the
          while-loop condition `onset > 2` is then False so the loop exits
          without a final onset==1 iteration.
          Tested by: final_onset_at_loop_exit, frame_0_state.

      H2: the INITIAL pre-loop segment assignment (the one built directly
          from `pos`/argmax, before the while loop even starts) has NO
          guard at all against onset <= 0 -- unlike every loop iteration,
          which does have `if onset < 2: onset = 1`. If that first onset
          computes to <= 0, `assigned_states[onset-1:offset]` uses a
          NEGATIVE slice start, which numpy does not clamp to 0 -- it wraps
          around and silently writes into the far END of the array instead
          of the intended early frames. This would look like corrupted/
          overlapping assignments near the tail, not a clean gap.
          Tested by: initial_onset_raw, initial_onset_lt_1,
          negative_onset_events.

      H3: general gap/overlap coverage -- every backtrack assignment (the
          initial one AND every loop iteration) increments a per-frame
          coverage counter over the assigned_states-sized array. Any frame
          in [0, seq_len) with count==0 is a gap (never assigned); any
          frame with count>1 is an overlap (assigned twice, meaning the
          second write silently clobbers/duplicates over the first). This
          catches boundary bugs beyond just frame 0 or just onset==2,
          without assuming where they'd be.
          Tested by: frames_never_covered, frames_covered_more_than_once,
          and their index lists.

    Nothing below assumes any of H1/H2/H3 is true -- it only records the
    raw facts needed to check each one against the printed backtrack
    trace."""
    pi_vector = np.array([0.25, 0.25, 0.25, 0.25])
    seq_len = observation_sequence.shape[0]
    num_states = 4
    max_duration_D = int(np.round((60. / heart_rate) * recording_frequency))
    delta = np.ones((seq_len + max_duration_D - 1, num_states)) * -np.inf
    psi = np.zeros((seq_len + max_duration_D - 1, num_states)).astype(int)
    psi_duration = np.zeros((seq_len + max_duration_D - 1, num_states))
    observation_probs = np.zeros((seq_len, num_states))

    diagnostics = {
        "seq_len": seq_len,
        "max_duration_D": max_duration_D,
        "accepted_updates": [],   # every accepted update, all regions
        "tail_records": [],       # accepted updates in the padded tail only
        "duration_sum": None,
        "delta_tail_grid": None,
        "backtrack_trace": [],
    }

    for state_i in range(num_states):
        pihat = models[state_i].predict_proba(observation_sequence)
        pihat = pihat[:, ::-1]
        for time_ in range(seq_len):
            Po_correction = multivariate_normal.pdf(observation_sequence[time_, :].reshape(1, -1),
                                                    mean=total_obs_distribution[0],
                                                    cov=total_obs_distribution[1])
            observation_probs[time_, state_i] = (pihat[time_, 1] * Po_correction) / pi_vector[state_i]

    observation_probs = np.where(observation_probs == 0, np.finfo(float).tiny, observation_probs)
    d_distributions, max_S1, min_S1, \
    max_S2, min_S2, max_systole, min_systole, \
    max_diastole, min_diastole = distribution.get_distributions(heart_rate, systolic_time)

    duration_probs = np.zeros((num_states, 3 * recording_frequency))
    duration_sum = np.zeros(num_states)

    for state_j in [1, 2, 3, 4]:
        for duration in range(1, max_duration_D + 1):
            # BUG-FOR-BUG PARITY with segmenter/viterbi.py -- see the matching comment
            # in _compute_probs() above. Real viterbi_segment uses Systole's
            # variance (index 1) for S1's (state_j==1) duration prior instead
            # of S1's own variance (index 0). Reproduced here so this
            # instrumented copy's output matches the real decode exactly.
            cov_index = state_j if state_j == 1 else state_j - 1
            duration_probs[state_j - 1, duration] = multivariate_normal.pdf(
                duration, mean=d_distributions[state_j - 1, 0], cov=d_distributions[cov_index, 1])
        duration_sum[state_j - 1] = np.sum(duration_probs[state_j - 1, :])

    diagnostics["duration_sum"] = duration_sum.copy()

    assigned_states = np.zeros(delta.shape[0])
    delta[0, :] = np.log(pi_vector) + np.log(observation_probs[0, :])
    psi[0, :] = -1

    a_matrix = np.array([[0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
    a_matrix_nonzero = np.where(a_matrix == 0, np.finfo(float).tiny, a_matrix)

    for time_ in range(2, seq_len + max_duration_D):
        end_time = time_
        if time_ > seq_len:
            end_time = seq_len
        old_start_time = end_time

        probs = observation_probs[end_time - 1, :]
        for duration in range(1, max_duration_D + 1):
            start_time = time_ - duration
            region = "none"
            if start_time < 1:
                start_time = 1
                region = "start_clamp"
            if start_time > seq_len - 1:
                start_time = seq_len - 1
                region = "tail_clamp"

            deltas = delta[start_time - 1, :] + np.log(a_matrix_nonzero.T)
            max_index = np.argmax(deltas, axis=1)
            max_delta = deltas[np.arange(4), max_index]

            if start_time < old_start_time:
                assert start_time == old_start_time - 1
                probs = probs * observation_probs[start_time - 1]
                old_start_time = start_time

            probs = np.where(probs == 0, np.finfo(float).tiny, probs)
            emission_probs = np.log(probs)
            emission_probs = np.where(emission_probs == 0, np.finfo(float).tiny, emission_probs)

            duration_probs_normalised = duration_probs[:, duration] / duration_sum
            duration_probs_normalised = np.where(duration_probs_normalised == 0,
                                                 np.finfo(float).tiny,
                                                 duration_probs_normalised)
            delta_temp = max_delta + emission_probs + np.log(duration_probs_normalised)

            actual_span = end_time - start_time

            for state_j in [1, 2, 3, 4]:
                if delta_temp[state_j - 1] > delta[time_ - 1, state_j - 1]:
                    delta[time_ - 1, state_j - 1] = delta_temp[state_j - 1]
                    psi[time_ - 1, state_j - 1] = max_index[state_j - 1] + 1
                    psi_duration[time_ - 1, state_j - 1] = duration

                    record = {
                        "time_": time_, "state": state_j, "region": region,
                        "end_capped": time_ > seq_len,  # end_time capped to seq_len independent
                                                         # of whether start_time also got clamped --
                                                         # this is what actually causes the "none"
                                                         # mismatches: capping end_time alone is enough
                                                         # to make actual_span < nominal_duration even
                                                         # when start_time hasn't been clamped yet.
                        "nominal_duration": duration, "start_time": start_time,
                        "end_time": end_time, "actual_span": actual_span,
                        "mismatch": actual_span != duration,
                        "delta_value": float(delta_temp[state_j - 1]),
                    }
                    diagnostics["accepted_updates"].append(record)
                    if time_ > seq_len:
                        diagnostics["tail_records"].append(record)

    diagnostics["delta_tail_grid"] = delta[seq_len:, :].copy()

    # --- Final-argmax fix (matches the corrected segmenter/viterbi.py): the true
    # end-of-sequence row lives at delta[seq_len - 1, :] (time_ == seq_len),
    # not delta[seq_len, :] (time_ == seq_len + 1). The old slice
    # delta[seq_len:, :] / pos + seq_len + 1 skipped that row entirely.
    temp_delta = delta[seq_len - 1:, :]
    idx = np.argmax(temp_delta)
    pos, _ = np.unravel_index(idx, temp_delta.shape)
    pos = pos + seq_len

    state = int(np.argmax(delta[pos - 1, :])) + 1

    # Coverage counter over the SAME full-length array as assigned_states,
    # incremented on every write this function makes to assigned_states,
    # using the SAME slice bounds (including if those bounds are negative --
    # we do NOT pre-clamp here, precisely because we want to observe what
    # numpy's actual (non-clamping) negative-slice behavior does, matching
    # whatever the real, unmodified assignment line below does).
    coverage = np.zeros_like(assigned_states)
    negative_onset_events = []

    offset = pos
    preceding_state = psi[offset - 1, state - 1]
    dur_used = psi_duration[offset - 1, state - 1]
    onset = int(offset - dur_used + 1)  # <-- H2: NO GUARD on this first onset, unlike every loop iteration below

    initial_onset_raw = onset
    initial_onset_lt_1 = onset < 1
    if initial_onset_lt_1:
        negative_onset_events.append({
            "step": "initial_pre_loop_segment", "onset_raw": onset,
            "slice_start_used": onset - 1, "offset": offset, "state": state,
            "note": "onset <= 0 -> assigned_states[onset-1:offset] uses a NEGATIVE "
                    "slice start, which numpy does not clamp to 0 -- it wraps and "
                    "writes near/at the END of the array instead of the intended "
                    "early frames.",
        })

    assigned_states[onset - 1:offset] = state
    coverage[onset - 1:offset] += 1
    diagnostics["backtrack_trace"].append({
        "offset": offset, "state": state, "preceding_state": int(preceding_state),
        "psi_duration_used": dur_used, "onset": onset,
        "actual_frames_assigned": offset - (onset - 1),
        "matches": dur_used == (offset - (onset - 1)),
        "is_initial_segment": True,
        "onset_raw": initial_onset_raw,
        "onset_was_clamped": False,  # this step has no clamp in the real code at all
    })

    state = int(preceding_state)
    count = 0

    while onset > 2:
        offset = int(onset - 1)
        preceding_state = psi[offset - 1, state - 1]
        dur_used = psi_duration[offset - 1, state - 1]
        onset_raw = int(offset - dur_used + 1)
        onset = onset_raw

        if onset < 2:
            onset = 1

        if onset_raw < 1:
            # loop DOES guard this (onset gets forced to 1), so this should
            # never actually produce a negative slice start here -- recorded
            # anyway so we can confirm the guard is doing its job, not assume it.
            negative_onset_events.append({
                "step": f"loop_iteration_{count}", "onset_raw": onset_raw,
                "onset_after_clamp": onset, "slice_start_used": onset - 1,
                "offset": offset, "state": state,
                "note": "onset_raw <= 0 but the loop's own guard forced onset=1 "
                        "before slicing, so slice_start_used should be 0, not negative.",
            })

        assigned_states[onset - 1:offset] = state
        coverage[onset - 1:offset] += 1
        diagnostics["backtrack_trace"].append({
            "offset": offset, "state": state, "preceding_state": int(preceding_state),
            "psi_duration_used": dur_used, "onset": onset,
            "actual_frames_assigned": offset - (onset - 1),
            "matches": dur_used == (offset - (onset - 1)),
            "is_initial_segment": False,
            "onset_raw": onset_raw,
            "onset_was_clamped": onset_raw < 2,
        })

        state = int(preceding_state)
        count = count + 1
        if count > 1000:
            diagnostics["backtrack_hit_iteration_limit"] = True
            break

    # --- H1: how did the loop actually end, and is frame 0 specifically
    # left unassigned? ---
    diagnostics["final_onset_at_loop_exit"] = onset
    diagnostics["loop_exit_reason"] = (
        "hit_iteration_limit" if diagnostics.get("backtrack_hit_iteration_limit")
        else "onset_not_gt_2"
    )
    diagnostics["frame_0_state"] = int(assigned_states[0]) if seq_len > 0 else None

    # --- H2: any negative-onset (unguarded) events on the initial segment? ---
    diagnostics["negative_onset_events"] = negative_onset_events
    diagnostics["initial_onset_raw"] = initial_onset_raw
    diagnostics["initial_onset_lt_1"] = initial_onset_lt_1

    # --- H3: general gap/overlap coverage over the WHOLE array, not just
    # frame 0, and not just assuming gaps -- also checks for overlaps
    # (a frame written more than once), which is what H2's wraparound would
    # actually look like if it happened (extra writes near the tail, not a
    # gap at the front). ---
    gap_mask_in_seq = (coverage[:seq_len] == 0)
    overlap_mask_in_seq = (coverage[:seq_len] > 1)
    diagnostics["unassigned_frame_count"] = int(np.sum(gap_mask_in_seq))  # kept for backward compat
    diagnostics["unassigned_frame_indices"] = np.nonzero(gap_mask_in_seq)[0][:10].tolist()
    diagnostics["frames_never_covered"] = int(np.sum(gap_mask_in_seq))
    diagnostics["frames_never_covered_indices"] = np.nonzero(gap_mask_in_seq)[0][:10].tolist()
    diagnostics["frames_covered_more_than_once"] = int(np.sum(overlap_mask_in_seq))
    diagnostics["frames_covered_more_than_once_indices"] = np.nonzero(overlap_mask_in_seq)[0][:10].tolist()
    # also check OUTSIDE [0, seq_len) -- i.e. in the padded tail portion of
    # the full-length array -- for anomalous extra writes, which is exactly
    # where H2's wraparound would land if it happened.
    tail_overlap_mask = (coverage[seq_len:] > 1)
    diagnostics["tail_region_frames_covered_more_than_once"] = int(np.sum(tail_overlap_mask))

    assigned_states = assigned_states[:seq_len]

    return delta, psi, assigned_states, diagnostics


def verify_against_real(observation_sequence, models, total_obs_distribution,
                        distribution, heart_rate, systolic_time, recording_frequency):
    """Runs both the real viterbi_segment (imported from segmenter/viterbi.py) and this
    instrumented copy on identical inputs, and confirms they produce
    identical assigned_states. If this returns False, don't trust the
    diagnostics from viterbi_segment_debug -- the instrumented copy has
    drifted from the real decode."""
    from segmenter.viterbi import viterbi_segment
    _, _, real_assigned = viterbi_segment(observation_sequence, models, total_obs_distribution,
                                          distribution, heart_rate, systolic_time, recording_frequency)
    _, _, debug_assigned, diagnostics = viterbi_segment_debug(
        observation_sequence, models, total_obs_distribution,
        distribution, heart_rate, systolic_time, recording_frequency)
    ok = bool(np.array_equal(real_assigned, debug_assigned))
    return ok, real_assigned, debug_assigned, diagnostics


def print_search_internals(clip_name, diagnostics):
    print(f"\n{'#' * 70}\n{clip_name}: SEARCH INTERNALS\n{'#' * 70}")
    print(f"seq_len={diagnostics['seq_len']}  max_duration_D={diagnostics['max_duration_D']}")
    print(f"duration_sum per state [S1, Systole, S2, Diastole]: {diagnostics['duration_sum']}")
    if np.any(diagnostics["duration_sum"] == 0):
        print("  *** WARNING: at least one state has duration_sum == 0 -- "
              "normalization is degenerate for that state ***")

    all_updates = diagnostics["accepted_updates"]
    # Cross-tab by (start_time-clamp region, whether end_time was independently
    # capped). region alone undercounts tail distortion: when time_ > seq_len,
    # end_time gets capped to seq_len even on iterations where start_time
    # hasn't been clamped yet (region=="none"), and that alone is enough to
    # make actual_span < nominal_duration. So "none" is NOT a safe "nothing
    # unusual happening here" bucket -- check end_capped separately.
    buckets = {}
    for u in all_updates:
        key = (u["region"], u["end_capped"])
        if key not in buckets:
            buckets[key] = {"count": 0, "mismatches": 0}
        buckets[key]["count"] += 1
        if u["mismatch"]:
            buckets[key]["mismatches"] += 1

    print("\nAccepted updates by (start_time region, end_time capped) -- "
          "'none, end_capped=True' is the mismatch source region-alone hides:")
    for (region, end_capped), stats in sorted(buckets.items()):
        print(f"    region={region:<12} end_capped={str(end_capped):<5} "
              f"count={stats['count']:>6}  mismatches={stats['mismatches']:>6}")

    tail = diagnostics["tail_records"]
    print(f"\n--- Full detail: {len(tail)} accepted updates in the padded tail region "
          f"(time_ > seq_len) ---")
    for r in tail:
        flag = " <-- MISMATCH" if r["mismatch"] else ""
        print(f"    time_={r['time_']:>4} state={r['state']} "
              f"nominal_duration={r['nominal_duration']:>3} "
              f"actual_span=[{r['start_time']}:{r['end_time']}]={r['actual_span']:>3} "
              f"delta={r['delta_value']:.2f}{flag}")

    print(f"\n--- delta[seq_len:, :] (the grid the final argmax is taken from) ---")
    grid = diagnostics["delta_tail_grid"]
    print("    row(offset from seq_len)   S1        Systole    S2        Diastole")
    for i, row in enumerate(grid):
        print(f"    {i:>4}   " + "  ".join(f"{v:>10.2f}" for v in row))

    print(f"\n--- Backtrack trace ({len(diagnostics['backtrack_trace'])} segments) ---")
    for step in diagnostics["backtrack_trace"]:
        flag = "" if step["matches"] else "  <-- psi_duration != actual frames assigned"
        print(f"    offset={step['offset']:>4} state={step['state']} "
              f"preceding_state={step['preceding_state']} onset={step['onset']:>4} "
              f"psi_duration_used={step['psi_duration_used']:>5} "
              f"actual_frames_assigned={step['actual_frames_assigned']:>4}{flag}")
    if diagnostics.get("backtrack_hit_iteration_limit"):
        print("    *** WARNING: backtrack hit the 1000-iteration safety limit ***")

    # --- NEW: backtrack termination / coverage report, testing 3 hypotheses ---
    print(f"\n--- Backtrack termination & coverage check (H1/H2/H3) ---")
    print(f"    [H1: onset==2-at-exit -> frame 0 gap]")
    print(f"        final_onset_at_loop_exit = {diagnostics.get('final_onset_at_loop_exit')}")
    print(f"        loop_exit_reason         = {diagnostics.get('loop_exit_reason')}")
    print(f"        frame_0_state            = {diagnostics.get('frame_0_state')}  "
          f"(0 is NOT a valid state -- valid states are 1-4)")
    print(f"    [H2: unguarded initial-segment onset -> negative-slice wraparound]")
    print(f"        initial_onset_raw        = {diagnostics.get('initial_onset_raw')}")
    print(f"        initial_onset_lt_1       = {diagnostics.get('initial_onset_lt_1')}")
    neg_events = diagnostics.get("negative_onset_events") or []
    print(f"        negative_onset_events    = {len(neg_events)} event(s)")
    for ev in neg_events:
        print(f"            {ev}")
    print(f"    [H3: general gap/overlap coverage across whole sequence]")
    print(f"        frames_never_covered           = {diagnostics.get('frames_never_covered')} "
          f"indices(first 10)={diagnostics.get('frames_never_covered_indices')}")
    print(f"        frames_covered_more_than_once   = {diagnostics.get('frames_covered_more_than_once')} "
          f"indices(first 10)={diagnostics.get('frames_covered_more_than_once_indices')}")
    print(f"        tail_region_frames_covered_more_than_once = "
          f"{diagnostics.get('tail_region_frames_covered_more_than_once')}")


# ---------------------------------------------------------------------------
# Combined entry point -- run this one for "print everything"
# ---------------------------------------------------------------------------

def diagnose_clip_deep(clip_name, recording, ground_truth, model,
                       top_k_heart_rates=3, verbose_segments=False,
                       min_heart_rate=DEFAULT_MIN_HEART_RATE, max_heart_rate=DEFAULT_MAX_HEART_RATE):
    """Runs BOTH diagnostics for one clip in a single call:

      1. diagnose_clip()'s rotation-scoring audit across the top heart-rate
         candidates (unchanged behavior).
      2. The DP search-internals dump (viterbi_segment_debug +
         print_search_internals), for the heart_rate/systolic_time pair
         predict() actually used (heart_rates[0]).

    Step 2 is preceded by verify_against_real(); if the instrumented copy's
    output doesn't match the real viterbi_segment exactly, this prints a
    loud warning and SKIPS the internals dump for this clip rather than
    printing numbers that might not reflect the real decode.

    This prints EVERYTHING for one clip -- full tail-region record list,
    full delta grid, full backtrack trace. Fine for one clip; across many
    clips it's unreadable and slow to send back and forth. For scanning a
    batch of clips, use summarize_clip() / print_clip_summary_table()
    instead, which run these same two layers but keep only the numbers
    that answer "did the search behave, and is the tail-clamp distortion
    anywhere near where the flip actually happens."
    """
    diagnose_clip(clip_name, recording, ground_truth, model,
                  top_k_heart_rates=top_k_heart_rates, verbose_segments=verbose_segments,
                  min_heart_rate=min_heart_rate, max_heart_rate=max_heart_rate)

    features = model.feature_extractor(recording, model.sampling_frequency)
    # same min/max as diagnose_clip above -- must match model.predict()'s
    # actual search window or hr/st here won't be the pair predict() used
    heart_rates, systolic_times = get_heart_rate(recording, model.sampling_frequency,
                                                  min_heart_rate=min_heart_rate,
                                                  max_heart_rate=max_heart_rate,
                                                  multiple_rates=True)
    hr, st = heart_rates[0], systolic_times[0]

    ok, real_assigned, debug_assigned, diagnostics = verify_against_real(
        features, model.models, model.total_obs_distribution,
        model.data_distribution, hr, st, model.feature_frequency)

    print(f"\n--- {clip_name}: instrumented-vs-real verification ---")
    if not ok:
        print("    *** MISMATCH: the instrumented copy's output differs from the "
              "real viterbi_segment on identical inputs. Search-internals dump "
              "SKIPPED for this clip -- it would not reflect the real decode. "
              "This mismatch is itself worth reporting: either viterbi_segment_debug "
              "has drifted from segmenter/viterbi.py, or there's nondeterminism in the real "
              "function (e.g. an array reused/mutated across calls). ***")
        min_len = min(len(real_assigned), len(debug_assigned), 30)
        print(f"    real_assigned[:{min_len}]  = {real_assigned[:min_len]}")
        print(f"    debug_assigned[:{min_len}] = {debug_assigned[:min_len]}")
        diff_idx = np.nonzero(real_assigned[:min(len(real_assigned), len(debug_assigned))]
                              != debug_assigned[:min(len(real_assigned), len(debug_assigned))])[0]
        print(f"    first mismatching frame indices: {diff_idx[:20]}"
              f"{' ...' if len(diff_idx) > 20 else ''} (total mismatching frames: {len(diff_idx)})")
        return

    print("    OK -- instrumented copy matches real decode exactly.")
    print_search_internals(clip_name, diagnostics)


# ---------------------------------------------------------------------------
# Condensed batch view -- one row per clip instead of the full dump
# ---------------------------------------------------------------------------

def summarize_clip(clip_name, recording, ground_truth, model,
                   min_heart_rate=DEFAULT_MIN_HEART_RATE, max_heart_rate=DEFAULT_MAX_HEART_RATE):
    """Runs the same two layers as diagnose_clip_deep() (rotation-scoring
    audit at the heart-rate candidate predict() actually used, plus the
    verified search-internals instrumentation) but returns a small dict of
    just the numbers needed to tell, at a glance, whether a clip's rotation
    flip lines up with tail-clamp distortion -- instead of printing the
    full per-frame dump. Prints nothing itself; pass a list of these to
    print_clip_summary_table() to display many clips compactly.

    Fields:
      accuracy               -- fraction of frames where model's actual
                                 output matches ground truth
      heart_rate_used        -- heart_rates[0], what predict() used
      best_scoring_candidate -- which of {actual_output, ground_truth,
                                 half_flip, quarter_flip_1, quarter_flip_3}
                                 scores highest under the model's own math.
                                 If this isn't "actual_output", the search
                                 didn't pick its own best-scoring path --
                                 i.e. a segmenter/viterbi.py bug, not just a
                                 prior/data problem.
      verify_ok               -- whether the instrumented copy matched the
                                 real decode (see diagnose_clip_deep). If
                                 False, none of the fields below (including
                                 the score_path_* ones) are present -- they'd
                                 be untrustworthy.
      real_dp_final_loglik / score_path_actual_output_loglik / score_path_matches_real_dp
                              -- cross-check of whether score_path() (used
                                 for the whole rotation-scoring audit)
                                 actually measures the same objective the
                                 real DP optimizes. real_dp_final_loglik is
                                 the real search's OWN cumulative score for
                                 the path it actually returned (read
                                 straight off delta_tail_grid, before
                                 backtracking); score_path_actual_output_loglik
                                 is score_path()'s score for that exact same
                                 label sequence. If score_path_matches_real_dp
                                 is False, best_scoring_candidate/scores
                                 above are NOT reliable evidence about
                                 whether the real search missed its own
                                 optimum -- the two scorers disagree on what
                                 "optimum" means (most likely boundary/
                                 first-segment treatment), so a rotation or
                                 ground truth "winning" under score_path
                                 doesn't mean the real DP could have reached
                                 it.
      total_accepted_updates / total_mismatches
                              -- across ALL regions, how many accepted DP
                                 updates had actual_span != nominal_duration
      start_clamp_count / start_clamp_mismatches
      tail_clamp_count / tail_clamp_mismatches
      none_end_capped_count / none_end_capped_mismatches
                              -- same breakdown, split by region (the
                                 region-alone table undercounts tail
                                 distortion -- see print_search_internals's
                                 docstring note -- so 'tail_clamp' and
                                 'none_end_capped' are reported separately
                                 and should usually be read together as
                                 "tail-region distortion")
      backtrack_segments / backtrack_mismatches
                              -- how many segments the final backtrack
                                 produced, and how many of those had
                                 psi_duration_used != actual_frames_assigned
      first_backtrack_mismatch
                              -- only present if backtrack_mismatches > 0.
                                 The first such segment IN BACKTRACK ORDER
                                 (i.e. nearest the end of the clip, since
                                 backtrack runs backward from the tail) --
                                 this is the segment where the search's own
                                 bookkeeping stopped agreeing with what got
                                 assigned, closest to the final decode
                                 decision. `in_padded_tail` tells you
                                 whether that segment's offset falls after
                                 seq_len, i.e. whether it's in the region
                                 the tail-clamp hypothesis is about.
      final_onset_at_loop_exit / unassigned_frame_count / unassigned_frame_indices / frame_0_state
                              -- NEW. Raw facts about how the backtrack
                                 while-loop terminated and whether any
                                 frames in [0, seq_len) were left at their
                                 np.zeros() default (state 0, not a real
                                 state). See viterbi_segment_debug's
                                 docstring for the hypothesis this is meant
                                 to test (onset==2 at loop exit -> frame 0
                                 never assigned). frame_0_state == 0 means
                                 frame 0 specifically was never assigned.
    """
    features = model.feature_extractor(recording, model.sampling_frequency)
    heart_rates, systolic_times = get_heart_rate(recording, model.sampling_frequency,
                                                  min_heart_rate=min_heart_rate,
                                                  max_heart_rate=max_heart_rate,
                                                  multiple_rates=True)
    hr0, st0 = heart_rates[0], systolic_times[0]

    predicted = model.predict(recording, min_heart_rate=min_heart_rate, max_heart_rate=max_heart_rate)

    obs_probs, dur_probs, dur_sum, max_dur, bounds = _compute_probs(
        features, model.models, model.total_obs_distribution,
        model.data_distribution, hr0, st0, model.feature_frequency)

    min_len = min(len(predicted), len(ground_truth), obs_probs.shape[0])
    gt = np.asarray(ground_truth)[:min_len]
    pred = np.asarray(predicted)[:min_len]
    op = obs_probs[:min_len]

    candidates = {
        "actual_output": pred,
        "ground_truth": gt,
        "half_flip": rotate_labels(gt, 2),
        "quarter_flip_1": rotate_labels(gt, 1),
        "quarter_flip_3": rotate_labels(gt, 3),
    }
    scores = {name: score_path(labels, op, dur_probs, dur_sum)[0] for name, labels in candidates.items()}
    best_candidate = max(scores, key=scores.get)

    summary = {
        "clip_name": clip_name,
        "accuracy": float(np.mean(pred == gt)),
        "heart_rate_used": float(hr0),
        "best_scoring_candidate": best_candidate,
        "scores": scores,
    }

    ok, real_assigned, debug_assigned, diagnostics = verify_against_real(
        features, model.models, model.total_obs_distribution,
        model.data_distribution, hr0, st0, model.feature_frequency)
    summary["verify_ok"] = ok
    if not ok:
        min_len2 = min(len(real_assigned), len(debug_assigned))
        diff_idx = np.nonzero(real_assigned[:min_len2] != debug_assigned[:min_len2])[0]
        summary["verify_mismatch_frame_count"] = int(len(diff_idx))
        return summary

    # Does score_path() actually measure the same objective the real DP
    # optimizes? diagnostics["delta_tail_grid"].max() IS the real DP's own
    # cumulative log-likelihood for the path it backtracked out (that's
    # exactly the value viterbi_segment's final argmax selects, before
    # backtracking even runs) -- so re-scoring that SAME label sequence
    # (pred/"actual_output") with score_path() and comparing the two
    # numbers directly tests whether score_path's accounting matches the
    # real DP's, independent of any question about ground truth or
    # rotations. If they don't match, "ground_truth scores higher than
    # actual_output under score_path" doesn't prove the real search missed
    # its own optimum -- it proves score_path isn't measuring what the
    # real DP measures (most likely boundary/first-segment treatment: the
    # real DP's delta[0,:] = log(pi_vector) + log(obs[0,:]) has no duration
    # term at all for the first frame, while score_path charges every
    # segment, including the first, full emission+duration cost uniformly.
    # ALSO: if frame 0 is unassigned (state 0), score_path() will see an
    # illegal transition from "state 0" into whatever the real first
    # assigned state is, and slap on a huge log(tiny) transition penalty --
    # see unassigned_frame_count / frame_0_state below, which is a second,
    # independent reason score_path_actual_output_loglik could diverge from
    # real_dp_final_loglik that has nothing to do with boundary accounting.)
    real_dp_final_loglik = float(np.max(diagnostics["delta_tail_grid"]))
    score_path_actual_output_loglik = scores["actual_output"]
    summary["real_dp_final_loglik"] = real_dp_final_loglik
    summary["score_path_actual_output_loglik"] = score_path_actual_output_loglik
    summary["score_path_matches_real_dp"] = bool(np.isclose(
        real_dp_final_loglik, score_path_actual_output_loglik, rtol=1e-6, atol=1e-3))

    updates = diagnostics["accepted_updates"]
    start_clamp = [u for u in updates if u["region"] == "start_clamp"]
    tail_clamp = [u for u in updates if u["region"] == "tail_clamp"]
    none_end_capped = [u for u in updates if u["region"] == "none" and u["end_capped"]]

    summary.update({
        "total_accepted_updates": len(updates),
        "total_mismatches": sum(1 for u in updates if u["mismatch"]),
        "start_clamp_count": len(start_clamp),
        "start_clamp_mismatches": sum(1 for u in start_clamp if u["mismatch"]),
        "tail_clamp_count": len(tail_clamp),
        "tail_clamp_mismatches": sum(1 for u in tail_clamp if u["mismatch"]),
        "none_end_capped_count": len(none_end_capped),
        "none_end_capped_mismatches": sum(1 for u in none_end_capped if u["mismatch"]),
    })

    backtrack = diagnostics["backtrack_trace"]
    bt_mismatches = [s for s in backtrack if not s["matches"]]
    summary["backtrack_segments"] = len(backtrack)
    summary["backtrack_mismatches"] = len(bt_mismatches)
    if bt_mismatches:
        first = bt_mismatches[0]
        summary["first_backtrack_mismatch"] = {
            "offset": first["offset"],
            "state": first["state"],
            "onset": first["onset"],
            "psi_duration_used": first["psi_duration_used"],
            "actual_frames_assigned": first["actual_frames_assigned"],
            "in_padded_tail": bool(first["offset"] > diagnostics["seq_len"]),
        }

    # NEW: pass through all three hypotheses' raw facts
    summary["final_onset_at_loop_exit"] = diagnostics.get("final_onset_at_loop_exit")
    summary["loop_exit_reason"] = diagnostics.get("loop_exit_reason")
    summary["frame_0_state"] = diagnostics.get("frame_0_state")  # H1

    summary["initial_onset_raw"] = diagnostics.get("initial_onset_raw")
    summary["initial_onset_lt_1"] = diagnostics.get("initial_onset_lt_1")
    summary["negative_onset_events"] = diagnostics.get("negative_onset_events")  # H2

    summary["unassigned_frame_count"] = diagnostics.get("unassigned_frame_count")
    summary["unassigned_frame_indices"] = diagnostics.get("unassigned_frame_indices")
    summary["frames_never_covered"] = diagnostics.get("frames_never_covered")
    summary["frames_never_covered_indices"] = diagnostics.get("frames_never_covered_indices")
    summary["frames_covered_more_than_once"] = diagnostics.get("frames_covered_more_than_once")
    summary["frames_covered_more_than_once_indices"] = diagnostics.get("frames_covered_more_than_once_indices")
    summary["tail_region_frames_covered_more_than_once"] = diagnostics.get(
        "tail_region_frames_covered_more_than_once")  # H3

    return summary


_SUMMARY_HEADER = (
    f"{'clip':<16} {'acc':>5} {'hr':>6} {'best-scoring':<15} {'scorers_agree':>13} {'verify':<6} "
    f"{'upd':>6} {'mis':>5} {'tail_upd':>8} {'tail_mis':>8} "
    f"{'bt_seg':>6} {'bt_mis':>6} {'mis_in_tail':>11} "
    f"{'final_onset':>11} {'frame0_st':>9} "
    f"{'gaps':>5} {'overlaps':>8} {'tail_ovl':>8} {'neg_onset':>9}"
)


def format_clip_summary_row(s):
    """One fixed-width line for a single summarize_clip() result. Kept
    separate from print_clip_summary_table() so callers writing straight to
    a file (e.g. diagnostics/summarize_flipped_clips.py) can build up rows without
    re-running the diagnostics."""
    if not s.get("verify_ok", False):
        note = s.get("verify_mismatch_frame_count", "?")
        return (f"{s['clip_name']:<16} {s['accuracy']:>5.2f} {s['heart_rate_used']:>6.1f} "
                f"{s['best_scoring_candidate']:<15} {'n/a':>13} {'FAIL':<6}  "
                f"-- verify_against_real mismatch on {note} frames, internals skipped --")

    fb = s.get("first_backtrack_mismatch")
    mismatch_in_tail = str(fb["in_padded_tail"]) if fb else "n/a"
    tail_upd = s["tail_clamp_count"] + s["none_end_capped_count"]
    tail_mis = s["tail_clamp_mismatches"] + s["none_end_capped_mismatches"]
    scorers_agree = str(s["score_path_matches_real_dp"])
    neg_onset_count = len(s.get("negative_onset_events") or [])
    return (f"{s['clip_name']:<16} {s['accuracy']:>5.2f} {s['heart_rate_used']:>6.1f} "
            f"{s['best_scoring_candidate']:<15} {scorers_agree:>13} {'OK':<6} "
            f"{s['total_accepted_updates']:>6} {s['total_mismatches']:>5} "
            f"{tail_upd:>8} {tail_mis:>8} "
            f"{s['backtrack_segments']:>6} {s['backtrack_mismatches']:>6} {mismatch_in_tail:>11} "
            f"{str(s.get('final_onset_at_loop_exit')):>11} "
            f"{str(s.get('frame_0_state')):>9} "
            f"{str(s.get('frames_never_covered')):>5} "
            f"{str(s.get('frames_covered_more_than_once')):>8} "
            f"{str(s.get('tail_region_frames_covered_more_than_once')):>8} "
            f"{neg_onset_count:>9}")


def print_clip_summary_table(summaries):
    """Prints one compact row per clip. Columns:
      acc          -- fraction of frames matching ground truth
      hr           -- heart_rate predict() used
      best-scoring -- which candidate (actual_output / ground_truth /
                      half_flip / quarter_flip_1 / quarter_flip_3) the
                      model's own math scores highest
      scorers_agree -- True/False: does score_path()'s score for the
                      model's actual output match the real DP's own final
                      score for that same path? False means best-scoring
                      above is not trustworthy evidence about whether the
                      real search missed its own optimum -- see
                      summarize_clip()'s docstring.
      verify       -- OK/FAIL for verify_against_real (FAIL means the rest
                      of that row's numbers would be untrustworthy and are
                      omitted)
      upd/mis      -- total accepted DP updates / how many had
                      actual_span != nominal_duration, across ALL regions
      tail_upd/tail_mis
                   -- same, restricted to tail_clamp + 'none, end_capped'
                      regions (the combined tail-distortion count)
      bt_seg/bt_mis
                   -- segments in the final backtrack trace / how many had
                      psi_duration_used != actual_frames_assigned
      mis_in_tail  -- whether the backtrack-order-first such mismatch sits
                      after seq_len (True/False), i.e. whether the
                      clearest sign of tail distortion actually reached
                      the final decoded path, not just the search's
                      internal bookkeeping
      final_onset  -- NEW. value of `onset` when the backtrack while-loop
                      exits. Tests the hypothesis that onset==2 at exit
                      correlates with frame 0 being left unassigned.
      unassigned   -- NEW. count of frames in [0, seq_len) never touched
                      by any backtrack assignment (still at the
                      np.zeros() default of state 0). Should be 0.
      frame0_state -- NEW. literal assigned_states[0]. 0 means frame 0
                      specifically was never assigned a real state.
    """
    print(_SUMMARY_HEADER)
    print("-" * len(_SUMMARY_HEADER))
    for s in summaries:
        print(format_clip_summary_row(s))


if __name__ == "__main__":
    print(__doc__)