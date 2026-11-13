"""
diagnostics/viterbi_instrumented.py

An instrumented copy of viterbi_segment (matching the fixed version, with
the hard duration-bound zeroing removed) that records detailed internal
state instead of just returning the final answer. Nothing about the actual
arithmetic/control-flow is changed from the real decode -- every line that
computes delta/psi/psi_duration/assigned_states is identical. The only
additions are bookkeeping (diagnostics dict) and appends, so this should be
behaviorally indistinguishable from segmenter/viterbi.py's viterbi_segment. That
claim is checked, not assumed -- see verify_against_real() below.

Records, per call:
  - every accepted delta-update (a segment hypothesis that won), tagged by
    which clamp region it happened in ("none", "start_clamp", "tail_clamp"),
    with the nominal duration used to look up the duration prior vs. the
    actual (start_time, end_time) span that produced the emission evidence
    for that update, and whether they mismatch.
  - full detail for every accepted update in the tail region (time_ > seq_len)
    specifically, since that's where the terminal decode point is chosen
    from and it's small enough to print in full.
  - duration_sum per state (catches degenerate/near-zero normalization).
  - the full delta[seq_len:, :] grid (the padded tail, where the final
    argmax is taken from).
  - a step-by-step backtrack trace: offset, state, preceding_state, onset,
    the psi_duration value used, and whether it matches offset - onset + 1.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths


import numpy as np
from scipy.stats import multivariate_normal


def viterbi_segment_debug(observation_sequence,
                          models,
                          total_obs_distribution,
                          distribution,
                          heart_rate,
                          systolic_time,
                          recording_frequency):
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
            duration_probs[state_j - 1, duration] = multivariate_normal.pdf(
                duration, mean=d_distributions[state_j - 1, 0], cov=d_distributions[state_j - 1, 1])
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
                        "nominal_duration": duration, "start_time": start_time,
                        "end_time": end_time, "actual_span": actual_span,
                        "mismatch": actual_span != duration,
                        "delta_value": float(delta_temp[state_j - 1]),
                    }
                    diagnostics["accepted_updates"].append(record)
                    if time_ > seq_len:
                        diagnostics["tail_records"].append(record)

    diagnostics["delta_tail_grid"] = delta[seq_len:, :].copy()

    temp_delta = delta[seq_len:, :]
    idx = np.argmax(temp_delta)
    pos, _ = np.unravel_index(idx, temp_delta.shape)
    pos = pos + seq_len + 1

    state = int(np.argmax(delta[pos - 1, :])) + 1

    offset = pos
    preceding_state = psi[offset - 1, state - 1]
    dur_used = psi_duration[offset - 1, state - 1]
    onset = int(offset - dur_used + 1)

    assigned_states[onset - 1:offset] = state
    diagnostics["backtrack_trace"].append({
        "offset": offset, "state": state, "preceding_state": int(preceding_state),
        "psi_duration_used": dur_used, "onset": onset,
        "actual_frames_assigned": offset - (onset - 1),
        "matches": dur_used == (offset - (onset - 1)),
    })

    state = int(preceding_state)
    count = 0

    while onset > 2:
        offset = int(onset - 1)
        preceding_state = psi[offset - 1, state - 1]
        dur_used = psi_duration[offset - 1, state - 1]
        onset = int(offset - dur_used + 1)

        if onset < 2:
            onset = 1

        assigned_states[onset - 1:offset] = state
        diagnostics["backtrack_trace"].append({
            "offset": offset, "state": state, "preceding_state": int(preceding_state),
            "psi_duration_used": dur_used, "onset": onset,
            "actual_frames_assigned": offset - (onset - 1),
            "matches": dur_used == (offset - (onset - 1)),
        })

        state = int(preceding_state)
        count = count + 1
        if count > 1000:
            diagnostics["backtrack_hit_iteration_limit"] = True
            break

    assigned_states = assigned_states[:seq_len]

    return delta, psi, assigned_states, diagnostics


def verify_against_real(observation_sequence, models, total_obs_distribution,
                        distribution, heart_rate, systolic_time, recording_frequency):
    """Runs both the real viterbi_segment (imported from your segmenter/viterbi.py --
    assumed to already have the hard-zero fix applied) and this instrumented
    copy on identical inputs, and confirms they produce identical
    assigned_states. If this returns False, don't trust the diagnostics
    below it -- the instrumented copy has drifted from the real decode."""
    from segmenter.viterbi import viterbi_segment
    _, _, real_assigned = viterbi_segment(observation_sequence, models, total_obs_distribution,
                                          distribution, heart_rate, systolic_time, recording_frequency)
    _, _, debug_assigned, _ = viterbi_segment_debug(observation_sequence, models, total_obs_distribution,
                                                    distribution, heart_rate, systolic_time, recording_frequency)
    return bool(np.array_equal(real_assigned, debug_assigned)), real_assigned, debug_assigned


def print_diagnostics(clip_name, diagnostics, ground_truth=None):
    print(f"\n{'#' * 70}\n{clip_name}: SEARCH INTERNALS\n{'#' * 70}")
    print(f"seq_len={diagnostics['seq_len']}  max_duration_D={diagnostics['max_duration_D']}")
    print(f"duration_sum per state [S1, Systole, S2, Diastole]: {diagnostics['duration_sum']}")
    if np.any(diagnostics["duration_sum"] == 0):
        print("  *** WARNING: at least one state has duration_sum == 0 -- "
              "normalization is degenerate for that state ***")

    all_updates = diagnostics["accepted_updates"]
    by_region = {"none": 0, "start_clamp": 0, "tail_clamp": 0}
    mismatches_by_region = {"none": 0, "start_clamp": 0, "tail_clamp": 0}
    for u in all_updates:
        by_region[u["region"]] += 1
        if u["mismatch"]:
            mismatches_by_region[u["region"]] += 1

    print(f"\nAccepted updates by region: {by_region}")
    print(f"Of those, nominal-duration != actual-span (mismatch) by region: {mismatches_by_region}")

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