# viterbi_decode_pcg_springer.py
import numpy as np
from scipy.stats import norm, multivariate_normal

from .default_Springer_HSMM_options import default_springer_hsmm_options
from .get_duration_distributions import get_duration_distributions

def viterbi_decode_pcg_springer(observation_sequence, pi_vector, b_matrix,
                                 total_obs_distribution, heartrate, systolic_time, fs):
    options = default_springer_hsmm_options()

    T = len(observation_sequence)
    N = 4  # number of states

    max_duration_D = round((60 / heartrate) * fs)

    total_len = T + max_duration_D - 1
    delta = np.full((total_len, N), -np.inf)
    psi = np.zeros((total_len, N), dtype=int)
    psi_duration = np.zeros((total_len, N), dtype=int)

    # --- Observation probabilities ---
    observation_probs = np.zeros((T, N))
    obs_mean, obs_cov = total_obs_distribution
    for n in range(N):
        model = b_matrix[n]
        pihat = model.predict_proba(observation_sequence)  # columns: [P(class0), P(class1)]
        for t in range(T):
            po_correction = multivariate_normal.pdf(observation_sequence[t, :], mean=obs_mean, cov=obs_cov)
            observation_probs[t, n] = (pihat[t, 1] * po_correction) / pi_vector[n]

    # --- Duration probabilities ---
    (d_distributions, max_S1, min_S1, max_S2, min_S2,
     max_systole, min_systole, max_diastole, min_diastole) = get_duration_distributions(heartrate, systolic_time)

    duration_probs = np.zeros((N, 3 * fs))
    duration_sum = np.zeros(N)
    bounds = {0: (min_S1, max_S1), 1: (min_systole, max_systole),
              2: (min_S2, max_S2), 3: (min_diastole, max_diastole)}

    realmin = np.finfo(float).tiny

    for state_j in range(N):
        mean_j, var_j = d_distributions[state_j]
        lo, hi = bounds[state_j]
        for d in range(1, max_duration_D + 1):
            val = norm.pdf(d, loc=mean_j, scale=np.sqrt(var_j))
            if d < lo or d > hi:
                val = realmin
            duration_probs[state_j, d - 1] = val
        duration_sum[state_j] = np.sum(duration_probs[state_j, :])

    if duration_probs.shape[1] > 3 * fs:
        duration_probs = duration_probs[:, :3 * fs]

    # --- Viterbi initialization ---
    qt = np.zeros(total_len, dtype=int)
    delta[0, :] = np.log(pi_vector) + np.log(observation_probs[0, :])
    psi[0, :] = -1

    a_matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])

    # --- Recursion (pure-Python fallback, since use_mex=False) ---
    for t in range(1, total_len):  # t = 2:T+max_duration_D-1 in MATLAB (1-indexed) -> here 0-indexed, start at 1
        for j in range(N):
            for d in range(1, max_duration_D + 1):
                start_t = t - d
                if start_t < 0:
                    start_t = 0
                if start_t > T - 2:
                    start_t = T - 2

                end_t = t
                if t > T - 1:
                    end_t = T - 1

                with np.errstate(divide='ignore'):
                    trans = delta[start_t, :] + np.log(a_matrix[:, j])
                max_index = np.argmax(trans)
                max_delta = trans[max_index]

                probs = np.prod(observation_probs[start_t:end_t + 1, j])
                if probs == 0:
                    probs = realmin
                emission_probs = np.log(probs)
                if emission_probs == 0 or np.isnan(emission_probs):
                    emission_probs = realmin

                dur_prob = duration_probs[j, d - 1] / duration_sum[j]
                with np.errstate(divide='ignore'):
                    delta_temp = max_delta + emission_probs + np.log(dur_prob)

                if delta_temp > delta[t, j]:
                    delta[t, j] = delta_temp
                    psi[t, j] = max_index
                    psi_duration[t, j] = d

    # --- Termination / backtrace ---
    temp_delta = delta[T:, :]
    idx = np.unravel_index(np.argmax(temp_delta), temp_delta.shape)
    pos = idx[0] + T

    state = np.argmax(delta[pos, :])

    offset = pos
    preceding_state = psi[offset, state]
    onset = offset - psi_duration[offset, state] + 1
    qt[max(onset, 0):offset + 1] = state + 1  # +1 to match MATLAB's 1-indexed state labels

    state = preceding_state
    count = 0
    while onset > 1:
        offset = onset - 1
        preceding_state = psi[offset, state]
        onset = offset - psi_duration[offset, state] + 1
        if onset < 1:
            onset = 0
        qt[onset:offset + 1] = state + 1
        state = preceding_state
        count += 1
        if count > 1000:
            break

    qt = qt[:T]
    return delta, psi, qt