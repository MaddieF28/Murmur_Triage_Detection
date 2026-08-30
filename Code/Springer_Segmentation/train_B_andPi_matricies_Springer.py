import numpy as np
from typing import List, Optional
from sklearn.linear_model import LogisticRegression

def train_b_and_pi_matrices_springer(state_observation_values):
    """Python port of trainBandPiMatricesSpringer.m
    state_observation_values: list of length N (recordings), each a list of
    4 arrays (one per state: S1, systole, S2, diastole), each array shaped
    (num_samples_in_state, num_features)."""
    number_of_states = 4
    rng = np.random.default_rng()

    pi_vector = np.array([0.25, 0.25, 0.25, 0.25])

    # Pool all recordings' observations by state
    statei_values = [[] for _ in range(number_of_states)]
    for pcg_states in state_observation_values:
        for state_i in range(number_of_states):
            if len(pcg_states[state_i]) > 0:
                statei_values[state_i].append(pcg_states[state_i])
    statei_values = [np.vstack(v) if v else np.empty((0, 0)) for v in statei_values]

    # total_obs_distribution: mean & covariance across ALL states pooled together
    total_observations = np.vstack(statei_values)
    total_obs_mean = np.mean(total_observations, axis=0)
    total_obs_cov = np.cov(total_observations, rowvar=False)
    total_obs_distribution = (total_obs_mean, total_obs_cov)

    B_matrix: List[Optional[LogisticRegression]] = [None] * number_of_states

    for state in range(number_of_states):
        length_of_state_samples = len(statei_values[state])
        length_per_other_state = length_of_state_samples // (number_of_states - 1)

        min_length_other_class = min(
            len(statei_values[s]) for s in range(number_of_states) if s != state
        )
        if length_per_other_state > min_length_other_class:
            length_per_other_state = min_length_other_class

        # Balanced sample: (n-1)*length_per_other_state from the main state,
        # length_per_other_state from each other state
        main_indices = rng.choice(
            length_of_state_samples,
            size=length_per_other_state * (number_of_states - 1),
            replace=False,
        )
        main_data = statei_values[state][main_indices]

        other_data = []
        for other_state in range(number_of_states):
            if other_state == state:
                continue
            idx = rng.choice(
                len(statei_values[other_state]), size=length_per_other_state, replace=False
            )
            other_data.append(statei_values[other_state][idx])
        other_data = np.vstack(other_data)

        X = np.vstack([main_data, other_data])
        y = np.concatenate([np.ones(len(main_data)), np.zeros(len(other_data))])  # 1 = this state

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        B_matrix[state] = model  # predict_proba(X)[:, 1] gives P(this state)

    return B_matrix, pi_vector, total_obs_distribution