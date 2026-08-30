# train_springer_segmentation_algorithm.py
import numpy as np
from typing import List, Optional
from .get_Springer_PCG_features import get_springer_pcg_features
from .label_PCG_states import label_pcg_states
from .train_B_andPi_matricies_Springer import train_b_and_pi_matrices_springer

def train_springer_segmentation_algorithm(pcg_cell_array, annotations_array, fs, figures=False):
    number_of_states = 4
    num_pcgs = len(pcg_cell_array)
    state_observation_values: List[List[Optional[np.ndarray]]] = [[None] * number_of_states for _ in range(num_pcgs)]


    for i in range(num_pcgs):
        pcg_audio = pcg_cell_array[i]
        s1_locations = annotations_array[i][0]
        s2_locations = annotations_array[i][1]

        pcg_features, features_fs = get_springer_pcg_features(pcg_audio, fs)
        pcg_states = label_pcg_states(pcg_features[:, 0], s1_locations, s2_locations, features_fs)

        print(f"recording {i}: pcg_features shape={pcg_features.shape}, pcg_states unique values={np.unique(pcg_states)}")

        for state_i in range(number_of_states):
            values = pcg_features[pcg_states == (state_i + 1)]
            state_observation_values[i][state_i] = values
            print(f"  state {state_i}: assigned shape={values.shape}")

    for i, row in enumerate(state_observation_values):
        for state_i, val in enumerate(row):
            if val is None:
                print(f"MISSING: recording {i}, state {state_i}")

    b_matrix, pi_vector, total_obs_distribution = train_b_and_pi_matrices_springer(state_observation_values)
    return b_matrix, pi_vector, total_obs_distribution