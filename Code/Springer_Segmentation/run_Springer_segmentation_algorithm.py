# run_springer_segmentation_algorithm.py
from .get_Springer_PCG_features import get_springer_pcg_features
from .get_heartrate_Schmidt import get_heart_rate_schmidt
from .viterbi_decode_PCG_Springer import viterbi_decode_pcg_springer
from .expand_qt import expand_qt
import numpy as np

def run_springer_segmentation_algorithm(audio_data, fs, b_matrix, pi_vector, total_obs_distribution, figures=False):
    pcg_features, features_fs = get_springer_pcg_features(audio_data, fs)
    heart_rate, systolic_time_interval = get_heart_rate_schmidt(audio_data, fs)

    _, _, qt = viterbi_decode_pcg_springer(
        pcg_features, pi_vector, b_matrix, total_obs_distribution,
        heart_rate, systolic_time_interval, features_fs
    )

    print(f"raw qt shape={qt.shape}, zero count={np.sum(qt == 0)}, unique={np.unique(qt)}")

    assigned_states = expand_qt(qt, features_fs, fs, len(audio_data))
    return assigned_states