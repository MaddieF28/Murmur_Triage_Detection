# get_springer_pcg_features.py
import numpy as np
from scipy.signal import resample_poly, resample

from .default_Springer_HSMM_options import default_springer_hsmm_options
from .butterwoth_low_pass_filter import butterworth_low_pass_filter
from .butterworth_high_pass_filter import butterworth_high_pass_filter
from .Schmidt_spike_removal import schmidt_spike_removal
from .Homomorphic_Envelope_with_Hilbert import homomorphic_envelope_with_hilbert
from .Hilbert_Envelope import hilbert_envelope
from .get_PSD_feature_Springer_HMM import get_psd_feature_springer_hmm
from .normalise_signal import normalise_signal

def get_springer_pcg_features(audio_data, fs):
    options = default_springer_hsmm_options()
    features_fs = options.audio_segmentation_Fs

    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, fs)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, fs)
    audio_data = schmidt_spike_removal(audio_data, fs)

    homomorphic_env = homomorphic_envelope_with_hilbert(audio_data, fs)
    downsampled_homomorphic = normalise_signal(resample_poly(homomorphic_env, features_fs, fs))

    hilbert_env = hilbert_envelope(audio_data)
    downsampled_hilbert = normalise_signal(resample_poly(hilbert_env, features_fs, fs))

    psd = get_psd_feature_springer_hmm(audio_data, fs, 40, 60)
    psd = normalise_signal(resample_poly(psd, len(downsampled_homomorphic), len(psd)))

    if options.include_wavelet_feature:
        raise NotImplementedError(
            "include_wavelet_feature is True but getDWT was never ported "
            "(it's off by default). Port getDWT.m first if you enable this."
        )

    pcg_features = np.column_stack([downsampled_homomorphic, downsampled_hilbert, psd])
    return pcg_features, features_fs