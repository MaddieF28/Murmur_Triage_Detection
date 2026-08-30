# get_heart_rate_schmidt.py
import numpy as np

from .butterwoth_low_pass_filter import butterworth_low_pass_filter
from .butterworth_high_pass_filter import butterworth_high_pass_filter
from .Schmidt_spike_removal import schmidt_spike_removal
from .Homomorphic_Envelope_with_Hilbert import homomorphic_envelope_with_hilbert

def get_heart_rate_schmidt(audio_data, fs):
    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, fs)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, fs)
    audio_data = schmidt_spike_removal(audio_data, fs)

    homomorphic_env = homomorphic_envelope_with_hilbert(audio_data, fs)

    y = homomorphic_env - np.mean(homomorphic_env)
    n = len(y)
    c = np.correlate(y, y, mode='full')
    c = c / c[n - 1]  # normalize to match MATLAB xcorr(y,'coeff')
    signal_autocorrelation = c[n:]  # positive lags, starting at lag 1

    min_index = round(0.5 * fs)
    max_index = round(2 * fs)
    segment = signal_autocorrelation[min_index - 1:max_index]
    idx = np.argmax(segment)
    true_index = idx + min_index
    heart_rate = 60 / (true_index / fs)

    max_sys_duration = round(((60 / heart_rate) * fs) / 2)
    min_sys_duration = round(0.2 * fs)
    segment2 = signal_autocorrelation[min_sys_duration - 1:max_sys_duration]
    pos = np.argmax(segment2)
    systolic_time_interval = (min_sys_duration + pos + 1) / fs

    return heart_rate, systolic_time_interval