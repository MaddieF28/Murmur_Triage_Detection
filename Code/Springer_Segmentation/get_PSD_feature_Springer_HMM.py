import numpy as np
from scipy.signal import spectrogram

def get_psd_feature_springer_hmm(data, sampling_frequency, frequency_limit_low, frequency_limit_high):
    nperseg = round(sampling_frequency / 40)
    noverlap = round(sampling_frequency / 80)

    freqs, times, Sxx = spectrogram(
        data, fs=sampling_frequency, window='hamming',
        nperseg=nperseg, noverlap=noverlap, nfft=sampling_frequency,
        mode='psd'
    )

    low_idx = np.argmin(np.abs(freqs - frequency_limit_low))
    high_idx = np.argmin(np.abs(freqs - frequency_limit_high))

    psd = np.mean(Sxx[low_idx:high_idx + 1, :], axis=0)
    return psd