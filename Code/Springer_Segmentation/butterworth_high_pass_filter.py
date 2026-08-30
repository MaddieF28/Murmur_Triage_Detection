from scipy.signal import butter, filtfilt
import numpy as np

def butterworth_high_pass_filter(original_signal, order, cutoff, sampling_frequency):
    b, a = butter(
        order,
        2 * cutoff / sampling_frequency,
        btype='high',
        output='ba'
    )

    return filtfilt(b, a, original_signal)