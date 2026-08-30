# butterworth_low_pass_filter.py
from scipy.signal import butter, filtfilt

def butterworth_low_pass_filter(original_signal, order, cutoff, sampling_frequency):
    b, a = butter(order, 2 * cutoff / sampling_frequency, btype='low')
    return filtfilt(b, a, original_signal)