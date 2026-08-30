# homomorphic_envelope_with_hilbert.py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def homomorphic_envelope_with_hilbert(input_signal, sampling_frequency, lpf_frequency=8):
    b, a = butter(1, 2 * lpf_frequency / sampling_frequency, btype='low')
    envelope = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(input_signal)))))
    envelope[0] = envelope[1]
    return envelope