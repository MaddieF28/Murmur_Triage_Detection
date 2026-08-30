# hilbert_envelope.py
import numpy as np
from scipy.signal import hilbert

def hilbert_envelope(input_signal: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(input_signal))  # type: ignore[arg-type]