# normalise_signal.py
import numpy as np

def normalise_signal(signal):
    return (np.asarray(signal) - np.mean(signal)) / np.std(signal)