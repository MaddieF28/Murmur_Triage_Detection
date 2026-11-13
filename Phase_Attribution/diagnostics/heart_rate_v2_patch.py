"""
Patch for get_heart_rate() in segmenter/heart_rate.py.
Only the `multiple_rates` branch changes - candidates are now sorted by
autocorrelation peak height (strongest first) instead of by lag position
(which previously meant "fastest candidate first" regardless of strength).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: makes `paths` and the packages importable from any cwd
import paths

import numpy as np
from scipy.signal import correlate, find_peaks
from segmenter.extract_features import get_butterworth_high_pass_filter, get_butterworth_low_pass_filter, schmidt_spike_removal


def get_heart_rate(audio_data, Fs, min_heart_rate=40, max_heart_rate=190, multiple_rates=True):
    audio_data = get_butterworth_low_pass_filter(audio_data, 2, 100, Fs)
    audio_data = get_butterworth_high_pass_filter(audio_data, 2, 20, Fs)
    audio_data = schmidt_spike_removal(audio_data, Fs)

    audio_data = np.abs(audio_data - audio_data.mean())
    c = correlate(audio_data, audio_data)
    c /= c[int(c.shape[0] / 2)]
    signal_autocorrelation = c[audio_data.shape[0]:]

    min_idx = 60 * Fs / max_heart_rate
    max_idx = 60 * Fs / min_heart_rate
    min_sys_duration = np.round(0.1 * Fs)

    if multiple_rates:
        cut_autocorr = signal_autocorrelation[int(min_idx):int(max_idx)]
        peaks, _ = find_peaks(cut_autocorr, width=4, prominence=0.1)

        # --- THE FIX: sort by peak height (strongest first), ---
        # --- not by array position (which meant "fastest first") ---
        if len(peaks) > 0:
            peak_heights = cut_autocorr[peaks]
            order = np.argsort(peak_heights)[::-1]
            peaks = peaks[order]
        # --- end fix ---

        true_indices = peaks + min_idx
        heart_rates = 60. / (true_indices / Fs)

        max_sys_durations = [np.round(((60. / hr) * Fs) / 2) for hr in heart_rates]

        systolic_time_intervals = []
        for max_sd in max_sys_durations:
            pos = np.argmax(signal_autocorrelation[int(min_sys_duration): int(max_sd)])
            systolic_time_intervals.append((min_sys_duration + pos) / Fs)

    if not multiple_rates or len(heart_rates) == 0:
        index = np.argmax(signal_autocorrelation[int(min_idx): int(max_idx)])
        true_idx = index + min_idx
        heart_rate = 60. / (true_idx / Fs)

        max_sys_duration = np.round(((60. / heart_rate) * Fs) / 2)
        pos = np.argmax(signal_autocorrelation[int(min_sys_duration): int(max_sys_duration)])
        systolic_time_interval = (min_sys_duration + pos) / Fs
        systolic_time_intervals = [systolic_time_interval]
        heart_rates = [heart_rate]

    return heart_rates, systolic_time_intervals