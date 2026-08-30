# get_duration_distributions.py
from .default_Springer_HSMM_options import default_springer_hsmm_options

def get_duration_distributions(heartrate, systolic_time):
    options = default_springer_hsmm_options()
    fs = options.audio_segmentation_Fs

    mean_S1 = round(0.122 * fs)
    std_S1 = round(0.022 * fs)
    mean_S2 = round(0.094 * fs)
    std_S2 = round(0.022 * fs)

    mean_systole = round(systolic_time * fs) - mean_S1
    std_systole = (25 / 1000) * fs

    mean_diastole = ((60 / heartrate) - systolic_time - 0.094) * fs
    std_diastole = 0.07 * mean_diastole + (6 / 1000) * fs

    # d_distributions[state] = (mean, variance) — states 0=S1, 1=systole, 2=S2, 3=diastole
    d_distributions = {
        0: (mean_S1, std_S1 ** 2),
        1: (mean_systole, std_systole ** 2),
        2: (mean_S2, std_S2 ** 2),
        3: (mean_diastole, std_diastole ** 2),
    }

    min_systole = mean_systole - 3 * (std_systole + std_S1)
    max_systole = mean_systole + 3 * (std_systole + std_S1)

    min_diastole = mean_diastole - 3 * std_diastole
    max_diastole = mean_diastole + 3 * std_diastole

    min_S1 = mean_S1 - 3 * std_S1
    if min_S1 < fs / 50:
        min_S1 = fs / 50

    min_S2 = mean_S2 - 3 * std_S2
    if min_S2 < fs / 50:
        min_S2 = fs / 50

    max_S1 = mean_S1 + 3 * std_S1
    max_S2 = mean_S2 + 3 * std_S2

    return (d_distributions, max_S1, min_S1, max_S2, min_S2,
            max_systole, min_systole, max_diastole, min_diastole)