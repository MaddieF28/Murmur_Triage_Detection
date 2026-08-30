from dataclasses import dataclass

@dataclass
class SpringerHSMMOptions:
    # The sampling frequency at which to extract signal features:
    audio_Fs: int = 1000

    # The downsampled frequency (set to 50 in Springer paper):
    audio_segmentation_Fs: int = 50

    # Tolerance for S1 and S2 localization, in seconds:
    segmentation_tolerance: float = 0.1

    # Whether to use compiled/optimized Viterbi decoding (mex equivalent) or not:
    use_mex: bool = False

    # Whether to include the wavelet feature or not:
    include_wavelet_feature: bool = False


def default_springer_hsmm_options() -> SpringerHSMMOptions:
    """Python equivalent of default_Springer_HSMM_options.m"""
    return SpringerHSMMOptions()