import pandas as pd

from utils.analysis import evaluate_nfft_mels_interaction


def test_evaluate_nfft_mels_interaction_returns_anova_table() -> None:
    df = pd.DataFrame(
        [
            {"condition_name": "a", "n_fft": 256, "n_mels": 64, "fmax": 1000, "auc_recording": 0.72},
            {"condition_name": "b", "n_fft": 256, "n_mels": 64, "fmax": 1000, "auc_recording": 0.74},
            {"condition_name": "c", "n_fft": 256, "n_mels": 128, "fmax": 1000, "auc_recording": 0.76},
            {"condition_name": "d", "n_fft": 512, "n_mels": 64, "fmax": 1000, "auc_recording": 0.77},
            {"condition_name": "e", "n_fft": 512, "n_mels": 128, "fmax": 1000, "auc_recording": 0.79},
            {"condition_name": "f", "n_fft": 2048, "n_mels": 64, "fmax": 1000, "auc_recording": 0.80},
            {"condition_name": "g", "n_fft": 2048, "n_mels": 128, "fmax": 1000, "auc_recording": 0.82},
        ]
    )

    condition_level, anova_results = evaluate_nfft_mels_interaction(df)

    assert condition_level.shape[0] == 7
    assert anova_results.shape[0] >= 1
    assert "partial_eta_squared" in anova_results.columns
