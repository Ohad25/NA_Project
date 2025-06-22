import matplotlib.pyplot as plt
import numpy as np


def plot_power_spectrum(ff_left, pxx_left, pxx_right, trial_num: int, channel, plot_limit: tuple, mean=True):
    strat_freq = plot_limit[0]
    end_freq = plot_limit[1]
    mask = (ff_left >= strat_freq) & (ff_left <= end_freq)  # Take only the frequencies in the specified range
    ff3_masked = ff_left[mask]

    if mean:
        pxx_left = np.mean(pxx_left, axis=0)  # Average across trials
        pxx_right = np.mean(pxx_right, axis=0)  # Average across trials
        pxx_left = pxx_left[mask]
        pxx_right = pxx_right[mask]

        std_pxx_left = np.std(pxx_left, axis=0)
        std_pxx_right = np.std(pxx_right, axis=0)

    else:
        pxx_left = pxx_left[trial_num, mask]
        pxx_right = pxx_right[trial_num, mask]
    # Plotting the power spectrum for C3 and C4 channels

    plt.figure(figsize=(12, 6))

    plt.plot(ff3_masked, pxx_left, label='Left Hand', color='blue')
    plt.plot(ff3_masked, pxx_right, label='Right Hand', color='orange')

    if mean:
        plt.fill_between(ff3_masked, pxx_left - std_pxx_left, pxx_left + std_pxx_left, alpha=0.3)
        plt.fill_between(ff3_masked, pxx_right - std_pxx_right, pxx_right + std_pxx_right, alpha=0.3)

    plt.title(f'Power Spectrum of EEG Data channel {channel}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid()
    plt.show()
