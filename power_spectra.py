from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def calculate_power_spectrum(dataset, time_segment: tuple):
    """Computes the Welch power spectrum for left and right trials of a given EEG channel and time segment."""
    t0, t1 = time_segment
    windowed = dataset.get_windowed_data(t0, t1)

    c3 = windowed[f'C3']
    c4 = windowed[f'C4']
    n_per_segment = c3.shape[1] // 2

    f, psd_c3 = welch(c3, dataset.fs, nperseg=n_per_segment, axis=1)
    _, psd_c4 = welch(c4, dataset.fs, nperseg=n_per_segment, axis=1)

    return f, psd_c3, psd_c4


def plot_power_spectrum(dataset, f, psd_c3, psd_c4, plot_limit: tuple = None):
    """Plot power spectrum with 1-standard-deviation"""
    left_indices = dataset.left_indices
    right_indices = dataset.right_indices
    psd_c3_left = psd_c3[left_indices]
    psd_c3_right = psd_c3[right_indices]

    psd_c4_left = psd_c4[left_indices]
    psd_c4_right = psd_c4[right_indices]

    if plot_limit is not None:
        start_freq = plot_limit[0]
        end_freq = plot_limit[1]
    else:
        start_freq = f[0]
        end_freq = f[-1]
    mask = (f >= start_freq) & (f <= end_freq)  # Take only the frequencies in the specified range
    f_masked = f[mask]

    psd_c3_left = np.mean(psd_c3_left, axis=0)[mask]  # Average across trials
    psd_c3_right = np.mean(psd_c3_right, axis=0)[mask]  # Average across trials
    psd_c4_left = np.mean(psd_c4_left, axis=0)[mask]  # Average across trials
    psd_c4_right = np.mean(psd_c4_right, axis=0)[mask]  # Average across trials

    std_c3_left = np.std(psd_c3_left, axis=0)
    std_c3_right = np.std(psd_c3_left, axis=0)
    std_c4_left = np.std(psd_c4_left, axis=0)
    std_c4_right = np.std(psd_c4_right, axis=0)

    # Plotting the power spectrum for C3 and C4 channels
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    ax_c3 = axes[0]
    ax_c4 = axes[1]

    # Subplot for C3
    ax_c3.plot(f_masked, psd_c3_left, label='Left Hand', color='blue')
    ax_c3.plot(f_masked, psd_c3_right, label='Right Hand', color='orange')
    ax_c3.fill_between(f_masked, psd_c3_left - std_c3_left, psd_c3_left + std_c3_left, alpha=0.3)
    ax_c3.fill_between(f_masked, psd_c3_right - std_c3_right, psd_c3_right + std_c3_right, alpha=0.3)
    ax_c3.set_title(f'Power Spectrum of C3 Channel')
    ax_c3.set_xlabel('Frequency [Hz]')
    ax_c3.set_ylabel('Power/Frequency [V²/Hz]')
    ax_c3.legend()
    ax_c3.grid()
    # Subplot for C4
    ax_c4.plot(f_masked, psd_c4_left, label='Left Hand', color='blue')
    ax_c4.plot(f_masked, psd_c4_right, label='Right Hand', color='orange')
    ax_c4.fill_between(f_masked, psd_c4_left - std_c4_left, psd_c4_left + std_c4_left, alpha=0.3)
    ax_c4.fill_between(f_masked, psd_c4_right - std_c4_right, psd_c4_right + std_c4_right, alpha=0.3)
    ax_c4.set_title(f'Power Spectrum of C4 Channel')
    ax_c4.set_xlabel('Frequency [Hz]')
    ax_c4.set_ylabel('Power/Frequency [V²/Hz]')
    ax_c4.legend()
    ax_c4.grid()
    plt.tight_layout()
    plt.show()


def create_spectrogram(left_data, right_data, fs, baseline):
    """Generates spectrogram and baseline-corrected versions for left/right EEG data."""
    output = {}
    f, t, psd_left = spectrogram(left_data, fs=fs, nperseg=256, axis=1)
    f2, t2, psd_right = spectrogram(right_data, fs=fs, nperseg=256, axis=1)

    psd_diff = psd_left - psd_right  # Avoid division by zero

    baseline_mask = t <= baseline  # The assignment says to use the first second as baseline, but we start at 1 seconds
    # Average across time baseline period:
    baseline_Sxx_left = np.mean(psd_left[:, :, baseline_mask])  # Select the first second of the spectrogram
    baseline_Sxx_right = np.mean(psd_right[:, :, baseline_mask])  # Select the first second of the spectrogram

    output['left'] = (f, t, psd_left)
    output['right'] = (f2, t2, psd_right)
    output['difference'] = (f, t, psd_diff)
    output['baseline_left'] = (f, t, psd_left - baseline_Sxx_left)
    output['baseline_right'] = (f2, t2, psd_right - baseline_Sxx_right)
    return output


def plot_spectrogram(spectrogram_data, data_Sxx, trial, channel, mean=True):
    """Plots spectrogram from given data using mean or single-trial visualization."""
    if mean:
        for sxx in data_Sxx:
            f, t, Sxx = spectrogram_data[sxx]  # Get the frequencies, times, and spectrogram data
            spectrogram_data[sxx] = (f, t, (np.mean(Sxx, axis=0)))  # Average across trials and store with label

    else:
        for sxx in data_Sxx:
            f, t, Sxx = spectrogram_data[sxx]
            spectrogram_data[sxx] = (f, t, (Sxx[trial]))  # Select the trial data

    fig, axes = plt.subplots(len(data_Sxx), 1, figsize=(15, 10))
    for i, Sxx in enumerate(data_Sxx):
        ax = axes[i] if len(data_Sxx) > 1 else axes
        data = data_Sxx[i]
        f, t, Sxx = spectrogram_data[data]  # Get the frequencies, times, and spectrogram data
        im = ax.pcolormesh(t, f, Sxx, shading='gouraud',
                           cmap='jet')  # To change it to --> dB 10 * np.log10(mean_Sxx + 1e-10)
        ax.set_title(f'Spectrogram for Channel {channel} ({data})')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylim(0, 20)  # Limit the frequency range

        cbar = fig.colorbar(im)
        cbar.set_label('power per Hz ', fontsize=12)  # If you change to decibels, change this to 'dB'

    plt.tight_layout()
    plt.show()


def power_spectra(dataset):
    """Computes and plots spectrogram for C3 and C4 channels from a dataset."""
    baseline = 1  # First second as baseline
    sg3_info = create_spectrogram(dataset.c3_left, dataset.c3_right, dataset.fs, baseline)
    sg4_info = create_spectrogram(dataset.c4_left, dataset.c4_right, dataset.fs, baseline)

    data_sg = ['left', 'right', 'baseline_left', 'baseline_right', 'difference']
    plot_spectrogram(sg3_info, data_sg, channel=3, trial=0, mean=True)
    plot_spectrogram(sg4_info, data_sg, channel=4, trial=0, mean=True)
