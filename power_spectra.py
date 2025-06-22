from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np


def create_spectrogram(left_data, right_data, fs, baseline):
    """Generates spectrogram and baseline-corrected versions for left/right EEG data."""
    output = {}
    f, t, psd_left = spectrogram(left_data, fs=fs, nperseg=256, axis=1)
    f2, t2, psd_right = spectrogram(right_data, fs=fs, nperseg=256, axis=1)

    psd_diff = psd_left / psd_right  # Avoid division by zero

    baseline_mask = t <= baseline  # The assignment says to use the first second as baseline, but we start at 1 seconds
    # Average across time baseline period:
    baseline_Sxx_left = np.mean(psd_left[:, :, baseline_mask])  # Select the first second of the spectrogram
    baseline_Sxx_right = np.mean(psd_right[:, :, baseline_mask])  # Select the first second of the spectrogram\

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
        ax.set_ylim(0, 30)  # Limit the frequency range

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
