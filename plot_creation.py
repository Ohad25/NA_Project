import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random 
from scipy.signal import welch, spectrogram
import torch


def plot_power_spectrum(ff_left, pxx_left,pxx_right,trial_num:int,channel ,plot_limit:tuple,mean=True ):
     
    strat_freq = plot_limit[0]
    end_freq = plot_limit[1]
    mask = (ff_left >= strat_freq) & (ff_left <= end_freq) #Take only the frequencies in the specified range
    ff3_masked = ff_left[mask]


    if mean:
        pxx_left = np.mean(pxx_left, axis=0)  # Average across trials
        pxx_right = np.mean(pxx_right, axis=0)  # Average across trials
        pxx_left = pxx_left[mask]
        pxx_right = pxx_right[mask]
    
        std_pxx_left = np.std(pxx_left, axis=0)
        std_pxx_right = np.std(pxx_right, axis=0)

    else:
        pxx_left = pxx_left[trial_num,mask]
        pxx_right = pxx_right[trial_num,mask]
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


def plot_spectogram(spectogram_data,data_Sxx,trial,channel,mean=True):
    """The function gets a dictionary with information from a spectrogram and plots it.
    The dictionary should contain the following keys:
    'left': (frequencies, times, spectrogram) for left hand movement
    'right': (frequencies, times, spectrogram) for right hand movement
    'difference': (frequencies, times, spectrogram) for the difference between left and right hand movements
    'baseline_left': (frequencies, times, spectrogram) for the baseline of left hand movement
    'baseline_right': (frequencies, times, spectrogram) for the baseline of right hand movement
    args:
    spectogram_data: dictionary with the spectrogram data
    data_sxx: A list that says which data to use (left and right, difference, baseline_left and baseline_right)
    """
    if mean:
        for sxx in data_Sxx:
            f, t, Sxx = spectogram_data[sxx]  # Get the frequencies, times, and spectrogram data
            spectogram_data[sxx] = (f, t, (np.mean(Sxx, axis=0)))  # Average across trials and store with label

    else:
        for sxx in data_Sxx:
            f, t, Sxx = spectogram_data[sxx]
            spectogram_data[sxx] = (f, t, (Sxx[trial]))  # Select the trial data

    fig, axes = plt.subplots(len(data_Sxx),1, figsize=(15, 10))\
    
    for i ,Sxx in enumerate(data_Sxx): 
        ax = axes[i] if len(data_Sxx) > 1 else axes
        data = data_Sxx[i] 
        f, t, Sxx = spectogram_data[data]  # Get the frequencies, times, and spectrogram data
        im = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='jet') #To change it to --> dB 10 * np.log10(mean_Sxx + 1e-10)
        ax.set_title(f'Spectrogram for Channel {channel} ({data})')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylim(0, 30)  # Limit the frequency range 

        cbar = fig.colorbar(im)
        cbar.set_label('power per Hz ', fontsize=12) #If you change to decibels, change this to 'dB'
    plt.tight_layout()
    plt.show()
