import matplotlib.pyplot as plt
import numpy as np


def plot_EEG(data, fs, title, channels_idx, side):
    """Plots 20 random EEG trials from C3 and C4 channels with time on the x-axis."""
    random_indices = np.random.randint(0, data.shape[0], size=20)
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    c3_idx, c4_idx = channels_idx

    for i in range(20):
        ax = axes[i // 5, i % 5]
        random_index = random_indices[i]
        ax.plot(np.arange(data.shape[1]) / fs, data[random_index, :, c3_idx], color='blue',alpha=0.9)
        ax.plot(np.arange(data.shape[1]) / fs, data[random_index, :, c4_idx], color='red',alpha=0.7)
        ax.set_title(f'Trial {random_index}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [µV]')
        ax.grid(True)


    plt.suptitle(f'EEG Data from {title} - {side} - Channels C3 (blue) & C4 (red)', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def visualization(dataset):
    """Displays raw EEG waveforms from both left and right motor imagery trials."""
    title = 'Training Data'

    plot_EEG(dataset.data_left, dataset.fs, title, dataset.channel_names.keys(), side='Left Hand')
    plot_EEG(dataset.data_right, dataset.fs, title, dataset.channel_names.keys(), side='Right Hand')


