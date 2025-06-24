import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Constants
start_fixation = 2.15
end_fixation = 6.0
time_segments = [(1, 3), (3, 5), (5, 6)]
frequency_bands = [(6, 12), (12, 20), (20, 25)]


def calculate_power_spectrum(dataset, channel_idx: int, time_segment=(start_fixation, end_fixation)):
    """Computes the Welch power spectrum for left and right trials of a given EEG channel and time segment."""
    t0, t1 = time_segment
    windowed = dataset.get_windowed_data(t0, t1)
    channel = dataset.channel_names[channel_idx]

    left = windowed[f'{channel}-Left']
    right = windowed[f'{channel}-Right']
    n_per_segment = left.shape[1] // 2

    f, psd_left = welch(left, dataset.fs, nperseg=n_per_segment, axis=1)
    _, psd_right = welch(right, dataset.fs, nperseg=n_per_segment, axis=1)

    return f, psd_left, psd_right


def psd_feature(frequencies, psd, frequency_band):
    """Calculates band power from a PSD array within a specified frequency range."""
    mask = (frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1])
    delta_f = frequencies[1] - frequencies[0]

    # Ensure PSD shape is (n_trials, n_frequencies)
    if psd.shape[0] == len(frequencies):
        psd = psd.T  # Transpose to (n_trials, n_frequencies)

    return np.sum(psd[:, mask], axis=1) * delta_f  # shape: (n_trials,)


def compute_band_features(dataset, channel_idx: int):
    """Extracts band power features across time segments and frequency bands for a given channel."""
    feats_left, feats_right = [], []

    for seg in time_segments:
        frequencies, psd_left, psd_right = calculate_power_spectrum(dataset, channel_idx, time_segment=seg)
        for band in frequency_bands:
            feats_left.append(psd_feature(frequencies, psd_left, band))
            feats_right.append(psd_feature(frequencies, psd_right, band))

    return np.array(feats_left).T, np.array(feats_right).T  # (n_trials, n_features)


def extract_time_features(trials):
    """Computes RMS and variance features for each EEG trial."""
    feats = []
    for trial in trials:
        rms = np.sqrt(np.mean(trial**2, axis=0))
        var = np.var(trial, axis=0)
        feats.append(np.array([rms, var]))
    return np.array(feats)  # (n_trials, 4)


def compute_time_features(dataset, channel_idx: int):
    """Extracts RMS and variance features for left and right trials of a specific EEG channel."""
    if channel_idx == 0:
        left_trials = dataset.c3_left
        right_trials = dataset.c3_right
    elif channel_idx == 1:
        left_trials = dataset.c4_left
        right_trials = dataset.c4_right
    else:
        raise ValueError("channel_idx must be 0 (C3) or 1 (C4)")

    left_feats = extract_time_features(left_trials)
    right_feats = extract_time_features(right_trials)

    return left_feats, right_feats


def spectral_entropy_from_psd(psd_trials):
    """Computes spectral entropy from PSD values for each trial."""
    entropy_vals = []
    for psd in psd_trials:
        psd_norm = psd / np.sum(psd)  # Normalize to get a probability distribution
        ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  # Shannon entropy
        entropy_vals.append(ent)
    return np.array(entropy_vals)


def compute_entropy_features(dataset):
    """Computes spectral entropy features from both EEG channels for left and right trials."""
    ff3, pxx_left3, pxx_right3 = calculate_power_spectrum(dataset, 0)
    ff4, pxx_left4, pxx_right4 = calculate_power_spectrum(dataset, 1)

    entropy_left3 = spectral_entropy_from_psd(pxx_left3)
    entropy_right3 = spectral_entropy_from_psd(pxx_right3)

    entropy_left4 = spectral_entropy_from_psd(pxx_left4)
    entropy_right4 = spectral_entropy_from_psd(pxx_right4)

    left_entropy = np.column_stack((entropy_left3, entropy_left4))  # (n_trials, 2)
    right_entropy = np.column_stack((entropy_right3, entropy_right4))

    return left_entropy, right_entropy


def build_feature_matrix(band_feats, time_feats, entropy_feats):
    """Combines band power, time-domain, and entropy features into a single feature matrix."""
    left = np.concatenate([band_feats[0], time_feats[0], entropy_feats[0]], axis=1)
    right = np.concatenate([band_feats[1], time_feats[1], entropy_feats[1]], axis=1)
    return left, right


def create_features_matrix(dataset):
    """Extracts and concatenates all EEG features from the dataset for both classes."""
    all_band_feats = []
    all_time_feats = []

    for channel_idx in dataset.channel_names.keys():
        print(f"Extracting features from channel {dataset.channel_names[channel_idx]}...")
        band_feats = compute_band_features(dataset, channel_idx)
        time_feats = compute_time_features(dataset, channel_idx)

        all_band_feats.append(band_feats)
        all_time_feats.append(time_feats)

    entropy_feats = compute_entropy_features(dataset)

    # Concatenate band + time features across channels
    band_feats_left = np.concatenate([f[0] for f in all_band_feats], axis=1)
    band_feats_right = np.concatenate([f[1] for f in all_band_feats], axis=1)

    time_feats_left = np.concatenate([f[0] for f in all_time_feats], axis=1)
    time_feats_right = np.concatenate([f[1] for f in all_time_feats], axis=1)

    left, right = build_feature_matrix((band_feats_left, band_feats_right),
                                       (time_feats_left, time_feats_right),
                                       entropy_feats)

    print("Final feature shapes:")
    print("Left:", left.shape, "Right:", right.shape)
    return left, right


def calculate_PCA(features_left, features_right, n_components=3):
    """Applies PCA to combined features from both classes and returns transformed features and labels."""
    X = np.concatenate((features_left, features_right), axis=0)  # shape: (n_trials_total, n_features_total)
    y_pca = np.array([0]*len(features_left) + [1]*len(features_right))  # 0 = LEFT, 1 = RIGHT

    # Normalize features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y_pca, pca


def plot_PCA(X_pca, y):
    """Visualizes PCA-reduced features in 2D or 3D based on the number of components."""
    if len(X_pca) != len(y):
        raise ValueError(f"Mismatch: X_pca has {len(X_pca)} samples but y has {len(y)} labels.")

    n_components = X_pca.shape[1]
    assert n_components in [2, 3], "Only 2D or 3D PCA plots are supported."

    classes = [0, 1]
    labels = ['LEFT', 'RIGHT']
    colors = ['red', 'blue']

    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for cls, label, color in zip(classes, labels, colors):
            indices = (y == cls)
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                        label=label, c=color, alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA: 2D Projection of EEG Features')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif n_components == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for cls, label, color in zip(classes, labels, colors):
            indices = (y == cls)
            ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2],
                       label=label, c=color, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA: 3D Projection of EEG Features')
        ax.legend()
        plt.tight_layout()
        plt.show()


def features(dataset):
    """Runs the full feature extraction pipeline and returns raw and PCA-reduced features with labels."""
    left_features, right_features = create_features_matrix(dataset)

    n_components = 10
    X_pca, y, pca = calculate_PCA(left_features, right_features, n_components=n_components)
    if n_components < 4:
        plot_PCA(X_pca, y)

    return left_features, right_features, X_pca, y
