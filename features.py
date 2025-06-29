import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from power_spectra import calculate_power_spectrum
# Constants

time_segments = [(1, 3),(1,2.5) ,(3, 5), (5, 6)]
frequency_bands = [(6, 12), (12, 16),(16,20), (20, 25)]

def psd_feature(frequencies, psd, frequency_band):
    """Calculates band power from a PSD array within a specified frequency range."""
    mask = (frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1])
    delta_f = frequencies[1] - frequencies[0]

    # Ensure PSD shape is (n_trials, n_frequencies)
    if psd.shape[0] == len(frequencies):
        psd = psd.T  # Transpose to (n_trials, n_frequencies)

    return np.sum(psd[:, mask], axis=1) * delta_f  # shape: (n_trials,)


def compute_band_features(dataset):
    """Extracts band power features across time segments and frequency bands for a given channel."""
    feats_c3, feats_c4 = [], []

    for seg in time_segments:
        frequencies, psd_c3, psd_c4 = calculate_power_spectrum(dataset,time_segment=seg)
        for band in frequency_bands:
            feats_c3.append(psd_feature(frequencies, psd_c3, band))
            feats_c4.append(psd_feature(frequencies, psd_c4, band))

    return np.array(feats_c3).T, np.array(feats_c4).T  # (n_trials, time_segments * frequency_bands)


def extract_time_features(trials):
    """Computes RMS and variance features for each EEG trial."""
    feats_loop = []
    rms = np.sqrt(np.mean(trials**2, axis=1))  # RMS for each trial
    var = np.var(trials, axis=1)  # Variance for each trial
    feats = np.column_stack((rms, var))  # shape: (n_trials, 2)
    for trial in trials:
        rms_forloop = np.sqrt(np.mean(trial**2, axis=0))
        var_forloop = np.var(trial, axis=0)
        feats_loop.append(np.array([rms_forloop, var_forloop]))
    assert np.all(np.isclose(feats,np.array(feats_loop))), "Mismatch between vectorized and for-loop features"
    return np.array(feats)  # (n_trials, 4)


def compute_time_features(dataset):
    """Extracts RMS and variance features for left and right trials of a specific EEG channel."""

    c3_trials = dataset.c3_data
    c4_trials = dataset.c4_data

    c3_feats = extract_time_features(c3_trials)
    c4_feats = extract_time_features(c4_trials)

    return c3_feats, c4_feats


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
    ch3_entropy = np.empty((len(dataset.c3_data), len(time_segments)))
    ch4_entropy = np.empty((len(dataset.c4_data), len(time_segments)))
    
    for i,seg in enumerate(time_segments):
        ff3, pxx_c3, pxx_c4 = calculate_power_spectrum(dataset,time_segment=seg)

        entropy_ch3 = spectral_entropy_from_psd(pxx_c3)
        entropy_ch4 = spectral_entropy_from_psd(pxx_c4)

        entropy_left4 = spectral_entropy_from_psd(entropy_ch3)
        entropy_right4 = spectral_entropy_from_psd(entropy_ch4)
        ch3_entropy[:, i] = entropy_left4
        ch4_entropy[:, i] = entropy_right4

    return ch3_entropy, ch4_entropy



def create_features_matrix(dataset):
    """Extracts and concatenates all EEG features from the dataset for both classes."""

    c3_band_feats,c4_band_feats = compute_band_features(dataset)
    c3_time_feats,c4_time_feats = compute_time_features(dataset)
    c3_entropy_feats, c4_entropy_feats = compute_entropy_features(dataset)

    c3_feats = np.concatenate((c3_band_feats, c3_time_feats, c3_entropy_feats), axis=1)
    c4_feats = np.concatenate((c4_band_feats, c4_time_feats, c4_entropy_feats), axis=1)

    if dataset.train_mode:
        c3_left_feats = c3_feats[dataset.left_indices]
        c3_right_feats = c3_feats[dataset.right_indices]
        c4_left_feats = c4_feats[dataset.left_indices]
        c4_right_feats = c4_feats[dataset.right_indices]
        left = np.concatenate((c3_left_feats, c4_left_feats), axis=1)
        right = np.concatenate((c3_right_feats, c4_right_feats), axis=1)
        print("Left:", left.shape, "Right:", right.shape)
        feats = (left,right)
    
        
    else:
        feats = (c3_feats, c4_feats)
    
    return feats


def calculate_PCA(features_left, features_right, n_components=3):
    """Applies PCA to combined features from both classes and returns transformed features and labels."""
    X = np.concatenate((features_left, features_right), axis=0)  # shape: (n_trials_total, n_features_total)
    y = np.array([0]*len(features_left) + [1]*len(features_right))  # 0 = LEFT, 1 = RIGHT

    # Normalize features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, pca


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
    features = create_features_matrix(dataset)
   
    if dataset.train_mode:
        left_features, right_features, = features

        
        X_pca_2d, y_2d, pca_2d = calculate_PCA(left_features, right_features, n_components=2)
        X_pca_3d, y_3d, pca_3d = calculate_PCA(left_features, right_features, n_components=2)

        plot_PCA(X_pca_2d, y_2d)
        plot_PCA(X_pca_3d, y_3d)



    return features
