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
from plot_creation import plot_power_spectrum, plot_spectrogram,plot_PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
#Hyperparameters
start_fixation = 2.15 #seconds - When Beep  was heard
end_fixation = 6  # seconds - End of imagination

with open(r'C:\Users\user\Desktop\Neuron\Task-5\motor_imagery_train_data.npy', 'rb') as f:  # TODO - add the file path!
    train_data = np.load(f,allow_pickle=True)
    train_data = train_data[:,:,:2] #Last dimension is not needed accordant to pdf
    labels = np.load(f,allow_pickle=True)
    # Extract left and right information:
    left,right = labels[2],labels[3]  # Extract left and right labels
    left_indices = np.where(left)[0]  # Indices of left hand movement
    right_indices = np.where(right)[0]  # Indices of right hand movement
    train_left = train_data[left_indices]  # Left hand movement data
    train_right = train_data[right_indices]  # Right hand movement data
    
    labels_name = np.load(f,allow_pickle=True)
    fs = np.load(f, allow_pickle=True)
    fs = fs.item()  # Convert to a scalar value


with open(r'C:\Users\user\Desktop\Neuron\Task-5\motor_imagery_test_data.npy', 'rb') as f:  # TODO - add the file path!
    labels_name = np.load(f,allow_pickle=True)

# Your code goes here

#Step 1-Plotting EEG data:
def plot_EEG(data, fs, title,channel_tuple):
    """Plots EEG data for 20 random trials from 
    C3 and C4 channels."""
    # Select 20 random trials
    random_indices = random.randint(0,data.shape[0],size=20)
    fig, axes = plt.subplots(4,5, figsize=(15, 10))
    channel = channel_tuple[0]  # Extract the channel index from the tuple
    channel_name=channel_tuple[1]  # Extract the channel name from the tuple
    for i in range(20):
        ax = axes[i // 5, i % 5]
        random_index = random_indices[i]
        ax.plot(np.arange(data.shape[1]) / fs, data[random_index,:,channel],color='blue')
        ax.set_title(f'Trial {random_index}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        ax.grid(True)
    plt.suptitle(f'EEG Data from {title} - Channel {channel_name}', fontsize=16)
    plt.tight_layout()
    plt.show()  
#TODO: Eyeball the data and see if you can identify qualitative differences between the different classes.

#Step 2
def calculate_power_spectrum(data_left,data_right,channel:int, fs:int,time_segment:tuple=(start_fixation, end_fixation)):
    """Calculates the power spectrum of the EEG data using Welch's method.
    This function calculates for C3 and C4 channels.
    args: 
    data: divided into data_left and data_right numpy array of shape (n_trials, n_samples, n_channels) 
    where data_left contains left hand movement data and data_right contains right hand movement data
    channel: int, channel index to analyze (0 for C3, 1 for C4)
    fs: sampling frequency of the EEG data
    time_segment: tuple of (start_time, end_time) in seconds to define the segment of interest for power spectrum calculation.
    
    Returns: tuple of (frequencies, power_spectrum) for left and right hand movements."""
    trial_start,trial_end = time_segment
    t0 = int(trial_start * fs)  # Convert start time to sample index
    t1 = int(trial_end * fs)  # Convert end time to sample index

    signal_left = data_left[:, t0:t1, channel]  
    signal_right = data_right[:,t0:t1, channel]  

    npersegment = signal_left.shape[1]//2  # Number of samples in each segment

    ff_left, psd_left = welch(signal_left, fs, nperseg=npersegment,axis=1)
    ff_right, psd_right = welch(signal_right, fs, nperseg=npersegment,axis=1) #ff4 not needed because its equal to ff3
    
    assert ff_left.shape == ff_right.shape, "Frequency vectors must be the same length for left and right hand movements."

    return ff_left, psd_left, psd_right


def create_spectrogram(data1,data2, fs, channel,baseline=1): #TODO: Divide the for time segments and then write the power spectrum for each segment.
    """Creates a spectrogram of the EEG data for a specific channel.
    data1 = EEG data of imagining left hand
    data2 = EEG data imagining right hand
    WARNING:TODO:If we subtract we need to convert it to dB first so it will be in log scale.
    
    returns: dictionary of (frequencies, times, spectrogram) for left and right hand movements."""

    signal_left = data1[:,:, channel]  # Select the specified channel
    signal_right = data2[:,:, channel]  # Select the specified channel

    f, t, psd_left = spectrogram(signal_left, fs=fs, nperseg=256,axis=1)  # Calculate the spectrogram for left hand movement
    f2, t2, psd_right = spectrogram(signal_right, fs=fs, nperseg=256,axis=1)  # Calculate the spectrogram for right hand movement

    output = {}
    #Sxx_left = 10* np.log10(Sxx_left + 1e-10)  # Convert to dB, add small value to avoid log(0)
    #Sxx_right = 10* np.log10(Sxx_right + 1e-10)  # Convert to dB, add small value to avoid log(0)

    #Spectrogram for subtraction of left and right channels

    psd_diff = psd_left/psd_right # Calculate the difference between left and right spectrograms
        

    #Spectrogram for left and right channels  #TODO: mean over all spectrogram.  

    baseline_mask = t <= baseline #The assignment says to use the first second as baseline but we start at 1 seconds
    #Average across time baseline period:
    baseline_Sxx_left = np.mean(psd_left[:,:,baseline_mask])  # Select the first second of the spectrogram
    baseline_Sxx_right = np.mean(psd_right[:,:,baseline_mask])  # Select the first second of the spectrogram\

    output['left'] = (f, t, psd_left)
    output['right'] = (f2, t2, psd_right)
    output['difference'] = (f, t, psd_diff)
    output['baseline_left'] = (f, t, psd_left - baseline_Sxx_left)
    output['baseline_right'] = (f2, t2, psd_right - baseline_Sxx_right)

    return output




def psd_feature(ff,psd,frequency_band):
    """
    data1: signal of left hand movement (Can be raw signal or PSD or spectrogram)
    data2: signal of right hand movement "-"
    channel: channel index to analyze (0 for C3, 1 for C4)
    frequency_band1: tuple of (start_freq, end_freq) for the first frequency band in Hz

    returns: 
    power band over the frequency band for left/right hand movement.
    shape is a scalar value for each trial or time. depending on the shape of the power spectrum.


    """

    #TODO: Check if IAF needed. Start without it and if needed add it later.

    mask = (ff >= frequency_band[0]) & (ff <= frequency_band[1])  # Mask for the frequency band
    
    ff_len = len(ff)  # Length of the frequency vector
    
    #Calculate delta_f for frequency resolution (Right and left hand movement should be the same)
    delta_f = ff[1] - ff[0]  # Frequency resolution
    power_shape1 , power_shape2 = psd.shape  # Shape of the power spectrum
    #Calculate power: equivalent to the area under the curve in the frequency band
    psd = np.transpose(psd) if power_shape1==ff_len else psd #Take the right shape for masking can be (trials,frequencies) or (frequencies,time)

    psd = np.sum(psd[:,mask],axis=1) * delta_f  # Area under the curve in the frequency band
    feature_vector = np.array([psd])

    return feature_vector




# Time-Domain Features (RMS, Variance) for C3 and C4
def extract_time_features(trials):  # shape: (n_trials, n_samples, 2)
    feats = []
    for trial in trials:
        rms = np.sqrt(np.mean(trial ** 2, axis=0))  # shape (2,)
        var = np.var(trial, axis=0)  # shape (2,)
        feats.append(np.array([rms, var]))  # total: 4 features per trial
    return np.array(feats)



def time_feature(train_left, train_right,channel):
    train_left_ch = train_left[:, :, channel]  # C3 channel for left hand
    train_right_ch = train_right[:, :, channel]  # C3 channel for
    X = np.concatenate([train_left_ch, train_right_ch])  # shape (n_trials_total, n_samples, 2)
    y = np.array([0] * len(train_left_ch) + [1] * len(train_right_ch))  # 0: left, 1: right

    X_time_feats = extract_time_features(X)  # shape: (n_trials, 4)

    feature_names = ['Root Mean squared', 'Variance']  # update if different


    for i in range(len(feature_names)):
        plt.figure(figsize=(8, 5))
        plt.hist(X_time_feats[y == 0, i], bins=50, alpha=0.6, label='Left', color='blue')
        plt.hist(X_time_feats[y == 1, i], bins=50, alpha=0.6, label='Right', color='red')
        plt.title(f'Feature {i + 1}: {feature_names[i]}')
        plt.xlabel('Value')
        plt.ylabel('Number of Trials')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return X_time_feats

def spectral_entropy_from_psd(freqs, psd_trials):
    """
    Compute spectral entropy for each trial from PSD data.

    Args:
      freqs: 1D array of frequency bins from Welch (e.g., ff3)
      psd_trials: 2D array, shape (n_trials, n_freqs), PSD values from Welch

    Returns:
      entropy_vals: 1D array of spectral entropy per trial
    """
    entropy_vals = []
    for psd in psd_trials:
        psd_norm = psd / np.sum(psd)  # Normalize to get a probability distribution
        ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  # Shannon entropy
        entropy_vals.append(ent)
    return np.array(entropy_vals)


def spectral_entropy(ff3, pxx_left3, pxx_right3, ff4, pxx_left4, pxx_right4):
    entropy_left3 = spectral_entropy_from_psd(ff3, pxx_left3)  # C3 left trials
    entropy_right3 = spectral_entropy_from_psd(ff3, pxx_right3)  # C3 right trials

    entropy_left4 = spectral_entropy_from_psd(ff4, pxx_left4)  # C4 left trials
    entropy_right4 = spectral_entropy_from_psd(ff4, pxx_right4)  # C4 right trials

    # For left trials, stack features from both channels (n_trials_left, 2)
    X_left = np.column_stack((entropy_left3, entropy_left4))

    # For right trials, stack features from both channels (n_trials_right, 2)
    X_right = np.column_stack((entropy_right3, entropy_right4))

    # Combine all trials vertically (total_trials, 2)
    X_spectral_entropy = np.vstack((X_left, X_right)) 

    # Create labels vector (0 for left, 1 for right)
    y = np.array([0] * X_left.shape[0] + [1] * X_right.shape[0])

    # Split features by class
    entropy_C3_left = X_left[:, 0]
    entropy_C3_right = X_right[:, 0]

    entropy_C4_left = X_left[:, 1]
    entropy_C4_right = X_right[:, 1]

    # Plot histograms
    plt.figure(figsize=(12, 5))

    # Histogram 1: Entropy C3
    plt.subplot(1, 2, 1)
    plt.hist(entropy_C3_left, bins=50, alpha=0.6, label='Left', color='blue')
    plt.hist(entropy_C3_right, bins=50, alpha=0.6, label='Right', color='red')
    plt.title('Feature 1: Entropy C3')
    plt.xlabel('Entropy Value')
    plt.ylabel('Number of Trials')
    plt.legend()

    # Histogram 2: Entropy C4
    plt.subplot(1, 2, 2)
    plt.hist(entropy_C4_left, bins=50, alpha=0.6, label='Left', color='blue')
    plt.hist(entropy_C4_right, bins=50, alpha=0.6, label='Right', color='red')
    plt.title('Feature 2: Entropy C4')
    plt.xlabel('Entropy Value')
    plt.ylabel('Number of Trials')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return entropy_left3, entropy_right3, entropy_left4, entropy_right4

def create_feature_vector(time_segment, frequency_bands,train_left, train_right, fs):
    channel_3 = {}
    channel_4 = {}
    total_feature = len(channel_3) * len(frequency_bands) + 1

    for time_segment in time_segment:
        """Each output contains the power spectrum for left and right hand movements for each channel.
        as well as the frequency vector.
        {'segment': (ff, psd_left, psd_right)}"""

        print(f"Calculating power spectrum for segment: {time_segment}")
        output_3 = calculate_power_spectrum(train_left,train_right,channel=0,fs=fs,time_segment=time_segment)
        output_4 = calculate_power_spectrum(train_left,train_right,channel=1,fs=fs,time_segment=time_segment)
        channel_3[time_segment] = output_3  # Store the output for C3 channel
        channel_4[time_segment] = output_4

    #For spectral entropy features
    ff3, psd_left3, psd_right3 = calculate_power_spectrum(train_left,train_right,channel=0,fs=fs,time_segment=(2.15, 6))  # C3 channel
    ff4, psd_left4, psd_right4 = calculate_power_spectrum(train_left,train_right,channel=1,fs=fs,time_segment=(2.15, 6))  # C4 channel
    entropy_left3, entropy_right3, entropy_left4, entropy_right4 = spectral_entropy(ff3, psd_left3, psd_right3,ff4, psd_left4, psd_right4)
   
    
    # Add spectral entropy features to total feature count


    
    print("\nCalculating features for C3 and C4 channels")
    print("Number of segments:", len(channel_3))
    print("Number of frequency bands:", len(frequency_bands))
    print("In total there are", len(channel_3) * len(frequency_bands), "feature vectors to calculate. for each channel.\n")
    total_feature = len(channel_3) * len(frequency_bands)  # Total number of feature vectors to calculate
    channel_3_features = {}
    channel_4_features = {}
    #Channel 3 features
    ch3_feature_list_left = []
    ch3_feature_list_right = []
    

    #np.empty((2,2,64,)) # shape (channels,hand,trials,features)
    X_time_left_ch3, X_time_right_ch3 = time_feature(train_left,train_right,channel=0).reshape(2,64,2)  # shape: (n_trials, 2)
    X_time_left_ch4, X_time_right_ch4 = time_feature(train_left,train_right,channel=1).reshape(2,64,2)  # shape: (n_trials, 2)
    total_feature += X_time_left_ch3.shape[1]  # Add time features to total feature count
    
    for seg, (ff3, psd_left3, psd_right3) in channel_3.items():
        ff, psd_left, psd_right = (ff3, psd_left3, psd_right3)
        # Calculate features for each frequency band
        for band in frequency_bands:
            feature_left = psd_feature(ff, psd_left, frequency_band=band)
            feature_right = psd_feature(ff, psd_right, frequency_band=band)
            ch3_feature_list_left.append(feature_left[0])
            ch3_feature_list_right.append(feature_right[0])

    channel_3_features['left'] = ch3_feature_list_left
    channel_3_features['right'] = ch3_feature_list_right

    #Channel 4 features
    ch4_feature_list_left = []
    ch4_feature_list_right = []
    for seg, (ff4, psd_left4, psd_right4) in channel_4.items():
        ff, psd_left, psd_right = (ff4, psd_left4, psd_right4)

        # Calculate features for each frequency band
        for band in frequency_bands:
            feature_left = psd_feature(ff, psd_left, frequency_band=band)
            feature_right = psd_feature(ff, psd_right, frequency_band=band)
            ch4_feature_list_left.append(feature_left[0])
            ch4_feature_list_right.append(feature_right[0])

    channel_4_features['left'] = ch4_feature_list_left
    channel_4_features['right'] = ch4_feature_list_right

    assert len(ch3_feature_list_left) == len(ch3_feature_list_right), "Left and right feature lists must have the same length."
    assert len(ch4_feature_list_left) == len(ch4_feature_list_right), "Left and right feature lists must have the same length."
    print("Length of feature vectors for C3 channel for each hand:", len(ch3_feature_list_left))
    print("Length of feature vectors for C4 channel for each hand:", len(ch4_feature_list_right))
    print("\nFeature vectors calculated successfully.\n")

    #Conctate everything: 
    channel_3_left = np.concatenate([np.array(ch3_feature_list_left).T, X_time_left_ch3,np.array([entropy_left3]).T], axis=1)  # shape: (n_trials, n_features_total)
    channel_3_right = np.concatenate([np.array(ch3_feature_list_right).T, X_time_right_ch3,np.array([entropy_right3]).T], axis=1)
    channel_4_left = np.concatenate([np.array(ch4_feature_list_left).T, X_time_left_ch4,np.array([entropy_left4]).T], axis=1)  # shape: (n_trials, n_features_total)
    channel_4_right = np.concatenate([np.array(ch4_feature_list_right).T, X_time_right_ch4,np.array([entropy_right4]).T], axis=1)

    left = np.vstack([channel_3_left,channel_4_left]) #(n_trials_left_ch3 + n_trials_right_ch4, n_features_total)
    right = np.vstack([channel_3_right,channel_4_right]) #(n_trials_left_ch3 + n_trials_right_ch4, n_features_total)


    return  left, right  # Return the feature vectors for C3 and C4 channels for left and right hand movements

def calculate_PCA(features_left, features_right, n_components=3):
    """
    Calculates PCA on the combined C3 and C4 features for LEFT and RIGHT trials.

    Args:
        features_channel3: dict with keys 'left' and 'right' (numpy arrays)
        features_channel4: dict with keys 'left' and 'right' (numpy arrays)
        n_components: number of PCA components to keep

    Returns:
        X_pca: PCA-transformed features (n_trials, n_components)
        y: class labels (0 for LEFT, 1 for RIGHT)
        pca: fitted PCA object
    """


    X = np.concatenate((features_left, features_right),axis=0) # shape: (n_trials_total, n_features_total)            
    y = np.array([0]*len(features_left) + [1]*len(features_right))  # 0 = LEFT, 1 = RIGHT

    # Normalize features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, pca


#TODO:Remember to plot the channel in subplots and not separately
def main():
    #plot_EEG(train_left, fs, 'Training Data',(0,'C3-Left'))
    #plot_EEG(train_left, fs, 'Training Data',(1,'C4-Left'))
    #plot_EEG(train_right, fs, 'Training Data',(0,'C3-Right'))
    #plot_EEG(train_right, fs, 'Training Data',(1,'C4-Right'))


    """From Spectrogram"""
    sg3_info = create_spectrogram(train_left,train_right, fs, channel=0)  # C3 channel
    sg4_info = create_spectrogram(train_left,train_right, fs, channel=1)  # C4 channel

    data_sg = ['baseline_left'] #data_sg can be a subset of the keys in sg3_info and sg4_info (i.e. 'left', 'right','baseline_left','baseline_right' ,'difference')
    plot_spectrogram(sg3_info, data_sg,channel=3,trial=0 ,mean=True)  # C3 channel
    #plot_spectrogram(sg4_info, data_sg,channel=4,trial=0 ,mean=True)  # C4 channel

    #TODO: Compare the power spectra of both classes. Are there any frequency bands that seem useful for separating the classes?


if __name__ == "__main__":
    #main()
    #Power Spectrum divided into time segments (1.5 seconds)
    segment = [(1.5, 3.0), (3.0, 4.5), (4.5, 6)]  # Define time segments for analysis
    frequency_bands = [(6,12), (12,15), (15,20),(20,25) ,(25,30)]  # Define frequency bands for analysis

    left_features, right_feature = create_feature_vector(segment, frequency_bands, train_left, train_right, fs)

    X_pca, y, pca = calculate_PCA(left_features, right_feature, n_components=2)
    plot_PCA(X_pca, y)
