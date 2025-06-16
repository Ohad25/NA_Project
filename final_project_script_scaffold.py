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
from plot_creation import plot_power_spectrum, plot_spectogram
#Hyperparameters
start_fixation = 2.15 #seconds - When Beep  was heard
end_fixation = 6  # seconds - End of imagination

with open(r'C:\Users\user\Desktop\Neuron\Task-5\motor_imagery_train_data.npy', 'rb') as f:  # TODO - add the file path!
    train_data = np.load(f,allow_pickle=True)
    train_data = train_data[:,:,:2] #Last dimension is not needed accrodint to pdf
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
def calculate_power_spectrum(data_left,data_right,channel:int, fs:int):
    """Calculates the power spectrum of the EEG data using Welch's method.
    This function calculates for C3 and C4 channels.
    args: 
    data: devided into data_left and data_right numpy array of shape (n_trials, n_samples, n_channels) 
    where data_left contains left hand movement data and data_right contains right hand movement data
    channel: int, channel index to analyze (0 for C3, 1 for C4)
    fs: sampling frequency of the EEG data
    plot_limit: tuple containing the start and end of Hz range to plot the power spectrum
    
    Returns: tuple of (frequencies, power_spectrum) for left and right hand movements."""
    trial_start = int(start_fixation * fs)
    trial_end = int(end_fixation * fs)

    signal_left = data_left[:, trial_start:trial_end, channel]  # C3 channel
    signal_right = data_right[:,trial_start:trial_end, channel]  # C4 channel

    
    ff_left, pxx_left = welch(signal_left, fs, nperseg=256,noverlap=128,axis=1)
    ff_right, pxx_right = welch(signal_right, fs, nperseg=256,noverlap=128,axis=1) #ff4 not needed because its equal to ff3

    return (ff_left,pxx_left), (ff_right,pxx_right)


def create_spectogram(data1,data2, fs, channel,baseline=1):
    """Creates a spectrogram of the EEG data for a specific channel.
    data1 = EEG data of imagining left hand
    data2 = EEG data imagining right hand
    WARNING:TODO:If we substract we need to convert it to dB first so it will be in log scale.
    
    returns: tuple of (frequencies, times, spectrogram) for left and right hand movements."""

    signal_left = data1[:,:, channel]  # Select the specified channel
    signal_right = data2[:,:, channel]  # Select the specified channel

    f, t, Sxx_left = spectrogram(signal_left, fs=fs, nperseg=256,axis=1)  # Calculate the spectrogram for left hand movement
    f2, t2, Sxx_right = spectrogram(signal_right, fs=fs, nperseg=256,axis=1)  # Calculate the spectrogram for right hand movement

    output = {}
    #Sxx_left = 10* np.log10(Sxx_left + 1e-10)  # Convert to dB, add small value to avoid log(0)
    #Sxx_right = 10* np.log10(Sxx_right + 1e-10)  # Convert to dB, add small value to avoid log(0)

    #Spectogram for substraction of left and right channels

    Sxx_diff = Sxx_left/Sxx_right # Calculate the difference between left and right spectrograms
        

    #Spectogram for left and right channels  #TODO: mean over all spectogram.  

    baseline_mask = t <= baseline #The assisngment says to use the first second as baseline but we start at 1 seconds
    #Average across time baseline period:
    baseline_Sxx_left = np.mean(Sxx_left[:,:,baseline_mask])  # Select the first second of the spectrogram
    baseline_Sxx_right = np.mean(Sxx_right[:,:,baseline_mask])  # Select the first second of the spectrogram\

    output['left'] = (f, t, Sxx_left)
    output['right'] = (f2, t2, Sxx_right)
    output['difference'] = (f, t, Sxx_diff)
    output['baseline_left'] = (f, t, Sxx_left - baseline_Sxx_left)
    output['baseline_right'] = (f2, t2, Sxx_right - baseline_Sxx_right)


    return output




def psd_feature(ff,power, fs,f_b):
    """
    data1: signal of left hand movement (Can be raw signal or PSD or spectogram)
    data2: signal of right hand movement "-"
    fs: sampling frequency
    channel: channel index to analyze (0 for C3, 1 for C4)
    frequancy_band1: tuple of (start_freq, end_freq) for the first frequency band in Hz
    st_fix: start fixation time in seconds
    en_fix: end fixation time in seconds
    """

    #TODO: Check if IAF needed. Start without it and if needed add it later.

    left_mask = (ff >= f_b[0]) & (ff <= f_b[1])  # Mask for the frequency band

    
    #Calculate delta_f for frequency resolution (Right and left hand movement should be the same)
    delta_f_left = ff[1] - ff[0]  # Frequency resolution

    #Calcualte power: equivalent to the area under the curve in the frequency band
    # For left hand movement:
    power_left = np.sum(power[:,left_mask],axis=1) * delta_f_left  # Area under the curve in the frequency band
    feature_vector_left = np.array([power_left])
    # For right hand movement:


    return feature_vector_left

def spectogram_feature(signal, fs, f_b):
    pass



#TODO:Remeber to plot the channel in subplots and not seperatly
def main():
    #plot_EEG(train_left, fs, 'Training Data',(0,'C3-Left'))
    #plot_EEG(train_left, fs, 'Training Data',(1,'C4-Left'))
    #plot_EEG(train_right, fs, 'Training Data',(0,'C3-Right'))
    #plot_EEG(train_right, fs, 'Training Data',(1,'C4-Right'))

    psd_left3_tuple, psd_right3_tuple = calculate_power_spectrum(train_left,train_right,channel=0,fs=fs) # C3 channel return (ff3, pxx_left), (ff4, pxx_right)
    psd_left4_tuple, psd_right4_tuple = calculate_power_spectrum(train_left,train_right,channel=1,fs=fs) # C4 channel return (ff3, pxx_left), (ff4, pxx_right)

    ff3, pxx_left3 = psd_left3_tuple
    ff4, pxx_left4 = psd_left4_tuple
    _, pxx_right3 = psd_right3_tuple
    _, pxx_right4 = psd_right4_tuple

    plot_power_spectrum(ff3,pxx_left3,pxx_right3,trial_num=0,channel=0, plot_limit=(0, 30),mean=True)  # C3 channel
    plot_power_spectrum(ff4,pxx_left4,pxx_right4, trial_num=0,channel=1, plot_limit=(0, 30),mean=True)  # C4 channel

    #Channel 3 features 
    feature_left = psd_feature(ff3,pxx_left3,fs, (15, 20))
    print("Feature vector for left hand movement:", feature_left)

    feature_right = psd_feature(ff3,pxx_right3,fs, (15, 20))
    print("Feature vector for right hand movement:", feature_right)

    #Channel 4 features
    feature_left4 = psd_feature(ff4,pxx_left4,fs, (7, 12))
    print("Feature vector for left hand movement C4 channel:", feature_left4)

    feature_right4 = psd_feature(ff4,pxx_right4,fs, (7, 12))
    print("Feature vector for right hand movement C4 channel:", feature_right4)



if __name__ == "__main__":
    #main()

    psd_left3_tuple, psd_right3_tuple = calculate_power_spectrum(train_left,train_right,channel=0,fs=fs) # C3 channel return (ff3, pxx_left), (ff4, pxx_right)
    psd_left4_tuple, psd_right4_tuple = calculate_power_spectrum(train_left,train_right,channel=1,fs=fs) # C4 channel return (ff3, pxx_left), (ff4, pxx_right)

    # Extract psd information from each hand and each channel
    ff3, pxx_left3 = psd_left3_tuple
    ff4, pxx_left4 = psd_left4_tuple
    _, pxx_right3 = psd_right3_tuple
    _, pxx_right4 = psd_right4_tuple


    plot_power_spectrum(ff3,pxx_left3,pxx_right3,trial_num=0,channel=0, plot_limit=(0, 30),mean=True)  # C3 channel
    plot_power_spectrum(ff4,pxx_left4,pxx_right4, trial_num=0,channel=1, plot_limit=(0, 30),mean=True)  # C4 channel



    sg3_info = create_spectogram(train_left,train_right, fs, channel=0)  # C3 channel
    sg4_info = create_spectogram(train_left,train_right, fs, channel=1)  # C4 channel

    data_sg = ['baseline_left','baseline_right','left'] #data_sg can be a subset of the keys in sg3_info and sg4_info (e.i. 'left', 'right','baseline_left','baseline_right' ,'difference')
    plot_spectogram(sg3_info, data_sg,channel=3,trial=0 ,mean=True)  # C3 channel

    #TODO: Compare the power spectra of both classes. Are there any frequency bands that seem useful for separating the classes?

    
