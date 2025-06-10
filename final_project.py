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
def calculate_power_spectrum(data_left,data_right,channel:int, fs:int,plot_limit:tuple):
    """Calculates the power spectrum of the EEG data using Welch's method.
    This function calculates for C3 and C4 channels.
    args: 
    data: devided into data_left and data_right numpy array of shape (n_trials, n_samples, n_channels) 
    where data_left contains left hand movement data and data_right contains right hand movement data
    channel: int, channel index to analyze (0 for C3, 1 for C4)
    fs: sampling frequency of the EEG data
    plot_limit: tuple containing the start and end of Hz range to plot the power spectrum"""
    trial_start = int(start_fixation * fs)
    trial_end = int(end_fixation * fs)
    signal_left = data_left[:, trial_start:trial_end, channel]  # C3 channel
    signal_right = data_right[:,trial_start:trial_end, channel]  # C4 channel

    signal_left_mean = np.mean(signal_left,axis=0)
    signal_right_mean = np.mean(signal_right,axis=0)
    

    ff3, pxx_left = welch(signal_left_mean, fs, nperseg=256,noverlap=128)
    _, pxx_right = welch(signal_right_mean, fs, nperseg=256,noverlap=128) #ff4 not needed because its equal to ff3
    
    strat_freq = plot_limit[0]
    end_freq = plot_limit[1]
    mask = (ff3 >= strat_freq) & (ff3 <= end_freq) #Take only the frequencies in the specified range
    ff3_masked = ff3[mask]
    pxx_left = pxx_left[mask]
    pxx_right = pxx_right[mask]

    std_pxx_left = np.std(pxx_left, axis=0)
    std_pxx_right = np.std(pxx_right, axis=0)

    # Plotting the power spectrum for C3 and C4 channels

    plt.figure(figsize=(12, 6))

    plt.plot(ff3_masked, pxx_left, label='Left Hand', color='blue')
    plt.plot(ff3_masked, pxx_right, label='Right Hand', color='orange')
    plt.fill_between(ff3_masked, pxx_left - std_pxx_left, pxx_left + std_pxx_left, alpha=0.3)
    plt.fill_between(ff3_masked, pxx_right - std_pxx_right, pxx_right + std_pxx_right, alpha=0.3)


    plt.title(f'Power Spectrum of EEG Data channel {channel+3}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid()
    plt.show()

    return pxx_left, pxx_right


def create_spectogram(data1,data2, fs, channel,difference=False,baseline=False):
    """Creates a spectrogram of the EEG data for a specific channel.
    data1 = EEG data of imagining left hand
    data2 = EEG data imagining right hand
    WARNING:TODO:If we substract we need to convert it to dB first so it will be in log scale."""
    #Mean over trials:
    signal_left = np.mean(data1,axis=0)
    signal_right = np.mean(data2,axis=0)
    signal_left = signal_left[:, channel]  # Select the specified channel
    signal_right = signal_right[:, channel]  # Select the specified channel

    f, t, Sxx_left = spectrogram(signal_left, fs=fs, nperseg=256)
    f2, t2, Sxx_right = spectrogram(signal_right, fs=fs, nperseg=256)

    #Sxx_left = 10* np.log10(Sxx_left + 1e-10)  # Convert to dB, add small value to avoid log(0)
    #Sxx_right = 10* np.log10(Sxx_right + 1e-10)  # Convert to dB, add small value to avoid log(0)

    #Spectogram for substraction of left and right channels
    if difference:
        Sxx_diff = Sxx_left - Sxx_right # Calculate the difference between left and right spectrograms
        data_Sxx = [(Sxx_diff,'Difference Spectrogram (left - right)')]

    #Spectogram for left and right channels  #TODO: mean over all spectogram.  
    elif baseline:    
        baseline_mask = t <= 1 #The assisngment says to use the first second as baseline but we start at 1 seconds
        #Average across time baseline period:
        baseline_Sxx_left = np.mean(Sxx_left[:,baseline_mask])  # Select the first second of the spectrogram
        baseline_Sxx_right = np.mean(Sxx_right[:,baseline_mask])  # Select the first second of the spectrogram


        difference_Sxx_left = Sxx_left - baseline_Sxx_left
        difference_Sxx_right = Sxx_right - baseline_Sxx_right
        data_Sxx = [(difference_Sxx_left,'Left (Substraction from baseline (1 [sec]))'), (difference_Sxx_right,'Right (Substraction from baseline (1 [sec]))')]

    #Baseline period
    else:
        data_Sxx = [(Sxx_left,'Left'), (Sxx_right,'Right')]

    fig, axes = plt.subplots(len(data_Sxx),1, figsize=(15, 10))

    for i ,Sxx in enumerate(data_Sxx): 
        ax = axes[i] if len(data_Sxx) > 1 else axes
        mean_Sxx, label = Sxx
        im = ax.pcolormesh(t, f, mean_Sxx, shading='gouraud', cmap='jet') #To change it to --> dB 10 * np.log10(mean_Sxx + 1e-10)
        ax.set_title(f'Spectrogram for Channel {channel} ({label})')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylim(0, 2)  # Limit the frequency range 

        cbar = fig.colorbar(im)
        cbar.set_label('power per Hz ', fontsize=12) #If you change to decibels, change this to 'dB'
    plt.tight_layout()
    plt.show()
    

def extract_features(data1,data2, fs, channel,frequancy_band1,frequancy_band2,st_fix,en_fix):
    """
    data1: signal of left hand movement
    data2: signal of right hand movement
    fs: sampling frequency
    channel: channel index to analyze (0 for C3, 1 for C4)
    frequancy_band1: tuple of (start_freq, end_freq) for the first frequency band
    frequancy_band2: tuple of (start_freq, end_freq) for the second frequency band
    st_fix: start fixation time in seconds
    en_fix: end fixation time in seconds
    """
    trial_start = int(st_fix * fs)
    trial_end = int(en_fix * fs)
    
    signal_left = data1[:, trial_start:trial_end, channel]  # C3 channel
    signal_right = data2[:, trial_start:trial_end, channel]  # C4 channel

    # Calculate the power spectrum for the left hand movement
    f_left, pxx_left = welch(signal_left, fs, nperseg=256,noverlap=128,axis=1)
    
    # Calculate the power spectrum for the right hand movement
    f_right, pxx_right = welch(signal_right, fs, nperseg=256,noverlap=128,axis=1)

    # Extract features for the specified frequency bands
    band1_mask_left = (f_left >= frequancy_band1[0]) & (f_left <= frequancy_band1[1])
    band2_mask_left = (f_left >= frequancy_band2[0]) & (f_left <= frequancy_band2[1])
    
    band1_mask_right = (f_right >= frequancy_band1[0]) & (f_right <= frequancy_band1[1])
    band2_mask_right = (f_right >= frequancy_band2[0]) & (f_right <= frequancy_band2[1])

    feature_vector_left = np.array([np.mean(pxx_left[band1_mask_left]), np.mean(pxx_left[band2_mask_left])])
    feature_vector_right = np.array([np.mean(pxx_right[band1_mask_right]), np.mean(pxx_right[band2_mask_right])])

    return feature_vector_left, feature_vector_right




#TODO:Remeber to plot the channel in subplots and not seperatly
def main():
    #plot_EEG(train_left, fs, 'Training Data',(0,'C3-Left'))
    #plot_EEG(train_left, fs, 'Training Data',(1,'C4-Left'))
    #plot_EEG(train_right, fs, 'Training Data',(0,'C3-Right'))
    #plot_EEG(train_right, fs, 'Training Data',(1,'C4-Right'))

    mean_ppxleft3, mean_pxxright3 = calculate_power_spectrum(train_left,train_right,channel=0,fs=fs,plot_limit=(0, 30))
    mean_ppxleft4, mean_pxxright4 = calculate_power_spectrum(train_left,train_right,channel=1,fs=fs,plot_limit=(0, 30))

if __name__ == "__main__":
    main()
    #Recall channel 0 is C3 and channel 1 is C4
    #Spectorgram for C3 and C4 channels left and right hand movements
    create_spectogram(train_left,train_right, fs, channel=0)  # C3 channel
    create_spectogram(train_left,train_right, fs, channel=1)  # C4 channel

    #Difference between left and right channels
    create_spectogram(train_left,train_right, fs, channel=0,difference=True)  # C3 channel
    create_spectogram(train_left,train_right, fs, channel=1,difference=True)  # C4 channel

    #Baseline period
    create_spectogram(train_left,train_right, fs, channel=0,baseline=True)  # C3 channel
    create_spectogram(train_left,train_right, fs, channel=1,baseline=True)  # C4 channel
    #TODO: Compare the power spectra of both classes. Are there any frequency bands that seem useful for separating the classes?
