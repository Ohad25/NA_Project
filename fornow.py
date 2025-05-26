import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random 


with open(r'C:\Users\user\Desktop\Neuron\Task-4\motor_imagery_train_data.npy', 'rb') as f:  # TODO - add the file path!
    train_data = np.load(f,allow_pickle=True)
    labels = np.load(f,allow_pickle=True)
    labels_name = np.load(f,allow_pickle=True)
    fs = np.load(f, allow_pickle=True)


with open(r'C:\Users\user\Desktop\Neuron\Task-4\motor_imagery_test_data.npy', 'rb') as f:  # TODO - add the file path!
    labels_name = np.load(f,allow_pickle=True)

# Your code goes here

#Subplots QUESTION1:
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




plot_EEG(train_data, fs, 'Training Data',(0,'C3'))
plot_EEG(train_data, fs, 'Training Data',(1,'C4'))
        
