import numpy as np

from classification import classification
from features import features
from motor_imagery_dataset import MotorImageryDataset
from power_spectra import power_spectra
from visualization import visualization


def load_test_data():
    with open(r'motor_imagery_test_data.npy', 'rb') as f:  # TODO - add the file path!
        labels_name = np.load(f, allow_pickle=True)


# Your code goes here

# TODO:Remember to plot the channel in subplots and not separately
# TODO: Compare the power spectra of both classes.
#  Are there any frequency bands that seem useful for separating the classes?
def main():
    # PART 1:
    train_dataset = MotorImageryDataset('motor_imagery_train_data.npy')
    visualization(train_dataset)

    # PART 2:
    power_spectra(train_dataset)

    # PART 3:
    left_features, right_features, X_pca, y_pca = features(train_dataset)

    X = np.concatenate((left_features, right_features), axis=0)
    y = np.array([0]*len(left_features) + [1]*len(right_features))  # 0 = LEFT, 1 = RIGHT

    # PART 4:
    classification(X_pca, y_pca, X, y)


if __name__ == "__main__":
    main()
