import numpy as np
from classification import classification
from features import features
from motor_imagery_dataset import MotorImageryDataset
from power_spectra import power_spectra
from visualization import visualization
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif, RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_best_method(X_data, y_data, reducer, k):
    """Build pipeline with LDA and given reducer (PCA, ANOVA, RFE)"""
    if reducer == 'pca':
        reducer_step = PCA(n_components=k, random_state=0)
    elif reducer == 'anova':
        reducer_step = SelectKBest(score_func=f_classif, k=k)
    elif reducer == 'rfe':
        # RFE needs a model with coef_ or feature_importances_
        selector_model = LogisticRegression(max_iter=1000, solver='liblinear')
        reducer_step = RFE(estimator=selector_model, n_features_to_select=k, step=1)
    else:
        raise ValueError("Reducer must be 'pca', 'anova' or 'rfe'")
    if reducer == 'anova':
        reducer = reducer_step.fit(X_data, y_data)
        input = reducer.transform(X_data)
    else:
        scaler = StandardScaler()
        norm_X = scaler.fit_transform(X_data)
        reducer = reducer_step.fit(norm_X, y_data)
        input = reducer.transform(norm_X)

    lda = LinearDiscriminantAnalysis()
    lda.fit(input, y_data)

    return scaler, reducer, lda


def test(test_ds, scaler, method, lda):
    test_ds_scaled = scaler.transform(test_ds)
    test_ds_reduced = method.transform(test_ds_scaled)
    y_pred = lda.predict(test_ds_reduced)
    return y_pred


# Your code goes here

# TODO:Remember to plot the channel in subplots and not separately
# TODO: Compare the power spectra of both classes.
#  Are there any frequency bands that seem useful for separating the classes?
def main():
    # PART 1:
    train_dataset = MotorImageryDataset('motor_imagery_train_data.npy')
    train_dataset.reduce_baseline(1)
    visualization(train_dataset)

    # PART 2:
    power_spectra(train_dataset)

    # PART 3:
    train_features = features(train_dataset)
    left_features, right_features = train_features
    X = np.concatenate((left_features, right_features), axis=0)
    y = np.array([0] * len(left_features) + [1] * len(right_features))  # 0 = LEFT, 1 = RIGHT

    # PART 4:
    best_method = classification(X, y)
    method, k = best_method['method'], best_method['features']

    # Training + Testing
    scaler, reducer, lda = train_best_method(X, y, method, k)
    test_class = MotorImageryDataset('motor_imagery_test_data.npy', train_mode=False)
    test_class.reduce_baseline(1, train_mode=False)

    c3_features, c4_features = features(test_class)
    # Concatenate left and right features for testing
    ts = np.concatenate((c3_features, c4_features), axis=1)

    y_pred = test(ts, scaler, reducer, lda)
    print(f"Using {method} with {k} features, the test predictions are:\n {y_pred}")

    # Save to disk
    np.save('y_pred.npy', y_pred)
    print("Saved predictions to y_pred.npy")


if __name__ == "__main__":
    main()
