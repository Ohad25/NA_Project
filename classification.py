import numpy as np
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def cross_validate_LDA(X_train_val, y_train_val, k=10):
    """Performs k-fold cross-validation using LDA on the training data and returns the model and scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_val)

    lda = LinearDiscriminantAnalysis()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_val_score(lda, X_scaled, y_train_val, cv=kf)

    print(f"{k}-Fold CV Accuracy on training set: {np.mean(scores)*100:.2f} ± {np.std(scores)*100:.2f}%")

    return lda, scaler


def train_and_test_on_holdout(X_train_val, y_train_val, X_test, y_test, lda):
    """Trains LDA on the full training data and evaluates accuracy on a held-out test set."""
    lda.fit(X_train_val, y_train_val)

    y_test_prediction = lda.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_prediction)

    print(f"Test Accuracy on held-out set: {test_acc * 100:.2f}%")
    return test_acc


def classification(X_pca, y):
    """Performs repeated train-test splits and runs LDA classification with cross-validation and test evaluation."""
    for j in range(1000):
        rnd_state = random.randint(0, 10000)
        print(f"Iteration {j + 1}:")
        print(f"Random state for train-test split: {rnd_state}")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_pca, y, test_size=0.1, stratify=y, random_state=rnd_state
        )

    lda, scaler = cross_validate_LDA(X_train_val, y_train_val, k=10)

    train_and_test_on_holdout(X_train_val, y_train_val, X_test, y_test, lda)
