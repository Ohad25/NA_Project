import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif, RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cross_validate_LDA(X_train_val, y_train_val, k=10):
    """Performs k-fold cross-validation using LDA on the training data and returns the model and scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_val)

    lda = LinearDiscriminantAnalysis()
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(lda, X_scaled, y_train_val, cv=cv)

    print(f"{k}-Fold CV Accuracy on training set: {np.mean(scores)*100:.2f} ± {np.std(scores)*100:.2f}%")
    lda.fit(X_scaled, y_train_val)
    return lda, scaler


def train_and_test_on_holdout(X_test, y_test, lda, scaler):
    """Trains LDA on the full training data and evaluates accuracy on a held-out test set."""
    X_test_scaled = scaler.transform(X_test)
    y_test_prediction = lda.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_prediction)

    print(f"Test Accuracy on held-out set: {test_acc * 100:.2f}%")
    return test_acc


def try_parameters(X, y):
    """Run a parameter sweep with PCA, ANOVA, and RFE to evaluate LDA performance."""
    results = feature_reduction_search(X, y, max_k=10)
    results_df = pd.DataFrame(results)

    plot_parameter_search(results_df)
    return results_df


def feature_reduction_search(X, y, max_k=40):
    """Search across different feature reduction methods and dimensions to evaluate accuracy."""
    methods = ['pca', 'anova', 'rfe']
    results = []

    max_k = min(max_k, X.shape[1])
    for method in methods:
        for k in range(1, max_k + 1):
            try:
                acc, std = evaluate_pipeline(X, y, method, k)
                results.append({
                    'method': method,
                    'features': k,
                    'accuracy': acc,
                    'std': std
                })
            except Exception as e:
                print(f"Skipped {method} with k={k}: {e}")

    best_result = sorted(results, key=lambda x: x['accuracy'], reverse=True)[0]
    print(f"\nBest result:")
    print(f"Method: {best_result['method']}, Features: {best_result['features']}")
    print(f"Accuracy: {best_result['accuracy'] * 100:.2f}% ± {best_result['std'] * 100:.2f}%")
    return results


def evaluate_pipeline(X, y, reducer, k):
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

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('reduce', reducer_step),
        ('lda', LinearDiscriminantAnalysis())
    ])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv)
    return scores.mean(), scores.std()


def plot_parameter_search(results_df):
    """Plot accuracy (± std) vs. number of features/components for all reduction methods."""
    plt.figure(figsize=(8, 5))

    for method in results_df['method'].unique():
        df = results_df[results_df['method'] == method]
        plt.errorbar(df['features'], df['accuracy'], yerr=df['std'],
                     label=method.upper(), capsize=3, lw=1.5)

    plt.axhline(0.88, ls='--', lw=1, label='88 % threshold')
    plt.xlabel('Number of retained dimensions / features')
    plt.ylabel('10-fold CV accuracy')
    plt.title('Accuracy vs. Feature Count for PCA / ANOVA / RFE')
    plt.legend()
    plt.tight_layout()
    plt.show()


def classification(X_pca, y_pca, X=None, y=None):
    """Runs LDA classification on preprocessed data and performs parameter search and visualization
    if original raw features are provided."""
    rnd_state = random.randint(0, 10000)
    print(f"\nUsing random state: {rnd_state} for train-test split")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_pca, y_pca, test_size=0.1, stratify=y_pca, random_state=rnd_state
    )

    lda, scaler = cross_validate_LDA(X_train_val, y_train_val, k=10)
    train_and_test_on_holdout(X_test, y_test, lda, scaler)

    if X is not None and y is not None:
        try_parameters(X, y)
