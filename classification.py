import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif,  RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




def cv_score(clf, X, y, n_splits=10, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores.mean(), scores.std()


def pca_sweep(X, y, max_comps=None):
    max_valid = min(X.shape[0] - X.shape[0] // 10, X.shape[1])
    if max_comps is None or max_comps > max_valid:
        max_comps = max_valid

    means, sds = [], []

    for n in range(1, max_comps + 1):
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=n, random_state=0)),
            ('lda', LinearDiscriminantAnalysis())
        ])
        try:
            mu, sd = cv_score(pipe, X, y)
            means.append(mu)
            sds.append(sd)
        except Exception as e:
            print(f"PCA sweep failed at {n} components: {e}")
            break

    return np.arange(1, len(means) + 1), np.array(means), np.array(sds)


def anova_sweep(X, y, max_feats=None):
    max_feats = min(X.shape[1], 40) if max_feats is None else min(max_feats, X.shape[1])
    means, sds = [], []

    for k in range(1, max_feats + 1):
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('k_best', SelectKBest(score_func=f_classif, k=k)),
            ('lda', LinearDiscriminantAnalysis())
        ])
        try:
            mu, sd = cv_score(pipe, X, y)
            means.append(mu)
            sds.append(sd)
        except Exception as e:
            print(f"ANOVA sweep failed at {k} features: {e}")
            break

    return np.arange(1, len(means) + 1), np.array(means), np.array(sds)




def improve_accuracy(X_raw, y):
    # X_raw is your original feature matrix (no PCA already applied!), y are labels
    pca_x, pca_mu, pca_sd = pca_sweep(X_raw, y, max_comps=15)
    anova_x, anova_mu, anova_sd = anova_sweep(X_raw, y, max_feats=15)

    plt.figure(figsize=(8, 5))
    plt.errorbar(pca_x, pca_mu, yerr=pca_sd, label='PCA', capsize=3, lw=1.5)
    plt.errorbar(anova_x, anova_mu, yerr=anova_sd, label='ANOVA', capsize=3, lw=1.5)
    plt.axhline(0.88, ls='--', lw=1, label='88 % threshold')
    plt.xlabel('Number of retained dimensions / features')
    plt.ylabel('10-fold CV accuracy')
    plt.title('Model accuracy as a function of feature count')
    plt.legend()
    plt.tight_layout()
    plt.show()


def classification(X_pca, y_pca, X, y):
    try_parameters(X, y)
    # improve_accuracy(X, y)


def evaluate_pipeline(X, y, reducer, k):
    """
    Build pipeline with LDA and given reducer (PCA, ANOVA, RFE)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
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
    """if reducer == 'pca':
        reducer_step = PCA(n_components=k, random_state=0)
        reduced = reducer_step.fit_transform(X_scaled)
    elif reducer == 'anova':
        reducer_step = SelectKBest(score_func=f_classif, k=k)
    elif reducer == 'rfe':
        # RFE needs a model with coef_ or feature_importances_
        selector_model = LogisticRegression(max_iter=1000, solver='liblinear')
        reducer_step = RFE(estimator=selector_model, n_features_to_select=k, step=1)
    else:
        raise ValueError("Reducer must be 'pca', 'anova' or 'rfe'")

    lda = LinearDiscriminantAnalysis()"""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv)
    return scores.mean(), scores.std()


def feature_reduction_search(X, y, max_k=40):
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
                print(f"⚠️ Skipped {method} with k={k}: {e}")

    best_result = sorted(results, key=lambda x: x['accuracy'], reverse=True)[0]
    print(f"\n🔍 Best result:")
    print(f"Method: {best_result['method']}, Features: {best_result['features']}")
    print(f"Accuracy: {best_result['accuracy'] * 100:.2f}% ± {best_result['std'] * 100:.2f}%")
    return results


def try_parameters(X, y):
    results = feature_reduction_search(X, y, max_k=10)
    pd.DataFrame(results).sort_values("accuracy", ascending=False).head(10)
