"""
src/train.py

Train a classification model for the prediction of churn.
Load processed data, train a model (RandomForest) handling the unbalanced classes,
evaluate performance metric and store the model and metrics.
"""

import yaml
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --- Paths ---
# create the paths where we want to store the object and load the data
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_PATH = Path("metrics.json")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- params from .yaml file ---
def load_params(section: str) -> dict:
    """Carica una sezione specifica da params.yaml"""
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params[section]


def load_processed_data():
    # function to load data and return the X and y for train and test
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    return X_train, X_test, y_train, y_test

# Using cross validation to find the best RandomForest model
def train_rf(X_train, y_train, params) -> RandomForestClassifier:
    model=RandomForestClassifier(class_weight='balanced', max_features='sqrt',
                                 random_state=params['random_state'], n_jobs=-1, max_depth=None)
    grid_values = {'n_estimators' : params['param_grid']['n_estimators'],
                   'min_samples_split' : params['param_grid']['min_sample_split'],
                   'min_samples_leaf' : params['param_grid']['min_samples_leaf']}
    grid = GridSearchCV(model, param_grid=grid_values, scoring="f1", cv=params['cv_folds'])
    grid.fit(X_train, y_train)  

    return grid.best_estimator_, grid.best_params_

def train_logistic(X_train, y_train, params) -> LogisticRegression:
    model = LogisticRegression(l1_ratio=1, solver='liblinear', class_weight='balanced',
                               random_state=params['random_state'], penalty='deprecated')
    cs = l1_min_c(X_train, y_train, loss="log") * np.logspace(0, 2.5, 50)
    grid_values = {'C' : params['param_grid']['C']}
    grid = GridSearchCV(model, param_grid=grid_values, scoring="f1", cv=params['cv_folds'])
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_

# A function to evaluate the model
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },
    }
    return metrics

def get_feature_importance(model, feature_names, top_n=10) -> dict:
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    return {name: float(score) for name, score in feature_importance[:top_n]}

def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print("Training model...")
    params_rf = load_params('train_rf')
    params_logistic = load_params('train_logistic')
    model_rf, best_params_rf = train_rf(X_train, y_train, params_rf)
    model_logistic, best_params_log = train_logistic(X_train=X_train, y_train=y_train, params=params_logistic)
    
    print("Evaluation...")
    metrics_rf = evaluate_model(model_rf, X_test, y_test)
    metrics_rf["feature_importance_top10"] = get_feature_importance(
        model_rf, X_train.columns.tolist()
    )
    metrics_rf["parameters"] = best_params_rf
    metrics_rf = {'model_type':'Random Forest', **metrics_rf}

    metrics_logistic = evaluate_model(model_logistic, X_test, y_test)
    metrics_logistic["parameters"]= best_params_log
    metrics_logistic = {'model_type':'logistic', **metrics_logistic}
    features = model_logistic.feature_names_in_[(model_logistic.coef_!=0).tolist()[0]].tolist()
    values = model_logistic.coef_[model_logistic.coef_!=0].tolist()
    coef = dict()
    for i, value in enumerate(values):
        feature = features[i]
        coef[feature]=round(value, ndigits=4)
    metrics_logistic['coefficients different from 0']=coef
    

    print("Saving model...")
    joblib.dump(model_rf, MODELS_DIR / "model_rf.pkl")
    joblib.dump(model_logistic, MODELS_DIR / "model_logistic.pkl")

    print("Saving metrics...")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_rf, f, indent=2)
        json.dump(metrics_logistic, f, indent=2)


    print(f"Done. Results Random Forest. Accuracy: {metrics_rf['accuracy']:.3f}, "
          f"F1: {metrics_rf['f1_score']:.3f}, "
          f"ROC-AUC: {metrics_rf['roc_auc']:.3f}")


if __name__ == "__main__":
    main()