import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from compute_cost import CLASSES as TARGET_CLASSES
from compute_cost import (
    aggregate_targets,
    get_ground_truth_df,
    get_segment_prediction_df,
    check_dataframe,
    total_cost
)

from data_split import read_files, load_split
from evaluation_functions import flatten_for_framewise_classification, evaluate_classifiers, evaluate_cost
from file_locations import DATASET_PATH

def train_logistic_regression(
    X_train: list[np.ndarray],
    Y_train: dict[str, list[np.ndarray]],
    classes: list[str]
) -> dict[str, callable]:
    """
    Trains one scaler+logistic-regression per class and returns a dict of
    inference functions. Each function takes a (T, D) feature array and
    returns a (T,) array of {0,1} predictions.
    """
    inference_funcs = {}
    for cls in classes:
        # prepare frame-wise training data
        X_tr, y_tr = flatten_for_framewise_classification(X_train, Y_train[cls])

        # fit scaler and model
        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ).fit(X_tr_scaled, y_tr)

        # define and store the joined inference function
        def make_inference(scaler, clf):
            return lambda x: clf.predict(scaler.transform(x))

        inference_funcs[cls] = make_inference(scaler, clf)

    return inference_funcs

if __name__ == '__main__':
    X_train, Y_train, train_files = load_split("train")
    X_test, Y_test, test_files = load_split("test")
    X_val, Y_val, val_files = load_split("val")

    lr_inference_funcs = train_logistic_regression(
        X_train, Y_train, TARGET_CLASSES
    )

    with open("lr_inference_funcs.pkl", "wb") as f:
        pickle.dump(lr_inference_funcs, f)

    val_metrics = evaluate_classifiers(
        classes=TARGET_CLASSES,
        X_val=X_val,
        Y_val=Y_val,
        inference_funcs=lr_inference_funcs
    )

    df = pd.DataFrame(val_metrics).T.round(3)
    df.columns = ["BAcc", "Precision", "Recall", "F1"]
    print(tabulate(df, headers='keys', tablefmt='github'))

    # inference_funcs from train_logistic_regression_inference(...)
    total, breakdown = evaluate_cost(
        val_files=val_files,
        dataset_path=DATASET_PATH,
        classes=TARGET_CLASSES,
        X_val=X_val,
        inference_funcs=lr_inference_funcs
    )

    df = pd.DataFrame({cls: {"Avg. Cost per minute": round(m["cost"], 4)} for cls, m in breakdown.items()}).T
    print(f"Total average cost per minute: {total:.4f}\n")
    print(tabulate(df, headers="keys", tablefmt="github"))