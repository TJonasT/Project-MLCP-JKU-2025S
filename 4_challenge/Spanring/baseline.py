import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from compute_cost import CLASSES as TARGET_CLASSES
from compute_cost import (
    aggregate_targets,
    get_ground_truth_df,
    get_segment_prediction_df,
    check_dataframe,
    total_cost
)

from data_split import load_split
from evaluation_functions import flatten_for_framewise_classification, evaluate_classifiers, evaluate_cost
from file_locations import DATASET_PATH

DATA_SUBSAMPLE = 3000  # works with available RAM in Colab

def baseline_most_frequent(
    Y_train: dict[str, list[np.ndarray]],
    classes: list[str]
) -> dict[str, callable]:
    """
    Returns inference functions that always predict each class’s majority label.
    """
    inference_funcs = {}
    for cls in classes:
        all_frames = np.concatenate(Y_train[cls])
        most_freq_label  = int(np.mean(all_frames) >= 0.5)
        # inference func ignores features, just returns most frequent label per frame
        inference_funcs[cls] = lambda x, ml=most_freq_label: np.full(x.shape[0], ml, dtype=int)
    return inference_funcs

if __name__ == '__main__':
    X_train, Y_train, train_files = load_split("train")
    X_test, Y_test, test_files = load_split("test")
    X_val, Y_val, val_files = load_split("val")

    # 1) Create baseline’s inference functions
    bl_inference_funcs = baseline_most_frequent(Y_train, TARGET_CLASSES)

    # metrics for most-frequent label baseline
    val_metrics = evaluate_classifiers(
        classes=TARGET_CLASSES,
        X_val=X_val,
        Y_val=Y_val,
        inference_funcs=bl_inference_funcs
    )

    df = pd.DataFrame(val_metrics).T.round(3)
    df.columns = ["BAcc", "Precision", "Recall", "F1"]
    print(tabulate(df, headers='keys', tablefmt='github'))

    # cost for most-frequent label baseline
    total, breakdown = evaluate_cost(
        val_files=val_files,
        dataset_path=DATASET_PATH,
        classes=TARGET_CLASSES,
        X_val=X_val,
        inference_funcs=bl_inference_funcs
    )

    df = pd.DataFrame({cls: {"Avg. Cost per minute": round(m["cost"], 4)} for cls, m in breakdown.items()}).T
    print(f"Total average cost per minute: {total:.4f}\n")
    print(tabulate(df, headers="keys", tablefmt="github"))