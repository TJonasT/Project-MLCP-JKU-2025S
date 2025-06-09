import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

from compute_cost import CLASSES as TARGET_CLASSES
from compute_cost import (
    aggregate_targets,
    get_ground_truth_df,
    get_segment_prediction_df,
    check_dataframe,
    total_cost
)

# Flatten: Each frame is a sample
def flatten_for_framewise_classification(X, Y_class):
    X_flat = np.concatenate(X)  # shape: (total_frames, num_features)
    Y_flat = np.concatenate(Y_class)  # shape: (total_frames,)
    return X_flat, Y_flat

def evaluate_classifiers(
    classes: list[str],
    Y_val: dict[str, list[np.ndarray]],
    X_val: list[np.ndarray] = None,
    inference_funcs: dict[str, callable] = None,
    Y_pred: dict[str, list[np.ndarray]] = None
) -> tuple[dict[str, list[np.ndarray]], dict[str, dict]]:
    """
    Evaluates per-frame binary classifiers and computes metrics per class.
    Uses either computed predictions or given inference functions.

    Args:
        classes: List of class names to evaluate.
        Y_val: Dict mapping class names to lists of ground-truth (T,) binary arrays.
        X_val: List of input feature arrays, one per validation file. Required if Y_pred not given.
        inference_funcs: Dict mapping class names to binary inference functions.
        Y_pred: Dict with precomputed predictions (same format as Y_val).

    Returns:
        metrics: Dict[class → {'balanced_accuracy', 'precision', 'recall', 'f1'}].
    """

    if Y_pred is None:
        assert inference_funcs is not None and X_val is not None, "If 'Y_pred' is not given, 'inference_funcs' \
                                                                    and 'X_val' must be given."

    Y_val_preds = {}
    metrics     = {}

    for cls in classes:
        # use predictions if given, else infer
        if Y_pred and cls in Y_pred:
            preds_per_file = Y_pred[cls]
        else:
            infer = inference_funcs[cls]
            preds_per_file = [infer(x_file) for x_file in X_val]
        Y_val_preds[cls] = preds_per_file

        # flatten to compute metrics
        y_true = np.concatenate(Y_val[cls])
        y_pred = np.concatenate(preds_per_file)

        metrics[cls] = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision":         precision_score(y_true, y_pred, zero_division=0),
            "recall":            recall_score(y_true, y_pred, zero_division=0),
            "f1":                f1_score(y_true, y_pred, zero_division=0),
        }

    return metrics

def evaluate_cost(
    val_files: list[str],
    dataset_path: str,
    classes: list[str],
    X_val: list[np.ndarray] = None,
    inference_funcs: dict[str, callable] = None,
    Y_pred: dict[str, list[np.ndarray]] = None
):
    """
    Computes segment-level cost based on predictions and ground truth.
    Uses either computed predictions or given inference functions.

    Args:
        val_files: List of filenames corresponding to X_val.
        dataset_path: Path to dataset root (used for loading ground truth).
        classes: List of class names to evaluate.
        X_val: List of input feature arrays, one per validation file. Required if Y_pred not given.
        inference_funcs: Dict mapping class names to binary inference functions.
        Y_pred: Dict with precomputed predictions (class → list of (T,) arrays).

    Returns:
        total: Total cost across all validation files.
        breakdown: Dict[class → segment-level cost].
    """

    if Y_pred is None:
        assert inference_funcs is not None and X_val is not None, "If 'Y_pred' is not given, 'inference_funcs' \
                                                                    and 'X_val' must be given."

    # 0) frame-wise predictions (per class)
    if Y_pred is None:
        Y_pred = {
            cls: [infer(x_file) for x_file in X_val]
            for cls, infer in inference_funcs.items()
        }

    # 1) restructure to filename -> class -> (T,) array
    preds_by_file = {}
    for i, fname in enumerate(val_files):
        preds_by_file[fname] = {
            cls: Y_pred[cls][i] for cls in classes
        }

    # 2) segment-level aggregation using compute_cost
    pred_df = get_segment_prediction_df(
        predictions=preds_by_file,
        class_names=classes
    )

    # 3) load & aggregate ground truth using compute_cost
    gt_df = get_ground_truth_df(val_files, dataset_path)

    # 4) sanity checks from compute_cost
    check_dataframe(pred_df, dataset_path)
    check_dataframe(gt_df, dataset_path)

    # 5) compute cost
    total, breakdown = total_cost(pred_df, gt_df)

    return total, breakdown