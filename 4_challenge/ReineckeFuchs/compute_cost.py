import argparse
import os
import math
import numpy as np
import pandas as pd

from typing import Iterable, Dict, List, Optional

CLASSES = ['Speech', 'Shout', 'Chainsaw', 'Jackhammer', 'Lawn Mower', 'Power Drill', 'Dog Bark', 'Rooster Crow', 'Horn Honk', 'Siren']

COST_MATRIX = {
    "Speech":         {"TP": 0,  "FP": 1,  "TN": 0, "FN": 5},
    "Dog Bark":       {"TP": 0,  "FP": 1,  "TN": 0, "FN": 5},
    "Rooster Crow":   {"TP": 0,  "FP": 1,  "TN": 0, "FN": 5},
    "Shout":          {"TP": 0,  "FP": 2,  "TN": 0, "FN": 10},
    "Lawn Mower":     {"TP": 0,  "FP": 3,  "TN": 0, "FN": 15},
    "Chainsaw":       {"TP": 0,  "FP": 3,  "TN": 0, "FN": 15},
    "Jackhammer":     {"TP": 0,  "FP": 3,  "TN": 0, "FN": 15},
    "Power Drill":    {"TP": 0,  "FP": 3,  "TN": 0, "FN": 15},
    "Horn Honk":      {"TP": 0,  "FP": 3,  "TN": 0, "FN": 15},
    "Siren":          {"TP": 0,  "FP": 3,  "TN": 0, "FN": 15},
}

def check_dataframe(data_frame, dataset_path):
    """
    Validates the integrity of a predictions or ground truth DataFrame.

    Parameters:
    ----------
    predictions_df : pandas.DataFrame
        A DataFrame containing model predictions or the ground truth.
        It must include columns:
        - 'filename': Name of the audio file (e.g., "xyz.wav")
        - 'onset': Onset times or frame indices
        - One column for each class in the global `CLASSES` list

    dataset_path : str
        Path to the root of the dataset directory. It must contain a
        subdirectory 'audio_features' with `.npz` files for each audio file.

    Raises:
    ------
    AssertionError:
        If any of the following checks fail:
        - The dataset or audio_features directory doesn't exist
        - The DataFrame is missing required columns
        - Expected feature files are missing
        - Number of predictions doesn't match the number of expected timesteps

    Example:
    -------
    check_dataframe(predicted_df, "MLPC2025_dataset")
    """
    audio_features_path = os.path.join(dataset_path, "audio_features")
    assert os.path.exists(dataset_path), f"Dataset path '{dataset_path}' does not exist."
    assert os.path.exists(audio_features_path), f"Audio features path '{audio_features_path}' does not exist."

    required_columns = set(CLASSES + ["filename", "onset"])
    missing_columns = required_columns - set(data_frame.columns)
    assert not missing_columns, f"Missing columns in predictions_df: {missing_columns}"

    assert ((data_frame["onset"] / 1.2) % 1).apply(lambda x: np.isclose(x, 0, atol=0.1)).all(), "Not all values are divisible by 1.2."
    assert data_frame[CLASSES].isin([0, 1]).all().all(), "Not all predictions are 0 or 1."

    for filename in data_frame["filename"].unique():
        file_id = os.path.splitext(filename)[0]
        feature_file = os.path.join(audio_features_path, f"{file_id}.npz")

        assert os.path.exists(feature_file), f"Feature file '{feature_file}' does not exist."

        embeddings = np.load(feature_file)["embeddings"]
        expected_timesteps = math.ceil(len(embeddings) / 10)
        actual_timesteps = len(data_frame[data_frame["filename"] == filename])

        assert actual_timesteps == expected_timesteps, (
            f"Mismatch in timesteps for '{filename}': expected {expected_timesteps}, found {actual_timesteps}."
        )


def total_cost(predictions_df, ground_truth_df):
    """
    Computes total cost of predictions based on a cost matrix for TP, FP, TN, and FN
    for each class in a multilabel classification problem.

    Parameters:
    ----------
    predictions_df : pandas.DataFrame
        DataFrame containing predicted binary labels (0 or 1) for each class in CLASSES.

    ground_truth_df : pandas.DataFrame
        DataFrame containing ground truth binary labels for each class in CLASSES.

    Returns:
    -------
    total_cost_value : float
        Total cost across all classes and samples.

    metrics_per_class : dict
        Dictionary with TP, FP, TN, FN counts and cost per class.
    """

    # Align rows by filename and onset
    merged = predictions_df.merge(
        ground_truth_df,
        on=["filename", "onset"],
        suffixes=("_pred", "_true"),
        how="inner",
        validate="one_to_one"
    )

    if merged.shape[0] != predictions_df.shape[0]:
        raise ValueError("Mismatch in alignment between prediction and ground truth rows")

    metrics_per_class = {}

    for cls in CLASSES:
        y_pred = predictions_df[cls].astype(int)
        y_true = ground_truth_df[cls].astype(int)

        TP = ((y_pred == 1) & (y_true == 1)).mean() * 50
        FP = ((y_pred == 1) & (y_true == 0)).mean() * 50
        TN = ((y_pred == 0) & (y_true == 0)).mean() * 50
        FN = ((y_pred == 0) & (y_true == 1)).mean() * 50

        cost = (
            COST_MATRIX[cls]["TP"] * TP +
            COST_MATRIX[cls]["FP"] * FP +
            COST_MATRIX[cls]["TN"] * TN +
            COST_MATRIX[cls]["FN"] * FN
        )

        metrics_per_class[cls] = {
            "TP": TP, "FP": FP, "TN": TN, "FN": FN, "cost": cost
        }

    return sum([metrics_per_class[c]["cost"] for c in metrics_per_class]), metrics_per_class


def aggregate_targets(arr: np.ndarray, f: int = 10) -> np.ndarray:
    """
    Aggregates frame-level ground truths into segment-level by taking the max over fixed-size chunks.

    Parameters:
    ----------
    arr : np.ndarray
        Array of shape (N, D) where N is the number of frames, D is number of classes.

    f : int
        Aggregation factor (number of frames per chunk).

    Returns:
    -------
    np.ndarray
        Aggregated labels of shape (ceil(N/f), D)
    """
    N, D = arr.shape
    full_chunks = N // f
    remainder = N % f

    # Aggregate full chunks
    aggregated = arr[:full_chunks * f].reshape(full_chunks, f, D).max(axis=1)

    # Handle leftover frames
    if remainder > 0:
        tail = arr[full_chunks * f:].max(axis=0, keepdims=True)
        aggregated = np.vstack([aggregated, tail])

    return aggregated


def get_ground_truth_df(filenames: Iterable[str], dataset_path: str) -> pd.DataFrame:
    """
    Loads and aggregates ground truth labels for an arbitrary list of files.

    Parameters:
    ----------
    filenames : Iterable[str]
        List or array of filenames (e.g., from a subset of metadata.csv) to process.

    dataset_path : str
        Path to dataset containing the 'labels/' folder with .npz files.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns: ["filename", "onset"] + CLASSES
    """
    rows = []

    for fname in filenames:
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(dataset_path, 'labels', f"{base}_labels.npz")
        assert os.path.exists(label_path), f"Missing label file: {label_path}"

        y = np.load(label_path)
        class_matrix = np.stack([y[cls].mean(-1) for cls in CLASSES], axis=1)
        aggregated = aggregate_targets(class_matrix)

        for i, row in enumerate(aggregated):
            onset = round(i * 1.2, 1)
            binary_labels = (row > 0).astype(int).tolist()
            rows.append([fname, onset] + binary_labels)

    return pd.DataFrame(data=rows, columns=["filename", "onset"] + CLASSES)


def get_segment_prediction_df(
    predictions: Dict[str, Dict[str, np.ndarray]],
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregates frame-level predictions into fixed-length segments for a set of files.

    Parameters:
    ----------
    predictions : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping each filename to another dictionary of class-wise frame-level predictions.
        Each class prediction is a 1D NumPy array of shape (T,), where T is time.

    class_names : List[str], optional
        List of class names to include in the output. If None, uses keys from the first file's prediction dict.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns: ["filename", "onset"] + class_names.
        Each row represents a segment and contains aggregated predictions for that segment.
    """
    if class_names is None:
        class_names = list(next(iter(predictions.values())).keys())

    rows = []

    for filename, class_preds in predictions.items():
        # Collect and stack predictions into shape (T, num_classes)
        frame_matrix = np.stack([class_preds[cls] for cls in class_names], axis=1)

        # Aggregate over fixed-length segments
        aggregated = aggregate_targets(frame_matrix, f=10)

        for seg_idx, segment in enumerate(aggregated):
            onset = round(seg_idx * 1.2, 1)
            rows.append([filename, onset] + segment.tolist())

    return pd.DataFrame(rows, columns=["filename", "onset"] + class_names)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute total cost for environmental noise predictions.")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the root directory of the dataset (must contain 'audio_features/')."
    )

    parser.add_argument(
        "--ground_truth_csv",
        type=str,
        default=None,
        help="Path to the CSV file containing the ground truth labels."
    )

    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing the predicted labels."
    )

    args = parser.parse_args()

    df_pred = pd.read_csv(args.predictions_csv)
    check_dataframe(df_pred, dataset_path=args.dataset_path)
    print("Predictions CSV formated correctly.")

    if args.ground_truth_csv is not None:
        df_gt = pd.read_csv(args.ground_truth_csv)
        check_dataframe(df_gt, dataset_path=args.dataset_path)
        print("Ground truth CSV formated correctly.")

        total, breakdown = total_cost(df_pred, df_gt)

        print("Total cost:", total)