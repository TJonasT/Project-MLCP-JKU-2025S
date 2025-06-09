import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from compute_cost import CLASSES as TARGET_CLASSES
from file_locations import AUDIO_FEATURES_DIR, LABELS_DIR, DEV_SET_FILES

def read_files(file_names, classes, features_dir=AUDIO_FEATURES_DIR, labels_dir=LABELS_DIR):
    """
    Loads features and binary labels for a list of files.

    Returns:
        X: list of np.ndarrays, each of shape (num_frames, num_features)
        Y: dict of lists of np.ndarrays, each of shape (num_frames,)
    """
    X = []
    Y = {c: [] for c in classes} if labels_dir is not None else None

    for fname in file_names:
        base = os.path.splitext(fname)[0]

        # Load features
        feat_path = os.path.join(features_dir, base + '.npz')
        features = np.load(feat_path)['embeddings']  # shape: (T, D)
        X.append(features)

        if labels_dir is not None:
            # Load labels
            label_path = os.path.join(labels_dir, base + '_labels.npz')
            labels = np.load(label_path)

            for c in classes:
                label_array = labels[c]  # shape: (T, num_annotators)
                binary_labels = (np.max(label_array, axis=1) > 0).astype(int)
                Y[c].append(binary_labels)  # shape: (T,)

    return X, Y

# Save the training, validation, and test splits
# X: Features, Y: Labels
def save_split(X, Y, files, prefix, OUTPUT_DIR='./data_files'):
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save features as a pickle file since they are a list of arrays
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_features.pkl"), 'wb') as f:
        pickle.dump(X, f)

    # Save labels as a pickle file since it's a dictionary
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_labels.pkl"), 'wb') as f:
        pickle.dump(Y, f)

    with open(os.path.join(OUTPUT_DIR, f"{prefix}_files.pkl"), 'wb') as f:
        pickle.dump(files, f)


# Load the dataset splits
def load_split(prefix, OUTPUT_DIR='./data_files'):
    # Load features
    feature_path = os.path.join(OUTPUT_DIR, f"{prefix}_features.pkl")
    with open(feature_path, 'rb') as f:
        X = pickle.load(f)

    # Load labels
    label_path = os.path.join(OUTPUT_DIR, f"{prefix}_labels.pkl")
    with open(label_path, 'rb') as f:
        Y = pickle.load(f)

    # Load files
    files_path = os.path.join(OUTPUT_DIR, f"{prefix}_files.pkl")
    with open(files_path, 'rb') as f:
        files = pickle.load(f)

    return X, Y, files


if __name__ == "__main__":
    # Get filenames for split based on filenames
    all_files = DEV_SET_FILES.unique()

    # First split: 60% train, 40% temp (val + test)
    train_files, temp_files = train_test_split(
        all_files, test_size=0.4, random_state=42, shuffle=True
    )

    # Second split: 50% val, 50% test from the remaining 40%
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=42, shuffle=True
    )

    # train_files = train_files[:DATA_SUBSAMPLE]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Load features and labels
    X_train, Y_train = read_files(train_files, TARGET_CLASSES)
    X_val, Y_val = read_files(val_files, TARGET_CLASSES)
    X_test, Y_test = read_files(test_files, TARGET_CLASSES)

    # Save the dataset splits
    save_split(X_train, Y_train, train_files, "train")
    save_split(X_val, Y_val, val_files, "val")
    save_split(X_test, Y_test, test_files, "test")


