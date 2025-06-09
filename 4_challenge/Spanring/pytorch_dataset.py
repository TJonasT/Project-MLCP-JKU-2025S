import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from data_split import load_split
from compute_cost import CLASSES as TARGET_CLASSES

class SequenceDataset(Dataset):
    """
    Dataset for sequence modeling tasks with optional per-frame binary labels.

    Args:
        X: List of input feature arrays (T_i, D), one per file.
        Y: Optional dict[class → list of (T_i,) label arrays], one per file and class.
        classes: List of class names to extract from Y.
        filenames: List of filenames corresponding to each input.

    Returns:
        Each item is a tuple:
        - (features, labels, filename): if Y is provided
        - (features, filename): if Y is None
    """
    def __init__(self, X, Y, classes, filenames):
      # in colab with limited RAM, we convert our files to
      # tensors only in __getitem__
      self.X = X  # Keep X as a list of np.ndarrays
      self.Y = Y
      self.classes = classes
      self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)  # Convert on access
        if self.Y is not None:
            y_tensor = torch.stack([
                torch.tensor(self.Y[c][idx], dtype=torch.long) for c in self.classes
            ], dim=1)
            return x_tensor, y_tensor, self.filenames[idx]
        else:
            return x_tensor, self.filenames[idx]

# collate_fn used to create batches from the individual dataset items
def collate_fn(batch):
    if len(batch[0]) == 3:
        Xs, Ys, filenames = zip(*batch)
        lengths = torch.tensor([x.size(0) for x in Xs], dtype=torch.long)
        X_padded = pad_sequence(Xs, batch_first=True)
        Y_padded = pad_sequence(Ys, batch_first=True)
        return X_padded, Y_padded, lengths, list(filenames)
    elif len(batch[0]) == 2:
        Xs, filenames = zip(*batch)
        lengths = torch.tensor([x.size(0) for x in Xs], dtype=torch.long)
        X_padded = pad_sequence(Xs, batch_first=True)
        return X_padded, lengths, list(filenames)
    else:
        raise ValueError("Unexpected batch format: expected 2 or 3 elements per item.")

if __name__ == '__main__':
    X_train, Y_train, train_files = load_split("train")
    X_test, Y_test, test_files = load_split("test")
    X_val, Y_val, val_files = load_split("val")

    ds = SequenceDataset(X_train, Y_train, TARGET_CLASSES, train_files)
    feat0, label0, file0 = ds[0]
    print("SequenceDataset[0] -> feature shape:", feat0.shape,
          "\nlabel shape:", label0.shape,
          "\nfile[0]:", file0)

    batch = [ds[i] for i in range(32)]
    X_pad, Y_pad, lengths, filenames = collate_fn(batch)

    print("collate_fn -> X_padded:", X_pad.shape,
          "\nY_padded:", Y_pad.shape,
          "\nlengths:", lengths,
          "\nfilenames:", filenames[:3], "...")