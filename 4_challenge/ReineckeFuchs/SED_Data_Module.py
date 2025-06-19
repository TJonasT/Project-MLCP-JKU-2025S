
# In a separate file, say SED_Data_Module.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

NUM_WORKERS = 6  # number of workers (maximum number of workers = CPU cores - 1)
BATCH_SIZE = 128

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



# DataModule is used by pytorch lightning
class SEDDataModule(pl.LightningDataModule):
    def __init__(self,
                 X_train, Y_train, train_files,
                 X_val,   Y_val,   val_files,
                 X_test,  Y_test,  test_files,
                 classes,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        super().__init__()
        self.X_train, self.Y_train, self.train_files = X_train, Y_train, train_files
        self.X_val,   self.Y_val,   self.val_files   = X_val,   Y_val,   val_files
        self.X_test,  self.Y_test,  self.test_files  = X_test,  Y_test,  test_files
        self.classes     = classes
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = SequenceDataset(self.X_train, self.Y_train, self.classes, self.train_files)
        self.val_ds   = SequenceDataset(self.X_val,   self.Y_val,   self.classes, self.val_files)
        self.test_ds  = SequenceDataset(self.X_test,  self.Y_test,  self.classes, self.test_files)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers)

