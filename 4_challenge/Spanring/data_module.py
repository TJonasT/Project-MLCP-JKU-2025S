import os
import pickle
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from data_split import load_split
from pytorch_dataset import SequenceDataset, collate_fn

from compute_cost import CLASSES as TARGET_CLASSES

# DataModule is used by pytorch lightning
class SEDDataModule(pl.LightningDataModule):
    def __init__(self,
                 X_train, Y_train, train_files,
                 X_val,   Y_val,   val_files,
                 X_test,  Y_test,  test_files,
                 classes,
                 batch_size=32,
                 num_workers=4):
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
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers,
                          persistent_workers=True)

if __name__ == '__main__':
    X_train, Y_train, train_files = load_split("train")
    X_test, Y_test, test_files = load_split("test")
    X_val, Y_val, val_files = load_split("val")

    dm = SEDDataModule(
        X_train=X_train, Y_train=Y_train, train_files=train_files,
        X_val=X_val, Y_val=Y_val, val_files=val_files,
        X_test=X_test, Y_test=Y_test, test_files=test_files,
        classes=TARGET_CLASSES,
        batch_size=32,
        num_workers=os.cpu_count() - 1
    )

    dm.setup()
    loader = dm.train_dataloader()
    X_batch, Y_batch, len_batch, filenames = next(iter(loader))

    print("DataModule batch -> X:", X_batch.shape,
          "\nY:", Y_batch.shape,
          "\nlengths:", len_batch,
          "\nfilenames:", filenames[:3], "...")

    # Create a dictionary to store the batch data
    batch_data = {
        "X_batch": X_batch,
        "Y_batch": Y_batch,
        "len_batch": len_batch,
        "filenames": filenames
    }

    # Specify the output file path
    output_file = "data_files/batch_data.pkl"

    # Save the dictionary as a .pkl file
    with open(output_file, 'wb') as file:
        pickle.dump(batch_data, file)

    print(f"Batch data is saved to {output_file}")
