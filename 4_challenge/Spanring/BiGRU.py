import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger
from tabulate import tabulate

from lightning_module import (SEDLightningModule,
                              process_training_step,
                              process_validation_step,
                              process_validation_epoch_end,
                              process_predict_step)
from data_module import SEDDataModule
from data_split import load_split
from compute_cost import CLASSES as TARGET_CLASSES

class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU classifier with a linear output layer.

    Args:
        input_dim: Input feature dimension (D).
        hidden_dim: Hidden size per GRU direction.
        num_layers: Number of stacked GRU layers.
        num_classes: Number of output classes (C).

    Input:
        x: Tensor of shape (B, T, D) — batch of padded sequences.
        lengths: Tensor of shape (B,) — actual lengths before padding.

    Returns:
        logits: Tensor of shape (B, T, C) — class scores for each time step.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        # x: (B, T, D), lengths: (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: (B, T, 2*hidden_dim)
        logits = self.classifier(out)  # (B, T, num_classes)
        return logits

if __name__ == '__main__':
    with open("data_files/batch_data.pkl", "rb") as file:
        batch_data = pickle.load(file)

    X_batch = batch_data["X_batch"]
    Y_batch = batch_data["Y_batch"]
    len_batch = batch_data["len_batch"]
    filenames = batch_data["filenames"]

    # Instantiate model
    model = BiGRUClassifier(
        input_dim=X_batch.shape[-1],
        hidden_dim=1024,
        num_layers=2,
        num_classes=Y_batch.shape[-1]
    )

    # Forward pass
    logits = model(X_batch, len_batch)

    # Print shapes
    print("Input X_batch shape:", X_batch.shape)  # (B, T_max, F)
    print("Output logits shape:", logits.shape)  # (B, T_max, C)

    SEDLightningModule.process_training_step = process_training_step
    SEDLightningModule.process_validation_step = process_validation_step
    SEDLightningModule.process_validation_epoch_end = process_validation_epoch_end
    SEDLightningModule.process_test_step = SEDLightningModule.process_validation_step
    SEDLightningModule.process_test_epoch_end = SEDLightningModule.process_validation_epoch_end
    SEDLightningModule.process_predict_step = process_predict_step

    hparams = dict(
        # not tuned by us - used out of the box
        input_dim=X_batch.shape[-1],
        hidden_dim=1024,
        num_layers=4,
        lr=1e-4,
        batch_size=64,
        max_epochs=50,
        threshold=0.5,
        patience=5,  # Early-stopping patience
    )

    with open(f"RNN/rnn_{hparams['hidden_dim']}x{hparams['num_layers']}_hparams.pkl", "wb") as file:
        pickle.dump(hparams, file)

    checkpoint_cb = ModelCheckpoint(
        monitor="val/total_cost",  # minimize cost
        mode="min",
        save_top_k=1,  # save top model on validation data
        filename=f"rnn_{hparams['hidden_dim']}x{hparams['num_layers']}",
        dirpath="RNN"
    )

    early_stop_cb = EarlyStopping(
        monitor="val/total_cost",
        mode="min",
        patience=hparams["patience"],
        verbose=True
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    # RichProgressBar generates minimal output compared to 'tqdm'
    progress_bar_cb = RichProgressBar()

    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor_cb, progress_bar_cb]

    wandb_logger = WandbLogger(
        project="mlpc2025-sed",
        name=f"BiGRU-{hparams['hidden_dim']}x{hparams['num_layers']}",
        config=hparams
    )

    X_train, Y_train, train_files = load_split("train")
    X_test, Y_test, test_files = load_split("test")
    X_val, Y_val, val_files = load_split("val")

    dm = SEDDataModule(
        X_train=X_train, Y_train=Y_train, train_files=train_files,
        X_val=X_val, Y_val=Y_val, val_files=val_files,
        X_test=X_test, Y_test=Y_test, test_files=test_files,
        classes=TARGET_CLASSES,
        batch_size=hparams["batch_size"],
        num_workers=2
    )

    model = SEDLightningModule(
        model_class=BiGRUClassifier,
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        classes=TARGET_CLASSES,
        lr=hparams["lr"]
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=hparams["max_epochs"],
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        deterministic=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )

    trainer.fit(model, datamodule=dm)  # train and validate

    test_results = trainer.test(model, datamodule=dm, ckpt_path="best")  # test

    # Extract relevant costs from pytorch lightning test results
    rnn_result = test_results[0]
    rnn_costs = {
        cls: rnn_result[f"test/cost/{cls}"]
        for cls in TARGET_CLASSES
        if f"test/cost/{cls}" in rnn_result
    }

    # Add total cost
    rnn_costs["TOTAL"] = rnn_result["test/total_cost"]

    # Create a DataFrame for comparison
    cost_df = pd.DataFrame({
        "RNN": rnn_costs
    }).round(2)

    print(tabulate(cost_df.reset_index().values,
                   headers=["Class", "Baseline", "Logistic Regression", "RNN"],
                   tablefmt="github"))

    with open(f"RNN/rnn_{hparams['hidden_dim']}x{hparams['num_layers']}_costs.pkl", "wb") as file:
        pickle.dump(rnn_costs, file)

