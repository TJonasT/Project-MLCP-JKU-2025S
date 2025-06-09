import pickle

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

from evaluation_functions import evaluate_classifiers, evaluate_cost
from file_locations import DATASET_PATH

class SEDLightningModule(pl.LightningModule):
    def __init__(self, model_class, input_dim, hidden_dim, num_layers, classes, lr=1e-4, threshold=0.5):
        super().__init__()
        # Core model
        self.model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=len(classes)
        )

        self.classes = classes

        # Loss (we'll apply masking later, thus reduction='none')
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.lr = lr
        self.threshold = threshold

        self._val_preds   = {c: [] for c in self.classes}
        self._val_targets = {c: [] for c in self.classes}
        self._val_filenames = []

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def predict_step(self, batch, batch_idx):
        # unpack batch (with or without labels)
        if len(batch) == 4:
            X, _, lengths, filenames = batch
        else:
            X, lengths, filenames = batch

        # 1) raw logits → probs → binary preds
        logits = self.model(X, lengths)
        probs  = torch.sigmoid(logits)
        preds  = (probs > self.threshold).int() # (B, T_max, C)

        # 2) remove padding
        batch_preds = [preds[b, :lengths[b]].cpu()
                      for b in range(X.size(0))]

        return {"filenames": filenames, "preds": batch_preds}

    # we will implement the processing steps one after the other in the following
    def training_step(self, batch, batch_idx):
        return self.process_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.process_validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        return self.process_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        return self.process_test_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.process_test_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def process_training_step(self, batch, batch_idx):
    X, Y, lengths, _ = batch      # X: (B, T, D), Y: (B, T, C), lengths: (B,)
    logits = self(X, lengths)     # calls self.forward, results in logits of shape (B, T, C)

    # raw per-element loss
    loss_raw = self.criterion(logits, Y.float())  # (B, T, C)

    # build mask to zero out padded frames
    mask = torch.arange(logits.size(1), device=logits.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).float()     # (B, T, 1)

    # apply mask and average
    loss = (loss_raw * mask).sum() / mask.sum()

    self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=X.size(0))
    return loss


def process_validation_step(self, batch, batch_idx):
    X, Y, lengths, filenames = batch      # X: (B, T, D), Y: (B, T, C), lengths: (B,)
    logits = self(X, lengths)             # calls self.forward, results in logits of shape (B, T, C)

    # Determine logging prefix
    prefix = "test" if self.trainer.testing else "val"

    # compute masked BCE loss
    loss_raw = self.criterion(logits, Y.float())     # (B, T, C)
    mask = torch.arange(logits.size(1), device=logits.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).float()                # (B, T, 1)
    loss = (loss_raw * mask).sum() / mask.sum()

    self.log(f'{prefix}/loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=X.size(0))

    # store frame-wise preds & targets for epoch_end
    # frame-wise logits are thresholded here
    preds = (torch.sigmoid(logits) > self.threshold).long()     # (B, T, C)
    self._val_filenames.extend(filenames)

    for i, c in enumerate(self.classes):
        for b in range(X.size(0)):
            T = lengths[b]
            self._val_preds[c].append(preds[b, :T, i])
            self._val_targets[c].append(Y[b, :T, i])

    return loss


def process_validation_epoch_end(self):
    # Determine current mode
    prefix = "test" if self.trainer.testing else "val"

    # --- 1) Convert buffered tensors to NumPy arrays ---
    preds_numpy = {
        cls: [p.cpu().numpy() for p in self._val_preds[cls]]
        for cls in self.classes
    }
    targets_numpy = {
        cls: [t.cpu().numpy() for t in self._val_targets[cls]]
        for cls in self.classes
    }

    # --- 2) Frame‐level metrics ---
    frame_metrics = evaluate_classifiers(
        classes=self.classes,
        Y_val=targets_numpy,
        Y_pred=preds_numpy
    )

    for cls, m in frame_metrics.items():
        self.log(f'{prefix}/{cls}_bacc',     m['balanced_accuracy'])
        self.log(f'{prefix}/{cls}_precision',m['precision'])
        self.log(f'{prefix}/{cls}_recall',   m['recall'])
        self.log(f'{prefix}/{cls}_f1',       m['f1'])

    # --- 3) Segment‐level cost ---
    total_cost, cost_breakdown = evaluate_cost(
        val_files=self._val_filenames,
        dataset_path=DATASET_PATH,
        classes=self.classes,
        Y_pred=preds_numpy
    )
    self.log(f'{prefix}/total_cost', total_cost, prog_bar=True)
    for cls, cls_cost in cost_breakdown.items():
        self.log(f"{prefix}/cost/{cls}", cls_cost["cost"], prog_bar=False)

    # --- 4) Clear buffers ---
    self._val_preds     = {c: [] for c in self.classes}
    self._val_targets   = {c: [] for c in self.classes}
    self._val_filenames = []

def process_predict_step(self, batch, batch_idx):
    # unpack batch (with or without labels)
    if len(batch) == 4:
        X, _, lengths, filenames = batch
    else:
        X, lengths, filenames = batch

    # 1) raw logits → probs → binary preds
    logits = self.model(X, lengths)
    probs  = torch.sigmoid(logits)
    preds  = (probs > self.threshold).int() # (B, T_max, C)

    # 2) remove padding
    batch_preds = [preds[b, :lengths[b]].cpu()
                   for b in range(X.size(0))]

    return {"filenames": filenames, "preds": batch_preds}

