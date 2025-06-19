#!/usr/bin/env python
# coding: utf-8

# ### Colab Instructions
# 
# - **Files and Libraries Are Handled Automatically**  
#   The dataset, model checkpoint, `compute_cost.py`, and all required libraries will be automatically downloaded and installed when you run the notebook.
# 
# - **Enable GPU Acceleration**  
#   Go to **Runtime** → **Change runtime type**, and select **GPU** (e.g., **T4 GPU**) as the hardware accelerator.
# 
# - **Run the Notebook**  
#   Click **Runtime** → **Run all** to execute all cells sequentially.
# 
# - **Logging with Weights and Biases**  
#   The notebook will prompt you to paste your API token. You can obtain the token by creating a free account at [Weights and Biases](https://wandb.ai/site/).
# 
# - **Working Directory**  
#   The working directory is set to `/content`.
# 
# - **Runtime Duration**  
#   Running the full notebook will take approximately 45 minutes.

# # Advanced Sound Event Detection Tutorial
# 
# In this tutorial, you will learn how to:
# - Create a train/validation/test split  
# - Evaluate classifiers using standard metrics (e.g., precision, recall, f1-score)  
# - Compute segment-level costs based on classifier output  
# - Establish a simple baseline
# - Train and assess a logistic regression model on audio embeddings  
# - Build and evaluate a bidirectional RNN for sequence modeling  
# - Compare cost performance across baseline, logistic regression, and RNN models on the test set
# - Run inference on the customer's secret test set and store the predictions

# In[1]:


# Install required packages
get_ipython().system('pip install --quiet numpy pandas matplotlib scikit-learn torch torchvision torchaudio pytorch-lightning wandb rich ipywidgets tabulate tqdm')


# In[2]:


import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
import zipfile
import shutil


# In[3]:


# download the compute_cost.py file
pyfile_path = hf_hub_download(
    repo_id="fschmid56/mlpc2025_dataset",
    filename="compute_cost.py",
    repo_type="dataset"
)

# move to current working directory (/content)
shutil.copy(pyfile_path, os.getcwd() + "/compute_cost.py")

# import required functions
from compute_cost import CLASSES as TARGET_CLASSES
from compute_cost import COST_MATRIX as TARGET_CLASSES_COST_MATRIX
CLASSES_FP_COSTS = np.array([TARGET_CLASSES_COST_MATRIX[cl]["FP"] for cl in TARGET_CLASSES])
CLASSES_FN_COSTS = np.array([TARGET_CLASSES_COST_MATRIX[cl]["FN"] for cl in TARGET_CLASSES])
print(f"CLASSES_FP_COSTS = {CLASSES_FP_COSTS}")
print(f"CLASSES_FN_COSTS = {CLASSES_FN_COSTS}")
from compute_cost import (
    aggregate_targets,
    get_ground_truth_df,
    get_segment_prediction_df,
    check_dataframe,
    total_cost
)


# ## Download and prepare MLPC2025 Dataset

# In[4]:


# Step 1: Download the ZIP file from HF Hub
zip_path = hf_hub_download(
    repo_id="fschmid56/mlpc2025_dataset",   # your dataset repo
    filename="mlpc2025_dataset.zip",        # your uploaded ZIP file
    repo_type="dataset"                     # specify that it's a dataset repo
)
print(f"✅ ZIP downloaded: {zip_path}")


# In[5]:


# Step 2: Extract the ZIP
extract_path = "/content/mlpc2025_dataset"
os.makedirs(extract_path, exist_ok=True)

# Check if already extracted
if not os.path.exists(os.path.join(extract_path, "data")):  # assuming 'data/' is inside the zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Dataset extracted to {extract_path}")
else:
    print(f"✅ Dataset already extracted at {extract_path}")


# In[6]:


# Step 3: Set your DATASET_PATH
DATASET_PATH = os.path.join(extract_path, "data")  # because you zipped the 'data' folder
print(f"✅ DATASET_PATH set to {DATASET_PATH}")

# Quick check
print("Files in DATASET_PATH:", os.listdir(DATASET_PATH))


# In[7]:


METADATA_CSV = os.path.join(DATASET_PATH, 'metadata.csv')
ANNOTATIONS_CSV = os.path.join(DATASET_PATH, 'annotations.csv')
AUDIO_DIR = os.path.join(DATASET_PATH, 'audio')
AUDIO_FEATURES_DIR = os.path.join(DATASET_PATH, 'audio_features')
LABELS_DIR = os.path.join(DATASET_PATH, 'labels')

METADATA = pd.read_csv(METADATA_CSV)
DEV_SET_FILES = METADATA['filename']

CUSTOMER_DATASET_PATH = os.path.join(DATASET_PATH, 'customer_test_data')
CUSTOMER_AUDIO_DIR = os.path.join(CUSTOMER_DATASET_PATH, 'audio')
CUSTOMER_AUDIO_FEATURES_DIR = os.path.join(CUSTOMER_DATASET_PATH, 'audio_features')
CUSTOMER_METADATA_CSV = os.path.join(CUSTOMER_DATASET_PATH, 'metadata.csv')
CUSTOMER_METADATA = pd.read_csv(CUSTOMER_METADATA_CSV)

DATA_SUBSAMPLE = None  # works with available RAM in Colab
NUM_WORKERS = 8  # number of workers (maximum number of workers = CPU cores / threads - 1)
BATCH_SIZE = 128 # batch size
LOSS_SELECT = 2  # 0 .. nn.BCELoss, 1 .. nn.BCEWithLogitsLoss, 2 .. nn.WeightedBCEWithLogitsLoss
THRESHOLD_SELECT = 1  # 0 .. one threshold for all, 1 .. best threshold for each class
OPTIMIZER = 2    # 1 .. ADAM, 2 .. ADAMW, 3 .. NADAM, 4 .. ADAFACTOR, 5 .. LBFGS, 6 .. RMSPROP, x .. SGD


# ## Create the Data Split

# In[8]:


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


# In[9]:


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

train_files = train_files[:DATA_SUBSAMPLE]

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Load features and labels
X_train, Y_train = read_files(train_files, TARGET_CLASSES)
X_val, Y_val = read_files(val_files, TARGET_CLASSES)
X_test, Y_test = read_files(test_files, TARGET_CLASSES)


# In[10]:


print(f"### X ### : ")
print(f"X_train: {type(X_train)}, X_val: {type(X_val)}, X_test: {type(X_test)}")
print(f"X_train[0]: {type(X_train[0])}, X_val[0]: {type(X_val[0])}, X_test[0]: {type(X_test[0])}")
print(f"X_train[0]: {X_train[0].shape}, X_val[0]: {X_val[0].shape}, X_test[0]: {X_test[0].shape}")

print(f"### Y ### : ")
print(f"### Y_train: {len(Y_train)} / {type(Y_train)}")
for y in Y_train :
    print(f"{y} / {type(y)} : {len(Y_train[y])} / {type(Y_train[y])} : {len(Y_train[y][0])} / {type(Y_train[y][0])}")
print(f"### Y_val: {type(Y_val)} / {len(Y_val)}")
for y in Y_val :
    print(f"{y} / {type(y)} : {len(Y_val[y])} / {type(Y_val[y])} : {len(Y_val[y][0])} / {type(Y_val[y][0])}")
print(f"### Y_test: {len(Y_test)} / {type(Y_test)}")
for y in Y_test :
    print(f"{y} / {type(y)} : {len(Y_test[y])} / {type(Y_test[y])} : {len(Y_test[y][0])} / {type(Y_test[y][0])}")


# ## Evaluation Functions (Metrics & Cost)

# In[11]:


# Flatten: Each frame is a sample
def flatten_for_framewise_classification(X, Y_class):
    X_flat = np.concatenate(X)  # shape: (total_frames, num_features)
    Y_flat = np.concatenate(Y_class)  # shape: (total_frames,)
    return X_flat, Y_flat


# In[12]:


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


# In[13]:


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


# ## Most-Frequent Label Baseline

# In[14]:


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

# 1) Create baseline’s inference functions
bl_inference_funcs = baseline_most_frequent(Y_train, TARGET_CLASSES)


# In[15]:


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


# In[16]:


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


# # Bidirectional Gated Recurrent Unit
# 
# Can a recurrent neural network, which models temporal dependencies across frames, outperform logistic regression, which treats each frame independently, in sound event detection?
# 
# We will implement the required ingredients in the following order:
# * Dataset
# * DataModule
# * RNN Model
# * PyTorch Lightning Module
# * Hyperparameter Configuration
# * Logging via Weights & Biases
# * Callbacks
# * PyTorch Lightning Trainer

# ## Dataset
# 
# 

# In[17]:


### Exported to SED_Data_Module.py
#class SequenceDataset(Dataset):

from SED_Data_Module import SequenceDataset


# In[18]:


ds = SequenceDataset(X_train, Y_train, TARGET_CLASSES, train_files)
feat0, label0, file0 = ds[0]
print("SequenceDataset[0] -> feature shape:", feat0.shape,
      "\nlabel shape:", label0.shape,
      "\nfile[0]:", file0)


# In[19]:


### Exported to SED_Data_Module.py
# collate_fn used to create batches from the individual dataset items
#def collate_fn(batch):

from SED_Data_Module import collate_fn


# In[20]:


batch = [ds[i] for i in range(BATCH_SIZE)]
X_pad, Y_pad, lengths, filenames = collate_fn(batch)

print("collate_fn -> X_padded:", X_pad.shape,
      "\nY_padded:", Y_pad.shape,
      "\nlengths:", lengths,
      "\nfilenames:", filenames[:3], "...")


# ## DataModule
# 
# A `LightningDataModule` which organizes **all data loading logic** in one place.
# 
# Implements the following core API.
# 
# | Method                 | Purpose                                |
# |------------------------|----------------------------------------|
# | `__init__()`           | Save paths, batch size, classes, etc.  |
# | `setup(stage)`         | Prepare datasets (train/val/test)      |
# | `train_dataloader()`   | Return DataLoader for training         |
# | `val_dataloader()`     | Return DataLoader for validation       |
# | `test_dataloader()`    | Return DataLoader for testing          |

# In[21]:


### Exported to SED_Data_Module.py
# DataModule is used by pytorch lightning
#class SEDDataModule(pl.LightningDataModule):

from SED_Data_Module import SEDDataModule


# In[22]:


dm = SEDDataModule(
    X_train=X_train, Y_train=Y_train, train_files=train_files,
    X_val=X_val,     Y_val=Y_val,     val_files=val_files,
    X_test=X_test,   Y_test=Y_test,   test_files=test_files,
    classes=TARGET_CLASSES,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

dm.setup()
loader = dm.train_dataloader()
X_batch, Y_batch, len_batch, filenames = next(iter(loader))
print("DataModule batch -> X:", X_batch.shape,
      "\nY:", Y_batch.shape,
      "\nlengths:", len_batch,
      "\nfilenames:", filenames[:3], "...")


# ### Bidirectional TCN

# In[23]:


class ConvFeatureExtractor(nn.Module):
    """
    Applies Conv1D layers over the frequency axis to extract better frame-level features.
    Input: [B, T, F]
    Output: [B, C, T]
    """
    def __init__(self, in_channels=64, out_channels=128, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Input: [B, T, F] → [B, F, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x  # [B, C, T]


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    def __init__(self, num_blocks=4, channels=128, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            dilation = 2 ** i
            layers.append(ResidualBlock(channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.tcn(x)


class FramewiseAudioClassifier(nn.Module):
    def __init__(self, in_features=768, conv_features=128, num_classes=10): # , conv_features=128,
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(in_channels=in_features, out_channels=conv_features)
        self.tcn = TCN(channels=conv_features)
        self.output_layer = nn.Conv1d(conv_features, num_classes, kernel_size=1)

    def forward(self, x, lengths):
        # x: [B, T, F]
        x = self.feature_extractor(x)     # [B, C, T]
        x = self.tcn(x)                   # [B, C, T]
        x = self.output_layer(x)         # [B, 10, T]
        x = x.transpose(1, 2)            # [B, T, 10]
        return x


# In[24]:


# Instantiate model
model = FramewiseAudioClassifier()

# Forward pass
logits = model(X_batch, len_batch)

# Print shapes
print("Input X_batch shape:", X_batch.shape)       # (B, T_max, F)
print("Output logits shape:", logits.shape)         # (B, T_max, C)


# In[25]:


# Custom Loss Function for incorporating class specific costs

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, fp_cost, fn_cost, reduction: str = 'mean'):
        """
        Custom BCEWithLogitsLoss that uses different costs for false positives and false negatives per class.

        Args:
            fp_cost (Tensor): 1D tensor of shape (num_classes,) with cost for false positives.
            fn_cost (Tensor): 1D tensor of shape (num_classes,) with cost for false negatives.
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super().__init__()
        assert reduction in ('none', 'mean', 'sum'), "Invalid reduction type"
        self.register_buffer('fp_cost', torch.tensor(fp_cost, dtype=torch.float32))
        self.register_buffer('fn_cost', torch.tensor(fn_cost, dtype=torch.float32))

        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (Tensor): Raw logits of shape (batch_size, num_classes)
            target (Tensor): Binary labels of shape (batch_size, num_classes)
        Returns:
            loss (Tensor): Loss scalar or tensor depending on reduction.
        """
        # Apply sigmoid to logits
        #probs = torch.sigmoid(input)

        # Compute weights: FP weight where target==0, FN weight where target==1
        # Shape: (batch_size, num_classes)
        weights = torch.where(target == 1, self.fn_cost, self.fp_cost)

        # Compute BCE loss per element
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Apply per-class weight
        weighted_loss = bce * weights

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss

        #criterion = torch.nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=weights)
        #return criterion(input, target)


# In[ ]:





# In[ ]:


python PretrainedSED-main/ex_dcase2016task2.py --task_path=hear_datasets/tasks/dcase2016_task2-hear2021-full --model_name=ATST-F --pretrained=strong --lr_decay=0.95


# In[ ]:





# In[ ]:





# ### PyTorch Lightning Module: `SEDLightningModule`
# 
# The `LightningModule` wraps your model and training logic, abstracting away boilerplate code and handling key training steps automatically.  
# It implements a **standardized API** to define how your model should behave during training, validation, testing, and prediction.
# 
# - **`__init__`**: initializes the model (`BiGRUClassifier`), loss function, and validation and test buffer storage, and sets important attributes (e.g., lr, threshold).
# - **`forward(x, lengths)`**: forward pass through the GRU model.
# - **`predict_step(batch, batch_idx)`**: applies sigmoid + thresholding, slices off padding → returns predictions.
# - **`training_step(batch, batch_idx)`**: handles training logic.
# - **`validation_step(batch, batch_idx)`**: handles validation logic.
# - **`on_validation_epoch_end()`**: aggregates validation results after each epoch.
# - **`configure_optimizers()`**: defines the optimizer (Adam).

# In[26]:


class SEDLightningModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, classes, classes_fp_costs, classes_fn_costs, lr=1e-4, threshold=torch.tensor([0.5]*10, dtype=torch.float32)):
        super().__init__()
        # Core model
        self.model = FramewiseAudioClassifier()

        self.classes = classes
        self.classes_fp_costs = classes_fp_costs
        self.classes_fn_costs = classes_fn_costs

        if 1 == LOSS_SELECT :
            # Loss (we'll apply masking later, thus reduction='none')
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif 2 == LOSS_SELECT :
            # Custom Loss for class specific costs
            self.criterion = WeightedBCEWithLogitsLoss(fp_cost=self.classes_fp_costs, fn_cost=self.classes_fn_costs, reduction='none')

        self.lr = lr
        self.register_buffer('threshold', torch.tensor(threshold, dtype=torch.float32))

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
        if 1 == OPTIMIZER : # ADAM
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif 2 == OPTIMIZER : # ADAMW
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif 3 == OPTIMIZER : # NADAM
            return torch.optim.NAdam(self.parameters(), lr=self.lr)
        elif 4 == OPTIMIZER : # ADAFACTOR
            return torch.optim.Adafactor(self.parameters(), lr=self.lr)
        elif 5 == OPTIMIZER : # LBFGS
            return torch.optim.LBFGS(self.parameters(), lr=self.lr)
        elif 6 == OPTIMIZER : # RMSPROP
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else : # SGD
            return torch.optim.SGD(self.parameters(), lr=self.lr)


# ### PyTorch Lightning Module: `SEDLightningModule`
# 
# In the following, we will implement the missing functions:
# 
# - `process_training_step`:  
#   computes masked BCE loss for each frame and logs the training loss.
# 
# - `process_validation_step`:  
#   collects per-frame predictions and targets for later metric computation.
# 
# - `process_validation_epoch_end`:  
#   aggregates predictions and targets, computes metrics, and logs results.
# 
# - `process_test_step`:  
#   same as validation but used for test-time evaluation.
# 
# - `process_test_epoch_end`:  
#   evaluates and logs performance after test epoch.
# 
# We will bind this functions to our `SEDLightningModule`.

# ### `process_training_step`
# 
# computes masked BCE loss for each frame and logs the training loss

# In[27]:


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

# Bind it to the LightningModule
SEDLightningModule.process_training_step = process_training_step


# ### `process_validation_step`
# 
# computes masked BCE loss, logs it, and stores frame-level predictions and targets for aggreation in `process_validation_epoch_end`

# In[28]:


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

    if 0 == THRESHOLD_SELECT :
        preds = (torch.sigmoid(logits) > self.threshold).long()     # (B, T, C)
    elif 1 == THRESHOLD_SELECT :
        # Use logits directly as preds if you want to use variable thresholds per class
        preds = logits

    self._val_filenames.extend(filenames)

    for i, c in enumerate(self.classes):
        for b in range(X.size(0)):
            T = lengths[b]
            self._val_preds[c].append(preds[b, :T, i])
            self._val_targets[c].append(Y[b, :T, i])

    return loss

# Bind it to the LightningModule
SEDLightningModule.process_validation_step = process_validation_step


# ### `process_validation_epoch_end`
# 
# computes dataset metrics and cost and logs them at the end of a validation epoch.

# In[29]:


def process_validation_epoch_end(self):
    # Determine current mode
    prefix = "test" if self.trainer.testing else "val"


    # --- 1) Convert buffered tensors to NumPy arrays ---

    # Replace preds_numpy definition if you want to use variable thresholds per class
    '''
    preds_numpy = {
        cls: [p for p in self._val_preds[cls]]
        for cls in self.classes
    }
    '''

    preds_numpy = {
        cls: [p.cpu().numpy() for p in self._val_preds[cls]]
        for cls in self.classes
    }
    targets_numpy = {
        cls: [t.cpu().numpy() for t in self._val_targets[cls]]
        for cls in self.classes
    }

    if 1 == THRESHOLD_SELECT :
        # --- 1.5) CODE FOR FINDING OPTIMAL VARIABLE THRESHOLDS
        # self._val_preds = dict for classes -> list of predictions per file
        # preds_numpy = dict for classes -> list of predictions per file

        thresholds = np.linspace(0, 1, 21)
        best_thresholds = []
        best_scores = []
        for cls in self.classes:
            best_score = 0
            best_thresh = 0.5  # default
            best_preds = None

            class_activations = preds_numpy[cls]
            class_targets = {cls: targets_numpy[cls]}

            for thresh in thresholds:

                class_predictions = {cls: [(torch.sigmoid(torch.tensor(file_pred)) > thresh).long().cpu().numpy() for file_pred in class_activations]}

                """
                print([cls])
                print(class_activations[0].shape)
                print(class_predictions[cls][0].shape)
                print(class_targets[cls][0].shape)
                """

                score = evaluate_classifiers(
                    classes=[cls],
                    Y_val=class_targets,
                    Y_pred=class_predictions
                )

                if score[cls]["f1"] > best_score:
                    best_score = score[cls]["f1"]
                    best_thresh = thresh
                    best_preds = class_predictions

            best_thresholds.append(best_thresh)
            best_scores.append(best_score)
            preds_numpy[cls] = best_preds[cls]

        print(f"bestthresh: {best_thresholds}")
        self.register_buffer('threshold', torch.tensor(best_thresholds, dtype=torch.float32))

    # --- 2) Frame‐level metrics ---
    frame_metrics = evaluate_classifiers(
        classes=[self.classes[1]],
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


SEDLightningModule.process_validation_epoch_end = process_validation_epoch_end


# ### `process_test_step` and `process_test_epoch_end` ...
# 
# fortunately require the same logic as validation, so we can reuse `process_validation_step` and `process_validation_epoch_end`

# In[30]:


# After you’ve attached the validation logic, simply reuse it for testing:

# Reuse the same step‐logic
SEDLightningModule.process_test_step = SEDLightningModule.process_validation_step

# Reuse the same epoch‐end logic
SEDLightningModule.process_test_epoch_end = SEDLightningModule.process_validation_epoch_end


# ### Hyperparameters
# 
# Key hyperparameters with reasonable initial values — most likely not guaranteed optimal.

# In[31]:


hparams = dict(
    # not tuned by us - used out of the box
    input_dim      = X_batch.shape[-1],
    hidden_dim     = 1024,
    num_layers     = 4,
    lr             = 1e-4,
    batch_size     = BATCH_SIZE,
    max_epochs     = 50,
    threshold      = 0.5,
    patience       = 5,         # Early-stopping patience
)


# ### Callbacks
# 
# Callbacks are modular hooks that enable custom actions during training (e.g., saving checkpoints, early stopping, or logging), triggered at specific stages.

# In[32]:


checkpoint_cb = ModelCheckpoint(
    monitor    = "val/total_cost",   # minimize cost
    mode       = "min",
    save_top_k = 1,                  # save top model on validation data
    filename   = "best-{epoch:02d}"
)

early_stop_cb = EarlyStopping(
    monitor  = "val/total_cost",
    mode     = "min",
    patience = hparams["patience"],
    verbose  = True
)

lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

# RichProgressBar generates minimal output compared to 'tqdm'
progress_bar_cb = RichProgressBar()

callbacks = [checkpoint_cb, early_stop_cb, lr_monitor_cb, progress_bar_cb]


# ### Logger
# 
# - [Weights & Biases (wandb)](https://wandb.ai/site/) is a powerful and free experiment tracking tool  
# - lets you log metrics, visualize training runs, compare models  
# - share results via an interactive online dashboard  
# - integrates seamlessly with PyTorch Lightning

# In[33]:


wandb_logger = WandbLogger(
    project     = "mlpc2025-sed",
    name        = f"TCN_FULLTRAIN-{hparams['hidden_dim']}x{hparams['num_layers']}-threshold={THRESHOLD_SELECT}+loss={LOSS_SELECT}+opt={OPTIMIZER}",
    config      = hparams
)


# ### Trainer
# 
# The `Trainer` is the central PyTorch Lightning component that orchestrates training, validation, and testing.
# 
# It brings everything together:
# - The `SEDDataModule` provides the data.
# - The `SEDLightningModule` defines the model and training logic.
# - The `Trainer` handles the training loop, evaluation, logging, and callbacks.

# In[34]:


dm = SEDDataModule(
    X_train=X_train, Y_train=Y_train, train_files=train_files,
    X_val=X_val,     Y_val=Y_val,     val_files=val_files,
    X_test=X_test,   Y_test=Y_test,   test_files=test_files,
    classes=TARGET_CLASSES,
    batch_size=hparams["batch_size"],
    num_workers=NUM_WORKERS
)

model = SEDLightningModule(
    input_dim  = hparams["input_dim"],
    hidden_dim = hparams["hidden_dim"],
    num_layers = hparams["num_layers"],
    classes    = TARGET_CLASSES,
    classes_fp_costs = CLASSES_FP_COSTS,
    classes_fn_costs = CLASSES_FN_COSTS,
    lr         = hparams["lr"]
)

trainer = pl.Trainer(
    accelerator             = "gpu",
    devices                 = 1,
    max_epochs              = hparams["max_epochs"],
    callbacks               = callbacks,
    logger                  = wandb_logger,
    log_every_n_steps       = 10,
    deterministic           = True,
    check_val_every_n_epoch = 1,
    num_sanity_val_steps    = 0
)


# ### Let's train!
# 
# The following command launches training and validation, alternating across epochs. All results will be logged to Weights & Biases.
# You can explore a completed training run here: https://api.wandb.ai/links/cp_tobi/plk26iu9
# 
# Checkpoints will stored in `mlpc2025-sed/<wandb_id>/checkpoints`.

# In[35]:


trainer.fit(model, datamodule=dm)   # train and validate


# ### Let's test!
# 
# This loads the checkpoint with the lowest validation cost and runs evaluation on the test set.
# Check example test results logged to Weights & Biases here: https://api.wandb.ai/links/cp_tobi/plk26iu9

# In[49]:


test_results = trainer.test(model, datamodule=dm, ckpt_path="best")   # test


# ## Compare Baseline, Logistic Regression and BiGRU Costs on Test Set

# In[37]:


# baseline inference on test set
bl_total, bl_breakdown = evaluate_cost(
    test_files,
    DATASET_PATH,
    TARGET_CLASSES,
    X_test,
    bl_inference_funcs
)


# ### Collect all costs in a pandas dataframe for pretty print

# In[38]:


# shuffle around format for pretty print

# Convert breakdowns into dict[class → cost]
bl_costs = {cls: d["cost"] for cls, d in bl_breakdown.items()}

# Add total cost
bl_costs["TOTAL"] = bl_total

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
    "Baseline": bl_costs,
    "TCN": rnn_costs
}).round(2)


# In[39]:


print(test_results[0])


# In[40]:


print(tabulate(cost_df.reset_index().values,
               headers=["Class", "Baseline", "TCN"], #"Logistic Regression", 
               tablefmt="github"))


# ## Compute predictions on customer's secret test set
# 
# Requires three functions:
# - `load_model_from_checkpoint`: loading the desired model checkpoint, checkpoints are in folder `mlpc2025-sed/<wandb_id>/checkpoints`; an example checkpoint will be downloaded below
# - `predict_dataset`: generate predictions for customer's dataset
# - `segment_and_save`: bring predictions into the required 1.2 second segement format and save as csv file

# In[41]:


def load_model_from_checkpoint(
    ckpt_path: str,
    hparams: dict,
    classes: list[str]
) -> pl.LightningModule:
    return SEDLightningModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        input_dim  = hparams["input_dim"],
        hidden_dim = hparams["hidden_dim"],
        num_layers = hparams["num_layers"],
        lr         = hparams["lr"],
        threshold  = hparams["threshold"],
        classes    = classes,
        classes_fp_costs = CLASSES_FP_COSTS,
        classes_fn_costs = CLASSES_FN_COSTS,
    )


# In[42]:


def predict_dataset(
    model: pl.LightningModule,
    loader: DataLoader
) -> dict[str, dict[str, np.ndarray]]:
    """
    Runs trainer.predict() on `loader` and returns:
      preds_by_file[filename][class] = 1D NumPy array of frame‐wise {0,1}.
    """
    trainer = pl.Trainer(accelerator="auto", devices=1)
    outputs = trainer.predict(model, dataloaders=loader)

    # flatten into lists
    all_preds = {c: [] for c in model.classes}
    all_files = []
    for batch_out in outputs:
        for fname, pred in zip(batch_out["filenames"], batch_out["preds"]):
            all_files.append(fname)
            arr = pred.numpy()  # shape (T_i, C)
            for i, cls in enumerate(model.classes):
                all_preds[cls].append(arr[:, i])

    # repackage into preds_by_file
    preds_by_file: dict[str, dict[str, np.ndarray]] = {}
    for idx, fname in enumerate(all_files):
        preds_by_file.setdefault(fname, {})
        for cls in model.classes:
            preds_by_file[fname][cls] = all_preds[cls][idx]

    return preds_by_file


# In[43]:


def segment_and_save(
    preds_by_file: dict[str, dict[str, np.ndarray]],
    class_names: list[str],
    dataset_path: str,
    out_csv: str,
    compute_cost: bool = False,
    test_files: list[str] = None,
) -> pd.DataFrame:
    """
    1) Build segment‐level DataFrame
    2) Sanity‐check with check_dataframe()
    3) (optional) compute & print cost if val_files is provided
    4) save CSV to out_csv
    """
    # 1) aggregate predictions using the function provided in compute_cost.py
    pred_df = get_segment_prediction_df(
        predictions = preds_by_file,
        class_names = class_names
    )

    # 2) sanity‐check (from compute_cost.py)
    check_dataframe(pred_df, dataset_path)

    # 3) cost (optional), for sanity check on our custom test split
    if compute_cost and test_files is not None:
        gt_df = get_ground_truth_df(test_files, dataset_path) # from compute_cost.py
        total, breakdown = total_cost(pred_df, gt_df) # from compute_cost.py
        print(f"\nTotal cost: {total:.4f}")

        gt_csv = os.path.splitext(out_csv)[0] + "_ground_truth.csv"
        gt_df.to_csv(gt_csv, index=False)
        print(f"Saved ground truth segments to {gt_csv}")

    # 4) save
    pred_df.to_csv(out_csv, index=False)
    print(f"Saved segment predictions to {out_csv}")

    return pred_df


# ### Load checkpoint
# 
# Download an example checkpoint from huggingface.

# In[50]:


# download example checkpoint from huggingface
ckpt_path = hf_hub_download(
    repo_id="fschmid56/mlpc2025_dataset",
    filename="colab_tutorial.ckpt",
    repo_type="model"
)

# alternatively, use your own local checkpoint
# replace wandb id 'lo9ygyg4' with your desired wandb id
# replace 'best-epoch=11.ckpt' with the name of your checkpoint
# ckpt_path = "/content/mlpc2025-sed/lo9ygyg4/checkpoints/best-epoch=11.ckpt"

ckpt_path = "./mlpc2025-sed/dgwmepda/checkpoints/best-epoch=04.ckpt"
ckpt_path = "./mlpc2025-sed/qgziv2et/checkpoints/best-epoch=05.ckpt"
ckpt_path = "./mlpc2025-sed/gidac3vu/checkpoints/best-epoch=10.ckpt"
ckpt_path = "./mlpc2025-sed/hwbfqhhy/checkpoints/best-epoch=03.ckpt"
ckpt_path = "./mlpc2025-sed/hwbfqhhy/checkpoints/best-epoch=03-v1.ckpt"
ckpt_path = "./mlpc2025-sed/cpqw2wpy/checkpoints/best-epoch=06.ckpt"
ckpt_path = "./mlpc2025-sed/5ff9n10r/checkpoints/best-epoch=04.ckpt"
ckpt_path = "./mlpc2025-sed/xpdmnfgl/checkpoints/best-epoch=01.ckpt"
ckpt_path = "./mlpc2025-sed/6n6c9beb/checkpoints/best-epoch=04.ckpt"

model = load_model_from_checkpoint(ckpt_path, hparams, TARGET_CLASSES)


# We sanity check model loading, the prediction routine and segmenting predictions by applying it our custom test split and calculating costs (as we have access to the labels).

# In[51]:


# 1) TEST SPLIT
test_dataset = SequenceDataset(X_test, Y_test, TARGET_CLASSES, test_files)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_preds   = predict_dataset(model, test_loader)
segment_and_save(
    preds_by_file = test_preds,
    class_names   = TARGET_CLASSES,
    dataset_path  = DATASET_PATH,
    out_csv       = "test_split_predictions.csv",
    compute_cost  = True,
    test_files     = test_files,
)


# Finally, compute predictions on the customer's secret test set and store as `/content/customer_predictions.csv`.

# In[52]:


# 2) CUSTOMER SET (no labels → compute_cost=False)
customer_files = CUSTOMER_METADATA["filename"].unique()
X_cust, _ = read_files(customer_files, TARGET_CLASSES,
                       features_dir=CUSTOMER_AUDIO_FEATURES_DIR,
                       labels_dir=None)
cust_dataset = SequenceDataset(X_cust, None, TARGET_CLASSES, customer_files)
cust_loader  = DataLoader(cust_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

cust_preds = predict_dataset(model, cust_loader)
segment_and_save(
    preds_by_file = cust_preds,
    class_names   = TARGET_CLASSES,
    dataset_path  = CUSTOMER_DATASET_PATH,
    out_csv       = "customer_predictions.csv",
    compute_cost  = False,  # can't compute on customer's secret test set
)


# ### Final checks as in Task Description
# 
# Instead of importing all the functions from `compute_cost.py` and working with DataFrames directly, you can also run the provided script as recommended in the Task Description. Just pass your generated .csv file(s) to verify correctness and compute cost.

# In[53]:


get_ipython().system('python compute_cost.py    --dataset_path="{DATASET_PATH}"    --ground_truth_csv="test_split_predictions_ground_truth.csv"    --predictions_csv="test_split_predictions.csv"')


# In[54]:


get_ipython().system('python compute_cost.py    --dataset_path="{CUSTOMER_DATASET_PATH}"    --predictions_csv="customer_predictions.csv"')


# In[ ]:




