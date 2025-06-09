import os
import pickle
import subprocess

import numpy as np
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

from compute_cost import CLASSES as TARGET_CLASSES
from compute_cost import (
    aggregate_targets,
    get_ground_truth_df,
    get_segment_prediction_df,
    check_dataframe,
    total_cost
)

from lightning_module import SEDLightningModule
from file_locations import CUSTOMER_FILES, DATASET_PATH, CUSTOMER_DATASET_PATH, CUSTOMER_AUDIO_FEATURES_DIR
from pytorch_dataset import SequenceDataset, collate_fn
from data_split import read_files, load_split
from BiGRU import BiGRUClassifier


def load_model_from_checkpoint(
        ckpt_path: str,
        hparams: dict,
        classes: list[str]
) -> pl.LightningModule:
    return SEDLightningModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        lr=hparams["lr"],
        threshold=hparams["threshold"],
        classes=classes,
        model_class=BiGRUClassifier,
    )

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

if __name__ == '__main__':
    X_train, Y_train, train_files = load_split("train")
    X_test, Y_test, test_files = load_split("test")
    X_val, Y_val, val_files = load_split("val")

    # Choose best model
    with open("models/bigru_1024x2.pkl", "rb") as file:
        bigru_hparams = pickle.load(file)

    ckpt_path_bigru = "models/BiGRU_1024x2.ckpt"

    model = load_model_from_checkpoint(ckpt_path_bigru, bigru_hparams, TARGET_CLASSES)

    # 1) TEST SPLIT
    test_dataset = SequenceDataset(X_test, Y_test, TARGET_CLASSES, test_files)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)
    test_preds = predict_dataset(model, test_loader)
    segment_and_save(
        preds_by_file=test_preds,
        class_names=TARGET_CLASSES,
        dataset_path=DATASET_PATH,
        out_csv="test_split_predictions.csv",
        compute_cost=True,
        test_files=test_files,
    )

    # 2) CUSTOMER SET (no labels → compute_cost=False)
    customer_files = CUSTOMER_FILES.unique()
    X_cust, _ = read_files(customer_files, TARGET_CLASSES,
                           features_dir=CUSTOMER_AUDIO_FEATURES_DIR,
                           labels_dir=None)
    cust_dataset = SequenceDataset(X_cust, None, TARGET_CLASSES, customer_files)
    cust_loader = DataLoader(cust_dataset, batch_size=8, collate_fn=collate_fn)

    cust_preds = predict_dataset(model, cust_loader)
    segment_and_save(
        preds_by_file=cust_preds,
        class_names=TARGET_CLASSES,
        dataset_path=CUSTOMER_DATASET_PATH,
        out_csv="customer_predictions.csv",
        compute_cost=False,  # can't compute on customer's secret test set
    )


    # Command to run python script
    command = ("python compute_cost.py "
               "--dataset_path={} --ground_truth_csv=test_split_predictions_ground_truth.csv --predictions_csv=test_split_predictions.csv")
    # Formatted command
    command = command.format(DATASET_PATH)
    # Use subprocess to run the command
    subprocess.run(command, shell=True, check=True)

    # Command to execute the python script with
    command = "python compute_cost.py --dataset_path={} --predictions_csv=customer_predictions.csv"
    # Formatted command to include your dataset path
    command = command.format(CUSTOMER_DATASET_PATH)
    # Use subprocess to run the command
    subprocess.run(command, shell=True, check=True)




