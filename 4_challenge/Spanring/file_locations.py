import os

import pandas as pd

DATASET_PATH = '../../MLPC2025_classification'
CUSTOMER_DATASET_PATH = '../../MLPC2025_test'

METADATA_CSV = os.path.join(DATASET_PATH, 'metadata.csv')
CUSTOMER_METADATA_CSV = os.path.join(CUSTOMER_DATASET_PATH, 'metadata.csv')
ANNOTATIONS_CSV = os.path.join(DATASET_PATH, 'annotations.csv')
AUDIO_DIR = os.path.join(DATASET_PATH, 'audio')
AUDIO_FEATURES_DIR = os.path.join(DATASET_PATH, 'audio_features')
LABELS_DIR = os.path.join(DATASET_PATH, 'labels')

METADATA = pd.read_csv(METADATA_CSV)
DEV_SET_FILES = METADATA['filename']

CUSTOMER_METADATA = pd.read_csv(CUSTOMER_METADATA_CSV)
CUSTOMER_FILES = CUSTOMER_METADATA['filename']

CUSTOMER_AUDIO_DIR = os.path.join(CUSTOMER_DATASET_PATH, 'audio')
CUSTOMER_AUDIO_FEATURES_DIR = os.path.join(CUSTOMER_DATASET_PATH, 'audio_features')


