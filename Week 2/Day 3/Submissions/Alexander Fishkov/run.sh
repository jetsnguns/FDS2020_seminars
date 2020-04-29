#!/usr/bin/env bash

# Directories and paths to intermediate objects
DATA_URL="https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"
DOWNLOAD_DIR=data
DATA_DIR=data
RAW_DIR=data/nycflights
PROCESSED_PATH=data/processed.hdf
RESULTS_DIR=results
TARGET_COL=ArrDelay

# Configuration files
TYPES_PATH=configs/dtypes.json
PREPROC_PATH=configs/preprocess.json
DESC1=configs/rf.json   # sklearn model
DESC2=configs/lr_a.json # dask model

python scripts/download.py -u $DATA_URL -d $DOWNLOAD_DIR -o $DATA_DIR
python scripts/preprocess.py -i $RAW_DIR -t $TYPES_PATH -p $PREPROC_PATH -o $PROCESSED_PATH
python scripts/train.py -i $PROCESSED_PATH -d $DESC1 -t $TARGET_COL -o $RESULTS_DIR
python scripts/train.py -i $PROCESSED_PATH -d $DESC2 -t $TARGET_COL -o $RESULTS_DIR
