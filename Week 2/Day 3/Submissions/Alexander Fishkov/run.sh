# Directories and paths to intermediate objects
DATA_URL="https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"
DOWNLOAD_DIR=data
DATA_DIR=data
RAW_DIR=data/nycflights
PROCESSED_PATH=data/processed.hdf
RESULTS_DIR=results
TARGET_COL=ArrDelay

# Configuration files
TYPES_PATH=dtypes.json
PREPROC_PATH=preprocess.json
DESC1=rf.json # sklearn model
DESC2=lr_a.json # dask model


python download.py -u $DATA_URL -d $DOWNLOAD_DIR -o $DATA_DIR
python preprocess.py -i $RAW_DIR -t $TYPES_PATH -p $PREPROC_PATH -o $PROCESSED_PATH
python train.py -i $PROCESSED_PATH -d $DESC1 -t $TARGET_COL -o $RESULTS_DIR
python train.py -i $PROCESSED_PATH -d $DESC2 -t $TARGET_COL -o $RESULTS_DIR
