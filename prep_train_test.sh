#!/usr/bin/env bash

DATASETFILE=ade.asu_chop.ioadrind.valid010.pkl
PROCESSDATADIR=./data/seq_labeling/processed
BASEDIR=./model_output

# Prep the dataset, creating a 90/10 train/valid split from the original training data
python ./prep.py -o $DATASETFILE -v 0.1

# Train, validate, and test model with default settings (single 256-dimensional LSTM layer 
#   in each direction, pre-trained word embeddings held fixed through training)
WORKING_DIR=$BASEDIR/train_val_test_asu_chop
mkdir -p $WORKING_DIR
python ./adr_label.py -P $PROCESSDATADIR/$DATASETFILE -b $WORKING_DIR > $WORKING_DIR/log