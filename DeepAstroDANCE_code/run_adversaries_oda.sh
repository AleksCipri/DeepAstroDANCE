#!/bin/sh
python training_dance.py --config 'configs/adversaries-train-config_ODA.yaml' $2 --source './data/images/images_Y10_train.npy' --target './data/images/images_Y1_train.npy' --gpu $1