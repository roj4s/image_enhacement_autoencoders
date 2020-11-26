#!/bin/bash

source activate autoencoder_image_enhacement
python train.py $X_ROOT $Y_ROOT --checkpoints-output $CHECKPOINTS_OUTPUT --tensorboard-output $TENSORBOARD_OUTPUT --model $MODEL --learning-rate $LEARNING_RATE
