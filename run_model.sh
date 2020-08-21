#!/bin/bash

# input arguments
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py cuda=1 -gm=DGCNN -data=MUTAG -retrain=1
