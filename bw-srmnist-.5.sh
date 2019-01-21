#!/bin/bash

source activate munit

python train.py --config ~/munit/configs/bw-srmnist-.5.yaml \
	--output_path /network/tmp1/binkowsm/munit/ \
	--trainer=MUNIT
