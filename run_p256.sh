#!/bin/bash

#module load cuda/8.0.44

module use ~/projects/rpp-bengioy/modules/*/Core
source activate munit

python train.py --config ~/project/binek/munit/configs/testDD.yaml \
	--output_path ~/project/binek/munit/ \
	--trainer=MUNITDD --requeue=3
