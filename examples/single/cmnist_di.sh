#!/bin/bash
I=0

./run_cmnist.py --config conf/sim_single_training.json \
    --env c_cause \
    --methods cnn_copa cnn_erm cnn_irm ora_erm \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-4 \
    --steps 20000 \
    --seed $I \
    --irm_lambda 0.01 \
    --out out/single_cmnist_cc/seed${I}
