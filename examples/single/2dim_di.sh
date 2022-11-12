#!/bin/bash
I=0

./run_2dim.py --config conf/sim_single_training.json \
    --env c_cause \
    --methods mlp_copa mlp_erm mlp_irm \
    --batch_size 512 \
    --hidden_dim 10 \
    --lr 1e-4 \
    --steps 20000 \
    --seed $I \
    --irm_lambda 0.01 \
    --out out/single_2dim_cc/seed${I}
