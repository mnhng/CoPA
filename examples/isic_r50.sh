#!/bin/bash
I=0

./run_isic.py \
    --zlabels anatom_site_general age_approx sex \
    --conf "conf/isic.json" \
    --methods r50_copa r50_erm r50_irm \
    --batch_size 64 \
    --hidden_dim 256 \
    --seed $I \
    --lr 3e-5 \
    --steps 12000 \
    --irm_lambda 0.01 \
    --out out/isic_r50/seed${I}
