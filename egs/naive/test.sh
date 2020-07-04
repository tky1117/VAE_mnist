#!/usr/bin/bash

. ./path.sh

latent_dim=10
L=1 # Number of samples
H=200

batch_size=100
lr=0.001
epochs=100

exp_dir="./exp/latent${latent_dim}_L${L}_H${H}_b${batch_size}_lr${lr}_epochs${epochs}"
model_path="${exp_dir}/best.pth"

test.py \
--latent_dim ${latent_dim} \
--hidden_channels ${H} \
--n_samples ${L} \
--model_path "${model_path}" \
--save_dir "${exp_dir}"