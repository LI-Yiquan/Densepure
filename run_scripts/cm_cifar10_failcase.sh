#!/usr/bin/env bash
cd ..

sigma=$1
steps=$2
reverse_seed=$3


python eval_certified_densepure.py \
--exp exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_$sigma-$steps-$reverse_seed \
--domain cifar10 \
--seed 0 \
--diffusion_type cm \
--lp_norm L2 \
--outfile results/cifar10-densepure-cm-sample_num_100000-noise_$sigma-failcase-12.14 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id 2300 3400 3600 4000 4550 5900 6500 7000 \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/cifar10/$sigma- \
--reverse_seed $reverse_seed \
--device 6