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
--outfile results/cifar10-densepure-cm-sample_num_100000-noise_$sigma-12-7-vanila \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 0 50 1950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/cifar10/$sigma- \
--reverse_seed $reverse_seed \
--device 2
"""
python eval_certified_densepure.py \
--exp exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_$sigma-$steps-$reverse_seed \
--domain cifar10 \
--seed 0 \
--diffusion_type cm \
--lp_norm L2 \
--outfile results/cifar10-densepure-cm-sample_num_100000-noise_$sigma-v1-2 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 2000 50 3950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/cifar10/$sigma- \
--reverse_seed $reverse_seed \
--device 2 & \
python eval_certified_densepure.py \
--exp exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_$sigma-$steps-$reverse_seed \
--domain cifar10 \
--seed 0 \
--diffusion_type cm \
--lp_norm L2 \
--outfile results/cifar10-densepure-cm-sample_num_100000-noise_$sigma-v1-3 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 4000 50 5950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/cifar10/$sigma- \
--reverse_seed $reverse_seed \
--device 3 & \
python eval_certified_densepure.py \
--exp exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_$sigma-$steps-$reverse_seed \
--domain cifar10 \
--seed 0 \
--diffusion_type cm \
--lp_norm L2 \
--outfile results/cifar10-densepure-cm-sample_num_100000-noise_$sigma-v1-4 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 6000 50 7950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/cifar10/$sigma- \
--reverse_seed $reverse_seed \
--device 4 & \
python eval_certified_densepure.py \
--exp exp/cifar10 \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_$sigma-$steps-$reverse_seed \
--domain cifar10 \
--seed 0 \
--diffusion_type cm \
--lp_norm L2 \
--outfile results/cifar10-densepure-cm-sample_num_100000-noise_$sigma-v1-5 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 8000 50 9950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/cifar10/$sigma- \
--reverse_seed $reverse_seed \
--device 5
"""

