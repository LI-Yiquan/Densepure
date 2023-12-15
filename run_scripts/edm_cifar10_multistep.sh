#!/usr/bin/env bash
cd ..

sigma=$1

python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type edm \
--lp_norm L2 \
--outfile results/cifar10-carlini22-edm-sample_num_100000-noise_$sigma-multistep_1_test \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 500 50 1950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--device 1 
'
python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type edm \
--lp_norm L2 \
--outfile results/cifar10-carlini22-edm-sample_num_100000-noise_$sigma-multistep_2_continue \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 2500 50 3950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--device 2 & \
python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type edm \
--lp_norm L2 \
--outfile results/cifar10-carlini22-edm-sample_num_100000-noise_$sigma-multistep_3_continue \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 4500 50 5950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--device 3 & \
python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type edm \
--lp_norm L2 \
--outfile results/cifar10-carlini22-edm-sample_num_100000-noise_$sigma-multistep_4_continue \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 6500 50 7950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--device 4 & \
python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type edm \
--lp_norm L2 \
--outfile results/cifar10-carlini22-edm-sample_num_100000-noise_$sigma-multistep_5_continue \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 8100 50 9950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--device 5 
'