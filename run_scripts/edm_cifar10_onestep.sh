#!/usr/bin/env bash
cd ..

sigma=$1

python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type edm_onestep \
--lp_norm L2 \
--outfile results/cifar10-edm-sample_num_100000-noise_$sigma-onestep_12-6 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 0 50 1950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--device 5
'
python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type edm_onestep \
--lp_norm L2 \
--outfile results/cifar10-edm-sample_num_100000-noise_$sigma-onestep_2 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 2000 50 3950) \
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
--diffusion_type edm_onestep \
--lp_norm L2 \
--outfile results/cifar10-edm-sample_num_100000-noise_$sigma-onestep_3 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 4000 50 5950) \
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
--diffusion_type edm_onestep \
--lp_norm L2 \
--outfile results/cifar10-edm-sample_num_100000-noise_$sigma-onestep_4 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 6000 50 7950) \
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
--diffusion_type edm_onestep \
--lp_norm L2 \
--outfile results/cifar10-edm-sample_num_100000-noise_$sigma-onestep_5 \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 8000 50 9950) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--device 5 
'