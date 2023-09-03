#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
train_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_train_cache_1
val_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_val_cache_1
shared_args="env.train_cache_path=${train_cache_path} env.val_cache_path=${val_cache_path} training.limit_train_size=10000"
# echo 'supervised clinical-bert (not interpretable) with risk factors'
# python train.py $shared_args actor.shared_params.embedder_type=bert
echo 'supervised interpretable with risk factors'
python -m pdb train.py $shared_args
# echo 'supervised interpretable with risk factors + randomly drop evidence'
# python train.py $shared_args actor.shared_params.randomly_drop_evidence=.2

# add clinical longformer and don't use evidence at all, only all raw reports


#other options:
# shape reward with attn?
# different diagnoses?
# different subsets?
# add a none of the above option?
