#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
train_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_train_cache_2
val_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_val_cache_2
shared_args="env.train_cache_path=${train_cache_path} env.val_cache_path=${val_cache_path}"
# echo 'supervised clinical-bert (not interpretable) with risk factors'
# python train.py $shared_args actor.shared_params.embedder_type=bert
echo 'supervised interpretable with risk factors'
python train.py $shared_args
# echo 'supervised interpretable with risk factors + randomly drop evidence'
# python train.py $shared_args actor.shared_params.randomly_drop_evidence=.2
# echo 'supervised interpretable using all sentences'
# python train.py $shared_args actor.shared_params.use_raw_sentences=true env.exclude_evidence=true

# add clinical longformer and don't use evidence at all, only all raw reports


#other options:
# shape reward with attn?
# different diagnoses?
# different subsets?
# add a none of the above option?
