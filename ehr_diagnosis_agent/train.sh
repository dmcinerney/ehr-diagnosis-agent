#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
train_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_train_cache_1
val_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_val_cache_1
shared_args="env.train_cache_path=${train_cache_path} env.val_cache_path=${val_cache_path} training.random_query_policy=true training.random_rp_policy=true"
#echo 'supervised bert ignore evidence'
#python train.py $shared_args actor.shared_params.embedder_type=bert actor.shared_params.ignore_evidence=true
#echo 'supervised bert'
#python train.py $shared_args actor.shared_params.embedder_type=bert
#echo 'supervised interpretable'
#python train.py $shared_args
#echo 'supervised bert for queries with risk factors'
#python train.py $shared_args actor.shared_params.embedder_type=bert env.add_risk_factor_queries=true env.top_k_evidence=100
echo 'supervised interpretable for queries with risk factors'
python train.py $shared_args env.add_risk_factor_queries=true env.top_k_evidence=100
#python train.py $shared_args env.add_risk_factor_queries=true env.top_k_evidence=100 training.resume_from=/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230804_090224-gzm7bwt8/files/ckpt_epoch=82_updates=5312.pt
#echo 'supervised + rl bert for queries with risk factors'
#python train.py $shared_args training.random_query_policy=false actor.shared_params.embedder_type=bert training.objective_optimization=mix_objectives env.add_risk_factor_queries=true
#echo 'supervised + rl interpretable for queries with risk factors'
#python train.py $shared_args training.random_query_policy=false training.objective_optimization=mix_objectives env.add_risk_factor_queries=true

#other options:
# shape reward with attn?
# different diagnoses?
# different subsets?
# add a none of the above option?
