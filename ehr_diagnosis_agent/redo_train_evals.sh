#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
# source activate /work/frink/mcinerney.de/envs/ehragent
echo 'supervised interpretable using all sentences'
# python redo_train_evals.py actor.shared_params.use_raw_sentences=true redo_train_evals=/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230925_104538-57jsyt8v/files
python3 redo_train_evals.py redo_train_evals=/PHShome/dzm44/code/ehr-diagnosis-agent/ehr_diagnosis_agent/outputs/wandb/offline-run-20240216_141857-tfvhvc4e/files

# add clinical longformer and don't use evidence at all, only all raw reports


#other options:
# shape reward with attn?
# different diagnoses?
# different subsets?
# add a none of the above option?
