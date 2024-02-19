#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
# source activate /work/frink/mcinerney.de/envs/ehragent
# echo 'supervised clinical-bert (not interpretable) with risk factors'
# python train.py actor.shared_params.embedder_type=bert
# echo 'supervised clinical-bert (not interpretable) using all sentences'
# python train.py actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true
# echo 'supervised clinical-longformer (not interpretable) using all sentences'
# python train.py actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true actor.shared_params.model_name=yikuan8/Clinical-Longformer
echo 'supervised interpretable with risk factors'
# python3 train.py
python3 -m pdb train.py device=cpu
## echo 'supervised interpretable with risk factors + randomly drop evidence'
## python train.py actor.shared_params.randomly_drop_evidence=.2
# echo 'supervised interpretable using all sentences'
# python train.py actor.shared_params.use_raw_sentences=true


# echo 'supervised interpretable with risk factors (male)'
# python train.py training.train_subset_file=train_subset_male.pkl training.val_subset_file=val_subset_male.pkl
# echo 'supervised interpretable with risk factors (female)'
# python train.py training.train_subset_file=train_subset_female.pkl training.val_subset_file=val_subset_female.pkl
# echo 'supervised interpretable with risk factors (black)'
# python train.py training.train_subset_file=train_subset_white.pkl training.val_subset_file=val_subset_white.pkl
# echo 'supervised interpretable with risk factors (white)'
# python train.py training.train_subset_file=train_subset_black.pkl training.val_subset_file=val_subset_black.pkl


#other options:
# shape reward with attn?
# different diagnoses?
# different subsets?
# add a none of the above option?
