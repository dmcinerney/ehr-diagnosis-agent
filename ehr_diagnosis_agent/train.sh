#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
# echo 'supervised clinical-bert (not interpretable) with risk factors'
# python train.py actor.shared_params.embedder_type=bert
echo 'supervised clinical-bert (not interpretable) using all sentences'
python train.py actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true
# echo 'supervised clinical-longformer (not interpretable) using all sentences'
# python train.py actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true actor.shared_params.model_name=yikuan8/Clinical-Longformer
# echo 'supervised interpretable with risk factors'
# python train.py
## echo 'supervised interpretable with risk factors + randomly drop evidence'
## python train.py actor.shared_params.randomly_drop_evidence=.2
# echo 'supervised interpretable using all sentences'
# python train.py actor.shared_params.use_raw_sentences=true

# add clinical longformer and don't use evidence at all, only all raw reports


#other options:
# shape reward with attn?
# different diagnoses?
# different subsets?
# add a none of the above option?
