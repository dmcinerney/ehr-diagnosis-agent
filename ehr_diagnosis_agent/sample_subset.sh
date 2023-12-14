#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
#python sample_subset.py
#python sample_subset.py sample_subset.subset_file=train_subset_male.pkl sample_subset.gender=M
#python sample_subset.py sample_subset.subset_file=val_subset_male.pkl sample_subset.gender=M sample_subset.split=val1 sample_subset.negatives_percentage=null
#python sample_subset.py sample_subset.subset_file=train_subset_female.pkl sample_subset.gender=F
#python sample_subset.py sample_subset.subset_file=val_subset_female.pkl sample_subset.gender=F sample_subset.split=val1 sample_subset.negatives_percentage=null
python sample_subset.py sample_subset.subset_file=train_subset_white.pkl sample_subset.race=WHITE
python sample_subset.py sample_subset.subset_file=val_subset_white.pkl sample_subset.race=WHITE sample_subset.split=val1 sample_subset.negatives_percentage=null
python sample_subset.py sample_subset.subset_file=train_subset_black.pkl sample_subset.race=BLACK
python sample_subset.py sample_subset.subset_file=val_subset_black.pkl sample_subset.race=BLACK sample_subset.split=val1 sample_subset.negatives_percentage=null
