#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
echo $1 $2 $3
split=$1
slice=$2
cache_evidence=$3
cache_dir=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_${split}_cache_2
shared_args="write_to_cache.slice=${slice} write_to_cache.split=${split} write_to_cache.cache_dir=${cache_dir} write_to_cache.run_through_episode=${cache_evidence}"
if [ "$cache_evidence" = "true" ] && [ "$split" = "train" ]; then
    echo "using train subset"
    shared_args="${shared_args} write_to_cache.subset_file=train_subset.pkl"
fi
python write_cache.py $shared_args
