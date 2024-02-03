#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
echo $1 $2 $3
split=$1
slice=$2
cache_evidence=$3
subset=$4
cache_dir=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_${split}_cache_flan
shared_args="write_to_cache.slice=${slice} write_to_cache.split=${split} write_to_cache.cache_dir=${cache_dir} write_to_cache.run_through_episode=${cache_evidence} write_to_cache.subset_file=${subset}"
python write_cache.py $shared_args
