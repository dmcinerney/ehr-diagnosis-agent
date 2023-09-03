#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
echo $1
split=val2
cache_dir=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_${split}_cache_1
run_through_episode=true
python write_cache.py write_to_cache.slice=$1 write_to_cache.split=$split write_to_cache.cache_dir=$cache_dir write_to_cache.run_through_episode=$run_through_episode
