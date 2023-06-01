export TRANSFORMERS_CACHE="/scratch/mcinerney.de/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/mcinerney.de/huggingface_cache"
module load anaconda3/3.7
module load cuda/11.8
source activate /work/frink/mcinerney.de/envs/ehragent
python -m pdb train.py
#python
