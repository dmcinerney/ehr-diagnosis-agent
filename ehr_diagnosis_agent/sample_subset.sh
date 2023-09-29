source activate /work/frink/mcinerney.de/envs/ehragent
train_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_train_cache_2
val_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_val_cache_2
shared_args="env.train_cache_path=${train_cache_path} env.val_cache_path=${val_cache_path}"
python sample_subset.py $shared_args
