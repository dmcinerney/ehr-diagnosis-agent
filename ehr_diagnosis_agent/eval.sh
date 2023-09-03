source activate /work/frink/mcinerney.de/envs/ehragent
train_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_train_cache_1
val_cache_path=/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/.cache/env_val1_cache_1
shared_args="env.train_cache_path=${train_cache_path} env.val_cache_path=${val_cache_path}"
python eval.py $shared_args env.add_risk_factor_queries=true env.top_k_evidence=100
