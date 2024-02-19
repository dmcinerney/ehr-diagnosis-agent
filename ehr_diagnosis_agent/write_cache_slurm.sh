# offset=0
offset=700
# njobs=200
njobs=1
split=train
subset=null
# njobs=1
# split=val1
# subset=null
jobsize=175
cache_evidence=true
for (( k = 0; k < njobs; ++k )); do
  slice="$(( jobsize*k + offset )),$(( jobsize*k + jobsize + offset ))"
  # sbatch write_cache.sh $split $slice $cache_evidence $subset
  sbatch /PHShome/dzm44/run_gpu_job.sbatch /PHShome/dzm44/code/ehr-diagnosis-agent/ehr_diagnosis_agent/write_cache.sh $split $slice $cache_evidence $subset
  sleep 1s
done
