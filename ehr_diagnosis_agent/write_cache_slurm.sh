offset=0
# njobs=200
# split=train
# subset=null
njobs=20
split=val1
subset=null
jobsize=175
cache_evidence=true
for (( k = 0; k < njobs; ++k )); do
  slice="$(( jobsize*k + offset )),$(( jobsize*k + jobsize + offset ))"
  sbatch write_cache.sh $split $slice $cache_evidence $subset
  sleep 1s
done
