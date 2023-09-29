offset=0
#njobs=200
#split=train
njobs=25
split=val2
jobsize=175
cache_evidence=true
for (( k = 0; k < njobs; ++k )); do
  slice="$(( jobsize*k + offset )),$(( jobsize*k + jobsize + offset ))"
  sbatch write_cache.sh $split $slice $cache_evidence
  sleep 1s
done
