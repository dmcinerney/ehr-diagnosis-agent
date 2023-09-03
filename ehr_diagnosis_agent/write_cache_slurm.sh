offset=0
#njobs=30
njobs=140
jobsize=250
for (( k = 0; k < njobs; ++k )); do
  slice="$(( jobsize*k + offset )),$(( jobsize*k + jobsize + offset ))"
  sbatch write_cache.sh $slice
  sleep 1s
done
