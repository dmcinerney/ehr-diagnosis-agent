import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from torch import negative_
from utils import get_args, get_mimic_demographics
import pandas as pd
import os
from tqdm import trange, tqdm
import pickle as pkl
import numpy as np


def main():
    args = get_args('config.yaml')
    print(f'loading {args.sample_subset.split} dataset...')
    df = pd.read_csv(os.path.join(
        args.data.path, args.data.dataset,
        f'{args.sample_subset.split}.data'), compression='gzip')
    print(f'length={len(df)}')
    env_args = dict(**args.env['other_args'])
    env_args.update(
        instances=eval_df,
        cache_path=args.env[f'{args.eval.split}_cache_path'],
        llm_name_or_interface=None,
        fmm_name_or_interface=None,
        fuzzy_matching_threshold=None,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        reward_type=args.env.reware_type,
        verbosity=1, # don't print anything when an environment is dead
    )
    env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/' + args.env['env_type'],
        **env_args,
    ) # type: ignore
    instances = env.get_cached_instance_dataframe()
    instances = pd.concat([df, instances], axis=1)
    if args.sample_subset.gender is not None or \
            args.sample_subset.race is not None:
        instances = instances.merge(get_mimic_demographics(
            args.data.mimic_folder_for_demographics), on="patient_id")
    if args.sample_subset.gender is not None:
        instances = instances[instances.gender == args.sample_subset.gender]
    if args.sample_subset.race is not None:
        instances = instances[instances.race == args.sample_subset.race]
    if args.sample_subset.negatives_percentage is not None:
        positive_filter = instances.apply(
            lambda r: r['is valid timestep'] is not None and \
                r['is valid timestep'] == r['is valid timestep'] and \
                sum(r['is valid timestep']) > 0 and \
                len(r['target diagnosis countdown'][0]) > 0, axis=1)
        positive_instances = instances[positive_filter]
        negative_filter = instances.apply(
            lambda r: r['is valid timestep'] is not None and \
                r['is valid timestep'] == r['is valid timestep'] and \
                sum(r['is valid timestep']) > 0 and \
                len(r['target diagnosis countdown'][0]) == 0, axis=1)
        negative_instances = instances[negative_filter]
        np.random.seed(args.sample_subset.seed)
        num_negatives = int(len(positive_instances) *
            args.sample_subset.negatives_percentage /
            (1 - args.sample_subset.negatives_percentage))
        #sampled_negatives = np.random.choice(
        #    negative_instances, size=num_negatives)
        #instances = set(positive_instances).union(set(sampled_negatives))
        sampled_negatives = negative_instances.sample(n=num_negatives)
        instances = pd.concat([positive_instances, sampled_negatives])
    instance_indices = set(sorted(list(instances.index)))
    with open(args.sample_subset.subset_file, 'wb') as f:
        pkl.dump(instance_indices, f)
    print('done')


if __name__ == '__main__':
    main()
