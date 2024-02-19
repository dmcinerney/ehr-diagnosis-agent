import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from utils import get_args
import pandas as pd
import os
from tqdm import trange, tqdm
import pickle as pkl
import numpy as np


def main():
    args = get_args('config.yaml')
    print(f'loading {args.write_to_cache.split} dataset...')
    df = pd.read_csv(os.path.join(
        args.data.path, args.data.dataset,
        f'{args.write_to_cache.split}.data'), compression='gzip')
    print(f'length={len(df)}')
    if args.write_to_cache.subset_file is not None:
        with open(args.write_to_cache.subset_file, 'rb') as f:
            subset = pkl.load(f)
    else:
        subset = None
    env_args = dict(**args.env['other_args'])
    env_args.update(
        instances=df,
        cache_path=args.write_to_cache.cache_dir,
        llm_name_or_interface=args.env.llm_name,
        fmm_name_or_interface=args.env.fmm_name,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        reward_type=args.env.reward_type,
        verbosity=1, # don't print anything when an environment is dead
        subset=subset,
    )
    env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/' + args.env['env_type'],
        **env_args,
    ) # type: ignore
    options = {}
    if args.data.max_reports_considered is not None:
        options['max_reports_considered'] = args.data.max_reports_considered
    dataset_length = env.total_num_examples()
    indices = list(range(dataset_length))
    if args.write_to_cache.slice is not None:
        slc = slice(*[int(i) for i in args.write_to_cache.slice.split(',')])
        print('Slicing dataset of length {} with {}'.format(
            dataset_length, str(slc)))
        indices = indices[slc]
    pbar = tqdm(indices, total=len(indices))
    valid_count = 0
    all_count = 0
    for i in pbar:
        if subset is not None and i not in subset:
            continue
        _, info = env.reset(options=dict(**options, instance_index=i))
        valid_count += info['is_valid_timestep']
        all_count += 1
        if args.write_to_cache.run_through_episode:
            # this writes queries to cache, which is helpful if a lot or
            # all of the queries are the same between different episodes
            terminated, truncated = not info['is_valid_timestep'], False
            step = 0
            while not (truncated or terminated):
                action = np.zeros_like(env.action_space.sample())
                _, _, terminated, truncated, info = env.step(action)
                step += 1
                max_observed_reports = args.write_to_cache[
                    'max_observed_reports']
                if max_observed_reports is not None and \
                        info['current_report'] + 1 >= max_observed_reports:
                    break
        pbar.set_postfix({
            'valid_instances': valid_count,
            'percentage_valid': valid_count / all_count
        })


if __name__ == '__main__':
    main()
