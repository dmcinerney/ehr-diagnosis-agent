import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from utils import get_args
import pandas as pd
import os
from tqdm import trange, tqdm
import pickle as pkl


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
    env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=df,
        cache_path=args.write_to_cache.cache_dir,
        llm_name_or_interface=args.env.llm_name,
        fmm_name_or_interface=args.env.fmm_name,
        reward_type=args.env.reward_type,
        num_future_diagnoses_threshold=args.env.num_future_diagnoses_threshold,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        top_k_evidence=args.env.top_k_evidence,
        verbosity=1, # don't print anything when an environment is dead
        add_risk_factor_queries=args.env.add_risk_factor_queries,
        limit_options_with_llm=args.env.limit_options_with_llm,
        add_none_of_the_above_option=args.env.add_none_of_the_above_option,
        true_positive_minimum=args.env.true_positive_minimum,
        use_confident_diagnosis_mapping=
            args.env.use_confident_diagnosis_mapping,
        skip_instances_with_gt_n_reports=
            args.env.skip_instances_with_gt_n_reports,
        subset=subset,
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
            while not (truncated or terminated):
                _, _, terminated, truncated, _ = env.step(
                    env.action_space.sample())
        pbar.set_postfix({
            'valid_instances': valid_count,
            'percentage_valid': valid_count / all_count
        })


if __name__ == '__main__':
    main()
