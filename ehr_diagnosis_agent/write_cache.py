import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from utils import get_args
import pandas as pd
import os
from tqdm import trange, tqdm


def main():
    args = get_args('config.yaml')
    print(f'loading {args.write_to_cache.split} dataset...')
    df = pd.read_csv(os.path.join(
        args.data.path, args.data.dataset,
        f'{args.write_to_cache.split}.data'), compression='gzip')
    print(f'length={len(df)}')
    env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=df,
        cache_path=args.env[f'{args.write_to_cache.split}_cache_path'],
        llm_name_or_interface=args.env.llm_name,
        fmm_name_or_interface=args.env.fmm_name,
        reward_type=args.env.reward_type,
        num_future_diagnoses_threshold=args.env.num_future_diagnoses_threshold,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        top_k_evidence=args.env.top_k_evidence,
        verbosity=1, # don't print anything when an environment is dead
        add_risk_factor_queries=args.env.add_risk_factor_queries,
        limit_options_with_llm=args.env.limit_options_with_llm,
    ) # type: ignore
    options = {}
    if args.data.max_reports_considered is not None:
        options['max_reports_considered'] = args.data.max_reports_considered
    dataset_length = env.num_examples()
    indices = list(range(dataset_length))
    if args.write_to_cache.slice is not None:
        slc = slice(*[int(i) for i in args.write_to_cache.slice.split(',')])
        print('Slicing dataset of length {} with {}'.format(dataset_length, str(slc)))
        indices = indices[slc]
    pbar = tqdm(indices, total=len(indices))
    valid_count = 0
    all_count = 0
    for i in pbar:
        obs, info = env.reset(options=dict(**options, instance_index=i))
        valid_count +=  not env.is_truncated(obs, info)
        all_count += 1
        pbar.set_postfix({'valid_instances': valid_count, 'percentage_valid': valid_count / all_count})


if __name__ == '__main__':
    main()
