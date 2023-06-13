import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from utils import get_args
import pandas as pd
import os
from tqdm import trange, tqdm


def main():
    args = get_args('config.yaml')
    train_df = pd.read_csv(os.path.join(args.data.path, args.data.dataset, 'train.data'), compression='gzip')
    train_env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=train_df,
        model_name=args.env.model_name,
        reward_type=args.env.reward_type,
        num_future_diagnoses_threshold=args.env.num_future_diagnoses_threshold,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        cache_path=args.env.cache_path,
        top_k_evidence=args.env.top_k_evidence,
        verbosity=1, # don't print anything when an environment is dead
    ) # type: ignore
    options = {}
    dataset_length = train_env.num_unseen_examples()
    indices = list(range(dataset_length))
    if args.write_to_cache_slice is not None:
        slc = slice(*[int(i) for i in args.write_to_cache_slice.split(',')])
        print('Slicing dataset of length {} with {}'.format(dataset_length, str(slc)))
        indices = indices[slc]
    pbar = tqdm(indices, total=len(indices))
    valid_count = 0
    all_count = 0
    for i in pbar:
        obs, info = train_env.reset(options=dict(**options, instance_index=i))
        valid_count +=  not train_env.is_truncated(obs, info)
        all_count += 1
        pbar.set_postfix({'valid_instances': valid_count, 'percentage_valid': valid_count / all_count})


if __name__ == '__main__':
    main()
