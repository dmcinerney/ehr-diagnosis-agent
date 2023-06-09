import ehr_diagnosis_env
import gymnasium
from utils import get_args
import pandas as pd
import os
from tqdm import trange, tqdm


def main():
    args = get_args('config.yaml')
    train_df = pd.read_csv(os.path.join(args.data.path, args.data.dataset, 'train.data'), compression='gzip')
    train_env = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=train_df,
        model_name=args.env.model_name,
        continuous_reward=args.env.continuous_reward,
        num_future_diagnoses_threshold=args.env.num_future_diagnoses_threshold,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        cache_path=args.env.cache_path,
    )
    options = {}
    if args.training.max_reports_considered is not None:
        options['max_reports_considered'] = args.training.max_reports_considered
    dataset_length = train_env.num_unseen_examples()
    for i in trange(dataset_length):
        train_env.reset(options=dict(**options, instance_index=i))


if __name__ == '__main__':
    main()
