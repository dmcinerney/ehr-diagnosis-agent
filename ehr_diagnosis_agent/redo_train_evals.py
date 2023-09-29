import warnings
import ehr_diagnosis_env
from ehr_diagnosis_env.utils import get_model_interface
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from eval import evaluate_on_environment
from utils import get_args, collect_trajectories_train, TqdmSpy, \
    sample_actor_policy, CustomPolicy, log_results
import pandas as pd
import os
from models.actor import InterpretableNormalActor, \
    InterpretableDirichletActor, InterpretableBetaActor, InterpretableDeltaActor
from models.critic import Critic
import torch
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf
from updates import update, ppo_dae_update
import gc
import numpy as np
import random
import pickle as pkl


actor_types = {
    'normal': InterpretableNormalActor,
    'dirichlet': InterpretableDirichletActor,
    'beta': InterpretableBetaActor,
    'delta': InterpretableDeltaActor,
}
recommended_reward_types = {
    'normal': ['continuous_dependent', 'ranking'],
    'dirichlet': ['continuous_dependent', 'ranking'],
    'beta': ['continuous_independent', 'ranking'],
    'delta': ['continuous_independent', 'continuous_dependent', 'ranking'],
}


def main():
    args = get_args('config.yaml')
    print('loading training dataset...')
    train_df = pd.read_csv(os.path.join(
        args.data.path, args.data.dataset, 'train.data'), compression='gzip')
    print(f'length={len(train_df)}')
    if args.training.limit_train_size is not None:
        # truncate to first instances because data indices have to be
        # maintained to use the cache
        train_df = train_df[:args.training.limit_train_size]
    llm_interface = get_model_interface(args.env.llm_name) \
        if args.env.llm_name is not None else None
    fmm_interface = SentenceTransformer(args.env.fmm_name) \
        if args.env.fmm_name is not None else None
    if args.training.train_subset_file is None:
        subset = None
    else:
        with open(args.training.train_subset_file, 'rb') as f:
            subset = pkl.load(f)
    train_env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=train_df,
        cache_path=args.env.train_cache_path,
        llm_name_or_interface=llm_interface,
        fmm_name_or_interface=fmm_interface,
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
    if args.training.val_every is not None:
        print('loading validation dataset...')
        val_df = pd.read_csv(os.path.join(
            args.data.path, args.data.dataset, 'val1.data'), compression='gzip')
        print('done')
        if args.training.limit_val_size is not None:
            # truncate to first instances because data indices have to be
            # maintained to use the cache
            val_df = val_df[:args.training.limit_val_size]
        val_env: EHRDiagnosisEnv | None = gymnasium.make(
            'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
            instances=val_df,
            cache_path=args.env.val1_cache_path,
            llm_name_or_interface=llm_interface,
            fmm_name_or_interface=fmm_interface,
            reward_type=args.env.reward_type,
            num_future_diagnoses_threshold=
                args.env.num_future_diagnoses_threshold,
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
        ) # type: ignore
    else:
        val_env = None
    if args.env.reward_type not in recommended_reward_types[args.actor.type]:
        warnings.warn(
            'Reward type "{}" does not align with the actor type "{}".'.format(
            args.env.reward_type, args.actor.type))
    actor_params = args.actor['{}_params'.format(args.actor.type)]
    actor_params.update(args.actor['shared_params'])
    actor = actor_types[args.actor.type](actor_params)
    actor.set_device('cuda')
    if args.training.objective_optimization in [
            'ppo_dae', 'ppo_gae', 'mix_objectives']:
        critic = Critic(args.critic)
    else:
        critic = None
    seed_offset = 0
    options = {}
    if args.data.max_reports_considered is not None:
        options['max_reports_considered'] = args.data.max_reports_considered
    updates = 0
    dataset_iterations = 0
    dataset_progress = TqdmSpy(
        desc='dataset progress (iteration {})'.format(dataset_iterations),
        total=train_env.num_unseen_examples(), leave=True)
    ckpt_files = [
        x for x in os.listdir(args.redo_train_evals) if x.endswith('.pt')]
    for epoch in trange(args.training.num_epochs, desc='epochs'):
        ckpt_file = [
            x for x in ckpt_files
            if x.startswith(f'ckpt_epoch={epoch}_updates=')]
        if len(ckpt_file) == 0:
            continue
        ckpt_file = ckpt_file[0]
        full_ckpt_file = os.path.join(args.redo_train_evals, ckpt_file)
        print(f'Resuming from {full_ckpt_file}')
        ckpt = torch.load(full_ckpt_file)
        actor.load_state_dict(ckpt['actor'])
        if critic is not None and 'critic' in ckpt.keys():
            critic.load_state_dict(ckpt['critic'])
        seed_offset = ckpt['seed_offset']
        del ckpt
        if args.training.clear_gpu: # only used for debugging
            gc.collect()
            torch.cuda.empty_cache()
        if critic is not None:
            critic.set_device('cpu')
            if args.training.clear_gpu: # only used for debugging
                gc.collect()
                torch.cuda.empty_cache()
        train_env.to('cuda') # train and val envs have connected models
        # Note: all the evaluation and trajectory collecting functions control
        #   their random seeds
        # validate current model
        results_file = os.path.join(
            args.redo_train_evals, 'val_metrics_{}.csv'.format(ckpt_file[:-3]))
        step_results, episode_results = evaluate_on_environment(
            val_env, actor, options=options,
            max_num_episodes=args.training.val_max_num_episodes,
            max_trajectory_length=args.training.val_max_trajectory_length,
            custom_policy=CustomPolicy(
                args.training.eval_random_query_policy,
                args.training.eval_random_rp_policy),
            filename=results_file,
            use_random_start_idx=
                args.training.eval_val_use_random_start_idx)
        results_file = os.path.join(
            args.redo_train_evals, 'train_metrics_{}.csv'.format(ckpt_file[:-3]))
        step_results, episode_results = evaluate_on_environment(
            train_env, actor, options=dict(add_to_seen=False, **options),
            max_num_episodes=args.training.trainmetrics_max_num_episodes,
            max_trajectory_length=
                args.training.trainmetrics_max_trajectory_length,
            custom_policy=CustomPolicy(
                args.training.eval_random_query_policy,
                args.training.eval_random_rp_policy),
            filename=results_file,
            use_random_start_idx=
                args.training.eval_train_use_random_start_idx)


if __name__ == '__main__':
    main()
