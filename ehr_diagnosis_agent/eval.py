import warnings
import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
from ehr_diagnosis_env.utils import sort_by_scores
import gymnasium
from utils import get_args, collect_episode, ReplayBuffer
import pandas as pd
import os
from models.actor import InterpretableNormalActor, \
    InterpretableDirichletActor, InterpretableBetaActor
import torch
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf
import gc
import io


actory_types = {
    'normal': InterpretableNormalActor,
    'dirichlet': InterpretableDirichletActor,
    'beta': InterpretableBetaActor,
}
recommended_reward_types = {
    'normal': ['continuous_dependent', 'ranking'],
    'dirichlet': ['continuous_dependent', 'ranking'],
    'beta': ['continuous_independent', 'ranking'],
}


def get_option_targets(options, option_to_target, env=None, all_targets=None):
    option_targets = []
    for o in options:
        if o in option_to_target.keys():
            option_targets.append(option_to_target[o][0])
        elif all_targets is not None:
            assert env is not None
            im, bm = env.is_match([o], all_targets)
            option_targets.append(bm[0] if im[0] else None)
        else:
            option_targets.append(None)
    return option_targets


def episode_to_df(
        env, actor, episode_buffer, top_ks=(1, 2, 5), all_targets=None):
    assert episode_buffer.num_episodes() == 1
    rows = []
    for t, (obs, info, action, action_info, next_info) in enumerate(
        zip(
            episode_buffer.observations,
            episode_buffer.infos,
            episode_buffer.actions,
            episode_buffer.action_infos,
            episode_buffer.next_infos,
        )
    ):
        if obs['evidence_is_retrieved']:
            option_to_target = {
                o: (t, r) for o, t, r in next_info['true_positives']}
            options = pd.read_csv(io.StringIO(obs['options']))
            assert (options.type == 'diagnosis').all()
            sorted_options = sort_by_scores(options.option.to_list(), action)
            option_targets = get_option_targets(
                sorted_options, option_to_target, all_targets=all_targets,
                env=env)
            dist = actor.parameters_to_dist(*action_info['params'])
            mean = actor.get_mean([dist])[0]
            sorted_options_deterministic = sort_by_scores(
                options.option.to_list(), mean)
            option_targets_deterministic = get_option_targets(
                sorted_options_deterministic, option_to_target,
                all_targets=all_targets, env=env)
            targets = info['current_targets'] if all_targets is None else \
                all_targets
            for target in targets:
                row = {
                    'time_step': t,
                    'target': target,
                    'time_to_target': info['target_countdown'][target] \
                        if target in info['current_targets'] else None,
                    'is_positive': target in option_targets,
                    'is_current_target': target in info['current_targets'],
                }
                row.update({
                    f'top_{k}': target in option_targets[:k] for k in top_ks})
                row.update({
                    f'top_{k}_deterministic':
                        target in option_targets_deterministic[:k]
                        for k in top_ks})
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_on_environment(
        env, actor, options=None, max_num_episodes=None,
        max_trajectory_length=None, filename=None):
    if filename is not None:
        if os.path.exists(filename):
            print(f'overwriting results at {filename}')
            os.remove(filename)
        else:
            print(f'writing results to {filename}')
    options = {} if options is None else dict(**options)
    num_examples = env.num_examples()
    episode_dfs = []
    num_rows = 0
    with torch.no_grad():
        pbar = trange(
            num_examples, desc='collecting episode trajectories', leave=False)
        for episode in pbar:
            options['instance_index'] = episode
            obs, info = env.reset(options=options)
            replay_buffer = ReplayBuffer()
            if not collect_episode(
                    env, actor, replay_buffer, obs, info,
                    max_trajectory_length=max_trajectory_length):
                continue
            episode_dfs.append(episode_to_df(
                env, actor, replay_buffer,
                all_targets=env.all_reference_diagnoses))
            episode_dfs[-1]['episode_idx'] = [episode] * len(episode_dfs[-1])
            num_rows += len(episode_dfs[-1])
            pbar.set_postfix(
                {'valid_episodes_collected': len(episode_dfs),
                 'num_rows_collected': num_rows})
            if filename is not None:
                episode_dfs[-1].to_csv(
                    filename, index=False, mode='a',
                    header=len(episode_dfs) == 1)
            if max_num_episodes is not None and \
                    len(episode_dfs) >= max_num_episodes:
                break
    return pd.concat(episode_dfs)


def main():
    args = get_args('config.yaml')
    # run = wandb.init(
    #     project="ehr-diagnosis-agent-eval",
    #     dir=args.output_dir,
    #     config=OmegaConf.to_container(args) # type: ignore
    # )
    # assert run is not None
    print(f'loading {args.eval.split} dataset...')
    eval_df = pd.read_csv(os.path.join(
        args.data.path, args.data.dataset, f'{args.eval.split}.data'),
        compression='gzip')
    print(f'length={len(eval_df)}')
    if args.eval.limit_eval_size is not None:
        # truncate to first instances because data indices have to be
        # maintained to use the cache
        print(f'truncating to first {args.eval.limit_eval_size}')
        eval_df = eval_df[:args.eval.limit_eval_size]
    env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=eval_df,
        cache_path=args.env[f'{args.eval.split}_cache_path'],
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
    if args.env.reward_type not in recommended_reward_types[args.actor.type]:
        warnings.warn(
            'Reward type "{}" does not align with the actor type "{}".'.format(
            args.env.reward_type, args.actor.type))
    actor = actory_types[args.actor.type](
        args.actor['{}_params'.format(args.actor.type)])
    actor.eval()
    actor.set_device('cuda')
    print(f'Evaluating {args.eval.checkpoint}')
    ckpt = torch.load(args.eval.checkpoint)
    actor.load_state_dict(ckpt['actor'])
    del ckpt
    if args.training.clear_gpu: # only used for debugging
        gc.collect()
        torch.cuda.empty_cache()
    options = {}
    if args.data.max_reports_considered is not None:
        options['max_reports_considered'] = args.data.max_reports_considered
    evaluate_on_environment(
        env, actor, options=options,
        max_num_episodes=args.eval.max_num_episodes,
        max_trajectory_length=args.eval.max_trajectory_length,
        filename=args.eval.checkpoint[:-2] + 'csv')


if __name__ == '__main__':
    main()
