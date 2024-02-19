import warnings
import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
from ehr_diagnosis_env.utils import sort_by_scores
import gymnasium
from utils import get_args, collect_episode, ReplayBuffer
import pandas as pd
import os
from models.actor import InterpretableNormalActor, \
    InterpretableDirichletActor, InterpretableBetaActor, InterpretableDeltaActor
import torch
from tqdm import trange, tqdm
from utils import CustomPolicy, fast_forward_to_random_idx
import gc
import io
import numpy as np


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


def get_top_k_evidence_param_info(actor, action_info, top_k):
    assert isinstance(actor, InterpretableDeltaActor)
    assert actor.attention is None
    if 'evidence_votes' not in action_info.keys():
        new_vote_info = {
            'diagnosis_strings': action_info['diagnosis_strings'],
            'context_strings': action_info['context_strings'],
            'context_info': action_info['context_info'],
            'param_votes': action_info['param_votes'],
        }
    else:
        scores = (action_info['evidence_votes'].squeeze(2) ** 2).mean(0)
        indices = torch.topk(
            scores, min(top_k, scores.shape[0]))[1].detach().cpu()
        if len(scores) != len(action_info['context_strings']):
            assert len(scores) == len(action_info['context_strings']) - 1
            indices = torch.cat([torch.tensor([0]), indices + 1])
        new_vote_info = {
            'diagnosis_strings': action_info['diagnosis_strings'],
            'context_strings': [
                action_info['context_strings'][i] for i in indices],
            'context_info': [
                action_info['context_info'][i] for i in indices],
            'param_votes': action_info['param_votes'][:, indices],
        }
    return new_vote_info, actor.votes_to_parameters(new_vote_info)


def get_prediction_rows(
        env, actor, timestep, obs, info, action, action_info, next_info,
        top_ks=(1, 2), all_targets=None):
    rows = []
    option_to_target = {
        o: (t, r) for o, t, r in next_info['true_positives']}
    options = pd.read_csv(io.StringIO(obs['options']))
    assert (options.type == 'diagnosis').all()
    option_targets = get_option_targets(
        options.option.to_list(), option_to_target,
        all_targets=all_targets, env=env)
    sorted_option_targets = sort_by_scores(option_targets, action)
    dist = actor.parameters_to_dist(*action_info['params'])
    action_det = actor.get_mean([dist])[0]
    sorted_option_targets_deterministic = sort_by_scores(
        option_targets, action_det)
    targets = set(option_targets).union(info['current_targets']) \
        if all_targets is None else all_targets
    for target in targets:
        action_target_score = 0
        action_target_score_deterministic = 0
        for t, score, score_det in zip(option_targets, action, action_det):
            if target == t:
                action_target_score += score.item()
                action_target_score_deterministic += score_det.item()
        row = {
            'time_step': timestep,
            'target': target,
            'time_to_target': info['target_countdown'][target] \
                if target in info['current_targets'] else np.nan,
            'is_positive': target in option_targets,
            'is_current_target': target in info['current_targets'],
            'action_target_score': action_target_score,
            'action_target_score_deterministic':
                action_target_score_deterministic,
        }
        row.update({
            f'top_{k}': target in sorted_option_targets[:k] for k in top_ks})
        row.update({
            f'top_{k}_deterministic':
                target in sorted_option_targets_deterministic[:k]
                for k in top_ks})
        rows.append(row)
    return pd.DataFrame(rows)


def episode_to_df(
        env, actor, episode_buffer, top_ks=(1, 2),
        top_k_evidence=(5, 10, 20, 40, 80), all_targets=None):
    assert episode_buffer.num_episodes() == 1
    prediction_dfs = []
    episode_row = {
        'cumulative_reward': 0,
        'num_steps': len(episode_buffer.rewards),
        'all_targets': episode_buffer.infos[0]['current_targets'],
    }
    evidence_rows = []
    for timestep, (obs, info, action, action_info, next_info, reward) in \
            enumerate(zip(
                episode_buffer.observations,
                episode_buffer.infos,
                episode_buffer.actions,
                episode_buffer.action_infos,
                episode_buffer.next_infos,
                episode_buffer.rewards,
            )):
        episode_row['cumulative_reward'] += reward
        if obs['evidence_is_retrieved']:
            if 'evidence_votes' in action_info.keys():
                evidence_votes = action_info['evidence_votes']
                num_evidence = action_info['num_evidence']
                evidence = action_info['context_strings'][-num_evidence:]
                for i, ev in enumerate(evidence):
                    params = actor.transform_parameters(
                        {'params': evidence_votes[:, i]})['params']
                    dist = actor.parameters_to_dist(*params)
                    ev_votes = actor.get_mean([dist])[0]
                    evidence_rows.append({
                        'time_step': timestep,
                        'evidence': ev,
                        **{f'{condition} vote': x.item()
                           for condition, x in zip(
                               action_info['diagnosis_strings'], ev_votes)}
                    })
            prediction_df = get_prediction_rows(
                env, actor, timestep, obs, info, action, action_info,
                next_info, top_ks=top_ks, all_targets=all_targets)
            if 'num_evidence' in action_info.keys() and \
                    isinstance(actor, InterpretableDeltaActor) and \
                    actor.attention is None and not prediction_df.empty:
                for k in top_k_evidence:
                    vote_info_temp, param_info_temp = \
                        get_top_k_evidence_param_info(
                        actor, action_info, k)
                    dist_temp = actor.parameters_to_dist(
                        *param_info_temp['params'])
                    action_temp = dist_temp.rsample()
                    action_info_temp = {
                        'action': action_temp,
                        'log_prob': dist_temp.log_prob(action),
                        'diagnosis_strings':
                            vote_info_temp['diagnosis_strings'],
                        'context_strings': vote_info_temp['context_strings'],
                        'context_info': vote_info_temp['context_info'],
                        **param_info_temp,
                    }
                    prediction_df_temp = get_prediction_rows(
                        env, actor, timestep, obs, info, action_temp,
                        action_info_temp, next_info, top_ks=top_ks,
                        all_targets=all_targets)
                    prediction_df = prediction_df.merge(
                        prediction_df_temp, on=[
                            'time_step', 'target'],
                            suffixes=('', f'_top-{k}-evidence'))
            prediction_dfs.append(prediction_df)
    episode_row['avg_reward'] = \
        episode_row['cumulative_reward'] / episode_row['num_steps']
    return pd.DataFrame(evidence_rows), pd.concat(prediction_dfs), episode_row


def evaluate_on_environment(
        env, actor, options=None, max_num_episodes=None,
        max_trajectory_length=None, custom_policy=None, filename=None,
        use_random_start_idx=False, seed_offset=0, max_observed_reports=None):
    actor.eval()
    if filename is not None:
        evidence_filename = filename.replace('.csv', '_evidence.csv')
        steps_filename = filename.replace('.csv', '_steps.csv')
        episodes_filename = filename.replace('.csv', '_episodes.csv')
        for x in [evidence_filename, steps_filename, episodes_filename]:
            if os.path.exists(x):
                print(f'overwriting results at {x}')
                os.remove(x)
            else:
                print(f'writing results to {x}')
    options = {} if options is None else dict(**options)
    num_examples = env.num_examples()
    evidence_dfs = []
    steps_dfs = []
    episodes_df = []
    num_rows = 0
    with torch.no_grad():
        pbar = trange(
            num_examples, desc='collecting episode trajectories', leave=False)
        for episode in pbar:
            options['instance_index'] = episode
            obs, info = env.reset(options=options)
            if not info['is_valid_timestep']:
                continue
            if use_random_start_idx:
                # obs, info = reset_at_random_idx(
                #     env, info, seed=episode + seed_offset, options=options)
                obs, info = fast_forward_to_random_idx(
                    env, obs, info, seed=episode + seed_offset,
                    max_observed_reports=max_observed_reports)
            replay_buffer = ReplayBuffer()
            assert collect_episode(
                env, actor, replay_buffer, obs, info,
                max_trajectory_length=max_trajectory_length,
                custom_policy=custom_policy,
                max_observed_reports=max_observed_reports)
            evidence_df, steps_df, episode_row = episode_to_df(
                env, actor, replay_buffer,
                all_targets=env.all_reference_diagnoses)
            evidence_dfs.append(evidence_df)
            steps_dfs.append(steps_df)
            episodes_df.append(episode_row)
            evidence_dfs[-1]['episode_idx'] = [episode] * len(evidence_dfs[-1])
            steps_dfs[-1]['episode_idx'] = [episode] * len(steps_dfs[-1])
            episodes_df[-1]['episode_idx'] = episode
            num_rows += len(steps_dfs[-1])
            pbar.set_postfix(
                {'valid_episodes_collected': len(steps_dfs),
                 'num_rows_collected': num_rows})
            if filename is not None:
                evidence_dfs[-1].to_csv(
                    evidence_filename, index=False,
                    mode='a', header=len(evidence_dfs) == 1)
                steps_dfs[-1].to_csv(
                    steps_filename, index=False,
                    mode='a', header=len(steps_dfs) == 1)
                pd.DataFrame(episodes_df[-1:]).to_csv(
                    episodes_filename, index=False,
                    mode='a', header=len(episodes_df) == 1)
            if max_num_episodes is not None and \
                    len(steps_dfs) >= max_num_episodes:
                break
    return pd.concat(evidence_dfs), pd.concat(steps_dfs), \
        pd.DataFrame(episodes_df)


def main():
    args = get_args('config.yaml')
    # run = wandb.init(
    #     project="ehr-diagnosis-agent-eval",
    #     dir=args.output_dir,
    #     config=OmegaConf.to_container(args) # type: ignore
    # )
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
    env_args = dict(**args.env['other_args'])
    env_args.update(
        instances=eval_df,
        cache_path=args.env[f'{args.eval.split}_cache_path'],
        llm_name_or_interface=args.env.llm_name,
        fmm_name_or_interface=args.env.fmm_name,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        reward_type=args.env.reware_type,
        verbosity=1, # don't print anything when an environment is dead
    )
    env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/' + args.env['env_type'],
        **env_args,
    ) # type: ignore
    if args.env.reward_type not in recommended_reward_types[args.actor.type]:
        warnings.warn(
            'Reward type "{}" does not align with the actor type "{}".'.format(
            args.env.reward_type, args.actor.type))
    actor_params = args.actor['{}_params'.format(args.actor.type)]
    actor_params.update(args.actor['shared_params'])
    actor_params['static_bias_params'] = actor_params['eval_static_bias_params']
    actor = actor_types[args.actor.type](actor_params)
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
    path = '/'.join(args.eval.checkpoint.split('/')[:-1])
    filename = f'full_{args.eval.split}_metrics_' + args.eval.checkpoint.split(
        '/')[-1][:-3] + f'_seed={args.eval.seed_offset}' + '.csv'
    evaluate_on_environment(
        env, actor, options=options,
        max_num_episodes=args.eval.max_num_episodes,
        max_trajectory_length=args.eval.max_trajectory_length,
        custom_policy=CustomPolicy(
            args.eval.query_policy,
            args.eval.rp_policy),
        filename=os.path.join(path, filename),
        use_random_start_idx=args.eval.use_random_start_idx,
        seed_offset=args.eval.seed_offset,
        max_observed_reports=args.eval.max_observed_reports)


if __name__ == '__main__':
    main()
