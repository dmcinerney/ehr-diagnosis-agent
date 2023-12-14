from collections import defaultdict
from omegaconf import OmegaConf
import torch
from tqdm import trange, tqdm
import wandb
import random
import pandas as pd
import io
from torch import nn
from sklearn.metrics import roc_auc_score
import os


def get_args(config_file):
    args_from_cli = OmegaConf.from_cli()
    args_from_yaml = OmegaConf.load(config_file)
    return OmegaConf.merge(args_from_yaml, args_from_cli)


class CustomPolicy:
    def __init__(self, random_query_policy, random_rp_policy):
        self.random_query_policy = random_query_policy
        self.random_rp_policy = random_rp_policy
    def __call__(self, actor, obs, env):
        device = next(iter(actor.parameters())).device
        if self.random_query_policy and not obs['evidence_is_retrieved']:
            return torch.tensor(env.action_space.sample(), device=device), \
                torch.tensor(0, device=device), {}
        if self.random_rp_policy and obs['evidence_is_retrieved']:
            return torch.tensor(env.action_space.sample(), device=device), \
                torch.tensor(0, device=device), {}
        return sample_actor_policy(actor, obs, env)


def sample_actor_policy(actor, obs, env):
    # sample from actor policy
    action_info = actor(obs)
    action = action_info['action']
    del action_info['action']
    action_log_prob = action_info['log_prob']
    del action_info['log_prob']
    return action, action_log_prob, action_info


def collect_episode(
        env, actor, replay_buffer, obs, info, max_trajectory_length=None,
        custom_policy=None):
    # Collect the episode trajectory given a starting state/environment.
    # If the starting state/environment is valid, return True and add
    # trajectory to the buffer, otherwise return False.
    if not info['is_valid_timestep']:
        return False
    t = 0
    sample_exploration_policy = sample_actor_policy \
        if custom_policy is None else custom_policy
    terminated, truncated = not info['is_valid_timestep'], False
    while not (terminated or truncated):
        action, action_log_prob, action_info = sample_exploration_policy(
            actor, obs, env)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        replay_buffer.append(
            obs, info, action, action_log_prob, reward,
            terminated, truncated, next_obs, next_info, action_info)
        obs = next_obs
        info = next_info
        t += 1
        if max_trajectory_length is not None and t >= max_trajectory_length:
            break
    return True


def reset_env_train(
        env, seed, options, dataset_iterations, dataset_progress, epoch,
        log=True):
    new_dataset_iteration = env.num_unseen_examples() == 0
    obs, info = env.reset(seed=seed, options=options)
    if new_dataset_iteration:
        dataset_progress.reset(total=env.num_unseen_examples())
        dataset_iterations += 1
    dataset_progress.update(1)
    if log:
        wandb.log({
            'epoch': epoch,
            'num_episodes_seen': dataset_progress.n +
                env.num_examples() * dataset_iterations,
            'percentage_dataset_seen': dataset_progress.n / env.num_examples()
                + dataset_iterations})
    return obs, info, dataset_iterations


def reset_at_random_idx(env, info, seed=None, options=None):
    if seed is not None:
        random.seed(seed)
    new_options = {
        'start_report_index': random.randint(
            info['current_report'],
            info['current_report'] + info['max_timesteps'] // 2 - 1),
        'instance_index': info['instance_index'],
    }
    if options is not None:
        new_options.update(options)
    return env.reset(seed=seed, options=new_options)


def collect_trajectories_train(
        args, train_env, options, actor, epoch, seed_offset,
        dataset_iterations, dataset_progress, log=True, custom_policy=None):
    replay_buffer = ReplayBuffer()
    actor.train()
    with torch.no_grad():
        for episode in trange(
                args.training.num_episodes,
                desc='collecting episode trajectories', leave=False):
            obs, info, dataset_iterations = reset_env_train(
                train_env, epoch * episode + seed_offset, options,
                dataset_iterations, dataset_progress, epoch)
            while not info['is_valid_timestep']:
                # dead environment, retrying...
                seed_offset += 1
                obs, info, dataset_iterations = reset_env_train(
                    train_env, epoch * episode + seed_offset, options,
                    dataset_iterations, dataset_progress, epoch)
            if args.training.use_random_start_idx:
                obs, info = reset_at_random_idx(
                    train_env, info, seed=epoch * episode + seed_offset,
                    options=options)
            assert collect_episode(
                train_env, actor, replay_buffer, obs, info,
                max_trajectory_length=args.training.max_trajectory_length,
                custom_policy=custom_policy)
            if log:
                wandb.log({
                    'epoch': epoch,
                    'num_episodes_collected':
                        epoch * args.training.num_episodes + episode + 1
                })
    if args.training.shape_reward_with_attn:
        replay_buffer.shape_reward_with_attn()
    return replay_buffer, seed_offset, dataset_iterations


class ReplayBuffer:
    def __init__(
            self, observations=None, infos=None, actions=None,
            action_log_probs=None, rewards=None, terminated=None,
            truncated=None, next_observations=None, next_infos=None,
            action_infos=None, episode_id=0, episode_ids=None):
        self.observations = [] if observations is None else observations
        self.infos = [] if infos is None else infos
        self.actions = [] if actions is None else actions
        self.action_log_probs = [] if action_log_probs is None else \
            action_log_probs
        self.rewards = [] if rewards is None else rewards
        self.terminated = [] if terminated is None else terminated
        self.truncated = [] if truncated is None else truncated
        self.next_observations = [] if next_observations is None else \
            next_observations
        self.next_infos = [] if next_infos is None else next_infos
        self.action_infos = [] if action_infos is None else action_infos
        self.episode_id = episode_id
        self.episode_ids = [] if episode_ids is None else episode_ids

    def append(
            self, observation, info, action, action_log_prob, reward,
            terminated, truncated, next_observation, next_info, action_info):
        self.observations.append(observation)
        self.infos.append(info)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.next_observations.append(next_observation)
        self.next_infos.append(next_info)
        self.action_infos.append(action_info)
        self.episode_ids.append(self.episode_id)
        if terminated or truncated:
            self.episode_id += 1

    def get_trajectories(self):
        trajectories = [[]]
        for o, i, a, a_log_prob, r, te, tr, no, ni, ai in zip(
                self.observations, self.infos, self.actions,
                self.action_log_probs, self.rewards, self.terminated,
                self.truncated, self.next_observations, self.next_infos,
                self.action_infos):
            trajectories[-1].append(
                (o, i, a, a_log_prob, r, te, tr, no, ni, ai))
            if te or tr:
                trajectories[-1].append((no,))
                trajectories.append([])
        return trajectories[:-1]

    def __len__(self):
        return len(self.observations)

    def num_episodes(self):
        return self.episode_id + int(
            not (self.terminated[-1] or self.truncated[-1]))

    def num_completed_episodes(self):
        return self.episode_id

    def shape_reward_with_attn(self):
        assert len(self) > 0
        assert 'context_attn_weights' in self.action_infos[0].keys()
        for i, (o, ai) in enumerate(
                zip(self.observations, self.action_infos)):
            if not self.observations[i]['evidence_is_retrieved']:
                assert self.rewards[i] == 0
                context = self.action_infos[i + 1]['context_strings']
                attn = self.action_infos[
                    i + 1]['context_attn_weights'].detach().mean(0)
                attn_on_each_query = defaultdict(lambda : 0)
                for a, c in zip(attn, context):
                    attn_on_each_query[c.split(':')[0]] += a.item()
                options = pd.read_csv(io.StringIO(
                    self.observations[i]['options'])).apply(
                    lambda r: f'{r.option} ({r.type})', axis=1).to_list()
                option_mask = torch.tensor(
                    [o in attn_on_each_query.keys() for o in options],
                    device=attn.device)
                query_attn = torch.tensor([
                    attn_on_each_query[o] for o in options
                    if o in attn_on_each_query.keys()],
                    device=attn.device)
                action_normalized = self.actions[i][option_mask].softmax(0)
                self.rewards[i] += (action_normalized * query_attn).mean()


# from https://stackoverflow.com/questions/55458145/how-to-get-the-current-value-of-tqdm
class TqdmSpy(tqdm):
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        self.__n = value


def log_results(reward_type, results, split, actor=None, suffix=''):
    return_dict = {}
    # for these metrics limit to targets that appear in the options list
    results = results[results.is_positive]
    targets = set(results.target)
    if reward_type in ['continuous_dependent', 'ranking']:
        precision_recall_micro = {
            f'{split}_precision_micro': results[results['top_1' + suffix]].is_current_target.mean(),
            f'{split}_recall_micro': results[results.is_current_target]['top_1' + suffix].mean(),
            f'{split}_precision_det_micro': results[results['top_1_deterministic' + suffix]].is_current_target.mean(),
            f'{split}_recall_det_micro': results[results.is_current_target]['top_1_deterministic' + suffix].mean(),
        }
        precision_recall = {
            f'{t}/{split}_{p_or_r}{det}': (
                results[
                    (results.target==t) & results['top_1' + suffix]].is_current_target.mean()
                if p_or_r == 'precision' else
                results[
                    (results.target==t) & results.is_current_target]['top_1' + suffix].mean()
            ) if det == '' else (
                results[
                    (results.target==t) &
                    results['top_1_deterministic' + suffix]].is_current_target.mean()
                if p_or_r == 'precision' else
                results[
                    (results.target==t) &
                    results.is_current_target]['top_1_deterministic' + suffix].mean()
            )
            for t in targets
            for p_or_r in ['precision', 'recall']
            for det in ['', '_det']
        }
    else:
        precision_recall_micro = {
            f'{split}_precision_micro': results[results['action_target_score' + suffix] > 0].is_current_target.mean(),
            f'{split}_recall_micro': (results[results.is_current_target]['action_target_score' + suffix] > 0).mean(),
            f'{split}_precision_det_micro': results[results['action_target_score_deterministic' + suffix] > 0].is_current_target.mean(),
            f'{split}_recall_det_micro': (results[results.is_current_target]['action_target_score_deterministic' + suffix] > 0).mean(),
        }
        precision_recall = {
            f'{t}/{split}_{p_or_r}{det}': (
                results[
                    (results.target==t) & (results['action_target_score' + suffix] > 0)].is_current_target.mean()
                if p_or_r == 'precision' else
                (results[
                    (results.target==t) & results.is_current_target]['action_target_score' + suffix] > 0).mean()
            ) if det == '' else (
                results[
                    (results.target==t) &
                    (results['action_target_score_deterministic' + suffix] > 0)].is_current_target.mean()
                if p_or_r == 'precision' else
                (results[
                    (results.target==t) &
                    results.is_current_target]['action_target_score_deterministic' + suffix] > 0).mean()
            )
            for t in targets
            for p_or_r in ['precision', 'recall']
            for det in ['', '_det']
        }
        # return_dict[f'{split}_bce_loss'] = -torch.nn.functional.logsigmoid(torch.tensor(
        #     results.apply(lambda r: r.action_target_score_deterministic if r.is_current_target else -r.action_target_score_deterministic, axis=1).to_numpy(),
        #     dtype=torch.float)).mean()
        return_dict[f'{split}_bce_loss_micro'] = nn.BCEWithLogitsLoss()(
            torch.tensor(results['action_target_score_deterministic' + suffix].to_numpy(), dtype=torch.float),
            torch.tensor(results.is_current_target.to_numpy(), dtype=torch.float)).item()
        if len(set(results.is_current_target)) > 1:
            return_dict[f'{split}_auroc_micro'] = roc_auc_score(
                results.is_current_target.to_numpy(),
                results['action_target_score_deterministic' + suffix].to_numpy())
        for target in targets:
            results_temp = results[results.target==target]
            return_dict[f'{target}/{split}_bce_loss'] = nn.BCEWithLogitsLoss()(
                torch.tensor(results_temp['action_target_score_deterministic' + suffix].to_numpy(), dtype=torch.float),
                torch.tensor(results_temp.is_current_target.to_numpy(), dtype=torch.float)).item()
            if len(set(results.is_current_target)) > 1:
                return_dict[f'{target}/{split}_auroc'] = roc_auc_score(
                    results_temp.is_current_target.to_numpy(),
                    results_temp['action_target_score_deterministic' + suffix].to_numpy())
        return_dict[f'{split}_bce_loss_macro'] = sum(
            [return_dict[f'{target}/{split}_bce_loss'] for target in targets]) / len(targets)
        if len(set(results.is_current_target)) > 1:
            return_dict[f'{split}_auroc_macro'] = sum(
                [return_dict[f'{target}/{split}_auroc'] for target in targets]) / len(targets)
    precision_recall_macro = {
        f'{split}_{p_or_r}{det}_macro': sum(
            [precision_recall[f'{t}/{split}_{p_or_r}{det}'] for t in targets])
            / len(targets)
        for p_or_r in ['precision', 'recall']
        for det in ['', '_det']
    }
    if actor is not None:
        if actor.has_bias and actor.config.static_diagnoses is not None:
            with torch.no_grad():
                diagnosis_strings = list(
                    actor.observation_embedder.diagnosis_mapping.keys())
                static_bias = actor.get_static_bias(diagnosis_strings)
                for diagnosis_string, params in zip(
                        diagnosis_strings, static_bias):
                    for i in range(params.shape[0]):
                        return_dict[f'{diagnosis_string}_param{i}'] = params[i].item()
    return_dict.update({
        **precision_recall_micro,
        **precision_recall_macro,
        **precision_recall,
    })
    return return_dict


def get_mimic_demographics(mimic_folder):
    patients = pd.read_csv(os.path.join(mimic_folder, 'PATIENTS.csv'))
    patients = patients[['SUBJECT_ID', 'GENDER']].drop_duplicates(
        ['SUBJECT_ID'])
    admissions = pd.read_csv(os.path.join(mimic_folder, 'ADMISSIONS.csv'))
    admissions = admissions[
        # ['SUBJECT_ID', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']
        ['SUBJECT_ID', 'ETHNICITY']
        ].drop_duplicates(['SUBJECT_ID'])
    df = patients.merge(admissions, on="SUBJECT_ID")
    df['race'] = df['ETHNICITY'].apply(
        lambda x: 'ASIAN' if 'ASIAN' in x else
            'HISPANIC' if 'HISPANIC' in x else
            'NATIVE AMERICAN/ALASKAN' if 'NATIVE' in x else
            'BLACK' if 'BLACK' in x else
            'WHITE' if 'WHITE' in x else "")
    del df['ETHNICITY']
    return df.rename(
        columns={'GENDER': 'gender', 'SUBJECT_ID': 'patient_id'})
