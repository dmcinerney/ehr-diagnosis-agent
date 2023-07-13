from collections import defaultdict
from omegaconf import OmegaConf
import torch
from tqdm import trange, tqdm
import wandb
import random
import pandas as pd
import io


def get_args(config_file):
    args_from_cli = OmegaConf.from_cli()
    args_from_yaml = OmegaConf.load(config_file)
    return OmegaConf.merge(args_from_yaml, args_from_cli)


def reset_env(env, seed, options, dataset_iterations, dataset_progress, epoch, log=True):
    new_dataset_iteration = env.num_unseen_examples() == 0
    obs, info = env.reset(seed=seed, options=options)
    if new_dataset_iteration:
        dataset_progress.reset(total=env.num_unseen_examples())
        dataset_iterations += 1
    dataset_progress.update(1)
    if log:
        wandb.log({'epoch': epoch, 'num_episodes_seen': dataset_progress.n + env.num_examples() * dataset_iterations,
                   'percentage_dataset_seen': dataset_progress.n / env.num_examples() + dataset_iterations})
    return obs, info, dataset_iterations


def sample_exploration_policy(env, actor, obs, info, mixin_expert_policy=0):
    if mixin_expert_policy == 0 or (mixin_expert_policy != 1 and random.uniform(0, 1) < mixin_expert_policy):
        # sample from actor policy
        action_info = actor(obs)
        action = action_info['action']
        del action_info['action']
        action_log_prob = action_info['log_prob']
        del action_info['log_prob']
        action_is_expert = False
    else:
        # sample from "expert" policy
        action_is_expert = True
        raise NotImplementedError
    return action, action_log_prob, action_is_expert, action_info


def collect_trajectories(args, train_env, options, actor, epoch, seed_offset, dataset_iterations, dataset_progress, log=True, mixin_expert_policy=0):
    replay_buffer = ReplayBuffer()
    with torch.no_grad():
        for episode in trange(args.training.num_episodes, desc='collecting episode trajectories', leave=False):
            terminated, truncated = False, False
            obs, info, dataset_iterations = reset_env(
                train_env, epoch * episode + seed_offset, options, dataset_iterations, dataset_progress, epoch)
            while train_env.is_truncated(obs, info):
                # dead environment, retrying...
                seed_offset += 1
                obs, info, dataset_iterations = reset_env(
                    train_env, epoch * episode + seed_offset, options, dataset_iterations, dataset_progress, epoch)
            t = 0
            while not (terminated or truncated):
                action, action_log_prob, action_is_expert, action_info = sample_exploration_policy(
                    train_env, actor, obs, info, mixin_expert_policy=mixin_expert_policy)
                next_obs, reward, terminated, truncated, next_info = train_env.step(action)
                replay_buffer.append(
                    obs, info, action, action_log_prob, action_is_expert, reward, terminated, truncated, next_obs, next_info, action_info)
                obs = next_obs
                info = next_info
                t += 1
                if args.training.max_trajectory_length is not None and t >= args.training.max_trajectory_length:
                    break
            if log:
                wandb.log({'epoch': epoch, 'num_episodes_collected': epoch * args.training.num_episodes + episode + 1})
    if args.training.shape_reward_with_attn:
        replay_buffer.shape_reward_with_attn()
    return replay_buffer, seed_offset, dataset_iterations


class ReplayBuffer:
    def __init__(self):
        self.observations = []
        self.infos = []
        self.actions = []
        self.action_log_probs = []
        self.action_is_expert = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.next_observations = []
        self.next_infos = []
        self.action_infos = []
        self.episode_id = 0
        self.episode_ids = []

    def append(self, observation, info, action, action_log_prob, action_is_expert, reward, terminated, truncated,
               next_observation, next_info, action_info):
        self.observations.append(observation)
        self.infos.append(info)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.action_is_expert.append(action_is_expert)
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
                self.observations, self.infos, self.actions, self.action_log_probs, self.rewards, self.terminated,
                self.truncated, self.next_observations, self.next_infos, self.action_infos):
            trajectories[-1].append((o, i, a, a_log_prob, r, te, tr, no, ni, ai))
            if te or tr:
                trajectories[-1].append((no,))
                trajectories.append([])
        return trajectories[:-1]

    def __len__(self):
        return len(self.observations)

    def shape_reward_with_attn(self):
        assert len(self) > 0
        assert 'context_attn_weights' in self.action_infos[0].keys()
        for i, (o, ai) in enumerate(zip(self.observations, self.action_infos)):
            if not self.observations[i]['evidence_is_retrieved']:
                assert self.rewards[i] == 0
                context = self.action_infos[i + 1]['context_strings']
                attn = self.action_infos[i + 1]['context_attn_weights'].detach().mean(0)
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
