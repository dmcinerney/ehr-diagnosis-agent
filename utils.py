from omegaconf import OmegaConf
import torch
from tqdm import trange
import pandas as pd
import io


def get_args(config_file):
    args_from_cli = OmegaConf.from_cli()
    args_from_yaml = OmegaConf.load(config_file)
    return OmegaConf.merge(args_from_yaml, args_from_cli)


def collect_trajectories(args, train_env, options, actor, epoch, seed_offset):
    replay_buffer = ReplayBuffer()
    with torch.no_grad():
        for episode in trange(args.training.num_episodes, desc='collecting episode trajectories', leave=False):
            terminated, truncated = False, False
            obs, info = train_env.reset(seed=epoch * episode + seed_offset, options=options)
            while len(info['current_targets']) == 0 or len(pd.read_csv(io.StringIO(obs['potential_diagnoses']))) < 2:
                print('Dead enviornment, retrying...')
                seed_offset += 1
                obs, info = train_env.reset(seed=epoch * episode + seed_offset, options=options)
            t = 0
            while not (terminated or truncated):
                # sample action
                action, action_log_prob = actor(obs, return_log_prob=True)
                next_obs, reward, terminated, truncated, next_info = train_env.step(action)
                replay_buffer.append(
                    obs, info, action, action_log_prob, reward, terminated, truncated, next_obs, next_info)
                obs = next_obs
                info = next_info
                t += 1
                if args.training.max_trajectory_length is not None and t >= args.training.max_trajectory_length:
                    break
    return replay_buffer, seed_offset


class ReplayBuffer:
    def __init__(self):
        self.observations = []
        self.infos = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.next_observations = []
        self.next_infos = []
        self.episode_id = 0
        self.episode_ids = []

    def append(self, observation, info, action, action_log_prob, reward, terminated, truncated, next_observation,
               next_info):
        self.observations.append(observation)
        self.infos.append(info)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.next_observations.append(next_observation)
        self.next_infos.append(next_info)
        self.episode_ids.append(self.episode_id)
        if terminated or truncated:
            self.episode_id += 1

    def get_trajectories(self):
        trajectories = [[]]
        for o, a, a_log_prob, r, te, tr, info, no in zip(
                self.observations, self.infos, self.actions, self.action_log_probs, self.rewards, self.terminated,
                self.truncated, self.next_observations, self.next_infos):
            trajectories[-1].append((o, a, a_log_prob, r, te, tr, info))
            if te or tr:
                trajectories[-1].append((no,))
                trajectories.append([])
        return trajectories[:-1]

    def __len__(self):
        return len(self.observations)
