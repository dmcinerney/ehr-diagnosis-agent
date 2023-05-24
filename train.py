import ehr_diagnosis_env
import gymnasium
from utils import *
import pandas as pd
import os
from models.actor import InterpretableRankingPolicy
from models.critic import Critic
import torch
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf


def compute_ppo_loss(ratio, estimated_advantage, epsilon=.1):
    # implements https://arxiv.org/pdf/2109.06093.pdf equation 49 for one s, a pair
    return -torch.minimum(ratio * estimated_advantage, torch.clip(ratio, 1-epsilon, 1+epsilon) * estimated_advantage)


def compute_dae_loss(rewards, estimated_advantages, estimated_values, estimated_last_old_value):
    # implements https://arxiv.org/pdf/2109.06093.pdf equation 13 for one trajectory
    # TODO: check whether it makes sense to use DAE when the environment in this case might just be partially observed,
    #   causal rules might not hold
    estimated_advantages = torch.cat(estimated_advantages).unsqueeze(0)
    rewards = torch.tensor(rewards, device=estimated_advantages.device).unsqueeze(0)
    estimated_values = torch.cat(estimated_values).unsqueeze(1)
    # calculate t x t' matrix with the proper terms
    dae_loss = (rewards - estimated_advantages) + estimated_last_old_value - estimated_values[:-1]
    # we would like the entries where t <= t', this is the upper triangle (torch.triu):
    # [ 0 <= 0   0 <= 1   0 <= 2 ]
    # [ 1 !<= 0  1 <= 1   1 <= 2 ]
    # [ 2 !<= 0  2 !<= 1  2 <= 2 ]
    dae_loss = torch.triu(dae_loss)
    # First sum along the t' axis and square
    dae_loss = dae_loss.sum(-1) ** 2
    # Then sum along the t axis and return
    return dae_loss.sum()


def main():
    args = get_args('config.yaml')
    run = wandb.init(
        project="ehr-diagnosis-agent",
        config= OmegaConf.to_container(args)
    )
    train_df = pd.read_csv(os.path.join(args.data.path, args.data.dataset, 'train.data'), compression='gzip')
    train_env = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=train_df,
        model_name=args.env.model_name,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
    )
    old_actor = InterpretableRankingPolicy(args.actor)
    actor = InterpretableRankingPolicy(args.actor)
    actor.set_device('cuda')
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.training.lr)
    old_critic = Critic(args.critic)
    critic = Critic(args.critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.training.lr)
    seed_offset = 0
    options = {}
    if args.training.max_reports_considered is not None:
        options['max_reports_considered'] = args.training.max_reports_considered
    updates = 0
    for epoch in trange(args.training.num_epochs, desc='epochs'):
        # collect trajectories via rolling out with the current policy
        old_actor.set_device('cpu')
        old_critic.set_device('cpu')
        critic.set_device('cpu')
        train_env.to('cuda')
        trajectories = []
        with torch.no_grad():
            for episode in trange(args.training.num_episodes, desc='collecting episode trajectories', leave=False):
                terminated, truncated = False, False
                obs, info = train_env.reset(seed=epoch * episode + seed_offset, options=options)
                while len(info['current_targets']) == 0:
                    print('Dead enviornment, retrying...')
                    seed_offset += 1
                    obs, info = train_env.reset(seed=epoch * episode + seed_offset, options=options)
                trajectories.append([])
                t = 0
                while not (terminated or truncated):
                    # sample action
                    action = actor(obs)
                    next_obs, reward, terminated, truncated, info = train_env.step(action)
                    trajectories[-1].append((obs, action, reward))
                    obs = next_obs
                    t += 1
                    if args.training.max_trajectory_length is not None and t >= args.training.max_trajectory_length:
                        break
                trajectories[-1].append((obs,))
        train_env.to('cpu')
        old_actor.set_device('cuda')
        old_critic.set_device('cuda')
        critic.set_device('cuda')
        old_actor.load_state_dict(actor.state_dict())
        old_critic.load_state_dict(critic.state_dict())
        for trajectory_idx, trajectory in tqdm(
                enumerate(trajectories), total=len(trajectories), desc='updating on each episode', leave=False):
            estimated_values = [critic(trajectory[0][0])]
            estimated_advantages = []
            ratio_mean = 0
            ppo_loss = 0
            rewards = []
            for t, (obs, action, reward) in tqdm(
                    enumerate(trajectory[:-1]), total=len(trajectory[:-1]),
                    desc='estimating loss for each step and updating', leave=False):
                estimated_values.append(critic(trajectory[t+1][0]))
                estimated_advantages.append(reward + estimated_values[-1] - estimated_values[-2])
                policy_log_prob = actor.log_prob(obs, action).mean()
                with torch.no_grad():
                    old_policy_log_prob = old_actor.log_prob(obs, action).mean()
                ratio = torch.exp(policy_log_prob - old_policy_log_prob)
                ratio_mean += ratio.item()
                ppo_loss += compute_ppo_loss(ratio, estimated_advantages[-1].detach(), epsilon=args.training.epsilon)
                rewards.append(reward)
                is_last = (t + 1) == len(trajectory[:-1])
                if is_last or ((t + 1) % args.training.batch_size) == 0:
                    make_update = is_last or (
                            ((t + 1) / args.training.batch_size) % args.training.accumulate_grad_batches) == 0
                    # compute ppo loss and update actor
                    count = len(estimated_advantages)
                    ppo_loss /= count
                    ppo_loss.backward()
                    if make_update:
                        actor_optimizer.step()
                        actor_optimizer.zero_grad()
                    # compute dae loss and update critic
                    with torch.no_grad():
                        estimated_last_old_value = old_critic(trajectory[t+1][0])
                    dae_loss = compute_dae_loss(
                        rewards, estimated_advantages, estimated_values, estimated_last_old_value)
                    dae_loss.backward()
                    if make_update:
                        critic_optimizer.step()
                        critic_optimizer.zero_grad()
                        updates += 1
                    # log
                    wandb.log({
                        'epoch': epoch,
                        'trajectory_idx': trajectory_idx,
                        'updates': updates,
                        'ppo_loss': ppo_loss.item(),
                        'dae_loss': dae_loss.item(),
                        'avg_ratio': ratio_mean / count,
                        'ratio': ratio.item(),
                        'estimated_advantage': torch.cat(estimated_advantages).detach().mean().item(),
                        'estimated_value': torch.cat(estimated_values).detach().mean().item(),
                        'cumulative_reward': sum(rewards),
                    })
                    # reset trajectory
                    # can't reuse value because graph is not saved
                    estimated_values = [critic(trajectory[t+1][0])] if not is_last else []
                    estimated_advantages = []
                    ratio_mean = 0
                    ppo_loss = 0
                    rewards = []
        if args.training.checkpoint_every is None or ((epoch + 1) % args.training.checkpoint_every) == 0:
            file = os.path.join(run.dir, 'actor_epoch={}_updates={}.pt'.format(epoch + 1, updates))
            print(f'Saving checkpoint to {file}')
            torch.save(actor.state_dict, file)


if __name__ == '__main__':
    main()
