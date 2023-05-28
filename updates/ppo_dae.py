from tqdm import tqdm
import torch
import wandb


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


def ppo_dae_update(
        args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates):
    # TODO: add gamma?
    old_critic = critic.__cls__(args.critic)
    old_critic.set_device('cuda')
    old_critic.load_state_dict(critic.state_dict)
    # TODO: change this to preprocess the old critic values now so we do not need to create a second critic model
    trajectories = replay_buffer.get_trajectories()
    for trajectory_idx, trajectory in tqdm(
            enumerate(trajectories), total=len(trajectories), desc='updating on each episode', leave=False):
        estimated_values = [critic(trajectory[0][0])]
        estimated_advantages = []
        ratio_mean = 0
        ppo_loss = 0
        rewards = []
        for t, (obs, action, old_policy_log_prob, reward, _, _, _) in tqdm(
                enumerate(trajectory[:-1]), total=len(trajectory[:-1]),
                desc='estimating loss for each step and updating', leave=False):
            estimated_values.append(critic(trajectory[t + 1][0]))
            estimated_advantages.append(reward + estimated_values[-1] - estimated_values[-2])
            policy_log_prob = actor.log_prob(obs, action).mean()
            ratio = torch.exp(policy_log_prob - old_policy_log_prob)
            ratio_mean += ratio.item()
            ppo_loss += compute_ppo_loss(ratio, estimated_advantages[-1].detach(), epsilon=args.ppo_dae.epsilon)
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
                    estimated_last_old_value = old_critic(trajectory[t + 1][0])
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
                estimated_values = [critic(trajectory[t + 1][0])] if not is_last else []
                estimated_advantages = []
                ratio_mean = 0
                ppo_loss = 0
                rewards = []
    return updates
