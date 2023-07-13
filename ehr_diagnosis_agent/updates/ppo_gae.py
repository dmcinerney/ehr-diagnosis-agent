# adapted from https://github.com/warrenzha/ppo-pytorch/blob/ac0be1459668e5881c20b0752dc06f6329ee6f3d/agent/ppo_continous.py#L172
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def ppo_gae_init(args, replay_buffer, critic):
    with torch.no_grad():  # adv and v_target have no gradient
        advantage = []
        gae = 0
        done = []
        value = []
        next_value = []
        for obs, next_obs, terminated, truncated in zip(
            replay_buffer.observations, replay_buffer.next_observations, replay_buffer.terminated,
            replay_buffer.truncated
        ):
            value.append(critic(obs) if len(done) == 0 or done[-1] else next_value[-1])
            next_value.append(
                critic(next_obs) if not (terminated or truncated) else
                torch.tensor([0.], device=value[-1].device))
            done.append(terminated or truncated)
        value = torch.cat(value)
        next_value = torch.cat(next_value)
        done = torch.tensor(done, dtype=torch.bool, device=value.device)
        original_reward = torch.tensor(
            replay_buffer.rewards, dtype=torch.float, device=value.device)
        if args.ppo_gae.log_reward:
            if args.env.reward_type in ['continuous_dependent', 'ranking']:
                reward = (1e-1 + original_reward).log()
            elif args.env.reward_type in ['continuous_independent']:
                raise NotImplementedError
            else:
                raise Exception
        else:
            reward = original_reward
        reward = (1e-1 + original_reward).log() if args.ppo_gae.log_reward else original_reward
        deltas = reward + args.ppo_gae.gamma * next_value - value
        for delta, d in zip(reversed(deltas), reversed(done)):
            gae = delta + args.ppo_gae.gamma * args.ppo_gae.lam * gae * (1.0 - d.float())
            advantage.insert(0, gae)
        advantage = torch.tensor(advantage, device=value.device)
        value_target = advantage + value
        if args.ppo_gae.use_adv_norm:  # Advantage normalization
            advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))
        action_scores_entropy = torch.stack(
            [Categorical(logits=action).entropy() / len(action)
             for action in replay_buffer.actions])
    return advantage, value_target, original_reward, reward, action_scores_entropy


def ppo_gae_batch(args, replay_buffer, actor, critic, index, advantage, value_target):
    log_dict = {
        'batch_advantage': advantage[index].detach().mean().item(),
    }
    if not args.ppo_gae.only_train_critic:
        action_dists = [actor.get_dist(replay_buffer.observations[i]) for i in index]
        dist_entropy = actor.get_entropy(action_dists)
        action_log_prob = torch.stack(
            [dist.log_prob(replay_buffer.actions[i]).mean()
             for dist, i in zip(action_dists, index)])
        old_action_log_prob = torch.stack(
            [replay_buffer.action_log_probs[i].mean() for i in index])
        ratios = torch.exp(action_log_prob - old_action_log_prob)
        # actor loss
        surr1 = ratios * advantage[index]
        surr2 = torch.clamp(
            ratios, 1 - args.ppo_gae.epsilon, 1 + args.ppo_gae.epsilon) * advantage[index]
        ppo_objective = torch.min(surr1, surr2)
        actor_loss = -ppo_objective - args.ppo_gae.entropy_coefficient * dist_entropy
        actor_loss = actor_loss.mean()
        log_dict.update({
            'ppo_objective': ppo_objective.detach().mean().item(),
            'dist_entropy': dist_entropy.detach().mean().item(),
            'ratio': ratios.detach().mean().item(),
        })
        log_dict.update(actor.get_dist_stats(action_dists))
    else:
        actor_loss = None
    # critic loss
    batch_values = torch.cat([critic(replay_buffer.observations[i]) for i in index])
    critic_loss = F.mse_loss(value_target[index], batch_values)
    return actor_loss, critic_loss, log_dict
