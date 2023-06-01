import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
import wandb
from torch.distributions.categorical import Categorical


# adapted from https://github.com/warrenzha/ppo-pytorch/blob/ac0be1459668e5881c20b0752dc06f6329ee6f3d/agent/ppo_continous.py#L172
def ppo_gae_update(
        args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates, num_instances_seen,
        num_dataset_epochs):
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
            next_value.append(critic(next_obs))
            done.append(terminated or truncated)
        value = torch.cat(value)
        next_value = torch.cat(next_value)
        done = torch.tensor(done, dtype=torch.bool, device=value.device)
        original_reward = torch.tensor(replay_buffer.rewards, dtype=torch.float, device=value.device)
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
            [Categorical(logits=action).entropy() / len(action) for action in replay_buffer.actions])
    # log epoch training metrics
    log_dict = {
        'epoch': epoch,
        'updates': updates,
        'estimated_advantage': advantage.mean().item(),
        'estimated_value': value_target.mean().item(),
        'avg_reward': original_reward.mean().item(),
        'action_scores_normalized_entropy': action_scores_entropy.mean().item(),
        'num_instances_seen': num_instances_seen,
        'num_dataset_epochs': num_dataset_epochs,
    }
    if args.ppo_gae:
        log_dict['avg_log_reward'] = reward.mean().item()
    wandb.log(log_dict)
    replay_length = len(replay_buffer)
    for sub_epoch in tqdm(range(args.ppo_gae.sub_epochs), total=args.ppo_gae.sub_epochs, desc='sub-epochs'):
        # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples
        # in the last time is less than batch_size
        batch_sampler = BatchSampler(SubsetRandomSampler(range(replay_length)), args.training.batch_size, False)
        for batch_idx, index in tqdm(enumerate(batch_sampler), total=len(batch_sampler), desc='mini-batches'):
            action_dists = [actor.get_dist(replay_buffer.observations[i]) for i in index]
            dist_entropy = actor.get_entropy(action_dists)
            action_log_prob = torch.stack(
                [dist.log_prob(replay_buffer.actions[i]).mean() for dist, i in zip(action_dists, index)])
            old_action_log_prob = torch.stack([replay_buffer.action_log_probs[i].mean() for i in index])
            ratios = torch.exp(action_log_prob - old_action_log_prob)

            make_update = ((batch_idx + 1) % args.training.accumulate_grad_batches) == 0 or \
                          (batch_idx + 1) == len(batch_sampler)

            # actor loss
            surr1 = ratios * advantage[index]
            surr2 = torch.clamp(ratios, 1 - args.ppo_gae.epsilon, 1 + args.ppo_gae.epsilon) * advantage[index]
            ppo_objective = torch.min(surr1, surr2)
            actor_loss = -ppo_objective - args.ppo_gae.entropy_coefficient * dist_entropy

            # update actor
            actor_loss.mean().backward()
            if make_update:
                if args.ppo_gae.use_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()
                actor_optimizer.zero_grad()

            # critic loss
            batch_values = torch.cat([critic(replay_buffer.observations[i]) for i in index])
            critic_loss = F.mse_loss(value_target[index], batch_values)

            # update critic
            critic_loss.backward()
            if make_update:
                if args.ppo_gae.use_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()
                critic_optimizer.zero_grad()
                updates += 1

            # log batch training metrics
            log_dict = {
                'epoch': epoch,
                'sub_epoch': sub_epoch,
                'updates': updates,
                'actor_loss': actor_loss.item(),
                'ppo_objective': ppo_objective.detach().mean().item(),
                'dist_entropy': dist_entropy.detach().mean().item(),
                'critic_loss': critic_loss.item(),
                'ratio': ratios.detach().mean().item(),
                'batch_advantage': advantage[index].detach().mean().item(),
            }
            log_dict.update(actor.get_dist_stats(action_dists))
            wandb.log(log_dict)

    # TODO: add lr decay?
    # if self.use_lr_decay:  # Trick 6:learning rate Decay
    #     self.lr_decay(total_steps)
    return updates
