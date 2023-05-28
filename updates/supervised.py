import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
import wandb
import pandas as pd
import io
from torch.distributions.categorical import Categorical


# adapted from https://github.com/warrenzha/ppo-pytorch/blob/ac0be1459668e5881c20b0752dc06f6329ee6f3d/agent/ppo_continous.py#L172
def supervised_update(args, replay_buffer, actor, actor_optimizer, epoch, updates, env):
    if env.fuzzy_matching_model is not None:
        env.fuzzy_matching_model.to('cuda')
    with torch.no_grad():  # adv and v_target have no gradient
        reward = torch.tensor(replay_buffer.rewards, dtype=torch.float, device='cuda')
        action_scores_entropy = torch.stack(
            [Categorical(logits=action).entropy() / len(action) for action in replay_buffer.actions])
    # log epoch training metrics
    wandb.log({
        'epoch': epoch,
        'updates': updates,
        'avg_reward': reward.mean().item(),
        'action_scores_normalized_entropy': action_scores_entropy.mean().item(),
    })
    replay_length = len(replay_buffer)
    for sub_epoch in tqdm(range(args.supervised.sub_epochs), total=args.supervised.sub_epochs, desc='sub-epochs'):
        # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples
        # in the last time is less than batch_size
        batch_sampler = BatchSampler(SubsetRandomSampler(range(replay_length)), args.training.batch_size, False)
        for batch_idx, index in tqdm(enumerate(batch_sampler), total=len(batch_sampler), desc='mini-batches'):
            action_dists = [actor.get_dist(replay_buffer.observations[i]) for i in index]
            actions = [action_dist.rsample() for action_dist in action_dists]

            # compute supervised loss
            log_probs = []
            for i, action in zip(index, actions):
                potential_diagnoses = pd.read_csv(
                    io.StringIO(replay_buffer.observations[i]['potential_diagnoses'])).diagnoses.to_list()
                is_match, best_match, reward = env.reward_per_item(
                    action, potential_diagnoses, replay_buffer.infos[i]['current_targets'])
                log_probs.append(reward.sum().log())
            supervised_loss = -torch.stack(log_probs).mean()

            make_update = ((batch_idx + 1) % args.training.accumulate_grad_batches) == 0 or \
                          (batch_idx + 1) == len(batch_sampler)

            # update actor
            supervised_loss.mean().backward()
            if make_update:
                if args.supervised.use_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()
                actor_optimizer.zero_grad()
                updates += 1

            # log batch training metrics
            log_dict = {
                'epoch': epoch,
                'sub_epoch': sub_epoch,
                'updates': updates,
                'supervised_loss': supervised_loss,
            }
            log_dict.update(actor.get_dist_stats(action_dists))
            wandb.log(log_dict)

    # TODO: add lr decay?
    # if self.use_lr_decay:  # Trick 6:learning rate Decay
    #     self.lr_decay(total_steps)
    return updates
