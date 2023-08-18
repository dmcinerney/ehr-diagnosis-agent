# this file contains the loop for updating via 
# ppo_gae or supervised or a combination
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
import wandb
from .supervised import supervised_init, supervised_batch
from .ppo_gae import ppo_gae_init, ppo_gae_batch


def update(
        args, replay_buffer, actor, actor_optimizer, actor_scheduler,
        critic, critic_optimizer, critic_scheduler,
        epoch, updates, env):
    use_supervised = args.training.objective_optimization in [
        'supervised', 'mix_objectives']
    use_ppo_gae = args.training.objective_optimization in [
        'ppo_gae', 'mix_objectives']
    assert use_supervised or use_ppo_gae
    log_dict = {
        'epoch': epoch,
        'updates': updates,
    }
    if use_supervised:
        reward, action_scores_entropy = supervised_init(replay_buffer, env)
    if use_ppo_gae:
        advantage, value_target, reward, reward_pg, action_scores_entropy = \
            ppo_gae_init(args, replay_buffer, critic)
        # log epoch training metrics
        log_dict.update({
            'estimated_advantage': advantage.mean().item(),
            'estimated_value': value_target.mean().item(),
        })
        if args.ppo_gae.log_reward:
            log_dict['avg_log_reward'] = reward_pg.mean().item()
    log_dict.update({
        'avg_reward_risk_prediction': reward[1::2].mean().item(),
        'avg_reward': reward.mean().item(),
        'action_scores_normalized_entropy': 
            action_scores_entropy.mean().item(),
    })
    if args.training.shape_reward_with_attn:
        log_dict['avg_reward_query_prediction'] = reward[::2].mean().item()
    wandb.log(log_dict)
    indices = list(range(len(replay_buffer)))
    assert not (args.training.skip_risk_prediction
        and args.training.skip_query_prediction)
    if args.training.skip_risk_prediction:
        indices = [
            i for i in indices
            if not replay_buffer.observations[i]['evidence_is_retrieved']]
    if args.training.skip_query_prediction:
        indices = [
            i for i in indices
            if replay_buffer.observations[i]['evidence_is_retrieved']]
    for sub_epoch in tqdm(
            range(args.ppo_gae.sub_epochs), total=args.ppo_gae.sub_epochs,
            desc='sub-epochs'):
        # Random sampling and no repetition. 'False' indicates that
        # training will continue even if the number of samples in the
        # last time is less than batch_size
        batch_sampler = BatchSampler(
            SubsetRandomSampler(indices), args.training.batch_size, False)
        for batch_idx, index in tqdm(
                enumerate(batch_sampler), total=len(batch_sampler),
                desc='mini-batches'):
            make_update = (
                (batch_idx + 1) % args.training.accumulate_grad_batches) == 0 \
                or (batch_idx + 1) == len(batch_sampler)
            update_performed = False
            log_dict = {
                'epoch': epoch,
                'sub_epoch': sub_epoch,
                'updates': updates,
            }
            if use_supervised:
                actor_loss_s, log_dict_s = supervised_batch(
                    args, replay_buffer, actor, env, index)
                log_dict.update(log_dict_s)
            if use_ppo_gae:
                actor_loss_pg, critic_loss, log_dict_pg = ppo_gae_batch(
                    args, replay_buffer, actor, critic, index, advantage,
                    value_target)
                log_dict.update(log_dict_pg)
                # update critic
                critic_loss.backward()
                if make_update:
                    if args.ppo_gae.use_grad_clip is not None:
                        clip_grad_norm_(critic.parameters(), 0.5)
                    critic_optimizer.step()
                    critic_optimizer.zero_grad()
                    log_dict['critic_lr'] = critic_scheduler.get_last_lr()[0]
                    critic_scheduler.step()
                    update_performed = True
            if use_supervised and use_ppo_gae and actor_loss_s is not None \
                    and actor_loss_pg is not None:
                actor_loss = args.mix_objectives.coefficient * actor_loss_s \
                    + (1 - args.mix_objectives.coefficient) * actor_loss_pg
            elif use_supervised:
                actor_loss = actor_loss_s
            elif use_ppo_gae:
                actor_loss = actor_loss_pg
            else:
                raise Exception
            # update actor
            if actor_loss is not None:
                actor_loss.backward()
                if make_update:
                    if args.ppo_gae.use_grad_clip is not None:
                        clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optimizer.step()
                    actor_optimizer.zero_grad()
                    log_dict['actor_lr'] = actor_scheduler.get_last_lr()[0]
                    actor_scheduler.step()
                    update_performed = True
                log_dict['actor_loss'] = actor_loss.item()
            wandb.log(log_dict)
            if update_performed:
                updates += 1
    # TODO: add lr decay?
    # if self.use_lr_decay:  # Trick 6:learning rate Decay
    #     self.lr_decay(total_steps)
    return updates
