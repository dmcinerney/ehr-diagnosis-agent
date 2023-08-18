import torch
import pandas as pd
import io
from torch.distributions.categorical import Categorical


def supervised_init(replay_buffer, env):
    if env.fuzzy_matching_model is not None:
        env.fuzzy_matching_model.to('cuda')
    with torch.no_grad():  # adv and v_target have no gradient
        reward = torch.tensor(
            replay_buffer.rewards, dtype=torch.float, device='cuda')
        action_scores_entropy = torch.stack(
            [Categorical(logits=action).entropy() / len(action)
             for action in replay_buffer.actions])
    return reward, action_scores_entropy


def supervised_batch(args, replay_buffer, actor, env, index):
    action_dists = [actor.get_dist(replay_buffer.observations[i])
                    for i in index]
    dist_entropy = actor.get_entropy(action_dists)
    actions = [action_dist.rsample() for action_dist in action_dists]
    # compute supervised loss
    supervised_loss = []
    for i, action in zip(index, actions):
        if args.supervised.only_train_risk_prediction and \
                not replay_buffer.observations[i]['evidence_is_retrieved']:
            continue
        if replay_buffer.observations[i]['evidence_is_retrieved'] or \
                args.supervised.query_supervision_type in ['targets']:
            options = pd.read_csv(
                io.StringIO(replay_buffer.observations[i]['options']))
            options = options[options.type == 'diagnosis'].option.to_list()
            is_match, best_match, reward = env.reward_per_item(
                action, options, replay_buffer.infos[i]['current_targets'])
            supervised_loss.append(-reward.sum())
        if not replay_buffer.observations[i]['evidence_is_retrieved'] and \
                args.supervised.query_supervision_type == 'attention':
            # TODO: implement this for when training query retrieval
            #   using attention
            raise NotImplementedError
    if len(supervised_loss) == 0:
        return None, {}
    supervised_loss = torch.stack(supervised_loss)
    actor_loss = supervised_loss - \
        args.supervised.entropy_coefficient * dist_entropy
    actor_loss = actor_loss.mean()
    log_dict = {
        'supervised_loss': supervised_loss.detach().mean().item(),
        'dist_entropy': dist_entropy.detach().mean().item(),
        'actor_loss_s': actor_loss.item(),
    }
    log_dict.update(actor.get_dist_stats(action_dists))
    return actor_loss, log_dict
