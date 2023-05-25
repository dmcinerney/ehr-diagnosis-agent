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
from rl.ppo_dae import ppo_dae_update
from rl.ppo_gae import ppo_gae_update
from rl.utils import collect_trajectories
import gc


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
        continuous_reward=args.env.continuous_reward,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
    )
    actor = InterpretableRankingPolicy(args.actor)
    actor.set_device('cuda')
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.training.actor_lr)
    critic = Critic(args.critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.training.critic_lr)
    seed_offset = 0
    options = {}
    if args.training.max_reports_considered is not None:
        options['max_reports_considered'] = args.training.max_reports_considered
    updates = 0
    for epoch in trange(args.training.num_epochs, desc='epochs'):
        # collect trajectories via rolling out with the current policy
        critic.set_device('cpu')
        train_env.to('cuda')
        replay_buffer, seed_offset = collect_trajectories(args, train_env, options, actor, epoch, seed_offset)
        train_env.to('cpu')
        critic.set_device('cuda')
        if args.training.objective_optimization == 'ppo_dae':
            updates = ppo_dae_update(
                args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates)
        elif args.training.objective_optimization == 'ppo_gae':
            updates = ppo_gae_update(
                args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates)
        else:
            raise Exception
        if args.training.checkpoint_every is None or ((epoch + 1) % args.training.checkpoint_every) == 0:
            file = os.path.join(run.dir, 'actor_epoch={}_updates={}.pt'.format(epoch + 1, updates))
            print(f'Saving checkpoint to {file}')
            torch.save(actor.state_dict, file)
        if args.training.clear_gpu: # only used for debugging
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
