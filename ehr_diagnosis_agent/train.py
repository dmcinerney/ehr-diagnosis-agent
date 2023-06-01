import ehr_diagnosis_env
import gymnasium
from utils import get_args, collect_trajectories, TqdmSpy
import pandas as pd
import os
from models.actor import InterpretableNormalActor, InterpretableDirichletActor
from models.critic import Critic
import torch
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf
from updates.ppo_dae import ppo_dae_update
from updates.ppo_gae import ppo_gae_update
from updates.supervised import supervised_update
import gc


# TODO: partition training into two different scripts, one that trains with direct supervision,
#  and this one which trains via updates. The updates one should be able to warm start from the output of the supervised training.
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
        num_future_diagnoses_threshold=args.env.num_future_diagnoses_threshold,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
    )
    if args.actor.type == 'normal':
        actor = InterpretableNormalActor(args.actor.normal_params)
    elif args.actor.type == 'dirichlet':
        actor = InterpretableDirichletActor(args.actor.dirichlet_params)
    else:
        raise Exception
    actor.train()
    actor.set_device('cuda')
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.training.actor_lr)
    if args.training.objective_optimization in ['ppo_dae', 'ppo_gae']:
        critic = Critic(args.critic)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.training.critic_lr)
    seed_offset = 0
    if args.training.resume_from is not None:
        print(f'Resuming from {args.training.resume_from}')
        ckpt = torch.load(args.training.resume_from)
        actor.load_state_dict(ckpt['actor'])
        actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        if args.training.objective_optimization in ['ppo_dae', 'ppo_gae']:
            critic.load_state_dict(ckpt['critic'])
            critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        seed_offset = ckpt['seed_offset']
    options = {}
    if args.training.max_reports_considered is not None:
        options['max_reports_considered'] = args.training.max_reports_considered
    updates = 0
    dataset_iterations = 0
    dataset_length = train_env.num_unseen_examples()
    dataset_progress = TqdmSpy(
        desc='dataset progress (iteration {})'.format(dataset_iterations), total=dataset_length,
        leave=True)
    for epoch in trange(args.training.num_epochs, desc='epochs'):
        # collect trajectories via rolling out with the current policy
        if args.training.objective_optimization in ['ppo_dae', 'ppo_gae']:
            critic.set_device('cpu')
        train_env.to('cuda')
        replay_buffer, seed_offset, dataset_iterations = collect_trajectories(
            args, train_env, options, actor, epoch, seed_offset, dataset_iterations, dataset_progress)
        train_env.to('cpu')
        if args.training.objective_optimization in ['ppo_dae', 'ppo_gae']:
            critic.set_device('cuda')
        num_instances_seen = dataset_progress.n + dataset_length * dataset_iterations
        if args.training.objective_optimization == 'ppo_dae':
            updates = ppo_dae_update(
                args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates,
                num_instances_seen, num_instances_seen / dataset_length)
        elif args.training.objective_optimization == 'ppo_gae':
            updates = ppo_gae_update(
                args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates,
                num_instances_seen, num_instances_seen / dataset_length)
        elif args.training.objective_optimization == 'supervised':
            updates = supervised_update(
                args, replay_buffer, actor, actor_optimizer, epoch, updates, train_env,
                num_instances_seen, num_instances_seen / dataset_length)
        else:
            raise Exception
        if args.training.checkpoint_every is None or ((epoch + 1) % args.training.checkpoint_every) == 0:
            file = os.path.join(run.dir, 'ckpt_epoch={}_updates={}.pt'.format(epoch + 1, updates))
            print(f'Saving checkpoint to {file}')
            ckpt = {
                'actor': actor.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
            }
            if args.training.objective_optimization in ['ppo_dae', 'ppo_gae']:
                ckpt.update({
                    'critic': critic.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                })
            ckpt['seed_offset'] = seed_offset
            torch.save(ckpt, file)
        if args.training.clear_gpu: # only used for debugging
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
