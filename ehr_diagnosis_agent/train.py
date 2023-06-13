import warnings
import ehr_diagnosis_env
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from utils import get_args, collect_trajectories, TqdmSpy
import pandas as pd
import os
from models.actor import InterpretableNormalActor, InterpretableDirichletActor, InterpretableBetaActor
from models.critic import Critic
import torch
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf
from updates.ppo_dae import ppo_dae_update
from updates.ppo_gae import ppo_gae_update
from updates.supervised import supervised_update
import gc


actory_types = {
    'normal': InterpretableNormalActor,
    'dirichlet': InterpretableDirichletActor,
    'beta': InterpretableBetaActor,
}
recommended_reward_types = {
    'normal': ['continuous_dependent', 'ranking'],
    'dirichlet': ['continuous_dependent', 'ranking'],
    'beta': ['continuous_independent', 'ranking'],
}


def main():
    args = get_args('config.yaml')
    run = wandb.init(
        project="ehr-diagnosis-agent",
        config=OmegaConf.to_container(args) # type: ignore
    )
    assert run is not None
    train_df = pd.read_csv(os.path.join(args.data.path, args.data.dataset, 'train.data'), compression='gzip')
    if args.training.limit_train_size is not None:
        # truncate to first instances because data indices have to be maintained to use the cache
        train_df = train_df[:args.training.limit_train_size]
    train_env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=train_df,
        model_name=args.env.model_name,
        reward_type=args.env.reward_type,
        num_future_diagnoses_threshold=args.env.num_future_diagnoses_threshold,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        cache_path=args.env.cache_path,
        top_k_evidence=args.env.top_k_evidence,
        verbosity=1, # don't print anything when an environment is dead
    ) # type: ignore
    if args.env.reward_type not in recommended_reward_types[args.actor.type]:
        warnings.warn('Reward type "{}" does not align with the actor type "{}".'.format(args.env.reward_type, args.actor.type))
    actor = actory_types[args.actor.type](args.actor['{}_params'.format(args.actor.type)])
    actor.train()
    actor.set_device('cuda')
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.training.actor_lr)
    if args.training.objective_optimization in ['ppo_dae', 'ppo_gae']:
        critic = Critic(args.critic)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.training.critic_lr)
    else:
        critic = None
        critic_optimizer = None
    seed_offset = 0
    if args.training.resume_from is not None:
        print(f'Resuming from {args.training.resume_from}')
        ckpt = torch.load(args.training.resume_from)
        actor.load_state_dict(ckpt['actor'])
        actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        if critic is not None and critic_optimizer is not None and 'critic' in ckpt.keys():
            critic.load_state_dict(ckpt['critic'])
            critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        seed_offset = ckpt['seed_offset']
        del ckpt
        if args.training.clear_gpu: # only used for debugging
            gc.collect()
            torch.cuda.empty_cache()
    options = {}
    if args.training.max_reports_considered is not None:
        options['max_reports_considered'] = args.training.max_reports_considered
    updates = 0
    dataset_iterations = 0
    dataset_progress = TqdmSpy(
        desc='dataset progress (iteration {})'.format(dataset_iterations), total=train_env.num_unseen_examples(),
        leave=True)
    for epoch in trange(args.training.num_epochs, desc='epochs'):
        # collect trajectories via rolling out with the current policy
        if critic is not None:
            critic.set_device('cpu')
        train_env.to('cuda')
        replay_buffer, seed_offset, dataset_iterations = collect_trajectories(
            args, train_env, options, actor, epoch, seed_offset, dataset_iterations, dataset_progress)
        train_env.to('cpu')
        if critic is not None:
            critic.set_device('cuda')
        if args.training.freeze_actor is not None:
            if isinstance(args.training.freeze_actor, str):
                freeze_actor_mode = args.training.freeze_actor
            else:
                freeze_actor_mode = None
            actor.freeze(mode=freeze_actor_mode)
        if args.training.objective_optimization == 'ppo_dae':
            updates = ppo_dae_update(
                args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates)
        elif args.training.objective_optimization == 'ppo_gae':
            updates = ppo_gae_update(
                args, replay_buffer, actor, actor_optimizer, critic, critic_optimizer, epoch, updates)
        elif args.training.objective_optimization == 'supervised':
            updates = supervised_update(
                args, replay_buffer, actor, actor_optimizer, epoch, updates, train_env)
        else:
            raise Exception
        if args.training.checkpoint_every is None or ((epoch + 1) % args.training.checkpoint_every) == 0:
            file = os.path.join(run.dir, 'ckpt_epoch={}_updates={}.pt'.format(epoch + 1, updates))
            print(f'Saving checkpoint to {file}')
            ckpt = {
                'actor': actor.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
            }
            if critic is not None and critic_optimizer is not None:
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
