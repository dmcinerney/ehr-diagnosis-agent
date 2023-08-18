import warnings
import ehr_diagnosis_env
from ehr_diagnosis_env.utils import get_model_interface
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup
from ehr_diagnosis_env.envs import EHRDiagnosisEnv
import gymnasium
from eval import evaluate_on_environment
from utils import get_args, collect_trajectories_train, TqdmSpy, \
    sample_actor_policy
import pandas as pd
import os
from models.actor import InterpretableNormalActor, \
    InterpretableDirichletActor, InterpretableBetaActor, InterpretableDeltaActor
from models.critic import Critic
import torch
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf
from updates import update, ppo_dae_update
import gc


actor_types = {
    'normal': InterpretableNormalActor,
    'dirichlet': InterpretableDirichletActor,
    'beta': InterpretableBetaActor,
    'delta': InterpretableDeltaActor,
}
recommended_reward_types = {
    'normal': ['continuous_dependent', 'ranking'],
    'dirichlet': ['continuous_dependent', 'ranking'],
    'beta': ['continuous_independent', 'ranking'],
    'delta': ['continuous_independent', 'continuous_dependent', 'ranking'],
}


def log_results(results, split):
    precision_recall_micro = {
        f'{split}_precision_micro': results[results.top_1].is_current_target.mean(),
        f'{split}_recall_micro': results[results.is_current_target].top_1.mean(),
        f'{split}_precision_det_micro': results[results.top_1_deterministic].is_current_target.mean(),
        f'{split}_recall_det_micro': results[results.is_current_target].top_1_deterministic.mean(),
    }
    targets = set(results.target)
    precision_recall = {
        f'{t}/{split}_{p_or_r}{det}': (
            results[
                (results.target==t) & results.top_1].is_current_target.mean()
            if p_or_r == 'precision' else
            results[
                (results.target==t) & results.is_current_target].top_1.mean()
        ) if det == '' else (
            results[
                (results.target==t) &
                results.top_1_deterministic].is_current_target.mean()
            if p_or_r == 'precision' else
            results[
                (results.target==t) &
                results.is_current_target].top_1_deterministic.mean()
        )
        for t in targets
        for p_or_r in ['precision', 'recall']
        for det in ['', '_det']
    }
    precision_recall_macro = {
        f'{split}_{p_or_r}{det}_macro': sum(
            [precision_recall[f'{t}/{split}_{p_or_r}{det}'] for t in targets])
            / len(targets)
        for p_or_r in ['precision', 'recall']
        for det in ['', '_det']
    }
    return {
        **precision_recall_micro,
        **precision_recall_macro,
        **precision_recall,
    }


class CustomPolicy:
    def __init__(self, random_query_policy, random_rp_policy):
        self.random_query_policy = random_query_policy
        self.random_rp_policy = random_rp_policy
    def __call__(self, actor, obs, env):
        device = next(iter(actor.parameters())).device
        if self.random_query_policy and not obs['evidence_is_retrieved']:
            return torch.tensor(env.action_space.sample(), device=device), \
                torch.tensor(0, device=device), {}
        if self.random_rp_policy and obs['evidence_is_retrieved']:
            return torch.tensor(env.action_space.sample(), device=device), \
                torch.tensor(0, device=device), {}
        return sample_actor_policy(actor, obs, env)


def main():
    args = get_args('config.yaml')
    run = wandb.init(
        project="ehr-diagnosis-agent",
        dir=args.output_dir,
        config=OmegaConf.to_container(args) # type: ignore
    )
    assert run is not None
    print('loading training dataset...')
    train_df = pd.read_csv(os.path.join(
        args.data.path, args.data.dataset, 'train.data'), compression='gzip')
    print(f'length={len(train_df)}')
    if args.training.limit_train_size is not None:
        # truncate to first instances because data indices have to be
        # maintained to use the cache
        train_df = train_df[:args.training.limit_train_size]
    llm_interface = get_model_interface(args.env.llm_name) \
        if args.env.llm_name is not None else None
    fmm_interface = SentenceTransformer(args.env.fmm_name) \
        if args.env.fmm_name is not None else None
    train_env: EHRDiagnosisEnv = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=train_df,
        cache_path=args.env.train_cache_path,
        llm_name_or_interface=llm_interface,
        fmm_name_or_interface=fmm_interface,
        reward_type=args.env.reward_type,
        num_future_diagnoses_threshold=args.env.num_future_diagnoses_threshold,
        progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
        top_k_evidence=args.env.top_k_evidence,
        verbosity=1, # don't print anything when an environment is dead
        add_risk_factor_queries=args.env.add_risk_factor_queries,
        limit_options_with_llm=args.env.limit_options_with_llm,
        add_none_of_the_above_option=args.env.add_none_of_the_above_option,
        alternatives_dir=args.env.alternatives_dir,
        risk_factors_dir=args.env.risk_factors_dir,
        true_positive_minimum=args.env.true_positive_minimum,
    ) # type: ignore
    if args.training.val_every is not None:
        print('loading validation dataset...')
        val_df = pd.read_csv(os.path.join(
            args.data.path, args.data.dataset, 'val.data'), compression='gzip')
        print('done')
        if args.training.limit_val_size is not None:
            # truncate to first instances because data indices have to be
            # maintained to use the cache
            val_df = val_df[:args.training.limit_val_size]
        val_env: EHRDiagnosisEnv | None = gymnasium.make(
            'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
            instances=val_df,
            cache_path=args.env.val_cache_path,
            llm_name_or_interface=llm_interface,
            fmm_name_or_interface=fmm_interface,
            reward_type=args.env.reward_type,
            num_future_diagnoses_threshold=
                args.env.num_future_diagnoses_threshold,
            progress_bar=lambda *a, **kwa: tqdm(*a, **kwa, leave=False),
            top_k_evidence=args.env.top_k_evidence,
            verbosity=1, # don't print anything when an environment is dead
            add_risk_factor_queries=args.env.add_risk_factor_queries,
            limit_options_with_llm=args.env.limit_options_with_llm,
            add_none_of_the_above_option=args.env.add_none_of_the_above_option,
            alternatives_dir=args.env.alternatives_dir,
            risk_factors_dir=args.env.risk_factors_dir,
            true_positive_minimum=args.env.true_positive_minimum,
        ) # type: ignore
    else:
        val_env = None
    if args.env.reward_type not in recommended_reward_types[args.actor.type]:
        warnings.warn(
            'Reward type "{}" does not align with the actor type "{}".'.format(
            args.env.reward_type, args.actor.type))
    actor_params = args.actor['{}_params'.format(args.actor.type)]
    actor_params.update(args.actor['shared_params'])
    actor = actor_types[args.actor.type](actor_params)
    actor.train()
    actor.set_device('cuda')
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=args.training.actor_lr)
    actor_scheduler = get_linear_schedule_with_warmup(
        actor_optimizer, args.training.actor_warmup,
        args.training.actor_training_steps)
    if args.training.objective_optimization in [
            'ppo_dae', 'ppo_gae', 'mix_objectives']:
        critic = Critic(args.critic)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=args.training.critic_lr)
        critic_scheduler = get_linear_schedule_with_warmup(
            critic_optimizer, args.training.critic_warmup,
            args.training.critic_training_steps)
    else:
        critic = None
        critic_optimizer = None
        critic_scheduler = None
    seed_offset = 0
    if args.training.resume_from is not None:
        print(f'Resuming from {args.training.resume_from}')
        ckpt = torch.load(args.training.resume_from)
        actor.load_state_dict(ckpt['actor'])
        actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        actor_scheduler.load_state_dict(ckpt['actor_scheduler'])
        if critic is not None and critic_optimizer is not None \
                and critic_scheduler is not None and 'critic' in ckpt.keys():
            critic.load_state_dict(ckpt['critic'])
            critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
            critic_scheduler.load_state_dict(ckpt['critic_scheduler'])
        seed_offset = ckpt['seed_offset']
        del ckpt
        if args.training.clear_gpu: # only used for debugging
            gc.collect()
            torch.cuda.empty_cache()
    with warnings.catch_warnings():
        actor_scheduler.step()
        if critic_scheduler is not None:
            critic_scheduler.step()
    options = {}
    if args.data.max_reports_considered is not None:
        options['max_reports_considered'] = args.data.max_reports_considered
    updates = 0
    dataset_iterations = 0
    dataset_progress = TqdmSpy(
        desc='dataset progress (iteration {})'.format(dataset_iterations),
        total=train_env.num_unseen_examples(), leave=True)
    for epoch in trange(args.training.num_epochs, desc='epochs'):
        if critic is not None:
            critic.set_device('cpu')
            if args.training.clear_gpu: # only used for debugging
                gc.collect()
                torch.cuda.empty_cache()
        train_env.to('cuda') # train and val envs have connected models
        if val_env is not None and epoch % args.training.val_every == 0:
            # validate current model
            results_file = os.path.join(
                run.dir, 'val_metrics_ckpt_epoch={}_updates={}.csv'.format(
                    epoch, updates))
            step_results, episode_results = evaluate_on_environment(
                val_env, actor, options=options,
                max_num_episodes=args.training.val_max_num_episodes,
                max_trajectory_length=args.training.val_max_trajectory_length,
                filename=results_file)
            wandb.log({
                'epoch': epoch,
                'updates': updates,
                **log_results(step_results, 'val'),
            })
        if train_env is not None and epoch % args.training.trainmetrics_every == 0:
            results_file = os.path.join(
                run.dir, 'train_metrics_ckpt_epoch={}_updates={}.csv'.format(
                    epoch, updates))
            step_results, episode_results = evaluate_on_environment(
                train_env, actor, options=options,
                max_num_episodes=args.training.trainmetrics_max_num_episodes,
                max_trajectory_length=
                    args.training.trainmetrics_max_trajectory_length,
                filename=results_file)
            wandb.log({
                'epoch': epoch,
                'updates': updates,
                **log_results(step_results, 'train'),
            })
        # collect trajectories via rolling out with the current policy
        replay_buffer, seed_offset, dataset_iterations = \
            collect_trajectories_train(
                args, train_env, options, actor, epoch, seed_offset,
                dataset_iterations, dataset_progress,
                custom_policy=CustomPolicy(
                    args.training.random_query_policy,
                    args.training.random_rp_policy))
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
                args, replay_buffer, actor, actor_optimizer,
                actor_scheduler, critic, critic_optimizer,
                critic_scheduler, epoch, updates)
        elif args.training.objective_optimization in [
                'ppo_gae', 'supervised', 'mix_objectives']:
            updates = update(
                args, replay_buffer, actor, actor_optimizer,
                actor_scheduler, critic, critic_optimizer,
                critic_scheduler, epoch, updates, train_env)
        else:
            raise Exception
        if args.training.checkpoint_every is None or (
                (epoch + 1) % args.training.checkpoint_every) == 0:
            file = os.path.join(run.dir, 'ckpt_epoch={}_updates={}.pt'.format(
                epoch + 1, updates))
            print(f'Saving checkpoint to {file}')
            ckpt = {
                'actor': actor.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'actor_scheduler': actor_scheduler.state_dict(),
            }
            if critic is not None and critic_optimizer is not None \
                    and critic_scheduler is not None:
                ckpt.update({
                    'critic': critic.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'critic_scheduler': critic_scheduler.state_dict(),
                })
            ckpt['seed_offset'] = seed_offset
            torch.save(ckpt, file)
        if args.training.clear_gpu: # only used for debugging
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
