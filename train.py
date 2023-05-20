import ehr_diagnosis_env
import gymnasium
from utils import *
import pandas as pd
import os
from actor import ActorPreprocessor, InterpretableRankingPolicy
from critic import CriticPreprocessor, Critic
import torch


if __name__ == '__main__':
    args = get_args('config.yaml')
    train_df = pd.read_csv(os.path.join(args.data.path, args.data.dataset, 'train.data'), compression='gzip')
    train_env = gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=train_df,
        model_name='google/flan-t5-xl',
        # model_name='google/flan-t5-xxl'
    )
    actor_preprocessor = ActorPreprocessor(args.actor_preprocessing)
    last_actor = InterpretableRankingPolicy(args.actor)
    actor = InterpretableRankingPolicy(args.actor)
    last_actor.load_state_dict(actor.state_dict())
    critic_preprocessor = CriticPreprocessor(args.critic_preprocessing)
    critic = Critic(args.critic)
    for epoch in range(args.num_epochs):
        # collect trajectories via rolling out with the current policy
        trajectories = []
        with torch.no_grad():
            for episode in range(args.num_episodes):
                terminated, truncated = False, False
                obs, info = train_env.reset(seed=epoch * episode)
                trajectories.append([(obs, info)])
                while not (terminated or truncated):
                    # sample action
                    action = actor(*actor_preprocessor(obs))
                    obs, reward, terminated, truncated, info = train_env.step(action)
                    trajectories[-1].append((obs, reward, terminated, truncated, info))
        # TODO: compute values/advantages using critic, ratio with actor and last_actor, and update actor with ppo objective
        # TODO: compute update critic with MSE objective computed using true reward
        last_actor.load_state_dict(actor.state_dict())
