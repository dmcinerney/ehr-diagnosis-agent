from torch import nn
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import io
import torch
from torch.distributions.normal import Normal
from .observation_embedder import ObservationEmbedder


class InterpretableRankingPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observation_embedder = ObservationEmbedder(config)
        dim = self.observation_embedder.diagnosis_encoder.config.hidden_size
        self.stddev_weight_1 = nn.Linear(dim, dim)
        self.mean_weight_1 = nn.Linear(dim, dim)
        self.stddev_weight_2 = nn.Linear(dim, dim)
        self.mean_weight_2 = nn.Linear(dim, dim)

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def get_dist_parameter_votes_and_evidence_strings(self, observation):
        diagnosis_embeddings, context_embeddings, context_strings = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'])
        if not observation['evidence_is_retrieved']:
            diagnosis_embeddings_stddev = self.stddev_weight_1(diagnosis_embeddings)
            diagnosis_embeddings_mean = self.mean_weight_1(diagnosis_embeddings)
        else:
            diagnosis_embeddings_stddev = self.stddev_weight_1(diagnosis_embeddings)
            diagnosis_embeddings_mean = self.mean_weight_1(diagnosis_embeddings)
        means = util.dot_score(diagnosis_embeddings_mean, context_embeddings)
        log_stddevs = util.dot_score(diagnosis_embeddings_stddev, context_embeddings)
        return means, log_stddevs, context_strings

    def get_dist_parameters(self, observation):
        means, log_stddevs, context_strings = self.get_dist_parameter_votes_and_evidence_strings(observation)
        return means.mean(-1), torch.nn.functional.softplus(log_stddevs.mean(-1)) + self.config.stddev_bias

    def get_dist(self, observation):
        mean, stddev = self.get_dist_parameters(observation)
        return Normal(mean, stddev)

    def forward(self, observation, return_log_prob=False):
        dist = self.get_dist(observation)
        action = dist.sample()
        if return_log_prob:
            return action, dist.log_prob(action)
        else:
            return action

    def log_prob(self, observation, action):
        return self.get_dist(observation).log_prob(action)
