from torch import nn
from sentence_transformers import util
import torch
from torch.distributions import Normal, Dirichlet, TransformedDistribution, ExpTransform
from .observation_embedder import ObservationEmbedder


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observation_embedder = ObservationEmbedder(config)

    def freeze(self, mode=None):
        raise NotImplementedError

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def forward(self, observation, return_log_prob=False):
        dist = self.get_dist(observation)
        action = dist.rsample()
        if return_log_prob:
            return action, dist.log_prob(action)
        else:
            return action

    def log_prob(self, observation, action):
        return self.get_dist(observation).log_prob(action)

    def get_dist_parameter_votes_and_evidence_strings(self, observation):
        raise NotImplementedError

    def votes_to_parameters(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def parameters_to_dist(*args, **kwargs):
        raise NotImplementedError

    def get_dist_parameters(self, observation):
        return self.votes_to_parameters(*self.get_dist_parameter_votes_and_evidence_strings(observation)[:-1])

    def get_dist(self, observation):
        return self.parameters_to_dist(*self.get_dist_parameters(observation))

    @staticmethod
    def get_dist_stats(dists):
        return {}

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.entropy().sum() for dist in dists])


class InterpretableNormalActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.diagnosis_encoder.config.hidden_size
        self.stddev_weight_query = nn.Linear(dim, dim, bias=False)
        self.mean_weight_query = nn.Linear(dim, dim, bias=False)
        self.stddev_weight_risk = nn.Linear(dim, dim, bias=False)
        self.mean_weight_risk = nn.Linear(dim, dim, bias=False)
        self.entropy_control = nn.Parameter(torch.ones(1))

    def freeze(self, mode=None):
        if mode is None:
            return
        elif mode == 'everything':
            for n, p in self.named_parameters():
                p.requires_grad = False
        elif mode == 'everything_but_query':
            for n, p in self.named_parameters():
                p.requires_grad = n.startswith('stddev_weight_query.') or n.startswith('mean_weight_query.')
        else:
            raise Exception

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def get_dist_parameter_votes_and_evidence_strings(self, observation):
        diagnosis_embeddings, context_embeddings, context_strings = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'])
        if not observation['evidence_is_retrieved']:
            diagnosis_embeddings_stddev = self.stddev_weight_query(diagnosis_embeddings)
            diagnosis_embeddings_mean = self.mean_weight_query(diagnosis_embeddings)
        else:
            diagnosis_embeddings_stddev = self.stddev_weight_query(diagnosis_embeddings)
            diagnosis_embeddings_mean = self.mean_weight_query(diagnosis_embeddings)
        means = util.dot_score(diagnosis_embeddings_mean, context_embeddings)
        log_stddevs = util.dot_score(diagnosis_embeddings_stddev, context_embeddings)
        return means, log_stddevs, context_strings

    def votes_to_parameters(self, means, log_stddevs):
        mean = means.mean(-1)
        mean = self.entropy_control * (mean - mean.mean()) / mean.std()
        # stddev = torch.nn.functional.softplus(log_stddevs.mean(-1)) + self.config.stddev_bias
        stddev = self.config.stddev_min + (
            self.config.stddev_max - self.config.stddev_min) * torch.nn.functional.sigmoid(log_stddevs.mean(-1))
        return mean, stddev

    @staticmethod
    def parameters_to_dist(mean, stddev):
        return Normal(mean, stddev)

    @staticmethod
    def get_dist_stats(dists):
        return {
            'avg_normal_loc': torch.stack([dist.loc.mean() for dist in dists]).mean(),
            'avg_normal_loc_stddev': torch.stack([dist.loc.std() for dist in dists]).mean(),
            'avg_normal_scale': torch.stack([dist.scale.mean() for dist in dists]).mean()}


class InterpretableDirichletActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.diagnosis_encoder.config.hidden_size
        self.concentration_weight_query = nn.Linear(dim, dim, bias=False)
        self.concentration_weight_risk = nn.Linear(dim, dim, bias=False)

    def freeze(self, mode=None):
        if mode is None:
            return
        elif mode == 'everything':
            for n, p in self.named_parameters():
                p.requires_grad = False
        elif mode == 'everything_but_query':
            for n, p in self.named_parameters():
                p.requires_grad = n.startswith('concentration_weight_query.')
        else:
            raise Exception

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def get_dist_parameter_votes_and_evidence_strings(self, observation):
        diagnosis_embeddings, context_embeddings, context_strings = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'])
        if not observation['evidence_is_retrieved']:
            diagnosis_embeddings = self.concentration_weight_query(diagnosis_embeddings)
        else:
            diagnosis_embeddings = self.concentration_weight_risk(diagnosis_embeddings)
        concentrations = util.dot_score(diagnosis_embeddings, context_embeddings)
        return concentrations, context_strings

    def votes_to_parameters(self, concentrations):
        concentration = concentrations.mean(-1)
        return torch.nn.functional.softplus(concentration) + self.config.concentration_min,

    @staticmethod
    def parameters_to_dist(concentration):
        # need to transform the dirichlet because
        return TransformedDistribution(Dirichlet(concentration), [ExpTransform().inv])

    @staticmethod
    def get_dist_stats(dists):
        return {
            'avg_dirichlet_concentration': torch.stack(
                [dist.base_dist.concentration.mean() for dist in dists]).mean(),
            'avg_dirichlet_concentration_stddev': torch.stack(
                [dist.base_dist.concentration.std() for dist in dists]).mean()}

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.base_dist.entropy().sum() for dist in dists])
