from torch import nn
from sentence_transformers import util
import torch
from torch.distributions import Normal, Dirichlet, TransformedDistribution, ExpTransform, Beta
from .observation_embedder import ObservationEmbedder


# reimplementing a simple attention module because multihead attention modifies the values
class Attention(nn.Module):
    def __init__(self, in_dim, embed_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.query_weight = nn.Linear(self.in_dim, self.embed_dim)
        self.key_weight = nn.Linear(self.in_dim, self.embed_dim)

    def forward(self, query, key, value):
        # query (l, n, e_in), key (s, n, e_in), value (s, n, e_v)
        query = self.query_weight(query)
        key = self.key_weight(key)
        # (n, l, s) = (n, l, e_in) x (n, e_in, s)
        attn_output_weights = torch.softmax(torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)), -1)
        # (n, l, e_v) = (n, l, s) x (n, s, e_v)
        attn_output = torch.bmm(attn_output_weights, value.permute(1, 0, 2))
        # (l, n, e_v), (n, l, s)
        return attn_output.permute(1, 0, 2), attn_output_weights


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observation_embedder = ObservationEmbedder(config)
        dim = self.observation_embedder.diagnosis_encoder.config.hidden_size
        self.attention = Attention(dim, dim) if self.config.use_attn else None

    @property
    def param_vec_size(self):
        raise NotImplementedError

    def freeze(self, mode=None):
        raise NotImplementedError

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def forward(self, observation):
        vote_info = self.get_dist_parameter_votes(observation)
        param_info = self.votes_to_parameters(
            vote_info['diagnosis_embeddings'], vote_info['context_embeddings'], vote_info['param_votes'])
        dist = self.parameters_to_dist(*param_info['params'])
        action = dist.rsample()
        return {
            'action': action,
            'log_prob': dist.log_prob(action),
            'context_strings': vote_info['context_strings'],
            **param_info,
        }

    def log_prob(self, observation, action):
        return self.get_dist(observation).log_prob(action)

    def get_dist_parameter_votes(self, observation):
        raise NotImplementedError

    def votes_to_parameters(self, diagnosis_embeddings, context_embeddings, param_votes):
        """
        diagnosis embeddings (num_diagnoses, kdim)
        context_embeddings (num_contexts, kdim)
        param_votes (num_diagnoses, num_contexts, vdim)
        """
        if self.attention is None:
            return {'params': param_votes.mean(1)}
        else:
            kdim = diagnosis_embeddings.shape[-1]
            nd, nc, vdim = param_votes.shape
            # (1, num_diagnoses, vdim)
            diagnosis_embeddings = diagnosis_embeddings.unsqueeze(0)
            # (num_contexts, num_diagnoses, kdim)
            context_embeddings = context_embeddings.unsqueeze(1).expand(nc, nd, kdim)
            # (num_contexts, num_diagnoses, vdim)
            param_votes = param_votes.transpose(0, 1)
            # attn_output (1, num_diagnoses, vdim), attn_output_weights (num_diagnoses, 1, num_contexts)
            attn_output, attn_output_weights = self.attention(
                query=diagnosis_embeddings, key=context_embeddings, value=param_votes)
            return {'params': attn_output.squeeze(0), 'context_attn_weights': attn_output_weights.squeeze(1)}

    def get_dist_parameters(self, observation):
        vote_info = self.get_dist_parameter_votes(observation)
        return self.votes_to_parameters(
            vote_info['diagnosis_embeddings'], vote_info['context_embeddings'], vote_info['param_votes'])

    @staticmethod
    def parameters_to_dist(*args, **kwargs):
        raise NotImplementedError

    def get_dist(self, observation):
        return self.parameters_to_dist(*self.get_dist_parameters(observation)['params'])

    @staticmethod
    def get_dist_stats(dists):
        return {}

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.entropy().sum() for dist in dists])


# TODO: consider adding extra weights to transform the context embeddings
# TODO: consider collapsing parameter-specific weights into the same layer


class InterpretableNormalActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.diagnosis_encoder.config.hidden_size
        self.stddev_weight_query = nn.Linear(dim, dim, bias=False)
        self.mean_weight_query = nn.Linear(dim, dim, bias=False)
        self.stddev_weight_risk = nn.Linear(dim, dim, bias=False)
        self.mean_weight_risk = nn.Linear(dim, dim, bias=False)
        self.entropy_control = nn.Parameter(torch.ones(1))

    @property
    def param_vec_size(self):
        return 2

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

    def get_dist_parameter_votes(self, observation):
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
        param_votes = torch.stack([means, log_stddevs], 2)
        return {
            'param_votes': param_votes,
            'context_strings': context_strings,
            'context_embeddings': context_embeddings,
            'diagnosis_embeddings': diagnosis_embeddings,
        }

    def votes_to_parameters(self, diagnosis_embeddings, context_embeddings, param_votes):
        return_dict = super().votes_to_parameters(diagnosis_embeddings, context_embeddings, param_votes)
        mean, log_stddev = return_dict['params'][:, 0], return_dict['params'][:, 0]
        mean = self.entropy_control * (mean - mean.mean()) / mean.std()
        # stddev = torch.nn.functional.softplus(log_stddevs.mean(-1)) + self.config.stddev_bias
        stddev = self.config.stddev_min + (
            self.config.stddev_max - self.config.stddev_min) * torch.nn.functional.sigmoid(log_stddev)
        return_dict['params'] = (mean, stddev)
        return return_dict

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

    @property
    def param_vec_size(self):
        return 1

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

    def get_dist_parameter_votes(self, observation):
        diagnosis_embeddings, context_embeddings, context_strings = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'])
        if not observation['evidence_is_retrieved']:
            diagnosis_embeddings = self.concentration_weight_query(diagnosis_embeddings)
        else:
            diagnosis_embeddings = self.concentration_weight_risk(diagnosis_embeddings)
        concentrations = util.dot_score(diagnosis_embeddings, context_embeddings)
        param_votes = concentrations.unsqueeze(2)
        return {
            'param_votes': param_votes,
            'context_strings': context_strings,
            'context_embeddings': context_embeddings,
            'diagnosis_embeddings': diagnosis_embeddings,
        }

    def votes_to_parameters(self, diagnosis_embeddings, context_embeddings, param_votes):
        return_dict = super().votes_to_parameters(diagnosis_embeddings, context_embeddings, param_votes)
        concentration = torch.nn.functional.softplus(return_dict['params'][:, 0]) + self.config.concentration_min
        return_dict['params'] = (concentration,)
        return return_dict

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
    def get_mean(dists):
        return torch.stack([dist.base_dist.mean.log() for dist in dists])

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.base_dist.entropy().sum() for dist in dists])


class InterpretableBetaActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.diagnosis_encoder.config.hidden_size
        self.concentration1_weight_query = nn.Linear(dim, dim, bias=False)
        self.concentration0_weight_query = nn.Linear(dim, dim, bias=False)
        self.concentration1_weight_risk = nn.Linear(dim, dim, bias=False)
        self.concentration0_weight_risk = nn.Linear(dim, dim, bias=False)

    @property
    def param_vec_size(self):
        return 2

    def freeze(self, mode=None):
        if mode is None:
            return
        elif mode == 'everything':
            for n, p in self.named_parameters():
                p.requires_grad = False
        elif mode == 'everything_but_query':
            for n, p in self.named_parameters():
                p.requires_grad = n.startswith('prob_weight_query.')
        else:
            raise Exception

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def get_dist_parameter_votes(self, observation):
        diagnosis_embeddings, context_embeddings, context_strings = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'])
        if not observation['evidence_is_retrieved']:
            diagnosis_embeddings_con1 = self.concentration1_weight_query(diagnosis_embeddings)
            diagnosis_embeddings_con0 = self.concentration0_weight_query(diagnosis_embeddings)
        else:
            diagnosis_embeddings_con1 = self.concentration1_weight_risk(diagnosis_embeddings)
            diagnosis_embeddings_con0 = self.concentration0_weight_risk(diagnosis_embeddings)
        concentrations1 = util.dot_score(diagnosis_embeddings_con1, context_embeddings)
        concentrations0 = util.dot_score(diagnosis_embeddings_con0, context_embeddings)
        # under the assumption that the dot score output is naturally initially centered around 0,
        # this changes concentrations0 to be initially centered around init_concentration0
        concentrations0 = concentrations0 + self.config.init_concentration0
        param_votes = torch.stack([concentrations1, concentrations0], 2)
        return {
            'param_votes': param_votes,
            'context_strings': context_strings,
            'context_embeddings': context_embeddings,
            'diagnosis_embeddings': diagnosis_embeddings,
        }

    def votes_to_parameters(self, diagnosis_embeddings, context_embeddings, param_votes):
        return_dict = super().votes_to_parameters(diagnosis_embeddings, context_embeddings, param_votes)
        concentration1, concentration0 = return_dict['params'][:, 0], return_dict['params'][:, 1]
        concentration1 = torch.nn.functional.softplus(concentration1) + self.config.concentration_min
        concentration0 = torch.nn.functional.softplus(concentration0) + self.config.concentration_min
        return_dict['params'] = (concentration1, concentration0)
        return return_dict

    @staticmethod
    def parameters_to_dist(concentration1, concentration0):
        return TransformedDistribution(Beta(concentration1, concentration0), [ExpTransform().inv])

    @staticmethod
    def get_dist_stats(dists):
        return {
            'avg_beta_concentration1': torch.stack(
                [dist.base_dist.concentration1.mean() for dist in dists]).mean(),
            'avg_beta_concentration1_stddev': torch.stack(
                [dist.base_dist.concentration1.std() for dist in dists]).mean(),
            'avg_beta_concentration0': torch.stack(
                [dist.base_dist.concentration0.mean() for dist in dists]).mean(),
            'avg_beta_concentration0_stddev': torch.stack(
                [dist.base_dist.concentration0.std() for dist in dists]).mean()}

    @staticmethod
    def get_mean(dists):
        return torch.stack([dist.base_dist.mean.log() for dist in dists])

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.base_dist.entropy().sum() for dist in dists])
