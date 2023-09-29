from torch import nn
from sentence_transformers import util
import torch
from torch.distributions import Normal, Dirichlet, TransformedDistribution, \
    ExpTransform, Beta, SigmoidTransform
from .utils import Delta
from .observation_embedder import InterpretableObservationEmbedder, \
    BertObservationEmbedder


# reimplementing a simple attention module because multihead attention
# modifies the values
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
        attn_output_weights = torch.softmax(torch.bmm(
            query.permute(1, 0, 2), key.permute(1, 2, 0)), -1)
        # (n, l, e_v) = (n, l, s) x (n, s, e_v)
        attn_output = torch.bmm(attn_output_weights, value.permute(1, 0, 2))
        # (l, n, e_v), (n, l, s)
        return attn_output.permute(1, 0, 2), attn_output_weights


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observation_embedder = InterpretableObservationEmbedder(config) \
            if config.embedder_type == 'interpretable' else \
            BertObservationEmbedder(config)
        dim = self.observation_embedder.context_encoder.config.hidden_size
        self.attention = Attention(dim, dim) if self.config.use_attn else None
        if self.has_bias and self.config.static_diagnoses is not None:
            self.static_bias = nn.Embedding(
                len(self.config.static_diagnoses), self.param_vec_size)
            if self.config.static_bias_params is not None:
                self.static_bias.weight.data = torch.tensor(
                    self.config.static_bias_params)
                for p in self.static_bias.parameters():
                    p.requires_grad = False

    def load_state_dict(self, state_dict):
        if self.has_bias and self.config.static_diagnoses is not None and \
                self.config.static_bias_params is not None:
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('static_bias.')}
        keys = super().load_state_dict(state_dict, strict=False)
        print(keys)
        return keys

    @property
    def param_vec_size(self):
        raise NotImplementedError

    def freeze(self, mode=None):
        raise NotImplementedError

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def get_static_bias(self, diagnosis_strings):
        if self.has_bias and self.config.static_diagnoses is not None:
            return self.static_bias(
                torch.tensor(
                    [self.observation_embedder.diagnosis_mapping[x]
                     for x in diagnosis_strings],
                    device=self.static_bias.weight.device))

    def forward(self, observation):
        vote_info = self.get_dist_parameter_votes(observation)
        param_info = self.votes_to_parameters(vote_info)
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

    @property
    def has_bias(self):
        return self.config.diagnosis_bias

    def transform_parameters(self, param_info):
        raise NotImplementedError

    def votes_to_parameters(self, vote_info):
        """
        diagnosis embeddings (num_diagnoses, kdim)
        context_embeddings (num_contexts, kdim)
        param_votes (num_diagnoses, num_contexts, vdim)
        """
        param_votes = vote_info['param_votes']
        static_bias = self.get_static_bias(vote_info['diagnosis_strings'])
        param_info = {}
        if self.has_bias:
            if self.config.static_diagnoses is None:
                param_info['bias'] = param_votes[:, 0]
            else:
                assert static_bias is not None
                param_info['bias'] = static_bias
        if self.attention is None:
            if self.has_bias and self.config.separate_bias and \
                    self.config.static_diagnoses is None:
                params = param_votes[:, 0]
                if param_votes.shape[1] > 1:
                    params = params / self.config.constant_denominator \
                        + param_votes[:, 1:].mean(1)
            else:
                params = param_votes.mean(1)
            if static_bias is not None:
                if vote_info['context_info'][0] == 'no evidence':
                    params = static_bias
                else:
                    params = params / self.config.constant_denominator \
                        + static_bias
            param_info['params'] = params
        else:
            diagnosis_embeddings = vote_info['diagnosis_embeddings']
            context_embeddings = vote_info['context_embeddings']
            kdim = diagnosis_embeddings.shape[-1]
            nd, nc, vdim = param_votes.shape
            # (1, num_diagnoses, vdim)
            diagnosis_embeddings = diagnosis_embeddings.unsqueeze(0)
            # (num_contexts, num_diagnoses, kdim)
            context_embeddings = context_embeddings.unsqueeze(1).expand(
                nc, nd, kdim)
            # (num_contexts, num_diagnoses, vdim)
            param_votes = param_votes.transpose(0, 1)
            if self.has_bias and self.config.separate_bias and  \
                    self.config.static_diagnoses is None:
                if nc > 1:
                    # attn_output (1, num_diagnoses, vdim),
                    #   attn_output_weights
                    #   (num_diagnoses, 1, num_contexts - 1)
                    attn_output, attn_output_weights = self.attention(
                        query=diagnosis_embeddings,
                        key=context_embeddings[1:],
                        value=param_votes[1:])
                    # add the bias
                    params = attn_output.squeeze(0) / \
                        self.config.constant_denominator + param_votes[0]
                else:
                    params = param_votes[0]
            else:
                # attn_output (1, num_diagnoses, vdim), attn_output_weights
                #   (num_diagnoses, 1, num_contexts)
                attn_output, attn_output_weights = self.attention(
                    query=diagnosis_embeddings, key=context_embeddings,
                    value=param_votes)
                params = attn_output.squeeze(0)
            if static_bias is not None:
                if vote_info['context_info'][0] == 'no evidence':
                    params = static_bias
                else:
                    params = params / self.config.constant_denominator \
                        + static_bias
            param_info['params'] = params
            param_info['context_attn_weights'] = attn_output_weights.squeeze(1)
        return self.transform_parameters(param_info)

    def get_dist_parameters(self, observation):
        vote_info = self.get_dist_parameter_votes(observation)
        return self.votes_to_parameters(vote_info)

    @staticmethod
    def parameters_to_dist(*args, **kwargs):
        raise NotImplementedError

    def get_dist(self, observation):
        return self.parameters_to_dist(
            *self.get_dist_parameters(observation)['params'])

    def get_dist_stats(self, forward_pass_info):
        return {}

    @staticmethod
    def get_mean(dists):
        return torch.stack([dist.mean for dist in dists])

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.entropy().sum() for dist in dists])


# TODO: consider adding extra weights to transform the context embeddings
# TODO: consider collapsing parameter-specific weights into the same layer


class InterpretableNormalActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.context_encoder.config.hidden_size
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
                p.requires_grad = n.startswith('stddev_weight_query.') or \
                    n.startswith('mean_weight_query.')
        else:
            raise Exception

    def get_dist_parameter_votes(self, observation):
        diagnosis_embeddings, context_embeddings, diagnosis_strings, \
            context_strings, context_info = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'] \
                and not self.config.ignore_evidence,
            ignore_evidence=self.config.ignore_evidence)
        if not observation['evidence_is_retrieved']:
            diagnosis_embeddings_stddev = self.stddev_weight_query(
                diagnosis_embeddings)
            diagnosis_embeddings_mean = self.mean_weight_query(
                diagnosis_embeddings)
        else:
            diagnosis_embeddings_stddev = self.stddev_weight_query(
                diagnosis_embeddings)
            diagnosis_embeddings_mean = self.mean_weight_query(
                diagnosis_embeddings)
        means = util.dot_score(diagnosis_embeddings_mean, context_embeddings)
        log_stddevs = util.dot_score(
            diagnosis_embeddings_stddev, context_embeddings)
        param_votes = torch.stack([means, log_stddevs], 2)
        return {
            'param_votes': param_votes,
            'context_strings': context_strings,
            'context_embeddings': context_embeddings,
            'diagnosis_strings': diagnosis_strings,
            'diagnosis_embeddings': diagnosis_embeddings,
            'context_info': context_info,
        }
    
    def transform_parameters(self, param_info):
        mean = param_info['params'][:, 0]
        log_stddev = param_info['params'][:, 1]
        mean = self.entropy_control * (mean - mean.mean()) / mean.std()
        # stddev = torch.nn.functional.softplus(log_stddevs.mean(-1)) + \
        #     self.config.stddev_bias
        stddev = self.config.stddev_min + (
            self.config.stddev_max - self.config.stddev_min
            ) * torch.nn.functional.sigmoid(log_stddev)
        param_info['params'] = (mean, stddev)
        return param_info

    @staticmethod
    def parameters_to_dist(mean, stddev):
        return Normal(mean, stddev)

    def get_dist_stats(self, forward_pass_info):
        return {}
        # return {
        #     'avg_normal_loc': torch.stack([dist.loc.mean() for dist in dists]).mean(),
        #     'avg_normal_loc_stddev': torch.stack([dist.loc.std() for dist in dists]).mean(),
        #     'avg_normal_scale': torch.stack([dist.scale.mean() for dist in dists]).mean()}


class InterpretableDirichletActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.context_encoder.config.hidden_size
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

    def get_dist_parameter_votes(self, observation):
        diagnosis_embeddings, context_embeddings, diagnosis_strings, \
            context_strings, context_info = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'] \
                and not self.config.ignore_evidence,
            ignore_evidence=self.config.ignore_evidence)
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
            'diagnosis_strings': diagnosis_strings,
            'diagnosis_embeddings': diagnosis_embeddings,
            'context_info': context_info,
        }

    def transform_parameters(self, param_info):
        concentration = torch.nn.functional.softplus(param_info['params'][:, 0]) + self.config.concentration_min
        param_info['params'] = (concentration,)
        return param_info

    @staticmethod
    def parameters_to_dist(concentration):
        # need to transform the dirichlet because
        return TransformedDistribution(Dirichlet(concentration), [ExpTransform().inv])

    def get_dist_stats(self, forward_pass_info):
        return {}
        # return {
        #     'avg_dirichlet_concentration': torch.stack(
        #         [dist.base_dist.concentration.mean() for dist in dists]).mean(),
        #     'avg_dirichlet_concentration_stddev': torch.stack(
        #         [dist.base_dist.concentration.std() for dist in dists]).mean()}

    @staticmethod
    def get_mean(dists):
        return torch.stack([dist.base_dist.mean.log() for dist in dists])

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.base_dist.entropy().sum() for dist in dists])


class InterpretableBetaActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.context_encoder.config.hidden_size
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
                p.requires_grad = \
                    n.startswith('concentration0_weight_query.') or \
                    n.startswith('concentration1_weight_query.')
        else:
            raise Exception

    def get_dist_parameter_votes(self, observation):
        diagnosis_embeddings, context_embeddings, diagnosis_strings, \
            context_strings, context_info = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'] \
                and not self.config.ignore_evidence,
            ignore_evidence=self.config.ignore_evidence)
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
            'diagnosis_strings': diagnosis_strings,
            'diagnosis_embeddings': diagnosis_embeddings,
            'context_info': context_info,
        }

    def transform_parameters(self, param_info):
        concentration1, concentration0 = param_info['params'][:, 0], param_info['params'][:, 1]
        concentration1 = torch.nn.functional.softplus(concentration1) + self.config.concentration_min
        concentration0 = torch.nn.functional.softplus(concentration0) + self.config.concentration_min
        param_info['params'] = (concentration1, concentration0)
        return param_info

    @staticmethod
    def parameters_to_dist(concentration1, concentration0):
        return TransformedDistribution(Beta(concentration1, concentration0), [SigmoidTransform().inv])

    def get_dist_stats(self, forward_pass_info):
        return {}
        # return {
        #     'avg_beta_concentration1': torch.stack(
        #         [dist.base_dist.concentration1.mean() for dist in dists]).mean(),
        #     'avg_beta_concentration1_stddev': torch.stack(
        #         [dist.base_dist.concentration1.std() for dist in dists]).mean(),
        #     'avg_beta_concentration0': torch.stack(
        #         [dist.base_dist.concentration0.mean() for dist in dists]).mean(),
        #     'avg_beta_concentration0_stddev': torch.stack(
        #         [dist.base_dist.concentration0.std() for dist in dists]).mean()}

    @staticmethod
    def get_mean(dists):
        return torch.stack([dist.base_dist.mean.log() for dist in dists])

    @staticmethod
    def get_entropy(dists):
        return torch.stack([dist.base_dist.entropy().sum() for dist in dists])


class InterpretableDeltaActor(Actor):
    def __init__(self, config):
        super().__init__(config)
        dim = self.observation_embedder.context_encoder.config.hidden_size
        self.weight_query = nn.Linear(dim, dim, bias=False)
        self.weight_risk = nn.Linear(dim, dim, bias=False)

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
                p.requires_grad = n.startswith('weight_query.')
        else:
            raise Exception

    def get_dist_parameter_votes(self, observation):
        diagnosis_embeddings, context_embeddings, diagnosis_strings, \
            context_strings, context_info = self.observation_embedder(
            observation, ignore_report=observation['evidence_is_retrieved'] \
                and not self.config.ignore_evidence,
            ignore_evidence=self.config.ignore_evidence)
        if not observation['evidence_is_retrieved']:
            diagnosis_embeddings = self.weight_query(diagnosis_embeddings)
        else:
            diagnosis_embeddings = self.weight_risk(diagnosis_embeddings)
        param_votes = util.dot_score(diagnosis_embeddings, context_embeddings)
        param_votes = param_votes.unsqueeze(2)
        return {
            'param_votes': param_votes,
            'context_strings': context_strings,
            'context_embeddings': context_embeddings,
            'diagnosis_strings': diagnosis_strings,
            'diagnosis_embeddings': diagnosis_embeddings,
            'context_info': context_info,
        }

    def transform_parameters(self, param_info):
        params = param_info['params'][:, 0]
        param_info['params'] = (params,)
        return param_info

    @staticmethod
    def parameters_to_dist(params):
        return Delta(params)

    def get_dist_stats(self, forward_pass_info):
        return_dict = {
            'avg_delta_param': torch.stack([
                info['params'][0].mean()
                for info in forward_pass_info]).mean(),
        }
        if self.has_bias:
            return_dict['avg_evidence_delta'] = torch.stack([
                (info['params'][0] - info['bias'][0]).mean()
                for info in forward_pass_info]).mean()
        return return_dict
        # return {
        #     'avg_delta_param': torch.stack(
        #         [dist.param.mean() for dist in dists]).mean()}
