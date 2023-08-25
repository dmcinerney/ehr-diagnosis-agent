from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import io
import torch
from torch.utils.checkpoint import checkpoint


def run_model(model, example_param, keys, *args):
    kwargs = {k: v for k, v in zip(keys, args)}
    return model(**kwargs).pooler_output


class ObservationEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.diagnosis_encoder = AutoModel.from_pretrained(self.config.model_name)
        self.context_encoder = AutoModel.from_pretrained(self.config.model_name)
        self.tokenizer.model_max_length = self.diagnosis_encoder.config.max_position_embeddings
        self.device = 'cpu'

    def set_device(self, device):
        self.device = device
        self.to(device)

    def tokenize(self, inputs):
        return self.tokenizer(inputs, truncation=True, padding=True, return_tensors='pt').to(self.device)

    def embed(self, model, inputs):
        input_tensors = self.tokenize(inputs)
        keys = list(input_tensors.keys())
        args = [input_tensors[k]for k in keys]
        example_param = next(iter(model.parameters()))
        if not self.config.grad_checkpointing:
            return run_model(
                model, example_param, keys, *args)
        else:
            return checkpoint(
                run_model, model, example_param, keys, *args)

    def batch_embed(self, model, inputs):
        batch_size = self.config.embedding_batch_size if self.config.embedding_batch_size is not None else len(inputs)
        all_embeddings = []
        for offset in range(0, len(inputs), batch_size):
            embeddings = self.embed(model, inputs[offset:offset + batch_size])
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, 0)
    
    def get_evidence_strings(self, observation):
        evidence = pd.read_csv(io.StringIO(observation['evidence']))
        for k in evidence.columns:
            set_of_evidence = set()
            new_evidence_list = []
            for e in evidence[k][::-1]:
                if e not in set_of_evidence:
                    new_evidence_list.append(e)
                    set_of_evidence.add(e)
                else:
                    new_evidence_list.append(None)
            evidence[k] = new_evidence_list[::-1]
        evidence_strings = []
        metadata = []
        for i, row in evidence.iterrows():
            for k, v in row.items():
                if self.config.ignore_no_evidence_found and v == 'no evidence found':
                    continue
                if k != 'day' and v is not None and v == v:
                    k = eval(k)
                    evidence_strings.append('{} ({}): {} (day {})'.format(k[0], k[1], v, row.day))
                    metadata.append({'report_idx': i})
        return evidence_strings, metadata


class InterpretableObservationEmbedder(ObservationEmbedder):
    def __init__(self, config):
        super().__init__(config)
        dim = self.diagnosis_encoder.config.hidden_size
        self.diagnosis_bias_vector = nn.Parameter(torch.randn(dim)) \
            if self.config.diagnosis_bias else None

    def forward(self, observation, ignore_report=False, ignore_evidence=False):
        reports = pd.read_csv(io.StringIO(
            observation['reports']), parse_dates=['date'])
        potential_diagnoses = pd.read_csv(io.StringIO(
            observation['options'])).apply(
            lambda r: f'{r.option} ({r.type})', axis=1).to_list()
        last_report = reports.iloc[-1].text
        diagnosis_embeddings = self.batch_embed(self.diagnosis_encoder, potential_diagnoses)
        context_strings = []
        context_info = []
        context_embeddings = []
        if self.diagnosis_bias_vector is not None:
            context_strings.append('[bias vector]')
            context_info.append('bias')
            context_embeddings.append(self.diagnosis_bias_vector.unsqueeze(0))
        if ignore_report:
            assert observation['evidence'].strip() != '' or \
                self.diagnosis_bias_vector is not None
        else:
            context_strings.append(last_report)
            context_info.append('report')
            context_embeddings.append(
                self.batch_embed(self.context_encoder, context_strings))
        if not ignore_evidence and observation['evidence'].strip() != '':
            evidence_strings, evidence_metadata = self.get_evidence_strings(
                observation)
            if len(evidence_strings) > 0:
                context_strings += evidence_strings
                context_info += [
                    'evidence (report {})'.format(x['report_idx'] + 1)
                    for x in evidence_metadata]
                evidence_embeddings = self.batch_embed(
                    self.context_encoder, evidence_strings)
                context_embeddings.append(evidence_embeddings)
        context_embeddings = torch.cat(context_embeddings) \
            if len(context_embeddings) > 1 else context_embeddings[0]
        return diagnosis_embeddings, context_embeddings, context_strings, \
            context_info


class BertObservationEmbedder(ObservationEmbedder):
    def forward(self, observation, ignore_report=False, ignore_evidence=False):
        reports = pd.read_csv(io.StringIO(
            observation['reports']), parse_dates=['date'])
        potential_diagnoses = pd.read_csv(io.StringIO(
            observation['options'])).apply(
            lambda r: f'{r.option} ({r.type})', axis=1).to_list()
        last_report = reports.iloc[-1].text
        diagnosis_embeddings = self.batch_embed(
            self.diagnosis_encoder, potential_diagnoses)
        context_strings = []
        context_info = []
        if ignore_report:
            assert observation['evidence'].strip() != '' or \
                self.diagnosis_bias_vector is not None
        else:
            context_strings.append(last_report)
            context_info.append('report')
        if not ignore_evidence and observation['evidence'].strip() != '':
            evidence_strings, evidence_metadata = self.get_evidence_strings(
                observation)
            if len(evidence_strings) > 0:
                context_strings += evidence_strings
                context_info += [
                    'evidence (report {})'.format(x['report_idx'])
                    for x in evidence_metadata]
        context_embeddings = self.batch_embed(
            self.context_encoder, ['\n\n'.join(context_strings)])
        return diagnosis_embeddings, context_embeddings, context_strings, \
            context_info
