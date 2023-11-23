from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import io
import torch
from torch.utils.checkpoint import checkpoint
import random
import nltk


def run_model(model, example_param, keys, *args):
    kwargs = {k: v for k, v in zip(keys, args)}
    return model(**kwargs).pooler_output


class ObservationEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.context_encoder = AutoModel.from_pretrained(
            self.config.model_name)
        if self.config.static_diagnoses is None:
            self.diagnosis_encoder = AutoModel.from_pretrained(
                self.config.model_name)
        else:
            self.diagnosis_embeddings = nn.Embedding(
                len(self.config.static_diagnoses),
                self.context_encoder.config.hidden_size)
            self.diagnosis_mapping = {
                diagnosis: idx for idx, diagnosis in enumerate(
                    self.config.static_diagnoses)}
        self.tokenizer.model_max_length = \
            self.context_encoder.config.max_position_embeddings
        self.device = 'cpu'
        dim = self.context_encoder.config.hidden_size
        self.diagnosis_bias_vector = nn.Parameter(torch.randn(dim)) \
            if self.config.diagnosis_bias and \
            self.config.static_diagnoses is None else None

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

    def get_diagnosis_embeddings(self, potential_diagnoses):
        if self.config.static_diagnoses is None:
            return self.batch_embed(self.diagnosis_encoder, potential_diagnoses)
        else:
            return self.diagnosis_embeddings(
                torch.tensor(
                    [self.diagnosis_mapping[x] for x in potential_diagnoses],
                    device=self.diagnosis_embeddings.weight.device))

    def get_raw_sentences(self, observation):
        past_reports = pd.read_csv(
            io.StringIO(observation['past_reports']), parse_dates=['date'])
        reports = pd.read_csv(
            io.StringIO(observation['reports']), parse_dates=['date'])
        all_reports = pd.concat([past_reports, reports]).reset_index()
        evidence_strings = []
        metadata = []
        # TODO: limit the number of reports or sentences considered?
        for i, row in all_reports[::-1].iterrows():
            day = (row.date - reports.iloc[0].date).days
            sentences = nltk.sent_tokenize(row.text)
            for sent in sentences[::-1]:
                evidence_strings.insert(
                    0, '{} (day {})'.format(sent, day))
                metadata.insert(0, {'report_idx': i})
                if self.config.limit_num_sentences is not None and \
                        len(evidence_strings) >= \
                        self.config.limit_num_sentences:
                    break
            if self.config.limit_num_sentences is not None and \
                    len(evidence_strings) >= self.config.limit_num_sentences:
                break
        return evidence_strings, metadata

    def get_evidence_strings(self, observation):
        if self.config.use_raw_sentences:
            return self.get_raw_sentences(observation)
        evidence = pd.read_csv(io.StringIO(observation['evidence']))
        # remove all occurances before the last occurance of a
        # particular evidence snippet
        for k in evidence.columns:
            if k == 'day' or k.endswith(' certainty'):
                continue
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
                if k == 'day' or k.endswith(' confidence') or v is None \
                        or v != v:
                    continue
                v = str(v)
                if self.config.ignore_no_evidence_found and \
                        v.lower().strip() in 'no evidence found':
                    continue
                if self.training and \
                        self.config.randomly_drop_evidence is not None and \
                        random.random() > self.config.randomly_drop_evidence:
                    continue
                k = eval(k)
                evidence_strings.append('{} ({}): {} (day {})'.format(
                    k[0], k[1], v, row.day))
                metadata.append({'report_idx': i})
                if str(k) + ' confidence' in row.keys() and \
                        row[str(k) + ' confidence'] == row[
                            str(k) + ' confidence']:
                    metadata[-1]['confidence'] = row[str(k) + ' confidence']
        return evidence_strings, metadata


class InterpretableObservationEmbedder(ObservationEmbedder):
    def forward(self, observation, ignore_report=False, ignore_evidence=False):
        potential_diagnoses = pd.read_csv(io.StringIO(
            observation['options'])).apply(
            lambda r: f'{r.option} ({r.type})', axis=1).to_list()
        diagnosis_embeddings = self.get_diagnosis_embeddings(
            potential_diagnoses)
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
            reports = pd.read_csv(io.StringIO(
                observation['reports']), parse_dates=['date'])
            last_report = reports.iloc[-1].text
            context_strings.append(last_report)
            context_info.append('report')
            context_embeddings.append(
                self.batch_embed(self.context_encoder, context_strings))
        if not ignore_evidence and observation['evidence'].strip() != '':
            evidence_strings, evidence_metadata = self.get_evidence_strings(
                observation)
            if len(evidence_strings) > 0:
                context_strings += evidence_strings
                if self.config.use_raw_sentences:
                    context_info += [
                        'sentence (report {})'.format(x['report_idx'] + 1)
                        for x in evidence_metadata]
                else:
                    context_info += [
                        'evidence (report {}, confidence {})'.format(
                            x['report_idx'] + 1,
                            'unk' if 'confidence' not in x.keys() else
                            x['confidence'])
                        for x in evidence_metadata]
                evidence_embeddings = self.batch_embed(
                    self.context_encoder, evidence_strings)
                context_embeddings.append(evidence_embeddings)
        if len(context_embeddings) == 0:
            # if there is a static bias, this serves as a dummy vector
            context_strings.append('no evidence found')
            context_info.append('no evidence')
            context_embeddings.append(self.batch_embed(
                self.context_encoder, context_strings[-1:]))
        context_embeddings = torch.cat(context_embeddings) \
            if len(context_embeddings) > 1 else context_embeddings[0]
        return diagnosis_embeddings, context_embeddings, potential_diagnoses, \
            context_strings, context_info


class BertObservationEmbedder(ObservationEmbedder):
    def forward(self, observation, ignore_report=False, ignore_evidence=False):
        potential_diagnoses = pd.read_csv(io.StringIO(
            observation['options'])).apply(
            lambda r: f'{r.option} ({r.type})', axis=1).to_list()
        diagnosis_embeddings = self.get_diagnosis_embeddings(
            potential_diagnoses)
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
            reports = pd.read_csv(io.StringIO(
                observation['reports']), parse_dates=['date'])
            last_report = reports.iloc[-1].text
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
        offset = 0 if self.diagnosis_bias_vector is None else 1
        context_strings = context_strings[:offset] + ['\n\n'.join(context_strings[offset:])]
        context_info = context_info[:offset] + [' | '.join(context_info[offset:])]
        context_embeddings.append(self.batch_embed(
            self.context_encoder, context_strings[-1:]))
        return diagnosis_embeddings, context_embeddings, potential_diagnoses, \
            context_strings, context_info
