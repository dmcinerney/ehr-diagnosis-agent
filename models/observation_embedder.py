from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import io
import torch
import math


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

    def batch_embed(self, model, inputs):
        batch_size = self.config.embedding_batch_size if self.config.embedding_batch_size is not None else len(inputs)
        all_embeddings = []
        for offset in range(0, len(inputs), batch_size):
            embeddings = model(**self.tokenize(inputs[offset:offset + batch_size])).pooler_output
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, 0)

    def forward(self, observation, ignore_report=False):
        reports = pd.read_csv(io.StringIO(observation['reports']), parse_dates=['date'])
        potential_diagnoses = pd.read_csv(io.StringIO(observation['potential_diagnoses'])).diagnoses.to_list()
        last_report = reports.iloc[-1].text
        diagnosis_embeddings = self.batch_embed(self.diagnosis_encoder, potential_diagnoses)
        if ignore_report:
            assert observation['evidence'].strip() != ''
            context_strings = []
            context_embeddings = None
        else:
            context_strings = [last_report]
            context_embeddings = self.batch_embed(self.context_encoder, context_strings)
        if observation['evidence'].strip() != '':
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
            for i, row in evidence.iterrows():
                for k, v in row.items():
                    if k != 'day' and v is not None and v == v:
                        context_strings.append('{}: {} (day {})'.format(k, v, row.day))
            context_strings2 = context_strings[1:] if not ignore_report else context_strings
            context_embeddings2 = self.batch_embed(self.context_encoder, context_strings2)
            context_embeddings = torch.cat([context_embeddings, context_embeddings2], 0) \
                if not ignore_report else context_embeddings2
        return diagnosis_embeddings, context_embeddings, context_strings
