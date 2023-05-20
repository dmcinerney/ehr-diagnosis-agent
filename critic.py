from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import io


class CriticPreprocessor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, observation):
        reports = pd.read_csv(io.StringIO(observation['reports']))
        potential_diagnoses = pd.read_csv(io.StringIO(observation['potential_diagnoses']))
        evidence = pd.read_csv(io.StringIO(observation['evidence']))
        diagnoses_inputs = self.tokenizer(
            potential_diagnoses.diagnoses.to_list(), truncation=True, padding=True, return_tensors='pt')
        evidence_inputs = self.tokenizer(evidence, truncation=True, padding=True, return_tensors='pt')
        report_inputs = self.tokenizer(reports)


class Critic(nn.Module):
    def __init__(self, config):
        self.config = config

    def forward(self, diagnoses_ids, diagnoses_mask, evidence_ids, evidence_mask, report_ids, report_mask):
        return
