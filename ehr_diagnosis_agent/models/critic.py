from torch import nn
from .observation_embedder import ObservationEmbedder


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observation_embedder = ObservationEmbedder(config)
        dim = self.observation_embedder.diagnosis_encoder.config.hidden_size
        self.linear = nn.Linear(dim, 1)

    def set_device(self, device):
        self.to(device)
        self.observation_embedder.set_device(device)

    def forward(self, observation):
        diagnosis_embeddings, context_embeddings, context_strings = self.observation_embedder(observation)
        mean_embedding = diagnosis_embeddings.mean(0) + context_embeddings.mean(0)
        return self.linear(mean_embedding.unsqueeze(0))[0]
