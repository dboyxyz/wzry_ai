# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import copy
import torch
from torch import nn
from layers import DecoderLayer
from embeddings import Embedder
from sublayers import Norm, Dense
from config import TransformerConfig

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, N, heads, dropout, max_length=1024):
        super().__init__()
        self.N = N
        self.embedX = Embedder(vocab_size, d_model)
        self.embedP = Embedder(max_length, d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, features, actions, trg_mask):
        position = torch.arange(0, features.size(1), dtype=torch.long, device=features.device)
        x = features + self.embedP(position) + self.embedX(actions) * 0
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):

    def __init__(self, trg_vocab, d_model, N, heads, dropout, feature_dim=6 * 6 * 2048):
        super().__init__()
        self.feature_projection = Dense(feature_dim, d_model)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.outX = Dense(d_model, trg_vocab)
        self.value_head = Dense(d_model, 1)

    def forward(self, features, actions, trg_mask):
        features = self.feature_projection(features)
        d_output = self.decoder(features, actions, trg_mask)
        output = self.outX(d_output)
        value_head = self.value_head(d_output)
        return (output, value_head)

def get_model(config, vocab_size=130, model_weights='policy.pt'):
    from pathlib import Path
    model = Transformer(vocab_size, config.d_model, config.n_layers, config.heads, config.dropout)
    path = Path(config.load_weights) / model_weights
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model

class PolicyAgent:
    def __init__(self, weights, device):
        self.model = get_model(TransformerConfig(load_weights=str(weights.parent)), model_weights=weights.name).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5, betas=(0.9, 0.95), eps=1e-9)
