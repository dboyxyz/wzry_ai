from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / 'weights'

@dataclass
class TransformerConfig:
    d_model: int = 768
    n_layers: int = 12
    heads: int = 12
    dropout: float = 0.0
    load_weights: str = str(WEIGHTS_DIR)
