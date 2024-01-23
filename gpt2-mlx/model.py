from dataclasses import dataclass
from typing import Optional

import mlx
import mlx.nn as nn

@dataclass
class ModelArgs:
    n_layers: int = 6
    n_heads: int = 8
    embed_size: int = 512
    feed_fwd_size: int = 2048
    vocab_size: int = 10000


