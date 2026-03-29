# client/config.py
"""
Client Configuration Loader
Shares config.yaml with server but has client-specific settings
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClientConfig(BaseModel):
    """Client-specific configuration"""
    
    # Identity
    client_id: str = Field(default_factory=lambda: os.environ.get("CLIENT_ID", "1"))
    server_url: str = Field(default_factory=lambda: os.environ.get("SERVER_URL", "http://server:8000"))
    
    # Model (must match server)
    num_items: int = 1682  # MovieLens-100K
    embedding_dim: int = 32
    score_hidden_dims: List[int] = [64, 32]
    
    # Training hyperparameters (Paper Section 6.2)
    lr_score: float = 0.01      # η for score function θₛ
    lr_item: float = 0.001      # η' for item embedding θₘ
    batch_size: int = 256
    epochs_local: int = 1       # E in Algorithm 1
    
    # Negative sampling (Eq. 8)
    negative_sampling_ratio: int = 4
    
    # Federated learning
    total_rounds: int = 100
    eval_every: int = 1
    
    # Data
    data_file: str = Field(default_factory=lambda: os.environ.get("DATA_FILE", ""))
    test_ratio: float = 0.2     # Leave-one-out
    
    # Privacy (Section 6.6)
    enable_ldp: bool = False
    ldp_lambda: float = 0.0
    
    # System
    host: str = "0.0.0.0"
    port: int = 8001
    timeout: int = 30
    log_level: str = "INFO"
    debug: bool = False

    # Add these properties to ClientConfig class:

    @property
    def num_items(self) -> int:
        return self._num_items if hasattr(self, '_num_items') else 1682

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim if hasattr(self, '_embedding_dim') else 32

    @property
    def score_hidden_dims(self) -> List[int]:
        return self._score_hidden_dims if hasattr(self, '_score_hidden_dims') else [64, 32]

    @property
    def negative_sampling_ratio(self) -> int:
        return self._negative_sampling_ratio if hasattr(self, '_negative_sampling_ratio') else 4

    @property
    def total_rounds(self) -> int:
        return self._total_rounds if hasattr(self, '_total_rounds') else 100

    @property
    def eval_every(self) -> int:
        return self._eval_every if hasattr(self, '_eval_every') else 1

    @property
    def enable_ldp(self) -> bool:
        return self._enable_ldp if hasattr(self, '_enable_ldp') else False

    @property
    def ldp_lambda(self) -> float:
        return self._ldp_lambda if hasattr(self, '_ldp_lambda') else 0.0

    @property
    def host(self) -> str:
        return self._host if hasattr(self, '_host') else "0.0.0.0"

    @property
    def port(self) -> int:
        return self._port if hasattr(self, '_port') else 8001

    @property
    def timeout(self) -> int:
        return self._timeout if hasattr(self, '_timeout') else 30

    @property
    def log_level(self) -> str:
        return self._log_level if hasattr(self, '_log_level') else "INFO"

    @property
    def debug(self) -> bool:
        return self._debug if hasattr(self, '_debug') else False
    
    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> 'ClientConfig':
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            # Return defaults if file not found
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract relevant sections for client
        client_config = {}
        
        if 'model' in config_dict:
            client_config['embedding_dim'] = config_dict['model'].get('item_embedding_dim', 32)
            client_config['score_hidden_dims'] = config_dict['model'].get('score_function', {}).get('hidden_dims', [64, 32])
        
        if 'training' in config_dict:
            t = config_dict['training']
            client_config.update({
                'lr_score': t.get('lr_score', 0.01),
                'lr_item': t.get('lr_item', 0.001),
                'batch_size': t.get('batch_size', 256),
                'epochs_local': t.get('epochs_local', 1),
                'negative_sampling_ratio': t.get('negative_sampling_ratio', 4),
                'total_rounds': t.get('total_rounds', 100),
            })
        
        if 'dataset' in config_dict:
            d = config_dict['dataset']
            client_config['num_items'] = d.get('num_items', 1682)
            client_config['test_ratio'] = d.get('test_ratio', 0.2)
        
        if 'privacy' in config_dict:
            p = config_dict['privacy']
            client_config.update({
                'enable_ldp': p.get('enable_ldp', False),
                'ldp_lambda': p.get('ldp_lambda', 0.0),
            })
        
        if 'logging' in config_dict:
            client_config['log_level'] = config_dict['logging'].get('level', 'INFO')
        
        # Override with environment variables
        for field in ['client_id', 'server_url', 'data_file']:
            env_val = os.environ.get(field.upper())
            if env_val:
                client_config[field] = env_val
        
        return cls(**{**cls().model_dump(), **client_config})
    
    def to_dict(self) -> Dict:
        """Export config as dictionary for trainer"""
        return {
            "lr_score": self.lr_score,
            "lr_item": self.lr_item,
            "batch_size": self.batch_size,
            "epochs_local": self.epochs_local,
            "negative_sampling_ratio": self.negative_sampling_ratio,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
        }
