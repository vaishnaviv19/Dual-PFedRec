# client/config.py
"""Client configuration with YAML support"""
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ClientConfig:
    # Client Settings
    client_id: str = "1"
    host: str = "0.0.0.0"
    port: int = 8001
    server_url: str = "http://server:8000"
    
    # Model Settings
    num_items: int = 1682
    embedding_dim: int = 32
    score_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    # Training Settings
    batch_size: int = 256
    negative_sampling_ratio: int = 4
    learning_rate_score: float = 0.01
    learning_rate_item: float = 0.001
    epochs_local: int = 1
    
    # Federated Settings
    total_rounds: int = 100
    eval_every: int = 1
    
    # Privacy
    enable_ldp: bool = True
    ldp_lambda: float = 0.4
    
    # System
    log_level: str = "INFO"
    log_dir: str = "logs/"
    timeout: int = 30
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> 'ClientConfig':
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            print(f"Config file {config_path} not found, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        kwargs: Dict[str, Any] = {}

        # System
        defaults = cls()
        system_cfg = config.get('system', {})
        client_cfg = system_cfg.get('client', {})
        kwargs['host'] = client_cfg.get('host', defaults.host)
        kwargs['port'] = client_cfg.get('port', defaults.port)

        communication_cfg = system_cfg.get('communication', {})
        kwargs['timeout'] = communication_cfg.get('timeout', defaults.timeout)

        # Model
        model_cfg = config.get('model', {})
        kwargs['embedding_dim'] = model_cfg.get('item_embedding_dim', defaults.embedding_dim)
        score_cfg = model_cfg.get('score_function', {})
        kwargs['score_hidden_dims'] = score_cfg.get('hidden_dims', defaults.score_hidden_dims)

        # Training
        train_cfg = config.get('training', {})
        kwargs['batch_size'] = train_cfg.get('batch_size', defaults.batch_size)
        kwargs['negative_sampling_ratio'] = train_cfg.get('negative_sampling_ratio', defaults.negative_sampling_ratio)
        kwargs['learning_rate_score'] = train_cfg.get('lr_score', defaults.learning_rate_score)
        kwargs['learning_rate_item'] = train_cfg.get('lr_item', defaults.learning_rate_item)
        kwargs['epochs_local'] = train_cfg.get('epochs_local', defaults.epochs_local)
        kwargs['total_rounds'] = train_cfg.get('total_rounds', defaults.total_rounds)

        # Evaluation
        eval_cfg = config.get('evaluation', {})
        kwargs['eval_every'] = eval_cfg.get('eval_every_round', defaults.eval_every)

        # Privacy
        privacy_cfg = config.get('privacy', {})
        kwargs['enable_ldp'] = privacy_cfg.get('enable_ldp', defaults.enable_ldp)
        kwargs['ldp_lambda'] = privacy_cfg.get('ldp_lambda', defaults.ldp_lambda)

        # Logging
        logging_cfg = config.get('logging', {})
        kwargs['log_level'] = logging_cfg.get('level', defaults.log_level)
        kwargs['log_dir'] = logging_cfg.get('log_dir', defaults.log_dir)
        
        # Environment overrides
        env_overrides = {
            'client_id': os.environ.get('CLIENT_ID'),
            'server_url': os.environ.get('SERVER_URL'),
            'port': os.environ.get('CLIENT_PORT'),
        }
        for key, value in env_overrides.items():
            if value is not None:
                kwargs[key] = int(value) if key == 'port' else value
        
        return cls(**{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for trainer"""
        return {
            'batch_size': self.batch_size,
            'negative_sampling_ratio': self.negative_sampling_ratio,
            'epochs_local': self.epochs_local,
            'num_items': self.num_items,
            'learning_rate_score': self.learning_rate_score,
            'learning_rate_item': self.learning_rate_item,
        }