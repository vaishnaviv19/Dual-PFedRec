import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ServerConfig:
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Model Settings
    num_items: int = 1682  
    embedding_dim: int = 32
    
    # Federated Learning Settings
    total_rounds: int = 100
    clients_per_round: int = 10
    min_clients_per_round: int = 2
    aggregation_method: str = "fedavg"
    
    # Evaluation
    eval_every: int = 1
    metrics: list = field(default_factory=lambda: ["hr@10", "ndcg@10"])
    
    # Privacy
    privacy_enabled: bool = True
    ldp_lambda: float = 0.4
    
    # System
    log_level: str = "INFO"
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"
    save_checkpoints: bool = True
    checkpoint_every: int = 10
    debug: bool = False
    timeout: int = 30
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> 'ServerConfig':
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            print(f"Config file {config_path} not found, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        kwargs: Dict[str, Any] = {}

        # System
        system_cfg = config.get('system', {})
        server_cfg = system_cfg.get('server', {})
        defaults = cls()

        kwargs['host'] = server_cfg.get('host', defaults.host)
        kwargs['port'] = server_cfg.get('port', defaults.port)
        kwargs['workers'] = server_cfg.get('workers', defaults.workers)

        communication_cfg = system_cfg.get('communication', {})
        kwargs['timeout'] = communication_cfg.get('timeout', defaults.timeout)

        # Model
        model_cfg = config.get('model', {})
        kwargs['embedding_dim'] = model_cfg.get('item_embedding_dim', defaults.embedding_dim)

        # Training
        train_cfg = config.get('training', {})
        kwargs['total_rounds'] = train_cfg.get('total_rounds', defaults.total_rounds)
        kwargs['clients_per_round'] = train_cfg.get('clients_per_round', defaults.clients_per_round)
        kwargs['aggregation_method'] = train_cfg.get('aggregation', defaults.aggregation_method)

        # Evaluation
        eval_cfg = config.get('evaluation', {})
        kwargs['eval_every'] = eval_cfg.get('eval_every_round', defaults.eval_every)
        kwargs['metrics'] = eval_cfg.get('metrics', defaults.metrics)

        # Privacy
        privacy_cfg = config.get('privacy', {})
        kwargs['privacy_enabled'] = privacy_cfg.get('enable_ldp', defaults.privacy_enabled)
        kwargs['ldp_lambda'] = privacy_cfg.get('ldp_lambda', defaults.ldp_lambda)

        # Logging
        logging_cfg = config.get('logging', {})
        kwargs['log_level'] = logging_cfg.get('level', defaults.log_level)
        kwargs['log_dir'] = logging_cfg.get('log_dir', defaults.log_dir)
        kwargs['save_checkpoints'] = logging_cfg.get('save_checkpoints', defaults.save_checkpoints)
        kwargs['checkpoint_every'] = logging_cfg.get('checkpoint_every', defaults.checkpoint_every)
        
        # Override with environment variables
        env_overrides = {
            'host': os.environ.get('SERVER_HOST'),
            'port': os.environ.get('SERVER_PORT'),
            'num_items': os.environ.get('NUM_ITEMS'),
            'embedding_dim': os.environ.get('EMBEDDING_SIZE'),
        }
        for key, value in env_overrides.items():
            if value is not None:
                kwargs[key] = int(value) if key in ['port', 'num_items', 'embedding_dim'] else value
        
        return cls(**{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__})
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}