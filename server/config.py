# server/config.py
"""
Server Configuration Loader
Loads and validates configuration from config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model architecture configuration (Paper Section 6.2)"""
    item_embedding_dim: int = Field(32, ge=8, le=256)
    score_function: Dict = Field(
        default_factory=lambda: {
            "hidden_dims": [64, 32],
            "activation": "relu",
            "output_activation": "sigmoid"
        }
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    # Optimizers
    lr_score: float = Field(0.01, gt=0)  # η for θₛ
    lr_item: float = Field(0.001, gt=0)  # η' for θₘ
    batch_size: int = Field(256, ge=16, le=2048)
    epochs_local: int = Field(1, ge=1, le=10)
    
    # Negative sampling (Eq. 8)
    negative_sampling_ratio: int = Field(4, ge=1, le=10)
    
    # Federated learning
    total_rounds: int = Field(100, ge=10, le=1000)
    clients_per_round: int = Field(10, ge=1)
    min_clients_per_round: int = Field(2, ge=1)
    aggregation: str = Field("fedavg", pattern="^(fedavg|fedprox)$")


class DatasetConfig(BaseModel):
    """Dataset configuration"""
    name: str = "movielens-100k"
    path: str = "data/ml-100k/"
    num_items: int = 1682  # MovieLens-100K
    min_interactions: int = 20
    test_ratio: float = 0.2


class PrivacyConfig(BaseModel):
    """Privacy settings (Section 6.6)"""
    enable_ldp: bool = False
    ldp_lambda: float = Field(0.0, ge=0, le=1.0)


class SystemConfig(BaseModel):
    """System/network configuration"""
    class ServerConfig(BaseModel):
        host: str = "0.0.0.0"
        port: int = 8000
        workers: int = 4
    
    class ClientConfig(BaseModel):
        host: str = "0.0.0.0"
        port: int = 8001
    
    server: ServerConfig = Field(default_factory=ServerConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    
    class CommunicationConfig(BaseModel):
        timeout: int = 30
        retry_attempts: int = 3
    
    communication: CommunicationConfig = Field(
        default_factory=CommunicationConfig
    )


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_dir: str = "logs/"
    tensorboard: bool = True
    save_checkpoints: bool = True
    checkpoint_every: int = 10


class ServerConfig(BaseModel):
    """Complete server configuration"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Convenience properties
    @property
    def embedding_dim(self) -> int:
        return self.model.item_embedding_dim
    
    @property
    def num_items(self) -> int:
        return self.dataset.num_items
    
    @property
    def aggregation_method(self) -> str:
        return self.training.aggregation
    
    @property
    def total_rounds(self) -> int:
        return self.training.total_rounds
    
    @property
    def clients_per_round(self) -> int:
        return self.training.clients_per_round
    
    @property
    def min_clients_per_round(self) -> int:
        return self.training.min_clients_per_round
    
    @property
    def eval_every(self) -> int:
        return 1  # Evaluate every round
    
    @property
    def privacy_enabled(self) -> bool:
        return self.privacy.enable_ldp
    
    @property
    def ldp_lambda(self) -> float:
        return self.privacy.ldp_lambda
    
    @property
    def log_level(self) -> str:
        return self.logging.level
    
    @property
    def log_dir(self) -> str:
        return self.logging.log_dir
    
    @property
    def checkpoint_dir(self) -> str:
        return "checkpoints/"
    
    @property
    def save_checkpoints(self) -> bool:
        return self.logging.save_checkpoints
    
    @property
    def checkpoint_every(self) -> int:
        return self.logging.checkpoint_every
    
    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> 'ServerConfig':
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Export config as dictionary"""
        return self.model_dump()