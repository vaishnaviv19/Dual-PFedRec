# utils/logger.py
"""
Logging Utilities
Structured logging with TensorBoard integration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union
import json
import numpy as np

# Optional: TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def setup_logger(level: str = "INFO", 
                log_dir: str = "logs/",
                name: str = "pfedrec") -> logging.Logger:
    """
    Configure structured logging with file and console handlers
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File handler (JSON format for parsing)
    file_handler = logging.FileHandler(
        log_path / f"{name}_{timestamp}.log",
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance (creates default if not configured)"""
    if name is None:
        name = "pfedrec"
    
    logger = logging.getLogger(name)
    
    # If no handlers, set up basic config
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    return logger


class TensorBoardLogger:
    """
    TensorBoard wrapper for experiment tracking
    
    Usage:
        tb = TensorBoardLogger("logs/my_experiment")
        tb.log_scalar("loss", 0.45, step=10)
        tb.close()
    """
    
    def __init__(self, log_dir: str, flush_secs: int = 30):
        """
        Args:
            log_dir: TensorBoard log directory
            flush_secs: How often to flush data to disk
        """
        if not TENSORBOARD_AVAILABLE:
            self.writer = None
            get_logger(__name__).warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
            return
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            str(self.log_dir),
            flush_secs=flush_secs
        )
        get_logger(__name__).info(f"TensorBoard logging to {self.log_dir}")
    
    def log_scalar(self, 
                  tag: str, 
                  value: float, 
                  step: int,
                  walltime: Optional[float] = None):
        """Log a scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, value, step, walltime)
    
    def log_scalars(self,
                   main_tag: str,
                   tag_scalar_dict: Dict[str, float],
                   step: int,
                   walltime: Optional[float] = None):
        """Log multiple scalars with same step"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step, walltime)
    
    def log_histogram(self,
                     tag: str,
                     values: Union[list, np.ndarray],
                     step: int,
                     bins: str = 'tensorflow'):
        """Log histogram of values (e.g., embedding distributions)"""
        if self.writer:
            self.writer.add_histogram(tag, values, step, bins)
    
    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: int,
                   prefix: str = ""):
        """Log a dictionary of metrics"""
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(tag, value, step)
    
    def log_config(self, config: Dict, step: int = 0):
        """Log configuration as text"""
        if self.writer:
            config_text = json.dumps(config, indent=2, default=str)
            self.writer.add_text("config", config_text, step)
    
    def close(self):
        """Close writer and flush data"""
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class ExperimentLogger:
    """
    Combined logger for experiments (console + file + TensorBoard)
    """
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = "logs/",
                 level: str = "INFO",
                 use_tensorboard: bool = True):
        """
        Args:
            experiment_name: Name for this experiment run
            log_dir: Base directory for logs
            level: Logging level
            use_tensorboard: Enable TensorBoard logging
        """
        self.name = experiment_name
        self.step = 0
        
        # Setup structured logger
        self.logger = setup_logger(
            level=level,
            log_dir=log_dir,
            name=f"pfedrec.{experiment_name}"
        )
        
        # Setup TensorBoard if requested
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tb = TensorBoardLogger(f"{log_dir}/{experiment_name}")
        else:
            self.tb = None
        
        self.logger.info(f"🚀 Experiment '{experiment_name}' started")
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, extra=kwargs if kwargs else None)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, extra=kwargs if kwargs else None)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, extra=kwargs if kwargs else None)
    
    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, extra=kwargs if kwargs else None)
    
    def log_metric(self, name: str, value: float):
        """Log a metric and advance step"""
        self.logger.info(f"📊 {name}: {value:.4f}")
        if self.tb:
            self.tb.log_scalar(f"metrics/{name}", value, self.step)
    
    def log_round(self, round_num: int, metrics: Dict[str, float]):
        """Log metrics for a federated round"""
        self.step = round_num
        
        # Console log
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"🔄 Round {round_num}: {metrics_str}")
        
        # TensorBoard
        if self.tb:
            self.tb.log_metrics(metrics, step=round_num, prefix="round")
    
    def log_embedding_stats(self, 
                           embedding_name: str,
                           embedding: np.ndarray,
                           step: int):
        """Log embedding statistics for analysis"""
        stats = {
            "mean": float(np.mean(embedding)),
            "std": float(np.std(embedding)),
            "min": float(np.min(embedding)),
            "max": float(np.max(embedding)),
            "norm": float(np.linalg.norm(embedding))
        }
        
        self.logger.debug(f"📐 {embedding_name} stats: {stats}")
        
        if self.tb:
            self.tb.log_scalars(f"embeddings/{embedding_name}", stats, step)
            self.tb.log_histogram(f"embeddings/{embedding_name}/values", embedding, step)
    
    def save_checkpoint(self, 
                       checkpoint: Dict,
                       path: str,
                       round_num: int):
        """Save experiment checkpoint"""
        import torch
        
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        save_path = checkpoint_path / f"checkpoint_round_{round_num}.pt"
        torch.save({
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            **checkpoint
        }, save_path)
        
        self.logger.info(f"💾 Checkpoint saved: {save_path}")
    
    def close(self):
        """Cleanup resources"""
        self.logger.info(f"✅ Experiment '{self.name}' completed")
        if self.tb:
            self.tb.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()