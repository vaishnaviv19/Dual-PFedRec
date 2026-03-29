"""PFedRec client-side trainer (Algorithm 1)."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from client.model import PFedRecModel
from data.sampler import NegativeSampler
from utils.logger import get_logger

logger = get_logger(__name__)


class PFedRecTrainer:
    """Local trainer for dual-personalization updates."""

    def __init__(self, model: PFedRecModel, config: Dict, negative_sampler: NegativeSampler):
        self.model = model
        self.config = config
        self.negative_sampler = negative_sampler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.BCELoss()

        # keep model optimizer rates aligned with configuration
        for group in self.model.optimizer_score.param_groups:
            group["lr"] = float(config.get("learning_rate_score", 0.01))
        for group in self.model.optimizer_item.param_groups:
            group["lr"] = float(config.get("learning_rate_item", 0.001))

    def train_local(self, positive_items: np.ndarray, all_items: set) -> Tuple[torch.Tensor, Dict]:
        """Run local client update and return item embedding weights + summary metrics."""
        self.model.train()
        losses = []
        total_samples = 0

        negative_pool = list(all_items - set(positive_items.tolist()))
        if len(positive_items) == 0:
            return self.model.get_item_embedding_weights(), {"avg_loss": 0.0, "samples": 0}

        pos_items_tensor = torch.tensor(positive_items, dtype=torch.long, device=self.device)

        for _ in range(int(self.config["epochs_local"])):
            shuffled = pos_items_tensor[torch.randperm(len(pos_items_tensor))]

            for start in range(0, len(shuffled), int(self.config["batch_size"])):
                end = min(start + int(self.config["batch_size"]), len(shuffled))
                batch_pos = shuffled[start:end]

                batch_neg_np = self.negative_sampler.sample(
                    positive_items=batch_pos.detach().cpu().numpy(),
                    negative_pool=negative_pool,
                    ratio=int(self.config["negative_sampling_ratio"]),
                )
                if len(batch_neg_np) == 0:
                    continue

                batch_neg = torch.tensor(batch_neg_np, dtype=torch.long, device=self.device)

                # Step 1: update score function only
                self.model.set_requires_grad(score_fn=True, item_emb=False)
                pos_pred = self.model(batch_pos)
                neg_pred = self.model(batch_neg)
                loss_step1 = self.criterion(pos_pred, torch.ones_like(pos_pred)) + self.criterion(
                    neg_pred, torch.zeros_like(neg_pred)
                )
                self.model.optimizer_score.zero_grad()
                loss_step1.backward()
                self.model.optimizer_score.step()

                # Step 2: update item embedding only
                self.model.set_requires_grad(score_fn=False, item_emb=True)
                pos_pred = self.model(batch_pos)
                neg_pred = self.model(batch_neg)
                loss_step2 = self.criterion(pos_pred, torch.ones_like(pos_pred)) + self.criterion(
                    neg_pred, torch.zeros_like(neg_pred)
                )
                self.model.optimizer_item.zero_grad()
                loss_step2.backward()
                self.model.optimizer_item.step()

                losses.extend([loss_step1.item(), loss_step2.item()])
                total_samples += int(len(batch_pos) + len(batch_neg))

        avg_loss = float(np.mean(losses)) if losses else 0.0
        logger.info("Client training complete: avg_loss=%.4f, samples=%d", avg_loss, total_samples)

        return self.model.get_item_embedding_weights().cpu(), {
            "avg_loss": avg_loss,
            "samples": total_samples,
        }