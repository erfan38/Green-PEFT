"""
efficient_training.py

Single integrated module that replaces:
1) smart sampler (loss/length proxy for importance sampling)
2) adaptive batch size controller

Also includes:
3) early stopping (energy-aware + validation-aware)

Design notes (important):
- If you need *dynamic batch size per step*, you MUST use a BatchSampler that yields List[int].
- DataLoader must be created with batch_sampler=... and batch_size=None.
- This module assumes you can compute PER-SAMPLE losses (loss reduction="none") in the training loop.

Typical usage:
    monitor = EnergyMonitor(...)  # must have get_remaining_energy() and energy_budget_wh
    controller = EnergyAwareTrainingController(
        dataset=dataset,
        energy_monitor=monitor,
        base_batch_size=32,
        min_batch_size=4,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=controller.batch_sampler,  # dynamic bs + smart sampling
        collate_fn=collate_fn,
        num_workers=...
    )

    for step, batch in enumerate(loader):
        outputs = model(...)
        per_sample_loss = ...  # shape [bs], torch.Tensor
        lengths = ...          # shape [bs], token counts
        controller.on_train_step_end(batch_indices=batch["_indices"], per_sample_losses=per_sample_loss, lengths=lengths)

        # optionally run validation sometimes and call:
        stop = controller.early_stopper.should_stop(
            step=step,
            val_metric=val_score,  # higher-is-better or lower-is-better (config)
            remaining_energy_wh=monitor.get_remaining_energy(),
        )
        if stop:
            break
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Dict, Any

import numpy as np
import torch
from torch.utils.data import BatchSampler


# -----------------------------
# 1) Loss/Length proxy tracker
# -----------------------------

class LossEfficiencyTracker:
    """
    Tracks a compute-aware "value-per-cost" score per sample using:
        score = (utility^alpha) / (cost^beta)

    Default proxy:
      - utility: loss (or loss per token, recommended)
      - cost: token length (or token length^2 if you want closer attention compute proxy)

    Practical safeguards:
      - eps floors to avoid division by zero
      - clipping to avoid a few samples dominating
      - EMA smoothing for stability
    """

    def __init__(
        self,
        dataset_size: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        decay: float = 0.9,
        eps: float = 1e-8,
        loss_clip: float = 50.0,
        score_floor: float = 1e-6,
        use_loss_per_token: bool = True,
        cost_power: float = 1.0,  # 1.0 -> length, 2.0 -> length^2 proxy
    ):
        self.dataset_size = int(dataset_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)
        self.loss_clip = float(loss_clip)
        self.score_floor = float(score_floor)
        self.use_loss_per_token = bool(use_loss_per_token)
        self.cost_power = float(cost_power)

        # start uniform
        self.scores = np.ones(self.dataset_size, dtype=np.float32)

    def update(
        self,
        indices: Sequence[int],
        losses: Sequence[float],
        lengths: Sequence[int],
    ) -> None:
        for idx, loss, length in zip(indices, losses, lengths):
            if idx < 0 or idx >= self.dataset_size:
                continue

            # sanitize
            safe_len = max(1, int(length))
            safe_loss = float(loss)

            # clip + floor
            if not np.isfinite(safe_loss):
                continue
            safe_loss = max(self.eps, min(self.loss_clip, safe_loss))

            # utility proxy
            if self.use_loss_per_token:
                utility = safe_loss / float(safe_len)
            else:
                utility = safe_loss

            utility = max(self.eps, utility)

            # cost proxy
            cost = float(safe_len) ** self.cost_power
            cost = max(1.0, cost)

            new_score = (utility ** self.alpha) / (cost ** self.beta)
            new_score = max(self.score_floor, float(new_score))

            # EMA update
            old = float(self.scores[idx])
            self.scores[idx] = np.float32(self.decay * old + (1.0 - self.decay) * new_score)

    def get_probabilities(self, explore_eps: float = 0.05) -> np.ndarray:
        """
        Returns a stable probability distribution with a uniform exploration mixture:
            p = (1-explore_eps)*soft_p + explore_eps*(1/N)
        """
        explore_eps = float(explore_eps)
        explore_eps = min(max(explore_eps, 0.0), 0.5)

        scores = self.scores.astype(np.float64, copy=False)
        total = float(scores.sum())

        if not np.isfinite(total) or total <= 0.0:
            p = np.ones(self.dataset_size, dtype=np.float64) / float(self.dataset_size)
        else:
            p = scores / total

        # exploration mixture
        if explore_eps > 0.0:
            p = (1.0 - explore_eps) * p + explore_eps * (1.0 / float(self.dataset_size))

        # final normalization
        p_sum = float(p.sum())
        if p_sum <= 0.0 or not np.isfinite(p_sum):
            return np.ones(self.dataset_size, dtype=np.float64) / float(self.dataset_size)

        return p / p_sum


# ---------------------------------------
# 2) Energy-aware batch size adaptation
# ---------------------------------------

class EnergyAwareBatchController:
    """
    Dynamic batch size based on remaining energy and training progress.

    Assumes:
      - energy_monitor.get_remaining_energy() -> Wh
      - energy_monitor.energy_budget_wh -> Wh
    """

    def __init__(
        self,
        base_batch_size: int,
        energy_monitor: Any,
        min_batch_size: int = 1,
    ):
        self.base_batch_size = int(base_batch_size)
        self.energy_monitor = energy_monitor
        self.min_batch_size = int(min_batch_size)

    def get_adaptive_batch_size(self, convergence_progress: float) -> int:
        progress = float(convergence_progress)
        progress = min(max(progress, 0.0), 1.0)

        remaining_energy = float(self.energy_monitor.get_remaining_energy())

        # hard guard
        if remaining_energy <= 0.0:
            return self.min_batch_size

        budget = float(getattr(self.energy_monitor, "energy_budget_wh", 0.0))
        budget = max(1e-9, budget)
        remaining_energy_ratio = remaining_energy / budget

        # Energy factor: use MORE compute when budget is ample, LESS when running low.
        # This is the correct energy-aware behaviour:
        #   high ratio (>0.7) → plentiful energy → larger batches → better gradient estimates
        #   low ratio  (<0.3) → scarce energy    → smaller batches → conserve remaining budget
        if remaining_energy_ratio > 0.7:
            energy_factor = 1.2
        elif remaining_energy_ratio > 0.3:
            energy_factor = 1.0
        else:
            energy_factor = 0.8

        # progress factor: smaller later
        if progress > 0.8:
            progress_factor = 0.7
        elif progress > 0.5:
            progress_factor = 0.9
        else:
            progress_factor = 1.0

        adaptive = int(round(self.base_batch_size * energy_factor * progress_factor))
        return max(self.min_batch_size, adaptive)


# -----------------------------
# 3) Early stopping controller
# -----------------------------

@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "max"  # "max" for accuracy/F1, "min" for loss
    energy_stop_threshold_wh: float = 0.0  # stop if remaining_energy <= threshold
    check_requires_val: bool = True        # if True, ignore metric logic when val_metric is None


class EnergyAwareEarlyStopper:
    """
    Stops training when:
      - energy is exhausted (remaining_energy <= threshold), OR
      - validation metric plateaus for `patience` checks.

    Call should_stop(...) after each validation evaluation (recommended), or periodically.
    """

    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best: Optional[float] = None
        self.bad_counts = 0

    def _is_improvement(self, metric: float) -> bool:
        assert self.best is not None
        if self.cfg.mode == "min":
            return metric < (self.best - self.cfg.min_delta)
        return metric > (self.best + self.cfg.min_delta)

    def should_stop(
        self,
        step: int,
        remaining_energy_wh: float,
        val_metric: Optional[float] = None,
    ) -> bool:
        # energy hard stop
        if float(remaining_energy_wh) <= float(self.cfg.energy_stop_threshold_wh):
            return True

        # metric logic
        if val_metric is None:
            # No metric provided - don't stop (energy check already passed above)
            return False

        metric = float(val_metric)
        if not np.isfinite(metric):
            # treat non-finite metric as "bad"
            self.bad_counts += 1
            return self.bad_counts >= self.cfg.patience

        if self.best is None:
            self.best = metric
            self.bad_counts = 0
            return False

        if self._is_improvement(metric):
            self.best = metric
            self.bad_counts = 0
            return False

        self.bad_counts += 1
        return self.bad_counts >= self.cfg.patience


# ------------------------------------------
# 4) BatchSampler (smart sampling + bs adapt)
# ------------------------------------------

class EnergyAwareBatchSampler(BatchSampler):
    """
    BatchSampler that:
      - chooses batch size dynamically (EnergyAwareBatchController)
      - chooses samples using LossEfficiencyTracker probabilities
      - samples without replacement per epoch

    IMPORTANT: Use with DataLoader(batch_sampler=..., batch_size=None)
    """

    def __init__(
        self,
        dataset_size: int,
        batch_controller: EnergyAwareBatchController,
        tracker: LossEfficiencyTracker,
        max_steps: int,
        explore_eps: float = 0.05,
        seed: int = 1234,
        drop_last: bool = False,
    ):
        self.dataset_size = int(dataset_size)
        self.batch_controller = batch_controller
        self.tracker = tracker
        self.max_steps = max(1, int(max_steps))
        self.explore_eps = float(explore_eps)
        self.drop_last = bool(drop_last)

        self._rng = np.random.default_rng(int(seed))
        self._global_step = 0

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def __iter__(self) -> Iterator[List[int]]:
        # epoch-local pool
        remaining = np.arange(self.dataset_size, dtype=np.int64)

        # Refresh probabilities every refresh_every steps so the sampler starts
        # exploiting loss scores as they accumulate during the epoch, rather than
        # being locked to the uniform snapshot taken before any losses are known.
        refresh_every = max(10, self.dataset_size // 10)
        steps_this_epoch = 0
        p_full = self.tracker.get_probabilities(explore_eps=self.explore_eps).astype(np.float64)

        while remaining.size > 0:
            # Refresh the probability snapshot periodically mid-epoch
            if steps_this_epoch > 0 and steps_this_epoch % refresh_every == 0:
                p_full = self.tracker.get_probabilities(
                    explore_eps=self.explore_eps
                ).astype(np.float64)

            progress = min(max(self._global_step / float(self.max_steps), 0.0), 1.0)
            bs = int(self.batch_controller.get_adaptive_batch_size(progress))

            if remaining.size < bs and self.drop_last:
                break

            # renormalize over remaining indices
            p = p_full[remaining]
            p_sum = float(p.sum())
            if not np.isfinite(p_sum) or p_sum <= 0.0:
                p = None  # uniform fallback
            else:
                p = p / p_sum

            k = min(bs, int(remaining.size))
            chosen_local = self._rng.choice(remaining.size, size=k, replace=False, p=p)
            batch = remaining[chosen_local].tolist()

            # remove chosen from remaining efficiently
            mask = np.ones(remaining.size, dtype=bool)
            mask[chosen_local] = False
            remaining = remaining[mask]

            yield batch
            self._global_step += 1
            steps_this_epoch += 1

    def __len__(self) -> int:
        # approximate number of batches per epoch using base batch size
        # (dynamic bs means exact length depends on energy/progress)
        base = max(1, int(self.batch_controller.base_batch_size))
        return int(np.ceil(self.dataset_size / float(base)))


# -------------------------------------------------
# 5) One controller object to hold everything
# -------------------------------------------------

class EnergyAwareTrainingController:
    """
    Single integrated controller for:
      - smart sampling (loss/length proxy)
      - energy-aware batch size adaptation
      - early stopping

    You create it once and use:
      - controller.batch_sampler in the DataLoader
      - controller.on_train_step_end(...) after each training step
      - controller.early_stopper.should_stop(...) after validation checks
    """

    def __init__(
        self,
        dataset: Any,
        energy_monitor: Any,
        base_batch_size: int,
        min_batch_size: int,
        max_steps: int,
        # tracker params
        alpha: float = 1.0,
        beta: float = 0.5,
        decay: float = 0.9,
        explore_eps: float = 0.05,
        use_loss_per_token: bool = True,
        cost_power: float = 1.0,
        # early stopping params
        early_stop_cfg: Optional[EarlyStoppingConfig] = None,
        seed: int = 1234,
    ):
        self.dataset = dataset
        self.energy_monitor = energy_monitor
        self.dataset_size = len(dataset)
        self.max_steps = max(1, int(max_steps))

        # components
        self.tracker = LossEfficiencyTracker(
            dataset_size=self.dataset_size,
            alpha=alpha,
            beta=beta,
            decay=decay,
            use_loss_per_token=use_loss_per_token,
            cost_power=cost_power,
        )

        self.batch_controller = EnergyAwareBatchController(
            base_batch_size=base_batch_size,
            energy_monitor=energy_monitor,
            min_batch_size=min_batch_size,
        )

        if early_stop_cfg is None:
            early_stop_cfg = EarlyStoppingConfig(
                patience=3,          # Reduced from 5 to 3 for faster energy saving
                min_delta=0.005,     # Require at least some improvement
                mode="min",          # Training loss decreases
                energy_stop_threshold_wh=0.0,
                check_requires_val=False, # We will fallback to smoothed training loss
            )
        self.early_stopper = EnergyAwareEarlyStopper(early_stop_cfg)

        self.batch_sampler = EnergyAwareBatchSampler(
            dataset_size=self.dataset_size,
            batch_controller=self.batch_controller,
            tracker=self.tracker,
            max_steps=self.max_steps,
            explore_eps=explore_eps,
            seed=seed,
            drop_last=False,
        )

        # optional logging buffers
        self.history: Dict[str, List[float]] = {
            "batch_size": [],
            "remaining_energy_wh": [],
        }

    @staticmethod
    def _to_float_list(x: torch.Tensor | Sequence[float]) -> List[float]:
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu()
            return x.tolist()
        return [float(v) for v in x]

    @staticmethod
    def _to_int_list(x: torch.Tensor | Sequence[int]) -> List[int]:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            return [int(v) for v in x.tolist()]
        return [int(v) for v in x]

    def set_global_step(self, step: int) -> None:
        self.batch_sampler.set_global_step(step)

    def on_train_step_end(
        self,
        batch_indices: Sequence[int],
        per_sample_losses: torch.Tensor | Sequence[float],
        lengths: torch.Tensor | Sequence[int],
        global_step: Optional[int] = None,
    ) -> None:
        """
        Call after each optimizer step (or after forward/backward) with:
          - batch_indices: dataset indices for each example in the batch
          - per_sample_losses: loss per example (shape [bs])
          - lengths: token count per example (shape [bs])

        If you use a collate_fn, include indices in the batch to pass here.
        """
        if global_step is not None:
            self.set_global_step(int(global_step))

        idx = [int(i) for i in batch_indices]
        losses = self._to_float_list(per_sample_losses)
        lens = self._to_int_list(lengths)

        self.tracker.update(idx, losses, lens)

        # lightweight logging
        remaining = float(self.energy_monitor.get_remaining_energy())
        progress = min(max(self.batch_sampler._global_step / float(self.max_steps), 0.0), 1.0)
        bs = int(self.batch_controller.get_adaptive_batch_size(progress))

        self.history["remaining_energy_wh"].append(remaining)
        self.history["batch_size"].append(float(bs))


# -----------------------------
# Optional: helper collate idea
# -----------------------------
def attach_indices_collate(batch: List[Any]) -> Dict[str, Any]:
    """
    Example collate_fn pattern if your dataset returns dict-like items.
    Assumes each item is a dict with tensor fields, and adds '_indices'.

    If your dataset already returns (item, idx) you can adapt accordingly.
    """
    # This is a minimal example; adapt to your dataset schema.
    collated: Dict[str, Any] = {}
    indices = []
    for i, item in enumerate(batch):
        if isinstance(item, dict) and "_index" in item:
            indices.append(int(item["_index"]))
        else:
            # fallback: DataLoader doesn't give indices; you must provide them in dataset __getitem__
            indices.append(-1)

    collated["_indices"] = indices
    # If your items are dicts, stack tensors by key
    if isinstance(batch[0], dict):
        for k in batch[0].keys():
            if k in ("_index",):
                continue
            v0 = batch[0][k]
            if torch.is_tensor(v0):
                collated[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                collated[k] = [b[k] for b in batch]
    else:
        collated["data"] = batch
    return collated