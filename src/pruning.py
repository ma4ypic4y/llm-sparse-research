import torch.nn as nn
import wandb
from typing import Optional
from .metrics import PruningMetricsCollector


class WeightPruner:
    def __init__(self, model: nn.Module, target_sparsity: float,
                 warmup_steps: int, final_prune_step: int, prune_freq: int,
                 metrics_collector: Optional[PruningMetricsCollector] = None):
        from torch.nn.utils import prune

        self.to_prune = [(m, 'weight') for m in model.modules()
                        if isinstance(m, (nn.Linear, nn.Conv2d))]
        self.prune = prune
        self.target = target_sparsity
        self.warmup = warmup_steps
        self.final = final_prune_step
        self.freq = prune_freq
        self.applied = 0.0

        self.metrics_collector = metrics_collector or PruningMetricsCollector()
        self.pruning_history = []
        self.step_sparsities = {}

    def __call__(self, step: int, model: nn.Module):
        if step % self.freq == 0 or step == self.warmup:
            pre_prune_stats = self.metrics_collector.collect_all_metrics(
                model, step, log_to_wandb=False
            )

        prune_applied = self._apply_pruning(step)

        post_prune_stats = self.metrics_collector.collect_all_metrics(
            model, step, log_to_wandb=True
        )

        self._log_pruning_stats(step, prune_applied, post_prune_stats)
        return post_prune_stats

    def _apply_pruning(self, step: int) -> bool:
        """Apply pruning and return True if pruning was applied"""
        if step < self.warmup or step % self.freq != 0:
            return False

        progress = min(1.0, (step - self.warmup) / (self.final - self.warmup))
        desired = progress * self.target
        inc = desired - self.applied

        if inc <= 0:
            return False

        self.prune.global_unstructured(
            self.to_prune,
            pruning_method=self.prune.L1Unstructured,
            amount=inc
        )

        self.applied = desired

        self.pruning_history.append({
            'step': step,
            'sparsity_increase': inc,
            'total_sparsity': desired,
            'progress': progress
        })

        return True

    def _log_pruning_stats(self, step: int, prune_applied: bool, stats: dict):
        """Log pruning statistics"""
        total = 0
        zero = 0
        for m, _ in self.to_prune:
            tensor = m.weight.data
            total += tensor.numel()
            zero += int((tensor == 0).sum())
        sparsity = zero / total

        self.step_sparsities[step] = sparsity

        pruning_stats = {
            'pruning/sparsity': sparsity,
            'pruning/target_sparsity': self.target,
            'pruning/applied_sparsity': self.applied,
            'pruning/prune_applied': int(prune_applied)
        }

        if len(self.step_sparsities) >= 2:
            recent_steps = sorted(self.step_sparsities.keys())[-5:]
            if len(recent_steps) >= 2:
                sparsity_values = [self.step_sparsities[s] for s in recent_steps]
                sparsity_growth_rate = (sparsity_values[-1] - sparsity_values[0]) / len(recent_steps)
                pruning_stats['pruning/sparsity_growth_rate'] = sparsity_growth_rate

        wandb.log(pruning_stats, step=step)

    def get_pruning_summary(self) -> dict:
        """Return pruning process summary"""
        if not self.pruning_history:
            return {}

        total_increments = sum(h['sparsity_increase'] for h in self.pruning_history)
        num_prune_steps = len(self.pruning_history)

        return {
            'total_pruning_increments': total_increments,
            'num_pruning_steps': num_prune_steps,
            'avg_increment_per_step': total_increments / num_prune_steps if num_prune_steps > 0 else 0,
            'final_applied_sparsity': self.applied,
            'target_sparsity': self.target,
            'pruning_efficiency': self.applied / self.target if self.target > 0 else 0
        }