import torch
import torch.nn as nn
from torch.nn.utils import prune
from transformers import pytorch_utils
from typing import Optional


class WeightPruner:
    def __init__(self, model: nn.Module, target_sparsity: float,
                 warmup_steps: int, final_prune_step: int, prune_freq: int):
        self.to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, pytorch_utils.Conv1D))]
        self.target = target_sparsity
        self.warmup = warmup_steps
        self.final = final_prune_step
        self.freq = prune_freq
        self.applied = 0.0

        self.pruning_history = []
        self.step_sparsities = {}

    def __call__(self, step: int):
        if step < self.warmup or step % self.freq != 0:
            return False

        progress = min(1.0, (step - self.warmup) / (self.final - self.warmup))
        desired = progress * self.target
        inc = desired - self.applied

        if inc <= 0:
            return False

        prune.global_unstructured(
            self.to_prune,
            pruning_method=prune.L1Unstructured,
            amount=inc
        )

        self.applied = desired

        self.pruning_history.append({
            'step': step,
            'sparsity_increase': inc,
            'total_sparsity': desired,
            'progress': progress
        })

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


class ActivationsPruner:
    def __init__(self, model: nn.Module, target_sparsity: float,
                 warmup_steps: int, final_prune_step: int, prune_freq: int):
        self.to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, pytorch_utils.Conv1D))]
        self.target = target_sparsity
        self.warmup = warmup_steps
        self.final = final_prune_step
        self.freq = prune_freq
        self.applied = 0.0

        self.pruning_history = []
        self.step_sparsities = {}
        
        self.hidden_layers_hooks = []

    def __call__(self, step: int):
        if step < self.warmup or step % self.freq != 0:
            return False

        progress = min(1.0, (step - self.warmup) / (self.final - self.warmup))
        desired = progress * self.target
        inc = desired - self.applied

        if inc <= 0:
            return False

        
        def process_activation(module, input_, output):
            abs_values = torch.abs(output)
            k = int(self.applied * len(abs_values))
            if k:
                threshold = torch.max((torch.topk(abs_values, k=k, largest=False, dim=1)).values).item()
                output.data[torch.abs(output) < threshold] = 0
        
        for hook in self.hidden_layers_hooks:
            hook.remove()
        for (param, _) in self.to_prune:
            self.hidden_layers_hooks.append(param.register_forward_hook(process_activation))

        self.applied = desired

        self.pruning_history.append({
            'step': step,
            'sparsity_increase': inc,
            'total_sparsity': desired,
            'progress': progress
        })

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
