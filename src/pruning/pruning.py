import torch
import torch.nn as nn
from torch.nn.utils import prune

from transformers import pytorch_utils
from transformers.trainer_callback import TrainerCallback

from .sparsify_activations_layer import replace_linears_with_pruner


### Strategies ###

class PruneStategy:
    def __init__(self):
        self.name = "empty-stategy"
        self.module = None

    def set_prunable_module(self, module: nn.Module):
        self.module = module

    def __call__(self, target_sparsity: float):
        pass

class GlobalWeightPruneStategy(PruneStategy):
    def __init__(self):
        self.name = "global-weight"

        self.applied = 0.0
        self.to_prune = []

    def set_prunable_module(self, module: nn.Module):
        super().set_prunable_module(module)

        self.to_prune = [(m, 'weight') for m in module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, pytorch_utils.Conv1D))]

    def __call__(self, target_sparsity: float):
        inc = target_sparsity - self.applied

        if inc <= 0:
            return

        prune.global_unstructured(
            self.to_prune,
            pruning_method=prune.L1Unstructured,
            amount=inc
        )

        self.applied = target_sparsity

class LayerActivationV1PruneStategy(PruneStategy):
    def __init__(self):
        self.name = "layer-activation-v1"

        self.hidden_layers_hooks = []
        self.to_prune = []

    def set_prunable_module(self, module: nn.Module):
        super().set_prunable_module(module)

        self.to_prune = [m for m in module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, pytorch_utils.Conv1D))]

    def __call__(self, target_sparsity: float):
        #def process_activation(module, input_, output):
        #    abs_values = torch.abs(output)
        #    k = int(target_sparsity * len(abs_values))
        #    if k:
        #        threshold = torch.max((torch.topk(abs_values, k=k, largest=False, dim=1)).values).item()
        #        output.data[abs_values < threshold] = 0

        def process_activation(module, input_, output):
            abs_values = torch.abs(output)
            barier_values = abs_values.type(torch.float32).quantile(target_sparsity, dim=1).unsqueeze(dim=1).repeat_interleave(abs_values.shape[1], dim=1)
            output.data[abs_values < barier_values] = 0

        for hook in self.hidden_layers_hooks:
            hook.remove()
        for param in self.to_prune:
            self.hidden_layers_hooks.append(param.register_forward_hook(process_activation))

class LayerActivationV2PruneStategy(PruneStategy):
    def __init__(self):
        self.name = "layer-activation-v2"

    def __call__(self, target_sparsity: float):
        replace_linears_with_pruner(
            self.module,
            sparsity_ratio=target_sparsity
        )

### Strategy factory ###

def make_prune_strategy(name: str):
    strategies_dict = {
        "none": PruneStategy(),
        "global-weight": GlobalWeightPruneStategy(),
        "layer-activation-v1": LayerActivationV1PruneStategy(),
        "layer-activation-v2": LayerActivationV2PruneStategy(),
    }
    return strategies_dict[name]

### Pruner ###

class PruneCallback(TrainerCallback):
    def __init__(self, model: nn.Module, strategy: PruneStategy, target_sparsity: float,
                 warmup_steps: int, final_prune_step: int, prune_freq: int):
        self.strategy = strategy

        self.target = target_sparsity
        self.warmup = warmup_steps
        self.final = final_prune_step
        self.freq = prune_freq

        self.pruning_history = []

        self.applied = 0.0

        self.strategy.set_prunable_module(model)


    def on_step_begin(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)

        if state.global_step < self.warmup or state.global_step % self.freq != 0:
            return

        progress = min(1.0, (state.global_step - self.warmup) / (self.final - self.warmup))
        desired = progress * self.target
        inc = desired - self.applied

        if inc <= 0:
            return

        self.strategy(desired)

        self.applied = desired

        self.pruning_history.append({
            'step': state.global_step,
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
