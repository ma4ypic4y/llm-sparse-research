import torch
import torch.nn as nn
import wandb
import logging
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np


class WeightStabilityTracker:
    """Track weight sparsity stability - how long weights remain zero"""

    def __init__(self):
        self.zero_duration = defaultdict(int)
        self.previous_masks = {}
        self.first_zero_step = {}
        self.revived_count = defaultdict(int)

    def update(self, model: nn.Module, step: int) -> Dict[str, float]:
        """Update weight stability statistics"""
        stats = {}
        total_revived = 0
        total_stable_zeros = 0

        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue

            current_mask = (param.data == 0)

            if name in self.previous_masks:
                prev_mask = self.previous_masks[name]

                revived = (prev_mask == True) & (current_mask == False)
                revived_count = revived.sum().item()
                total_revived += revived_count
                self.revived_count[name] += revived_count

                stable_zeros = (prev_mask == True) & (current_mask == True)
                stable_zero_count = stable_zeros.sum().item()
                total_stable_zeros += stable_zero_count

                for idx in torch.nonzero(stable_zeros, as_tuple=False):
                    coord = tuple(idx.tolist())
                    self.zero_duration[(name, coord)] += 1

                newly_zeroed = (prev_mask == False) & (current_mask == True)
                for idx in torch.nonzero(newly_zeroed, as_tuple=False):
                    coord = tuple(idx.tolist())
                    if (name, coord) not in self.first_zero_step:
                        self.first_zero_step[(name, coord)] = step

            self.previous_masks[name] = current_mask.clone()

        stats['weight_revival/total_revived'] = total_revived
        stats['weight_revival/total_stable_zeros'] = total_stable_zeros
        stats['weight_revival/revival_rate'] = total_revived / (total_revived + total_stable_zeros + 1e-8)

        return stats


class LayerAnalyzer:
    """Analyze individual layer state"""

    def analyze_layer_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """Analyze sparsity by layers"""
        stats = {}

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                w = module.weight.data
                sparsity = (w == 0).float().mean().item()
                stats[f'layer_sparsity/{name}'] = sparsity

                if isinstance(module, nn.Linear):
                    dead_out = (w == 0).all(dim=1).sum().item()
                    dead_in = (w == 0).all(dim=0).sum().item()
                    stats[f'dead_neurons/{name}_out'] = dead_out / w.shape[0]
                    stats[f'dead_neurons/{name}_in'] = dead_in / w.shape[1]

        return stats

    def analyze_layernorm_weights(self, model: nn.Module) -> Dict[str, float]:
        """Analyze LayerNorm weights - zeroed weights indicate dead neurons"""
        stats = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    zero_weights = (module.weight.data == 0).sum().item()
                    total_weights = module.weight.numel()
                    stats[f'layernorm_dead/{name}'] = zero_weights / total_weights

        return stats

    def compute_effective_rank(self, model: nn.Module, threshold: float = 0.01) -> Dict[str, float]:
        """Compute effective rank of weight matrices via SVD"""
        stats = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                if w.numel() > 0:
                    try:
                        U, s, V = torch.svd(w)
                        if len(s) > 0:
                            max_singular = s[0].item()
                            if max_singular > 0:
                                effective_rank = (s > threshold * max_singular).sum().item()
                                theoretical_rank = min(w.shape[0], w.shape[1])
                                stats[f'effective_rank/{name}'] = effective_rank / theoretical_rank
                    except Exception as e:
                        logging.warning(f"Could not compute SVD for {name}: {e}")

        return stats


class GradientAnalyzer:
    """Analyze gradients of zeroed weights"""

    def analyze_zero_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Analyze gradients of zeroed weights"""
        stats = {}
        total_zero_weights = 0
        total_zero_grad_norms = 0

        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                zero_mask = (param.data == 0)
                zero_count = zero_mask.sum().item()
                total_zero_weights += zero_count

                if zero_count > 0:
                    zero_grads = param.grad[zero_mask]
                    grad_norm = torch.norm(zero_grads).item()
                    total_zero_grad_norms += grad_norm
                    stats[f'zero_grad_norm/{name}'] = grad_norm / (zero_count ** 0.5 + 1e-8)

        if total_zero_weights > 0:
            stats['zero_grad_norm/average'] = total_zero_grad_norms / total_zero_weights

        return stats


class PruningMetricsCollector:
    """Main class for collecting all sparsification metrics"""

    def __init__(self):
        self.stability_tracker = WeightStabilityTracker()
        self.layer_analyzer = LayerAnalyzer()
        self.gradient_analyzer = GradientAnalyzer()
        self.logger = logging.getLogger('sparse_weights.metrics')
        self.metric_history = defaultdict(list)

    def collect_all_metrics(self, model: nn.Module, step: int,
                          log_to_wandb: bool = True) -> Dict[str, Any]:
        """Collect all metrics and log them"""
        all_stats = {}

        try:
            total_all_params = 0
            zero_all_params = 0
            for param in model.parameters():
                if param.requires_grad:
                    total_all_params += param.numel()
                    zero_all_params += (param.data == 0).sum().item()

            all_params_sparsity = zero_all_params / (total_all_params + 1e-8)
            all_stats['sparsity/all_parameters'] = all_params_sparsity

            total_prunable_params = 0
            zero_prunable_params = 0

            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Handle both pruned and unpruned modules
                    if hasattr(module, 'weight_orig'):
                        # Module has been pruned, use original weights
                        param = module.weight_orig
                        actual_weights = module.weight.data
                    elif hasattr(module, 'weight'):
                        # Module not pruned yet
                        param = module.weight
                        actual_weights = param.data
                    else:
                        continue

                    if param.requires_grad:
                        total_prunable_params += param.numel()
                        zero_prunable_params += (actual_weights == 0).sum().item()

            prunable_sparsity = zero_prunable_params / (total_prunable_params + 1e-8)
            all_stats['sparsity/prunable_weights'] = prunable_sparsity
            all_stats['sparsity/overall'] = prunable_sparsity

            all_stats['stats/total_parameters'] = total_all_params
            all_stats['stats/prunable_parameters'] = total_prunable_params
            all_stats['stats/prunable_ratio'] = total_prunable_params / (total_all_params + 1e-8)

            stability_stats = self.stability_tracker.update(model, step)
            layer_sparsity_stats = self.layer_analyzer.analyze_layer_sparsity(model)
            layernorm_stats = self.layer_analyzer.analyze_layernorm_weights(model)
            effective_rank_stats = self.layer_analyzer.compute_effective_rank(model)
            gradient_stats = self.gradient_analyzer.analyze_zero_gradients(model)

            all_stats.update(stability_stats)
            all_stats.update(layer_sparsity_stats)
            all_stats.update(layernorm_stats)
            all_stats.update(effective_rank_stats)
            all_stats.update(gradient_stats)

            for key, value in all_stats.items():
                self.metric_history[key].append((step, value))

            if step % 100 == 0:
                self._compute_trend_metrics(all_stats, step)

            if log_to_wandb:
                wandb.log(all_stats, step=step)

            self.logger.debug(f"Collected {len(all_stats)} metrics at step {step}")

        except Exception as e:
            self.logger.error(f"Error collecting metrics at step {step}: {e}")

        return all_stats

    def _compute_trend_metrics(self, stats: Dict[str, Any], step: int):
        """Compute trend metrics"""
        sparsity_history = self.metric_history.get('sparsity/overall', [])
        if len(sparsity_history) >= 10:
            recent_values = [v for s, v in sparsity_history[-10:]]
            trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            stats['trends/sparsity_slope'] = trend_slope

        revival_history = self.metric_history.get('weight_revival/revival_rate', [])
        if len(revival_history) >= 10:
            recent_values = [v for s, v in revival_history[-10:]]
            avg_revival_rate = np.mean(recent_values)
            stats['trends/avg_revival_rate'] = avg_revival_rate

    def get_final_report(self) -> Dict[str, Any]:
        """Generate final metrics report"""
        report = {}

        total_revivals = sum(self.stability_tracker.revived_count.values())
        report['final/total_weight_revivals'] = total_revivals

        if self.stability_tracker.zero_duration:
            avg_duration = np.mean(list(self.stability_tracker.zero_duration.values()))
            report['final/avg_zero_duration'] = avg_duration

        total_first_zeros = len(self.stability_tracker.first_zero_step)
        never_revived = total_first_zeros - len([k for k in self.stability_tracker.revived_count.keys()
                                              if self.stability_tracker.revived_count[k] > 0])
        if total_first_zeros > 0:
            report['final/never_revived_percentage'] = never_revived / total_first_zeros

        return report