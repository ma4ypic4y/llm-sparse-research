"""Utilities for model sparsity analysis"""

import torch.nn as nn
from typing import Dict, Any


def calculate_sparsity_stats(model: nn.Module) -> Dict[str, Any]:
    """Calculate detailed model sparsity statistics"""
    stats = {}

    total_all_params = 0
    zero_all_params = 0
    total_prunable_params = 0
    zero_prunable_params = 0

    layer_sparsity = {}
    dead_neurons = {}
    param_counts = {
        'embedding': 0,
        'layernorm': 0,
        'linear': 0,
        'conv': 0,
        'bias': 0,
        'other': 0
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_size = param.numel()
        zero_count = (param.data == 0).sum().item()

        total_all_params += param_size
        zero_all_params += zero_count

        if 'embed' in name.lower():
            param_counts['embedding'] += param_size
        elif 'norm' in name.lower():
            param_counts['layernorm'] += param_size
        elif 'bias' in name.lower():
            param_counts['bias'] += param_size
        elif 'weight' in name:
            try:
                parent_module = model
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)

                if isinstance(parent_module, nn.Linear):
                    param_counts['linear'] += param_size
                    total_prunable_params += param_size
                    zero_prunable_params += zero_count
                    layer_sparsity[name] = zero_count / param_size

                    # Handle both pruned and unpruned weights
                    if hasattr(parent_module, 'weight_orig'):
                        w = parent_module.weight.data  # Masked weights
                    else:
                        w = param.data  # Original weights

                    if w.dim() == 2:
                        dead_out = (w == 0).all(dim=1).sum().item()
                        dead_in = (w == 0).all(dim=0).sum().item()
                        dead_neurons[name] = {
                            'dead_output_neurons': dead_out,
                            'dead_input_connections': dead_in,
                            'total_output_neurons': w.shape[0],
                            'total_input_connections': w.shape[1],
                            'dead_output_ratio': dead_out / w.shape[0],
                            'dead_input_ratio': dead_in / w.shape[1]
                        }

                elif isinstance(parent_module, nn.Conv2d):
                    param_counts['conv'] += param_size
                    total_prunable_params += param_size
                    zero_prunable_params += zero_count
                    layer_sparsity[name] = zero_count / param_size
                else:
                    param_counts['other'] += param_size
            except:
                if not any(exclude in name.lower() for exclude in ['norm', 'embed']) and param.dim() == 2:
                    param_counts['linear'] += param_size
                    total_prunable_params += param_size
                    zero_prunable_params += zero_count
                    layer_sparsity[name] = zero_count / param_size
                else:
                    param_counts['other'] += param_size
        else:
            param_counts['other'] += param_size
    stats['all_params_sparsity'] = zero_all_params / (total_all_params + 1e-8)
    stats['prunable_weights_sparsity'] = zero_prunable_params / (total_prunable_params + 1e-8)
    stats['layer_wise_sparsity'] = layer_sparsity
    stats['dead_neurons_stats'] = dead_neurons
    stats['parameter_counts'] = param_counts

    stats['totals'] = {
        'all_parameters': total_all_params,
        'prunable_parameters': total_prunable_params,
        'zero_all_parameters': zero_all_params,
        'zero_prunable_parameters': zero_prunable_params,
        'prunable_ratio': total_prunable_params / (total_all_params + 1e-8)
    }

    return stats


def print_sparsity_report(model: nn.Module, title: str = "Sparsity Report"):
    """Print detailed model sparsity report"""
    stats = calculate_sparsity_stats(model)

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"\n📊 OVERALL SPARSITY:")
    print(f"   All parameters: {stats['all_params_sparsity']:.3%}")
    print(f"   Prunable weights: {stats['prunable_weights_sparsity']:.3%}")

    totals = stats['totals']
    print(f"\n🔢 PARAMETERS:")
    print(f"   Total: {totals['all_parameters']:,}")
    print(f"   Prunable: {totals['prunable_parameters']:,} ({totals['prunable_ratio']:.1%})")
    print(f"   Zeroed (all): {totals['zero_all_parameters']:,}")
    print(f"   Zeroed (prunable): {totals['zero_prunable_parameters']:,}")

    param_counts = stats['parameter_counts']
    print(f"\n📋 BREAKDOWN BY TYPE:")
    for param_type, count in param_counts.items():
        if count > 0:
            print(f"   {param_type.capitalize()}: {count:,}")

    layer_sparsity = stats['layer_wise_sparsity']
    if layer_sparsity:
        print(f"\n🏆 TOP-10 MOST SPARSE LAYERS:")
        sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True)
        for name, sparsity in sorted_layers[:10]:
            print(f"   {name}: {sparsity:.3%}")

    dead_neurons = stats['dead_neurons_stats']
    if dead_neurons:
        print(f"\n☠️  DEAD NEURONS (top-5 by output):")
        sorted_dead = sorted(dead_neurons.items(),
                           key=lambda x: x[1]['dead_output_ratio'], reverse=True)
        for name, dead_stats in sorted_dead[:5]:
            print(f"   {name}: {dead_stats['dead_output_neurons']}/{dead_stats['total_output_neurons']} "
                  f"({dead_stats['dead_output_ratio']:.1%})")

    print(f"{'='*60}\n")


def compare_sparsity_targets(model: nn.Module, target_sparsity: float) -> Dict[str, str]:
    """Compare current sparsity with target and provide recommendations"""
    stats = calculate_sparsity_stats(model)
    current_sparsity = stats['prunable_weights_sparsity']

    analysis = {
        'current_sparsity': f"{current_sparsity:.3%}",
        'target_sparsity': f"{target_sparsity:.3%}",
        'progress': f"{current_sparsity / target_sparsity:.1%}",
        'status': '',
        'recommendations': []
    }

    progress_ratio = current_sparsity / target_sparsity

    if progress_ratio < 0.1:
        analysis['status'] = 'Initial stage'
        analysis['recommendations'].append('Pruning just started, this is normal')
    elif progress_ratio < 0.5:
        analysis['status'] = 'Active pruning'
        analysis['recommendations'].append('Pruning is on track')
    elif progress_ratio < 0.9:
        analysis['status'] = 'Approaching target'
        analysis['recommendations'].append('Most pruning completed')
    elif progress_ratio < 1.1:
        analysis['status'] = 'Target reached'
        analysis['recommendations'].append('Sparsity matches target')
    else:
        analysis['status'] = 'Target exceeded'
        analysis['recommendations'].append('Sparsity above target - check configuration')

    dead_neurons = stats['dead_neurons_stats']
    if dead_neurons:
        max_dead_ratio = max(dn['dead_output_ratio'] for dn in dead_neurons.values())
        if max_dead_ratio > 0.5:
            analysis['recommendations'].append('⚠️ High dead neuron ratio - possible degradation')
        elif max_dead_ratio > 0.3:
            analysis['recommendations'].append('Moderate number of dead neurons')

    return analysis