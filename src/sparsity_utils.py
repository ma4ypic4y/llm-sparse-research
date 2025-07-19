"""
Вспомогательные утилиты для анализа спарсности модели
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Any


def calculate_sparsity_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Вычисляет детальную статистику спарсности модели

    Returns:
        Dict с ключами:
        - all_params_sparsity: спарсность всех параметров
        - prunable_weights_sparsity: спарсность только прунимых весов
        - layer_wise_sparsity: спарсность по слоям
        - dead_neurons_stats: статистика мертвых нейронов
        - parameter_counts: количество параметров разных типов
    """
    stats = {}

    # 1. Общая статистика параметров
    total_all_params = 0
    zero_all_params = 0
    total_prunable_params = 0
    zero_prunable_params = 0

    # 2. Спарсность по слоям
    layer_sparsity = {}

    # 3. Статистика мертвых нейронов
    dead_neurons = {}

    # 4. Подсчет разных типов параметров
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

        # Общая статистика
        total_all_params += param_size
        zero_all_params += zero_count

        # Классификация параметров
        if 'embed' in name.lower():
            param_counts['embedding'] += param_size
        elif 'norm' in name.lower():
            param_counts['layernorm'] += param_size
        elif 'bias' in name.lower():
            param_counts['bias'] += param_size
        elif 'weight' in name:
            # Получаем родительский модуль
            try:
                parent_module = model
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)

                if isinstance(parent_module, nn.Linear):
                    param_counts['linear'] += param_size
                    # Это прунимый параметр
                    total_prunable_params += param_size
                    zero_prunable_params += zero_count

                    # Спарсность слоя
                    layer_sparsity[name] = zero_count / param_size

                    # Анализ мертвых нейронов
                    w = param.data
                    if w.dim() == 2:  # Матрица весов Linear слоя
                        dead_out = (w == 0).all(dim=1).sum().item()  # Мертвые выходные нейроны
                        dead_in = (w == 0).all(dim=0).sum().item()   # Мертвые входные связи
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
                param_counts['other'] += param_size
        else:
            param_counts['other'] += param_size

    # Собираем финальную статистику
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


def print_sparsity_report(model: nn.Module, title: str = "Отчет по спарсности"):
    """Печатает детальный отчет по спарсности модели"""
    stats = calculate_sparsity_stats(model)

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    # Общая статистика
    print(f"\n📊 ОБЩАЯ СПАРСНОСТЬ:")
    print(f"   Все параметры: {stats['all_params_sparsity']:.3%}")
    print(f"   Прунимые веса: {stats['prunable_weights_sparsity']:.3%}")

    # Параметры
    totals = stats['totals']
    print(f"\n🔢 ПАРАМЕТРЫ:")
    print(f"   Общее количество: {totals['all_parameters']:,}")
    print(f"   Прунимые: {totals['prunable_parameters']:,} ({totals['prunable_ratio']:.1%})")
    print(f"   Зануленные (все): {totals['zero_all_parameters']:,}")
    print(f"   Зануленные (прунимые): {totals['zero_prunable_parameters']:,}")

    # Разбивка по типам
    param_counts = stats['parameter_counts']
    print(f"\n📋 РАЗБИВКА ПО ТИПАМ:")
    for param_type, count in param_counts.items():
        if count > 0:
            print(f"   {param_type.capitalize()}: {count:,}")

    # Топ-10 самых разреженных слоев
    layer_sparsity = stats['layer_wise_sparsity']
    if layer_sparsity:
        print(f"\n🏆 ТОП-10 САМЫХ РАЗРЕЖЕННЫХ СЛОЕВ:")
        sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True)
        for name, sparsity in sorted_layers[:10]:
            print(f"   {name}: {sparsity:.3%}")

    # Мертвые нейроны
    dead_neurons = stats['dead_neurons_stats']
    if dead_neurons:
        print(f"\n☠️  МЕРТВЫЕ НЕЙРОНЫ (топ-5 по выходным):")
        sorted_dead = sorted(dead_neurons.items(),
                           key=lambda x: x[1]['dead_output_ratio'], reverse=True)
        for name, dead_stats in sorted_dead[:5]:
            print(f"   {name}: {dead_stats['dead_output_neurons']}/{dead_stats['total_output_neurons']} "
                  f"({dead_stats['dead_output_ratio']:.1%})")

    print(f"{'='*60}\n")


def compare_sparsity_targets(model: nn.Module, target_sparsity: float) -> Dict[str, str]:
    """
    Сравнивает текущую спарсность с целевой и дает рекомендации

    Returns:
        Dict с анализом и рекомендациями
    """
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
        analysis['status'] = 'Начальная стадия'
        analysis['recommendations'].append('Прунинг только начался, это нормально')
    elif progress_ratio < 0.5:
        analysis['status'] = 'Активный прунинг'
        analysis['recommendations'].append('Прунинг идет по плану')
    elif progress_ratio < 0.9:
        analysis['status'] = 'Приближение к цели'
        analysis['recommendations'].append('Большая часть прунинга выполнена')
    elif progress_ratio < 1.1:
        analysis['status'] = 'Цель достигнута'
        analysis['recommendations'].append('Спарсность соответствует цели')
    else:
        analysis['status'] = 'Превышение цели'
        analysis['recommendations'].append('Спарсность выше целевой - проверьте конфигурацию')

    # Дополнительные рекомендации на основе dead neurons
    dead_neurons = stats['dead_neurons_stats']
    if dead_neurons:
        max_dead_ratio = max(dn['dead_output_ratio'] for dn in dead_neurons.values())
        if max_dead_ratio > 0.5:
            analysis['recommendations'].append('⚠️ Высокий процент мертвых нейронов - возможна деградация')
        elif max_dead_ratio > 0.3:
            analysis['recommendations'].append('Умеренное количество мертвых нейронов')

    return analysis