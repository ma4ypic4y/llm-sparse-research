#!/usr/bin/env python3
"""
Быстрая проверка спарсности обученной модели
"""

import argparse
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Config

from src import (
    print_sparsity_report,
    compare_sparsity_targets,
    calculate_sparsity_stats,
    load_config
)


def main():
    parser = argparse.ArgumentParser(description='Check sparsity of trained model')
    parser.add_argument('--model_dir', type=str, default='exp',
                       help='Directory with trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file (for target sparsity)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed report')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Загружаем модель
    print("Загружаю модель...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        print(f"✅ Модель загружена из {model_dir}")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        print("Попробую создать модель с конфигурацией по умолчанию...")

        config = GPT2Config.from_pretrained('gpt2')
        model = GPT2LMHeadModel(config)
        print("⚠️  Создана новая модель (не обученная)")

    # Загружаем конфигурацию
    try:
        config = load_config(args.config)
        target_sparsity = config.get('pruning', {}).get('target_sparsity', 0.5)
        print(f"📋 Целевая спарсность из конфига: {target_sparsity:.1%}")
    except Exception as e:
        print(f"⚠️  Не удалось загрузить конфиг: {e}")
        target_sparsity = 0.5
        print(f"📋 Используется целевая спарсность по умолчанию: {target_sparsity:.1%}")

    print("\n" + "="*60)
    print("АНАЛИЗ СПАРСНОСТИ МОДЕЛИ")
    print("="*60)

    # Быстрая сводка
    stats = calculate_sparsity_stats(model)

    print(f"\n🎯 ЦЕЛЕВАЯ СПАРСНОСТЬ: {target_sparsity:.1%}")
    print(f"📊 ТЕКУЩАЯ СПАРСНОСТЬ:")
    print(f"   - Прунимые веса: {stats['prunable_weights_sparsity']:.3%}")
    print(f"   - Все параметры: {stats['all_params_sparsity']:.3%}")

    # Сравнение с целью
    analysis = compare_sparsity_targets(model, target_sparsity)
    print(f"\n📈 ПРОГРЕСС: {analysis['progress']} от цели")
    print(f"🔍 СТАТУС: {analysis['status']}")

    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    for rec in analysis['recommendations']:
        print(f"   • {rec}")

    # Краткая статистика параметров
    totals = stats['totals']
    print(f"\n📊 ПАРАМЕТРЫ:")
    print(f"   • Общее количество: {totals['all_parameters']:,}")
    print(f"   • Прунимые: {totals['prunable_parameters']:,} ({totals['prunable_ratio']:.1%})")
    print(f"   • Зануленные: {totals['zero_prunable_parameters']:,}")

    # Детальный отчет если запрошен
    if args.detailed:
        print_sparsity_report(model, "ДЕТАЛЬНЫЙ ОТЧЕТ ПО СПАРСНОСТИ")

        # Дополнительная статистика по мертвым нейронам
        dead_neurons = stats['dead_neurons_stats']
        if dead_neurons:
            print("\n🧠 АНАЛИЗ МЕРТВЫХ НЕЙРОНОВ:")
            total_dead_output = sum(dn['dead_output_neurons'] for dn in dead_neurons.values())
            total_output = sum(dn['total_output_neurons'] for dn in dead_neurons.values())
            print(f"   • Общее количество мертвых выходных нейронов: {total_dead_output}")
            print(f"   • Общий процент мертвых нейронов: {total_dead_output/total_output:.2%}")

    # Проверка на проблемы
    print(f"\n🔍 ДИАГНОСТИКА:")
    issues = []

    if stats['prunable_weights_sparsity'] == 0 and target_sparsity > 0:
        issues.append("❌ Спарсность равна нулю при ненулевой цели - прунинг не работает")

    if stats['prunable_weights_sparsity'] > target_sparsity * 1.2:
        issues.append("⚠️  Спарсность значительно превышает цель")

    dead_neurons = stats['dead_neurons_stats']
    if dead_neurons:
        max_dead = max(dn['dead_output_ratio'] for dn in dead_neurons.values())
        if max_dead > 0.7:
            issues.append("🚨 Очень высокий процент мертвых нейронов - критическая деградация")
        elif max_dead > 0.4:
            issues.append("⚠️  Высокий процент мертвых нейронов - возможна деградация")

    if not issues:
        print("   ✅ Проблем не обнаружено")
    else:
        for issue in issues:
            print(f"   {issue}")

    print(f"\n{'='*60}")
    print("📁 Для детального анализа используйте: python check_sparsity.py --detailed")
    print("📊 Для анализа с графиками: python analyze_results.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()