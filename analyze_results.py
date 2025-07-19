#!/usr/bin/env python3

import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import wandb


def load_metrics_summary(output_dir: str):
    """Загружает сохраненную сводку метрик"""
    summary_path = Path(output_dir) / 'metrics_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"Metrics summary not found at {summary_path}")

    with open(summary_path, 'r') as f:
        return json.load(f)


def analyze_weight_revival(wandb_run_id: str, project_name: str):
    """Анализирует динамику разморозки весов"""
    api = wandb.Api()
    run = api.run(f"{project_name}/{wandb_run_id}")

    # Получаем историю метрик
    history = run.history()

    # Фильтруем метрики разморозки
    revival_cols = [col for col in history.columns if 'weight_revival' in col]
    sparsity_cols = [col for col in history.columns if 'sparsity' in col and 'layer' not in col]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Анализ разморозки весов', fontsize=16)

    # 1. Revival rate во времени
    if 'weight_revival/revival_rate' in history.columns:
        axes[0,0].plot(history.index, history['weight_revival/revival_rate'])
        axes[0,0].set_title('Revival Rate во времени')
        axes[0,0].set_xlabel('Шаг')
        axes[0,0].set_ylabel('Revival Rate')
        axes[0,0].grid(True)

    # 2. Общая спарсность
    if 'sparsity/overall' in history.columns:
        axes[0,1].plot(history.index, history['sparsity/overall'])
        axes[0,1].set_title('Общая спарсность')
        axes[0,1].set_xlabel('Шаг')
        axes[0,1].set_ylabel('Спарсность')
        axes[0,1].grid(True)

    # 3. Количество оживших весов
    if 'weight_revival/total_revived' in history.columns:
        axes[1,0].plot(history.index, history['weight_revival/total_revived'])
        axes[1,0].set_title('Количество оживших весов')
        axes[1,0].set_xlabel('Шаг')
        axes[1,0].set_ylabel('Оживших весов')
        axes[1,0].grid(True)

    # 4. Стабильные нулевые веса
    if 'weight_revival/total_stable_zeros' in history.columns:
        axes[1,1].plot(history.index, history['weight_revival/total_stable_zeros'])
        axes[1,1].set_title('Стабильно зануленные веса')
        axes[1,1].set_xlabel('Шаг')
        axes[1,1].set_ylabel('Стабильно нулевых весов')
        axes[1,1].grid(True)

    plt.tight_layout()
    return fig


def analyze_layer_degradation(wandb_run_id: str, project_name: str):
    """Анализирует деградацию по слоям"""
    api = wandb.Api()
    run = api.run(f"{project_name}/{wandb_run_id}")

    history = run.history()

    # Получаем метрики по слоям
    layer_sparsity_cols = [col for col in history.columns if 'layer_sparsity/' in col]
    dead_neuron_cols = [col for col in history.columns if 'dead_neurons/' in col]
    effective_rank_cols = [col for col in history.columns if 'effective_rank/' in col]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Анализ деградации слоев', fontsize=16)

    # 1. Спарсность по слоям (последние значения)
    if layer_sparsity_cols:
        last_values = history[layer_sparsity_cols].iloc[-1]
        layer_names = [col.replace('layer_sparsity/', '') for col in layer_sparsity_cols]

        axes[0,0].bar(range(len(last_values)), last_values.values)
        axes[0,0].set_title('Финальная спарсность по слоям')
        axes[0,0].set_xlabel('Слой')
        axes[0,0].set_ylabel('Спарсность')
        axes[0,0].set_xticks(range(len(layer_names)))
        axes[0,0].set_xticklabels(layer_names, rotation=45)

    # 2. Мертвые нейроны
    if dead_neuron_cols:
        dead_out_cols = [col for col in dead_neuron_cols if '_out' in col]
        if dead_out_cols:
            last_values = history[dead_out_cols].iloc[-1]
            layer_names = [col.replace('dead_neurons/', '').replace('_out', '')
                         for col in dead_out_cols]

            axes[0,1].bar(range(len(last_values)), last_values.values)
            axes[0,1].set_title('Мертвые выходные нейроны')
            axes[0,1].set_xlabel('Слой')
            axes[0,1].set_ylabel('Доля мертвых нейронов')
            axes[0,1].set_xticks(range(len(layer_names)))
            axes[0,1].set_xticklabels(layer_names, rotation=45)

    # 3. Effective Rank во времени
    if effective_rank_cols:
        for col in effective_rank_cols[:5]:  # Показываем только первые 5 слоев
            layer_name = col.replace('effective_rank/', '')
            axes[1,0].plot(history.index, history[col], label=layer_name, alpha=0.7)

        axes[1,0].set_title('Effective Rank во времени')
        axes[1,0].set_xlabel('Шаг')
        axes[1,0].set_ylabel('Effective Rank (нормализованный)')
        axes[1,0].legend()
        axes[1,0].grid(True)

    # 4. Gradient norms зануленных весов
    grad_norm_cols = [col for col in history.columns if 'zero_grad_norm/' in col]
    if 'zero_grad_norm/average' in history.columns:
        axes[1,1].plot(history.index, history['zero_grad_norm/average'])
        axes[1,1].set_title('Средняя норма градиентов зануленных весов')
        axes[1,1].set_xlabel('Шаг')
        axes[1,1].set_ylabel('Норма градиента')
        axes[1,1].grid(True)
        axes[1,1].set_yscale('log')  # Логарифмическая шкала

    plt.tight_layout()
    return fig


def generate_report(output_dir: str, wandb_run_id: str = None, project_name: str = None):
    """Генерирует полный отчет по эксперименту"""
    print("Генерирую отчет по эксперименту...")

    # Загружаем локальную сводку
    try:
        metrics_summary = load_metrics_summary(output_dir)
        print("✓ Локальная сводка метрик загружена")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return

    # Анализируем локальные метрики
    final_metrics = metrics_summary.get('final_metrics', {})
    pruning_summary = metrics_summary.get('pruning_summary', {})
    config = metrics_summary.get('config', {})

    print("\n" + "="*60)
    print("ОТЧЕТ ПО ЭКСПЕРИМЕНТУ СПАРСИФИКАЦИИ ВЕСОВ")
    print("="*60)

    print("\n📊 КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА:")
    print(f"Target sparsity: {config.get('pruning', {}).get('target_sparsity', 'N/A'):.1%}")
    print(f"Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
    print(f"Learning rate: {config.get('training', {}).get('lr', 'N/A')}")
    print(f"Prune frequency: {config.get('pruning', {}).get('prune_freq', 'N/A')}")

    print("\n🎯 РЕЗУЛЬТАТЫ ПРУНИНГА:")
    if pruning_summary:
        print(f"Финальная спарсность: {pruning_summary.get('final_applied_sparsity', 'N/A'):.1%}")
        print(f"Эффективность прунинга: {pruning_summary.get('pruning_efficiency', 'N/A'):.1%}")
        print(f"Количество шагов прунинга: {pruning_summary.get('num_pruning_steps', 'N/A')}")
        print(f"Средний прирост спарсности за шаг: {pruning_summary.get('avg_increment_per_step', 'N/A'):.3f}")

    print("\n🔄 АНАЛИЗ РАЗМОРОЗКИ ВЕСОВ:")
    if final_metrics:
        total_revivals = final_metrics.get('final/total_weight_revivals', 'N/A')
        never_revived = final_metrics.get('final/never_revived_percentage', 'N/A')
        avg_duration = final_metrics.get('final/avg_zero_duration', 'N/A')

        print(f"Общее количество разморозок: {total_revivals}")
        if isinstance(never_revived, (int, float)):
            print(f"Процент весов, не размороженных никогда: {never_revived:.1%}")
        else:
            print(f"Процент весов, не размороженных никогда: {never_revived}")
        print(f"Средняя длительность зануления: {avg_duration}")

    print("\n📈 КЛЮЧЕВЫЕ ВЫВОДЫ:")

    # Анализируем результаты и даем рекомендации
    if isinstance(final_metrics.get('final/never_revived_percentage'), (int, float)):
        never_revived_pct = final_metrics['final/never_revived_percentage']
        if never_revived_pct > 0.8:
            print("🚨 КРИТИЧНО: Очень высокий процент навсегда зануленных весов (>80%)")
            print("   Рекомендация: Возможна деградация модели, рассмотрите снижение target_sparsity")
        elif never_revived_pct > 0.6:
            print("⚠️  ВНИМАНИЕ: Высокий процент навсегда зануленных весов (>60%)")
            print("   Рекомендация: Проверьте качество модели, возможна частичная деградация")
        else:
            print("✅ Хороший баланс между прунингом и активностью весов")

    if total_revivals == 0:
        print("🚨 КРИТИЧНО: Полное отсутствие разморозки весов")
        print("   Рекомендация: Проверьте реализацию прунинга и learning rate")

    # Если есть wandb данные, генерируем графики
    if wandb_run_id and project_name:
        print(f"\n📈 Генерирую графики из wandb run: {wandb_run_id}")
        try:
            # Анализ разморозки весов
            revival_fig = analyze_weight_revival(wandb_run_id, project_name)
            revival_fig.savefig(f"{output_dir}/weight_revival_analysis.png", dpi=300, bbox_inches='tight')
            print("✓ График анализа разморозки весов сохранен")

            # Анализ деградации слоев
            degradation_fig = analyze_layer_degradation(wandb_run_id, project_name)
            degradation_fig.savefig(f"{output_dir}/layer_degradation_analysis.png", dpi=300, bbox_inches='tight')
            print("✓ График анализа деградации слоев сохранен")

        except Exception as e:
            print(f"✗ Ошибка при генерации графиков: {e}")

    print(f"\n📁 Отчет сохранен в: {output_dir}/")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze pruning experiment results')
    parser.add_argument('--output_dir', type=str, default='exp',
                       help='Directory with experiment results')
    parser.add_argument('--wandb_run_id', type=str,
                       help='WandB run ID for detailed analysis')
    parser.add_argument('--project_name', type=str, default='sparse-weights',
                       help='WandB project name')

    args = parser.parse_args()

    generate_report(args.output_dir, args.wandb_run_id, args.project_name)


if __name__ == '__main__':
    main()