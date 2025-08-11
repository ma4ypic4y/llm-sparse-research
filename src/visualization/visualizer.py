from __future__ import annotations

from copy import deepcopy
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from scipy.stats import gaussian_kde
import numpy as np

from ..hooks.data_collector import _get_weights_distribution_statistics
from .data_worker import DataWorker


class Visualizer:
    """
    A class to visualize model weights and gradients.
    """

    def __init__(self, data_worker: DataWorker):
        """
        Initializes the Visualizer with a DataWorker instance.

        Args:
            data_worker: An instance of DataWorker containing model weights and gradients.
        """
        self.data_worker = data_worker

    def visualize_weights_statistics(
            self,
            sort_by: Literal['depth', 'layer_type', 'active_weights'] = 'depth',
            slice_direction: Literal['layers', 'steps'] = 'layers',
            slice_position: int | str = -1
        ) -> None:
        """
        Visualizes the statistics of weights in the model.

        Args:
            sort_by: A string indicating how to sort the layers. Options are 'depth', 'layer_type', or 'active_weights'.
            slice_direction: A string indicating how to slice the data. Options are 'layers' (to visualize per layer) or 'steps' (to visualize per step).
            slice_position: An integer or string indicating the position to slice the data. If an integer

        Note:
            The 'depth' option sorts layers by their depth in the model, 'layer_type' sorts by the type of layer, and 'active_weights' sorts by the ratio of active weights to total weights.
        """

        assert slice_direction != 'steps' or sort_by == 'depth', "Currently, only 'depth' sorting is supported for 'steps' slicing"

        weights_statistics = {}
        model_statistics = {
            'active_weights': 0,
            'zero_only_weights': 0,
            'dead_only_weights': 0,
            'zero_dead_weights': 0
        }

        data = self.data_worker.get_weights_statistics()
        if len(data) == 0:
            print("Warning: No data collected, skipping visualization")
            return

        if slice_direction == 'layers':
            assert type(slice_position) is int and -len(data) <= slice_position < len(data), "slice_position must be an integer within the range of collected data"
            weights_statistics = data[slice_position]
        elif slice_direction == 'steps':
            assert type(slice_position) is str and slice_position in list(data[-1].keys()), "slice_position must be a string representing the layer name in the collected data"
            model_statistics = deepcopy(data[-1][slice_position])
            model_statistics['zero_only_weights'] = model_statistics['zero_weights'] - model_statistics['zero_dead_weights']
            model_statistics['dead_only_weights'] = model_statistics['dead_weights'] - model_statistics['zero_dead_weights']
            weights_statistics = {
                self.data_worker.get_collectors_info()['weights_collector_warmup_steps'] + step * self.data_worker.get_collectors_info()['weights_collector_frequency']: data[step][slice_position]
                for step in range(len(data))
            }

        layers = list(weights_statistics.keys())

        if sort_by == 'layer_type':
            layers = sorted(layers, key=lambda x: x[::-1])
        elif sort_by == 'active_weights':
            layers = sorted(layers, key=lambda x: weights_statistics[x]['active_weights'] / weights_statistics[x]['total_weights'], reverse=True)

        layer_names = []
        for layer in layers:
            layer_names += [layer] * 4
        weight_states = ['active', 'zero_only', 'dead_only', 'zero_dead'] * len(layers)
        counts = []
        for layer in layers:
            bar_height = np.log10(weights_statistics[layer]['total_weights'])
            value_ration = bar_height / weights_statistics[layer]['total_weights']

            counts.append(weights_statistics[layer]['active_weights'] * value_ration)
            counts.append((weights_statistics[layer]['zero_weights'] - weights_statistics[layer]['zero_dead_weights']) * value_ration)
            counts.append((weights_statistics[layer]['dead_weights'] - weights_statistics[layer]['zero_dead_weights']) * value_ration)
            counts.append(weights_statistics[layer]['zero_dead_weights'] * value_ration)

            if slice_direction == 'layers':
                model_statistics['active_weights'] += weights_statistics[layer]['active_weights']
                model_statistics['zero_only_weights'] += weights_statistics[layer]['zero_weights'] - weights_statistics[layer]['zero_dead_weights']
                model_statistics['dead_only_weights'] += weights_statistics[layer]['dead_weights'] - weights_statistics[layer]['zero_dead_weights']
                model_statistics['zero_dead_weights'] += weights_statistics[layer]['zero_dead_weights']

        plt.figure(figsize=(60, 8))
        sns.histplot(
            x=layer_names, hue=weight_states, weights=counts,
            multiple="stack",
            palette=sns.color_palette("husl", 8)[3::-1],
            edgecolor=".3",
            linewidth=.5,
            bins=len(layers)
        )
        plt.xticks(rotation=90)
        plt.title(f"Weights Statistics | dead: {model_statistics['dead_only_weights']}, zero only: {model_statistics['zero_only_weights']}, active: {model_statistics['active_weights']}")
        plt.xlabel("Layer Names")
        plt.ylabel("Count (log scale)")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        bar_width = plt.gca().patches[0].get_width()
        bar_start = plt.gca().patches[0].get_x()

        for layer_index, layer in enumerate(layers):
            zero_dead = weights_statistics[layer]['zero_dead_weights']
            zero = weights_statistics[layer]['zero_weights'] - zero_dead
            dead = weights_statistics[layer]['dead_weights'] - zero_dead
            active = weights_statistics[layer]['active_weights']
            total = weights_statistics[layer]['total_weights']

            plt.text(
                x=bar_start + layer_index * bar_width + bar_width / 2, y = 0.5,
                s=f"{zero_dead} / {dead} / {zero} / {active} | {total}",
                ha='center', va='bottom',
                rotation=90
            )

    def visualize_masks_statistics(
            self,
            sort_by: Literal['depth', 'layer_type', 'active_weights'] = 'depth',
            slice_direction: Literal['layers', 'steps'] = 'layers',
            slice_position: int | str = -1
        ) -> None:
        """
        Visualizes the statistics of masks in the model.

        Args:
            sort_by: A string indicating how to sort the layers. Options are 'depth', 'layer_type', or 'active_weights'.
            slice_direction: A string indicating how to slice the data. Options are 'layers' (to visualize per layer) or 'steps' (to visualize per step).
            slice_position: An integer or string indicating the position to slice the data. If an integer

        Note:
            The 'depth' option sorts layers by their depth in the model, 'layer_type' sorts by the type of layer, and 'active_weights' sorts by the ratio of active weights to total weights.
        """

        assert slice_direction != 'steps' or sort_by == 'depth', "Currently, only 'depth' sorting is supported for 'steps' slicing"

        weights_statistics = {}

        data = self.data_worker.get_masks_statistics()

        if slice_direction == 'layers':
            assert type(slice_position) is int and -len(data) <= slice_position < len(data), "slice_position must be an integer within the range of collected data"
            weights_statistics = data[slice_position]
        elif slice_direction == 'steps':
            assert type(slice_position) is str and slice_position in list(data[-1].keys()), "slice_position must be a string representing the layer name in the collected data"
            weights_statistics = {
                self.data_worker.get_collectors_info()['masks_collector_warmup_steps'] + step * self.data_worker.get_collectors_info()['masks_collector_frequency']: data[step][slice_position]
                for step in range(len(data))
            }

        layers = [stat for stat in weights_statistics.keys() if slice_direction == 'steps' or stat.endswith('_mask')]

        if sort_by == 'layer_type':
            layers = sorted(layers, key=lambda x: x[::-1])
        elif sort_by == 'active_weights':
            layers = sorted(layers, key=lambda x: weights_statistics[x]['active_weights'] / weights_statistics[x]['total_weights'], reverse=True)

        layer_names = []
        for layer in layers:
            layer_names += [layer] * 3
        weight_states = ['unchanged', 'turned_on', 'turned_off'] * len(layers)
        counts = []
        for layer in layers:
            bar_height = np.log10(weights_statistics[layer]['total_weights'])
            value_ration = bar_height / weights_statistics[layer]['total_weights']

            counts.append((weights_statistics[layer]['total_weights'] - weights_statistics[layer]['turned_on_amount'] - weights_statistics[layer]['turned_off_amount']) * value_ration)
            counts.append(weights_statistics[layer]['turned_on_amount'] * value_ration)
            counts.append(weights_statistics[layer]['turned_off_amount'] * value_ration)

        plt.figure(figsize=(60, 8))
        sns.histplot(
            x=layer_names, hue=weight_states, weights=counts,
            multiple="stack",
            palette="light:b",
            edgecolor=".3",
            linewidth=.5,
            bins=len(layers)
        )
        plt.xticks(rotation=90)
        plt.title(f"Masks Statistics")
        plt.xlabel("Layer Names")
        plt.ylabel("Count (log scale)")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        bar_width = plt.gca().patches[0].get_width()
        bar_start = plt.gca().patches[0].get_x()

        for layer_index, layer in enumerate(layers):
            unchanged = weights_statistics[layer]['total_weights'] - weights_statistics[layer]['turned_on_amount'] - weights_statistics[layer]['turned_off_amount']
            turned_on = weights_statistics[layer]['turned_on_amount']
            turned_off = weights_statistics[layer]['turned_off_amount']
            total = weights_statistics[layer]['total_weights']

            plt.text(
                x=bar_start + layer_index * bar_width + bar_width / 2, y = 0.5,
                s=f"{unchanged} / {turned_on} / {turned_off} | {total}",
                ha='center', va='bottom',
                rotation=90
            )

    def visualize_masks_summary_statistics(
            self,
            sort_by: Literal['depth', 'layer_type', 'active_weights'] = 'depth',
            slice_direction: Literal['layers', 'steps'] = 'layers',
        ) -> None:
        """
        Visualizes the statistics of masks in the model.

        Args:
            sort_by: A string indicating how to sort the layers. Options are 'depth', 'layer_type', or 'active_weights'.
            slice_direction: A string indicating how to slice the data. Options are 'layers' (to visualize per layer) or 'steps' (to visualize per step).
            slice_position: An integer or string indicating the position to slice the data. If an integer

        Note:
            The 'depth' option sorts layers by their depth in the model, 'layer_type' sorts by the type of layer, and 'active_weights' sorts by the ratio of active weights to total weights.
        """

        assert slice_direction != 'steps' or sort_by == 'depth', "Currently, only 'depth' sorting is supported for 'steps' slicing"

        weights_statistics = {
        }
        data = deepcopy(self.data_worker.get_masks_statistics())

        steps = list(range(len(data)))
        layers = list(data[-1].keys())

        if slice_direction == 'layers':
            for layer in layers:
                weights_statistics[layer] = {
                    'total_weights': data[0][layer]['total_weights'],
                    'turned_off_amount': sum([data[step][layer]['turned_off_amount'] for step in steps]),
                    'turned_on_amount': sum([data[step][layer]['turned_on_amount'] for step in steps]),
                    'avg_active_weights': sum([data[step][layer]['total_weights'] - data[step][layer]['turned_off_amount'] - data[step][layer]['turned_on_amount'] for step in steps]) / len(steps),
                    'avg_prune_amount': sum([data[step][layer]['prune_amount'] for step in steps]) / len(steps)
                }
        elif slice_direction == 'steps':
            for step in steps:
                weights_statistics[self.data_worker.get_collectors_info()['masks_collector_warmup_steps'] + step * self.data_worker.get_collectors_info()['masks_collector_frequency']] = {
                    'total_weights': sum([data[step][layer]['total_weights'] for layer in layers]),
                    'turned_off_amount': sum([data[step][layer]['turned_off_amount'] for layer in layers]),
                    'turned_on_amount': sum([data[step][layer]['turned_on_amount'] for layer in layers]),
                    'avg_active_weights': sum([data[step][layer]['total_weights'] - data[step][layer]['turned_off_amount'] - data[step][layer]['turned_on_amount'] for layer in layers]) / len(layers),
                    'avg_prune_amount': sum([data[step][layer]['prune_amount'] for layer in layers]) / len(layers)
                }

        layers = [stat for stat in weights_statistics.keys() if slice_direction == 'steps' or stat.endswith('_mask')]

        if sort_by == 'layer_type':
            layers = sorted(layers, key=lambda x: x[::-1])
        elif sort_by == 'active_weights':
            layers = sorted(layers, key=lambda x: weights_statistics[x]['active_weights'] / weights_statistics[x]['total_weights'], reverse=True)

        layer_names = []
        for layer in layers:
            layer_names += [layer] * 2
        weight_states = ['turned_on', 'turned_off'] * len(layers)
        counts = []
        for layer in layers:
            change_amount = weights_statistics[layer]['turned_on_amount'] + weights_statistics[layer]['turned_off_amount']
            value_ration = np.log10(change_amount) / change_amount if change_amount > 0 else 0

            counts.append(weights_statistics[layer]['turned_on_amount'] * value_ration)
            counts.append(weights_statistics[layer]['turned_off_amount'] * value_ration)

        plt.figure(figsize=(60, 8))
        sns.histplot(
            x=layer_names, hue=weight_states, weights=counts,
            multiple="stack",
            palette="light:b",
            edgecolor=".3",
            linewidth=.5,
            bins=len(layers)
        )
        plt.xticks(rotation=90)
        plt.title(f"Masks Statistics")
        plt.xlabel("Layer Names")
        plt.ylabel("Count (log scale)")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        bar_width = plt.gca().patches[0].get_width()
        bar_start = plt.gca().patches[0].get_x()

        for layer_index, layer in enumerate(layers):
            turned_on = weights_statistics[layer]['turned_on_amount']
            turned_off = weights_statistics[layer]['turned_off_amount']

            plt.text(
                x=bar_start + layer_index * bar_width + bar_width / 2, y = 0.5,
                s=f"{turned_on} / {turned_off} | avg_act: {weights_statistics[layer]['avg_active_weights']:.2f} / avg_prn: {weights_statistics[layer]['avg_prune_amount']:.2f} / total: {weights_statistics[layer]['total_weights']}",
                ha='center', va='bottom',
                rotation=90
            )

    def visualize_masks_flick_distribution(
            self,
            weights_transform: Literal['norm', 'abs_log'] = 'norm',
            sort_by: Literal['depth', 'layer_type', 'active_weights'] = 'depth',
            plot_type: Literal['violine', 'boxplot', 'lineplot'] = 'lineplot',
        ) -> None:
        """
        Visualizes the distribution of weights in the model.

        Args:
            weights_transform: A string indicating how to transform the weights. Options are 'norm' (to take the absolute value) or 'abs_log' (to take the 10'th logarithm of the absolute value).
            sort_by: A string indicating how to sort the layers. Options are 'depth', 'layer_type', or 'active_weights'.
            plot_type: A string indicating the type of plot to use. Options are 'violine' or 'boxplot'.
        Note:
            The 'depth' option sorts layers by their depth in the model, 'layer_type' sorts by the type of layer, and 'active_weights' sorts by the ratio of active weights to total weights.
        """

        data = self.data_worker.get_masks_changes()
        statistics = {name: _get_weights_distribution_statistics(param.type(torch.float32), try_unique_count=True) for name, param in data.items()}

        self._visualize_weights_distribution(
            statistics,
            'Mask parameters changes',
            weights_transform,
            sort_by,
            plot_type,
        )

    def visualize_weights_distribution(
            self,
            weights_transform: Literal['norm', 'abs_log'] = 'abs_log',
            sort_by: Literal['depth', 'layer_type', 'active_weights'] = 'depth',
            plot_type: Literal['violine', 'boxplot', 'lineplot'] = 'lineplot',
            slice_direction: Literal['layers', 'steps'] = 'layers',
            slice_position: int | str = -1,
        ) -> None:
        """
        Visualizes the distribution of weights in the model.

        Args:
            weights_transform: A string indicating how to transform the weights. Options are 'norm' (to take the absolute value) or 'abs_log' (to take the 10'th logarithm of the absolute value).
            sort_by: A string indicating how to sort the layers. Options are 'depth', 'layer_type', or 'active_weights'.
            plot_type: A string indicating the type of plot to use. Options are 'violine' or 'boxplot'.
            slice_direction: A string indicating how to slice the data. Options are 'layers' (to visualize per layer) or 'steps' (to visualize per step).
            slice_position: An integer or string indicating the position to slice the data.
        Note:
            The 'depth' option sorts layers by their depth in the model, 'layer_type' sorts by the type of layer, and 'active_weights' sorts by the ratio of active weights to total weights.
        """

        assert slice_direction != 'steps' or sort_by == 'depth', "Currently, only 'depth' sorting is supported for 'steps' slicing"

        statistics = {}
        data = self.data_worker.get_weights_data()

        if slice_direction == 'layers':
            assert type(slice_position) is int and -len(data) <= slice_position < len(data), "slice_position must be an integer within the range of collected data"
            statistics = data[slice_position]
        elif slice_direction == 'steps':
            assert type(slice_position) is str and slice_position in list(data[-1].keys()), "slice_position must be a string representing the layer name in the collected data"
            statistics = {
                self.data_worker.get_collectors_info()['weights_collector_warmup_steps'] + step * self.data_worker.get_collectors_info()['weights_collector_frequency']: data[step][slice_position]
                for step in range(len(data))
            }

        self._visualize_weights_distribution(
            statistics,
            'Weights',
            weights_transform,
            sort_by,
            plot_type,
        )

    def visualize_gradients_distribution(
            self,
            weights_transform: Literal['norm', 'abs_log'] = 'abs_log',
            sort_by: Literal['depth', 'layer_type', 'active_weights'] = 'depth',
            plot_type: Literal['violine', 'boxplot', 'lineplot'] = 'lineplot',
            slice_direction: Literal['layers', 'steps'] = 'layers',
            slice_position: int | str = -1
        ) -> None:
        """
        Visualizes the distribution of gradients in the model.

        Args:
            weights_transform: A string indicating how to transform the weights. Options are 'norm' (to take the absolute value) or 'abs_log' (to take the 10'th logarithm of the absolute value).
            sort_by: A string indicating how to sort the layers. Options are 'depth', 'layer_type', or 'active_weights'.
            plot_type: A string indicating the type of plot to use. Options are 'violine' or 'boxplot'.
            slice_direction: A string indicating how to slice the data. Options are 'layers' (to visualize per layer) or 'steps' (to visualize per step).
            slice_position: An integer or string indicating the position to slice the data.
        Note:
            The 'depth' option sorts layers by their depth in the model, 'layer_type' sorts by the type of layer, and 'active_weights' sorts by the ratio of active weights to total weights.
        """

        assert slice_direction != 'steps' or sort_by == 'depth', "Currently, only 'depth' sorting is supported for 'steps' slicing"

        statistics = {}
        data = self.data_worker.get_grad_data()

        if slice_direction == 'layers':
            assert type(slice_position) is int and -len(data) <= slice_position < len(data), "slice_position must be an integer within the range of collected data"
            statistics = data[slice_position]
        elif slice_direction == 'steps':
            assert type(slice_position) is str and slice_position in list(data[-1].keys()), "slice_position must be a string representing the step in the collected data"
            statistics = {
                self.data_worker.get_collectors_info()['weights_collector_warmup_steps'] + step * self.data_worker.get_collectors_info()['weights_collector_frequency']: data[step][slice_position]
                for step in range(len(data))
            }

        self._visualize_weights_distribution(
            statistics,
            'Gradients',
            weights_transform,
            sort_by,
            plot_type,
        )

    def _draw_discreate_violin_plots(
            self,
            ax: plt.Axes,
            position: int,
            mean: float,
            quantile_0: float,
            quantile_25: float | None,
            quantile_50: float | None,
            quantile_75: float | None,
            quantile_100: float,
            values: list[float] | None = None,
            weights: list[float] | None = None,
            color: str | None = None,
            section_width: float = 0.9
    ):
        if color is None:
            color = 'blue'

        if values is not None and weights is not None:
            kde = gaussian_kde(values, weights=weights)
            x_vals = np.linspace(quantile_0, quantile_100, 100)
            y_vals = kde(x_vals)

            y_vals_scaled = y_vals / y_vals.max() * (section_width / 2)
            ax.fill_between(x_vals, position - y_vals_scaled, position + y_vals_scaled, alpha=0.3, color=color, lw=1.5, edgecolor=color)

            ax.plot([quantile_0, quantile_100], [position, position], color=color, lw=1)

        cap = section_width * 0.05
        ax.plot([quantile_0, quantile_0], [position - cap, position + cap], color=color, lw=1)
        ax.plot([quantile_100, quantile_100], [position - cap, position + cap], color=color, lw=1)

        box_width = section_width * 0.1
        if quantile_25 is not None and quantile_75 is not None:
            ax.add_patch(plt.Rectangle(
                (quantile_25, position - box_width / 2),
                quantile_75 - quantile_25,
                box_width,
                facecolor=color,
                edgecolor='black',
                alpha=0.9
            ))

        if quantile_50 is not None:
            ax.plot([quantile_50, quantile_50], [position - box_width / 2, position + box_width / 2], color='black', lw=1)

        ax.plot([mean, mean], [position - cap, position + cap], '--', color='gray', lw=1)

    def _visualize_weights_distribution(
            self,
            weights_data: dict,
            data_suffix: str,
            weights_transform: Literal['norm', 'abs_log'] = 'abs_log',
            sort_by: Literal['depth', 'layer_type', 'active_weights'] = 'depth',
            plot_type: Literal['violine', 'boxplot', 'lineplot'] = 'lineplot'
        ) -> None:
        """
        Visualizes the distribution of weights in the model.

        Args:
            weights_data: A dictionary containing the weights data.
            data_suffix: A string indicating the type of data to visualize. Options are 'weights' or 'gradients'.
            weights_transform: A string indicating how to transform the weights. Options are 'norm' (to take the absolute value) or 'abs_log' (to take the 10'th logarithm of the absolute value).
            sort_by: A string indicating how to sort the layers. Options are 'depth', 'layer_type', or 'active_weights'.
            plot_type: A string indicating the type of plot to use. Options are 'violine' or 'boxplot'.
        Note:
            The 'depth' option sorts layers by their depth in the model, 'layer_type' sorts by the type of layer, and 'active_weights' sorts by the ratio of active weights to total weights.
        """

        weights_data = deepcopy(weights_data)

        layers = [layer for layer in weights_data.keys() if weights_data[layer] is not None]
        if len(layers) == 0:
            print("Warning: No layers with weights found, skipping visualization")
            return

        if sort_by == 'layer_type':
            layers = sorted(layers, key=lambda x: [int(part) if part.isdigit() else part for part in x.split('.')][::-1])
        elif sort_by == 'active_weights':
            weights_statistics = self.data_worker.get_weights_statistics()
            layers = sorted(layers, key=lambda x: weights_statistics[x]['active_weights'] / weights_statistics[x]['total_weights'], reverse=True)

        if weights_transform == 'abs_log':
            for layer in layers:
                if weights_data[layer]['quantile_0'] == 0:
                    weights_data[layer]['quantile_0'] = 1e-16
                if weights_data[layer]['quantile_25'] == 0:
                    weights_data[layer]['quantile_25'] = 1e-16
                if weights_data[layer]['quantile_50'] == 0:
                    weights_data[layer]['quantile_50'] = 1e-16
                if weights_data[layer]['quantile_75'] == 0:
                    weights_data[layer]['quantile_75'] = 1e-16
                if weights_data[layer]['quantile_100'] == 0:
                    weights_data[layer]['quantile_100'] = 1e-16

                if weights_data[layer]['between_dots'] is not None:
                    weights_data[layer]['between_dots'] = [max(x, 1e-16) for x in weights_data[layer]['between_dots']]

                weights_data[layer]['mean'] = weights_data[layer]['log10_mean']
                weights_data[layer]['quantile_0'] = np.log10(weights_data[layer]['quantile_0'])
                if weights_data[layer]['quantile_25'] is not None:
                    weights_data[layer]['quantile_25'] = np.log10(weights_data[layer]['quantile_25'])
                if weights_data[layer]['quantile_50'] is not None:
                    weights_data[layer]['quantile_50'] = np.log10(weights_data[layer]['quantile_50'])
                if weights_data[layer]['quantile_75'] is not None:
                    weights_data[layer]['quantile_75'] = np.log10(weights_data[layer]['quantile_75'])
                weights_data[layer]['quantile_100'] = np.log10(weights_data[layer]['quantile_100'])

                if weights_data[layer]['between_dots'] is not None:
                    weights_data[layer]['between_dots'] = [np.log10(x) for x in weights_data[layer]['between_dots']]

        plt.figure(figsize=(8, 2 + len(layers) // 3))
        if plot_type == 'violine':
            colors = sns.color_palette('Set2', len(layers))
            for i, layer in enumerate(layers):
                self._draw_discreate_violin_plots(
                    plt.gca(),
                    i,
                    weights_data[layer]['mean'],
                    weights_data[layer]['quantile_0'],
                    weights_data[layer]['quantile_25'],
                    weights_data[layer]['quantile_50'],
                    weights_data[layer]['quantile_75'],
                    weights_data[layer]['quantile_100'],
                    values=weights_data[layer]['between_dots'],
                    weights=weights_data[layer]['between_dots_cnt'],
                    color=colors[i],
                )
            plt.ylim(-0.5, len(layers) - 0.5)
        elif plot_type == 'boxplot':
            colors = sns.color_palette('Set2', len(layers))
            for i, layer in enumerate(layers):
                self._draw_discreate_violin_plots(
                    plt.gca(),
                    i,
                    weights_data[layer]['mean'],
                    weights_data[layer]['quantile_0'],
                    weights_data[layer]['quantile_25'],
                    weights_data[layer]['quantile_50'],
                    weights_data[layer]['quantile_75'],
                    weights_data[layer]['quantile_100'],
                    values=None,
                    weights=None,
                    color=colors[i],
                )
            plt.ylim(-0.5, len(layers) - 0.5)
        else:
            median_line = [
                weights_data[layer]['quantile_50'] if weights_data[layer]['quantile_50'] is not None else weights_data[layer]['mean']
                for layer in layers
            ]
            sns.lineplot(
                x=median_line, y=layers,
                label='Median', linewidth=2,
                orient='y'
            )

            mean_line = [weights_data[layer]['mean'] for layer in layers]
            plt.plot(mean_line, layers, color='orange', linestyle='--', label='Mean')

            percentile_0 = [weights_data[layer]['quantile_0'] for layer in layers]
            plt.plot(percentile_0, layers, color='lightblue', linestyle='--', label='Minimum')

            percentile_25 = [
                weights_data[layer]['quantile_25'] if weights_data[layer]['quantile_25'] is not None else weights_data[layer]['mean']
                for layer in layers
            ]
            plt.plot(percentile_25, layers, color='lightblue', linestyle='--', label='25th Percentile')

            percentile_75 = [
                weights_data[layer]['quantile_75'] if weights_data[layer]['quantile_75'] is not None else weights_data[layer]['mean']
                for layer in layers
            ]
            plt.plot(percentile_75, layers, color='lightblue', linestyle='--', label='75th Percentile')

            percentile_100 = [weights_data[layer]['quantile_100'] for layer in layers]
            plt.plot(percentile_100, layers, color='lightblue', linestyle='--', label='Maximum')

            plt.fill_betweenx(layers, percentile_25, percentile_75, color='lightblue', alpha=0.5, label='Interquartile Range')
            plt.fill_betweenx(layers, percentile_0, percentile_100, color='lightblue', alpha=0.2, label='Range')
            plt.legend()

        scale_type = 'log10' if weights_transform == 'abs_log' else 'absolute'

        plt.yticks(range(len(layers)), layers)
        plt.title(f"{data_suffix} distribution")
        plt.ylabel("Layer Names")
        plt.xlabel(f"Weights Values ({scale_type} scale)")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
