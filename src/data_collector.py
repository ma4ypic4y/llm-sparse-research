from __future__ import annotations

import math
import os

from tqdm import tqdm

import transformers
import torch


def _get_param_statistics(param: torch.Tensor, grad: torch.Tensor | None = None, zero_weight_threshold: float = 0.0, dead_grad_threshold: float = 0.0) -> dict:
    """
    Computes statistics for a single parameter tensor.

    Args:
        param: A torch tensor representing the parameter.
        grad: A torch tensor representing the gradient, if available.
        zero_weight_threshold: Threshold for considering a weight as zero.
        dead_grad_threshold: Threshold for considering a gradient as dead.

    Returns:
        A dictionary with statistics for the parameter.
    """
    zero_mask = torch.abs(param) <= zero_weight_threshold
    dead_mask = torch.zeros_like(param, dtype=torch.bool)

    if grad is not None:
        dead_mask = torch.abs(grad) <= dead_grad_threshold

    return {
        'total_weights': param.numel(),
        'zero_weights': zero_mask.sum().item(),
        'dead_weights': dead_mask.sum().item(),
        'zero_dead_weights': (zero_mask & dead_mask).sum().item(),
        'active_weights': (~zero_mask & ~dead_mask).sum().item(),
    }


def _lerp_between(min: float, max: float, between_count: int) -> float:
    """
    Performs linear interpolation between two values.

    Args:
        min: Minimum value.
        max: Maximum value.
        between_count: Number of values to generate between min and max.

    Returns:
        A list of interpolated values.
    """
    delta = (max - min) / (between_count + 1)
    return [min + delta * i for i in range(1, between_count + 1)]


def _get_weights_distribution_statistics(param: torch.Tensor) -> dict:
    """
    Computes distribution statistics for a single parameter tensor.

    Args:
        param: A torch tensor representing the parameter.

    Returns:
        A dictionary with distribution statistics for the parameter.
    """
    weights_magnitudes = param.abs()

    mean = weights_magnitudes.mean().item()
    std = weights_magnitudes.std().item()
    min_val = weights_magnitudes.min().item()
    max_val = weights_magnitudes.max().item()

    if weights_magnitudes.numel() < 100_000:
        return {
            'mean': mean,
            'std': std,
            'log10_mean': weights_magnitudes[weights_magnitudes >= 1e-10].abs().log10().mean().item(),
            'quantile_0': min_val,
            'quantile_25': torch.quantile(weights_magnitudes, 0.25).item(),
            'quantile_50': torch.quantile(weights_magnitudes, 0.5).item(),
            'quantile_75': torch.quantile(weights_magnitudes, 0.75).item(),
            'quantile_100': max_val,
        }

    mean_minus_std = mean - std
    mean_plus_std = mean + std

    if mean_minus_std < min_val:
        mean_minus_std = (min_val + mean) / 2
    if mean_plus_std > max_val:
        mean_plus_std = (max_val + mean) / 2

    distributed_dots = [min_val] + _lerp_between(min_val, mean_minus_std, 3) + [mean_minus_std] + \
                       _lerp_between(mean_minus_std, mean, 10) + [mean] + _lerp_between(mean, mean_plus_std, 10) + \
                       [mean_plus_std] + _lerp_between(mean_plus_std, max_val, 3) + [max_val]

    quantile_75, quantile_50, quantile_25 = min_val, min_val, min_val
    for dot in distributed_dots:
        less_dots = (weights_magnitudes < dot).sum() / weights_magnitudes.numel()
        if less_dots <= 0.75:
            quantile_75 = dot
        if less_dots <= 0.5:
            quantile_50 = dot
        if less_dots <= 0.25:
            quantile_25 = dot

    return {
        'mean': mean,
        'std': std,
        'log10_mean': weights_magnitudes[weights_magnitudes >= 1e-10].abs().log10().mean().item(),
        'quantile_0': min_val,
        'quantile_25': quantile_25,
        'quantile_50': quantile_50,
        'quantile_75': quantile_75,
        'quantile_100': max_val,
    }


class BaseCollector(transformers.TrainerCallback):
    """
    A base class for collector callbacks that initializes trackable layers and directories.
    """

    def __init__(
            self,
            trackable_layers: list[str] | None = None,
            collect_frequency: int = 1,
            dump_frequency: int = -1,
            output_dir: str = "./collector_output",
    ):
        self.trackable_layers_names = trackable_layers

        self.trackable_layers = []

        self.dump_frequency = dump_frequency
        self.collect_frequency = collect_frequency
        self.output_dir = output_dir

    def _initialize_trackable_layers(self, model: torch.nn.Module):
        """
        Initializes the trackable layers based on the provided layer names.

        Args:
            model: The model whose layers are to be tracked.
        """
        pass

    def on_init_end(self, args, state, control, model: torch.nn.Module, **kwargs):
        super().on_init_end(args, state, control, model=model, **kwargs)

        # Working with trackable weights

        self._initialize_trackable_layers(model)

        # Create output directory if it does not exist

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            for file in os.listdir(self.output_dir):
                if file.endswith(".pt"):
                    os.remove(os.path.join(self.output_dir, file))

    def _collect_data(self):
        """
        Collects distribution data about the model's weights and gradients.
        """

        pass

    def _dump_data(self, state: transformers.TrainerState):
        """
        Dumps collected data to disk.

        Args:
            state: Trainer state.
        """

        pass

    def on_optimizer_step(self, args, state, control, **kwargs):
        """
        Called at the end of each training step.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            model: The model being trained.
        """
        super().on_optimizer_step(args, state, control, **kwargs)

        if state.global_step % self.collect_frequency == 0:
            self._collect_data()

        if self.dump_frequency != -1 and state.global_step % self.dump_frequency == 0:
            self._dump_data(state)

    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            model: The model being trained.
        """
        super().on_train_end(args, state, control, **kwargs)

        self._collect_data()
        self._dump_data(state)


class MasksStatisticsCollector(BaseCollector):
    """
    A callback to collect statistics about masks during training.
    """

    def __init__(
            self,
            trackable_layers: list[str] | None = None,
            collect_frequency: int = 1,
            dump_frequency: int = -1,
            output_dir: str = "./masks_collector_output"
    ):
        """
        Initializes the MasksStatisticsCollector.

        Args:
            trackable_layers: A list of layer names to track masks for. (default: None)
            collect_frequency: Frequency of collecting masks during training. (default: 1)
            dump_frequency: Frequency of dumping collected data to disk. If -1, data will be dumped only at the end of training. (default: -1)
            output_dir: Directory to save collected data. (default: "./masks_collector_output")

        Note:
            If `trackable_layers` is provided, it will only track the specified layers. Otherwise, all layers of specific type will be tracked.
        """
        if math.ceil(dump_frequency / collect_frequency) > 255:
            print("Warning: over 255 collection steps per dump step. Unable to use uint8 for masks changes, using uint64 instead, which will increase memory usage.")
        if dump_frequency < 0:
            print("Warning: dump_frequency is less than 0, this will automatically use unint64 for masks changes instead of uint8, which will increase memory usage.")

        super().__init__(trackable_layers, collect_frequency, dump_frequency, output_dir)

        self.masks_statistics = []

        self.prev_masks = []
        self.masks_changes = []

    def _initialize_trackable_layers(self, model: torch.nn.Module):
        """
        Initializes the trackable layers based on the provided layer names.

        Args:
            model: The model whose layers are to be tracked.
        """

        # Working with trackable masks

        mask_type = torch.uint8 if self.dump_frequency >= 0 or math.ceil(self.dump_frequency / self.collect_frequency) <= 255 else torch.uint64

        if self.trackable_layers_names is not None:
            if any(mask not in model.state_dict() for mask in self.trackable_layers_names):
                print("Warning: Some trackable masks are not in the model state_dict, filtering them out.")
                self.trackable_layers_names = [mask for mask in self.trackable_layers_names if mask in model.state_dict()]
            self.trackable_layers = [model.state_dict()[mask] for mask in self.trackable_layers_names]
        else:
            self.trackable_layers_names = []
            self.trackable_layers = []

            for name, param in tqdm(model.named_buffers()):
                if name.endswith("_mask"):
                    self.trackable_layers_names.append(name)
                    self.trackable_layers.append(param)

        for name, layer in zip(self.trackable_layers_names, self.trackable_layers):
            self.prev_masks.append(layer.data.clone())
            self.masks_changes.append(torch.zeros_like(layer.data, dtype=mask_type))

    def _collect_data(self):
        """
        Collects general data about the model's weights and gradients.
        """

        masks_statistic = {}

        for mask, prev_mask, mask_change, name in zip(self.trackable_layers, self.prev_masks, self.masks_changes, self.trackable_layers_names):
            turned_on_mask = (mask.data != 0) & (prev_mask.data == 0)
            turned_off_mask = (mask.data == 0) & (prev_mask.data != 0)
            changes = turned_on_mask | turned_off_mask

            mask_change += changes
            prev_mask.copy_(mask.data)

            masks_statistic[name] = {
                'prune_amount': mask.mean().item(),
                'turned_on_amount': turned_on_mask.sum().item(),
                'turned_off_amount': turned_off_mask.sum().item()
            }


        self.masks_statistics.append(masks_statistic)

    def _dump_data(self, state: transformers.TrainerState):
        """
        Dumps collected data to disk.

        Args:
            state: Trainer state.
        """

        if self.output_dir is not None:
            torch.save({
                'masks_changes': self.masks_changes,
                'masks_statistics': self.masks_statistics,
            }, os.path.join(self.output_dir, f'collector_data_step_{state.global_step}.pt'))

        self.masks_changes = [torch.zeros_like(mask.data, dtype=mask.dtype) for mask in self.trackable_layers]
        self.masks_statistics = []


class WeightsStatisticsCollector(BaseCollector):
    """
    A callback to collect local statistics about weights and gradients during training.
    """

    def __init__(
            self,
            zero_weight_threshold: float = 0.0,
            dead_grad_threshold: float = 0.0,
            trackable_layers: list[str] | None = None,
            collect_frequency: int = 1,
            dump_frequency: int = -1,
            output_dir: str = "./weights_statistics_collector_output",
    ):
        """
        Initializes the WeightsStatisticsCollector.

        Args:
            zero_weight_threshold: A threshold below which weights are considered zero. (default: 0.0)
            dead_grad_threshold: A threshold below which gradients are considered to be representing dead weights. (default: 0.0)
            trackable_layers: A list of layer names to track weights for. (default: None)
            collect_frequency: Frequency of collecting weights and gradients during training. (default: 1)
            dump_frequency: Frequency of dumping collected data to disk. If -1, data will be dumped only at the end of training. (default: -1)
            output_dir: Directory to save collected data. (default: "./weights_collector_output")

        Note:
            If zero is used as a threshold, it will make an absolute comparison with 0 (`weight == 0.0`), which is suitable for activation-based pruning methods.
            If `trackable_layers` is provided, it will only track the specified layers. Otherwise, all layers of specific type will be tracked.
        """

        super().__init__(trackable_layers, collect_frequency, dump_frequency, output_dir)

        self.zero_weight_threshold = zero_weight_threshold
        self.dead_grad_threshold = dead_grad_threshold

        self.layer_masks = []

        self.weights_statistics = []

    def _initialize_trackable_layers(self, model: torch.nn.Module):
        """
        Initializes the trackable layers based on the provided layer names.

        Args:
            model: The model whose layers are to be tracked.
        """

        # Working with trackable weights

        if self.trackable_layers_names is not None:
            if any(layer not in model.state_dict() for layer in self.trackable_layers_names):
                print("Warning: Some trackable weights layers are not in the model state_dict, filtering them out.")
                self.trackable_layers_names = [layer for layer in self.trackable_layers_names if layer in model.state_dict()]

            self.trackable_layers = [model.state_dict()[layer] for layer in self.trackable_layers_names if layer]
            self.layer_masks = [model.state_dict()[layer.replace("_orig", "_mask")] if layer.endswith("_orig") else None
                                for layer in self.trackable_layers_names]
        else:
            self.trackable_layers_names = []
            self.trackable_layers = []

            for name, param in tqdm(model.named_parameters()):
                if param.requires_grad:
                    self.trackable_layers_names.append(name)
                    self.trackable_layers.append(param)
                    self.layer_masks.append(model.state_dict()[name.replace("_orig", "_mask")] if name.endswith("_orig") else None)

    def _collect_data(self):
        """
        Collects general data about the model's weights and gradients.
        """

        weights_statistics = {}

        for name, param, mask in zip(self.trackable_layers_names, self.trackable_layers, self.layer_masks):
            data, grad = param.data, param.grad

            if mask is not None:
                data = data * mask.data
                grad = grad * mask.data if grad is not None else None

            weights_statistics[name] = _get_param_statistics(
                data, grad,
                zero_weight_threshold=self.zero_weight_threshold,
                dead_grad_threshold=self.dead_grad_threshold
            )
            weights_statistics[name].update({
                'weights_distribution': _get_weights_distribution_statistics(param),
                'gradients_distribution': _get_weights_distribution_statistics(grad) if grad is not None else None,
            })

        self.weights_statistics.append(weights_statistics)

    def _dump_data(self, state: transformers.TrainerState):
        """
        Dumps collected data to disk.

        Args:
            state: Trainer state.
        """

        if self.output_dir is not None:
            torch.save({
                'weights_statistics': self.weights_statistics,
            }, os.path.join(self.output_dir, f'collector_data_step_{state.global_step}.pt'))

        self.weights_statistics = []


class WeightsGradientsCollector(BaseCollector):
    """
    A callback to collect weights and gradients during training.
    """

    def __init__(
            self,
            trackable_layers: list[str] | None = None,
            dump_frequency: int = 100,
            output_dir: str = "./weights_collector_output",
    ):
        """
        Initializes the WeightsGradientsCollector.

        Args:
            trackable_layers: A list of layer names to track weights for. (default: None)
            dump_frequency: Frequency of dumping collected data to disk. If -1, data will be dumped only at the end of training. (default: -1)
            output_dir: Directory to save collected data. (default: "./weights_gradients_collector_output")

        Note:
            If `trackable_layers` is provided, it will only track the specified layers. Otherwise, all layers of specific type will be tracked.
        """

        super().__init__(trackable_layers, dump_frequency, dump_frequency, output_dir)

        self.last_weight_dump = None

    def _initialize_trackable_layers(self, model: torch.nn.Module):
        """
        Initializes the trackable layers based on the provided layer names.

        Args:
            model: The model whose layers are to be tracked.
        """

        # Working with trackable weights

        if self.trackable_layers_names is not None:
            if any(layer not in model.state_dict() for layer in self.trackable_layers_names):
                print("Warning: Some trackable weights layers are not in the model state_dict, filtering them out.")
                self.trackable_layers_names = [layer for layer in self.trackable_layers_names if layer in model.state_dict()]

            self.trackable_layers = [model.state_dict()[layer] for layer in self.trackable_layers_names if layer]
        else:
            self.trackable_layers_names = []
            self.trackable_layers = []

            for name, param in tqdm(model.named_parameters()):
                if param.requires_grad:
                    self.trackable_layers_names.append(name)
                    self.trackable_layers.append(param)

    def _collect_data(self):
        """
        Collects distribution data about the model's weights and gradients.
        """

        self.last_weight_dump = {}

        for name, param in zip(self.trackable_layers_names, self.trackable_layers):
            self.last_weight_dump[name] = {
                'weights': param.data.clone(),
                'gradients': param.grad.clone() if param.grad is not None else None
            }

    def _dump_data(self, state: transformers.TrainerState):
        """
        Dumps collected data to disk.

        Args:
            state: Trainer state.
        """

        if self.output_dir is not None:
            torch.save({
                'weight_dump': self.last_weight_dump
            }, os.path.join(self.output_dir, f'collector_data_step_{state.global_step}.pt'))

        self.last_weight_dump = None


class GeneralStatisticsCollector(BaseCollector):
    """
    A callback to collect general statistics about model weights and gradients during training.
    """

    def __init__(
            self,
            weights_statistics_collector: WeightsStatisticsCollector | None = None,
            collect_frequency: int = 1,
            dump_frequency: int = -1,
            output_dir: str = "./general_collector_output",
    ):
        """
        Initializes the GeneralStatisticsCollector.

        Args:
            weights_statistics_collector: A collector for weights statistics. (default: None)
            trackable_layers: A list of layer names to track weights for. (default: None)
            collect_frequency: Frequency of collecting weights and gradients during training. (default: 1)
            dump_frequency: Frequency of dumping collected data to disk. If -1, data will be dumped only at the end of training. (default: -1)
            output_dir: Directory to save collected data. (default: "./general_collector_output")
        """

        self.is_local_collector = weights_statistics_collector is None

        if self.is_local_collector:
            weights_statistics_collector = WeightsStatisticsCollector(
                collect_frequency=collect_frequency,
                output_dir=None
            )

        assert collect_frequency >= weights_statistics_collector.collect_frequency and collect_frequency % weights_statistics_collector.collect_frequency == 0, \
            "collect_frequency must be multiple of weights_statistics_collector.collect_frequency."

        super().__init__(weights_statistics_collector.trackable_layers_names, collect_frequency, dump_frequency, output_dir)

        self.weights_statistics_collector = weights_statistics_collector

        self.model_statistics = []

    def _initialize_trackable_layers(self, model: torch.nn.Module):
        """
        Initializes the trackable layers based on the provided layer names.

        Args:
            model: The model whose layers are to be tracked.
        """
        # Working with trackable weights

        if self.is_local_collector:
            self.weights_statistics_collector._initialize_trackable_layers(model)

        assert self.weights_statistics_collector.trackable_layers != [], "WeightsStatisticsCollector must go first in the callbacks list."

        self.trackable_layers_names = self.weights_statistics_collector.trackable_layers_names

    def _collect_data(self):
        """
        Collects general data about the model's weights and gradients.
        """

        if self.is_local_collector:
            self.weights_statistics_collector._collect_data()

        model_statistics = {
            'total_weights': 0.0,
            "dead_weights": 0.0,
            "zero_weights": 0.0,
            "active_weights": 0.0
        }
        weights_statistics = self.weights_statistics_collector.weights_statistics[-1]

        for name in self.trackable_layers_names:
            model_statistics['total_weights'] += weights_statistics[name]['total_weights']
            model_statistics["zero_weights"] += weights_statistics[name]["zero_weights"]
            model_statistics["dead_weights"] += weights_statistics[name]["dead_weights"]
            model_statistics["active_weights"] += weights_statistics[name]["active_weights"]

        self.model_statistics.append(model_statistics)

    def _dump_data(self, state: transformers.TrainerState):
        """
        Dumps collected data to disk.

        Args:
            state: Trainer state.
        """

        if self.output_dir is not None:
            torch.save({
                'model_statistics': self.model_statistics,
            }, os.path.join(self.output_dir, f'collector_data_step_{state.global_step}.pt'))

        self.model_statistics = []


def summarize_statistics(
        masks_collector: MasksStatisticsCollector | None = None,
        weights_collector: WeightsStatisticsCollector | None = None,
        weights_gradients_collector: WeightsGradientsCollector | None = None,
        general_collector: GeneralStatisticsCollector | None = None,
) -> dict:
    """
    Summarizes the collected statistics.

    Args:
        masks_collector: An instance of MasksStatisticsCollector.
        weights_collector: An instance of WeightsStatisticsCollector.
        weights_gradients_collector: An instance of WeightsGradientsCollector.
        general_collector: An instance of GeneralStatisticsCollector.

    Returns:
        A dictionary with summarized statistics.
    """

    model_statistics = []
    if general_collector is not None:
        if general_collector.output_dir is not None:
            for file in os.listdir(general_collector.output_dir):
                if not file.endswith(".pt") or file == "collector_data_summary.pt":
                    continue

                data = torch.load(os.path.join(general_collector.output_dir, file))

                model_statistics += data['model_statistics']

        model_statistics += general_collector.model_statistics

    weights_statistics = []
    if weights_collector is not None:
        if weights_collector.output_dir is not None:
            for file in os.listdir(weights_collector.output_dir):
                if not file.endswith(".pt") or file == "collector_data_summary.pt":
                    continue

                data = torch.load(os.path.join(weights_collector.output_dir, file))

                weights_statistics += data['weights_statistics']

        weights_statistics += weights_collector.weights_statistics

    mask_changes = {}
    masks_statistics = []
    if masks_collector is not None:
        masks_changes = [torch.zeros_like(mask.data, dtype=mask.dtype) for mask in masks_collector.trackable_layers]

        if masks_collector.output_dir is not None:
            for file in os.listdir(masks_collector.output_dir):
                if not file.endswith(".pt") or file == "collector_data_summary.pt":
                    continue

                data = torch.load(os.path.join(masks_collector.output_dir, file))

                for idx, mask_change in enumerate(data['masks_changes']):
                    masks_changes[idx] += mask_change
                masks_statistics += data['masks_statistics']

        for idx, mask_change in enumerate(masks_collector.masks_changes):
            masks_changes[idx] += mask_change

        mask_changes = {name: change for name, change in zip(masks_collector.trackable_layers_names, mask_changes)}
        masks_statistics += masks_collector.masks_statistics

    weights_dumps = []
    if weights_gradients_collector is not None:
        if weights_gradients_collector.output_dir is not None:
            for file in os.listdir(weights_gradients_collector.output_dir):
                if not file.endswith(".pt") or file == "collector_data_summary.pt":
                    continue

                data = torch.load(os.path.join(weights_gradients_collector.output_dir, file))

                if data['weight_dump'] is not None:
                    weights_dumps.append(data['weight_dump'])

        if weights_gradients_collector.last_weight_dump is not None:
            weights_dumps.append(weights_gradients_collector.last_weight_dump)

    return {
        'model_statistics': model_statistics,
        'weights_statistics': weights_statistics,
        'masks_changes': mask_changes,
        'masks_statistics': masks_statistics,
        'weight_dumps': weights_dumps,
        'info': {
            'masks_collector_frequency': masks_collector.collect_frequency if masks_collector is not None else None,
            'weights_collector_frequency': weights_collector.collect_frequency if weights_collector is not None else None,
            'weights_gradients_collector_frequency': weights_gradients_collector.collect_frequency if weights_gradients_collector is not None else None,
            'general_collector_frequency': general_collector.collect_frequency if general_collector is not None else None,
        },
    }
