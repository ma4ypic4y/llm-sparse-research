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

    mean_minus_std = mean - std
    mean_plus_std = mean + std

    if mean_minus_std < min_val:
        mean_minus_std = (min_val + mean) / 2
    if mean_plus_std > max_val:
        mean_plus_std = (max_val + mean) / 2

    distributed_dots = [min_val] + _lerp_between(min_val, mean_minus_std, 2) + [mean_minus_std] + \
                       _lerp_between(mean_minus_std, mean, 2) + [mean] + _lerp_between(mean, mean_plus_std, 2) + \
                       [mean_plus_std] + _lerp_between(mean_plus_std, max_val, 2) + [max_val]
    dots_less_cnt = [
        (weights_magnitudes < dot).sum().item()
        for dot in distributed_dots
    ]

    between_dots = [
        (dot_r + dot_l) / 2 for dot_l, dot_r in zip(distributed_dots, distributed_dots[1:])
    ]
    between_dots_cnt = [
        dot_r - dot_l for dot_l, dot_r in zip(dots_less_cnt, dots_less_cnt[1:])
    ]

    return {
        'mean': mean,
        'std': std,
        'log10_mean': weights_magnitudes[weights_magnitudes >= 1e-10].abs().log10().mean().item(),

        'quantile_0': min_val,
        'quantile_25': torch.quantile(weights_magnitudes, 0.25).item() if weights_magnitudes.numel() < 100_000 else None,
        'quantile_50': torch.quantile(weights_magnitudes, 0.5).item() if weights_magnitudes.numel() < 100_000 else None,
        'quantile_75': torch.quantile(weights_magnitudes, 0.75).item() if weights_magnitudes.numel() < 100_000 else None,
        'quantile_100': max_val,

        'between_dots': between_dots,
        'between_dots_cnt': between_dots_cnt,
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
            warmup_steps: int = 0,
    ):
        self.trackable_layers_names = trackable_layers

        self.dump_frequency = dump_frequency
        self.collect_frequency = collect_frequency
        self.output_dir = output_dir

        self.warmup_steps = warmup_steps + (warmup_steps % collect_frequency)

        self.is_initialized = False
        self.model = None

    def _initialize_trackable_layers(self, model: torch.nn.Module):
        """
        Initializes the trackable layers based on the provided layer names.

        Args:
            model: The model whose layers are to be tracked.
        """
        pass

    def on_init_end(self, args, state, control, model: torch.nn.Module, **kwargs):
        super().on_init_end(args, state, control, model=model, **kwargs)

        self.model = model

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

        if self.warmup_steps > 0 and state.global_step < self.warmup_steps:
            return

        # Working with trackable weights
        if not self.is_initialized:
            self._initialize_trackable_layers(self.model)
            self.is_initialized = True

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
            output_dir: str = "./masks_collector_output",
            warmup_steps: int = 0,
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

        super().__init__(trackable_layers, collect_frequency, dump_frequency, output_dir, warmup_steps)

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

        self.model = model

        mask_type = torch.uint8 if self.dump_frequency >= 0 or math.ceil(self.dump_frequency / self.collect_frequency) <= 255 else torch.uint64

        self.prev_masks = []
        self.masks_changes = []

        if self.trackable_layers_names is not None:
            if any(mask not in model.state_dict() for mask in self.trackable_layers_names):
                print("Warning: Some trackable masks are not in the model state_dict, filtering them out.")
                self.trackable_layers_names = [mask for mask in self.trackable_layers_names if mask in model.state_dict()]

            for name in self.trackable_layers_names:
                mask = model.state_dict()[name]
                self.prev_masks.append(mask.data.clone())
                self.masks_changes.append(torch.zeros_like(mask.data, dtype=mask_type))
        else:
            self.trackable_layers_names = []

            for name, param in tqdm(model.named_buffers()):
                if name.endswith("_mask"):
                    self.trackable_layers_names.append(name)
                    self.prev_masks.append(param.data.clone())
                    self.masks_changes.append(torch.zeros_like(param.data, dtype=mask_type))

        print(f"Tracking masks for layers: {self.trackable_layers_names}")

    def _collect_data(self):
        """
        Collects general data about the model's weights and gradients.
        """

        masks_statistic = {}

        params = self.model.state_dict()

        for prev_mask, mask_change, name in zip(self.prev_masks, self.masks_changes, self.trackable_layers_names):
            mask = params[name]
            turned_on_mask = (mask.data != 0) & (prev_mask.data == 0)
            turned_off_mask = (mask.data == 0) & (prev_mask.data != 0)
            changes = turned_on_mask | turned_off_mask

            mask_change += changes
            prev_mask = mask.data.clone()

            masks_statistic[name] = {
                'prune_amount': 1.0 - mask.mean().item(),
                'total_weights': mask.numel(),
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

        self.masks_changes = [torch.zeros_like(mask, dtype=mask.dtype) for mask in self.masks_changes]
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
            warmup_steps: int = 0,
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

        super().__init__(trackable_layers, collect_frequency, dump_frequency, output_dir, warmup_steps)

        self.zero_weight_threshold = zero_weight_threshold
        self.dead_grad_threshold = dead_grad_threshold

        self.weights_statistics = []

    def _initialize_trackable_layers(self, model: torch.nn.Module):
        """
        Initializes the trackable layers based on the provided layer names.

        Args:
            model: The model whose layers are to be tracked.
        """

        self.model = model

        # Working with trackable weights

        if self.trackable_layers_names is not None:
            if any(layer not in model.state_dict() for layer in self.trackable_layers_names):
                print("Warning: Some trackable weights layers are not in the model state_dict, filtering them out.")
                self.trackable_layers_names = [layer for layer in self.trackable_layers_names if layer in model.state_dict()]

        else:
            self.trackable_layers_names = []

            for name, param in tqdm(model.named_parameters()):
                if param.requires_grad:
                    self.trackable_layers_names.append(name)

        print(f"Tracking weights: {self.trackable_layers_names}")

    def _collect_data(self):
        """
        Collects general data about the model's weights and gradients.
        """

        weights_statistics = {}

        params = dict(self.model.named_parameters())

        for name in self.trackable_layers_names:
            param = params[name]

            data, grad = param.data, param.grad

            if name.endswith("_orig"):
                mask = self.model.state_dict()[name.replace("_orig", "_mask")]

                data = data * mask.data
                grad = grad * mask.data

            weights_statistics[name] = _get_param_statistics(
                data, grad,
                zero_weight_threshold=self.zero_weight_threshold,
                dead_grad_threshold=self.dead_grad_threshold
            )
            weights_statistics[name].update({
                'weights_distribution': _get_weights_distribution_statistics(param),
                'gradients_distribution': _get_weights_distribution_statistics(grad),
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


def summarize_statistics(
        masks_collector: MasksStatisticsCollector | None = None,
        weights_collector: WeightsStatisticsCollector | None = None,
) -> dict:
    """
    Summarizes the collected statistics.

    Args:
        masks_collector: An instance of MasksStatisticsCollector.
        weights_collector: An instance of WeightsStatisticsCollector.

    Returns:
        A dictionary with summarized statistics.
    """

    weights_statistics = []
    if weights_collector is not None:
        if weights_collector.output_dir is not None:
            for file in os.listdir(weights_collector.output_dir):
                if not file.endswith(".pt") or file == "collector_data_summary.pt":
                    continue

                data = torch.load(os.path.join(weights_collector.output_dir, file), map_location='cpu')

                weights_statistics += data['weights_statistics']

        weights_statistics += weights_collector.weights_statistics

    mask_changes = {}
    masks_statistics = []
    if masks_collector is not None and masks_collector.trackable_layers_names is not None:
        masks_changes = [torch.zeros_like(mask, dtype=mask.dtype) for mask in masks_collector.masks_changes]

        if masks_collector.output_dir is not None:
            for file in os.listdir(masks_collector.output_dir):
                if not file.endswith(".pt") or file == "collector_data_summary.pt":
                    continue

                data = torch.load(os.path.join(masks_collector.output_dir, file), map_location='cpu')
                print(data)

                for idx, mask_change in enumerate(data['masks_changes']):
                    masks_changes[idx] += mask_change
                masks_statistics += data['masks_statistics']

        for idx, mask_change in enumerate(masks_collector.masks_changes):
            masks_changes[idx] += mask_change

        mask_changes = {name: change for name, change in zip(masks_collector.trackable_layers_names, masks_changes)}
        masks_statistics += masks_collector.masks_statistics

    return {
        'weights_statistics': weights_statistics,
        'masks_changes': mask_changes,
        'masks_statistics': masks_statistics,
        'info': {
            'masks_collector_frequency': masks_collector.collect_frequency if masks_collector is not None else None,
            'weights_collector_frequency': weights_collector.collect_frequency if weights_collector is not None else None,
        },
    }
