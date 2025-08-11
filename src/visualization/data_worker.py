from __future__ import annotations

import torch

from src.hooks.data_collector import WeightsStatisticsCollector, MasksStatisticsCollector, summarize_statistics


class DataWorker:
    """
    A class to handle data extraction and processing for model weights and gradients.
    """

    def __init__(self, zero_weight_threshold: float = 0.0, dead_grad_threshold: float = 0.0):
        """
        Initializes the DataWorker.

        Args:
            zero_weight_threshold: A threshold below which weights are considered zero. (default: 0.0)
            dead_grad_threshold: A threshold below which gradients are considered to be representing dead weights. (default: 0.0)

        Note:
            If zero is used as a threshold, it will make an absolute comparison with 0 (`weight == 0.0`), which is suitable for activation-based pruning methods.
        """
        self.zero_weight_threshold = zero_weight_threshold
        self.dead_grad_threshold = dead_grad_threshold

        self._weights_data, self._grad_data = None, None
        self._model_statistics, self._weights_statistics = None, None

        self._masks_statistics = None
        self._masks_changes = None

        self._collection_frequency = None
        self._dump_frequency = None

    def _load_collector_dict(self, collected_data: dict) -> None:
        """
        Loads the collected data into the DataWorker.

        Args:
            collected_data: The collected data dictionary.
        """

        self._weights_statistics = collected_data['weights_statistics']
        self._weights_data = [
            {
                name: collected_data['weights_statistics'][i][name]['weights_distribution']
                for name in collected_data['weights_statistics'][i].keys()
            }
            for i in range(len(collected_data['weights_statistics']))
        ]
        self._grad_data = [
            {
                name: collected_data['weights_statistics'][i][name]['gradients_distribution']
                for name in collected_data['weights_statistics'][i].keys()
            }
            for i in range(len(collected_data['weights_statistics']))
        ]

        self._masks_statistics = collected_data['masks_statistics']
        self._masks_changes = collected_data['masks_changes']

        self._collectors_info = collected_data['info']

    def load_model(self, model: torch.nn.Module, force_exact_quantiles: bool = False) -> DataWorker:
        """
        Loads a new model into the DataWorker.

        Args:
            model: The new model to load.
            force_exact_quantiles: Whether to force exact quantiles computation.

        Returns:
            The updated DataWorker instance.
        """

        m_collector = MasksStatisticsCollector(
            trackable_layers=None,
            collect_frequency=1,
            output_dir=None,
        )
        s_collector = WeightsStatisticsCollector(
            zero_weight_threshold=self.zero_weight_threshold,
            dead_grad_threshold=self.dead_grad_threshold,
            force_exact_quantiles=force_exact_quantiles,
            trackable_layers=None,
            collect_frequency=1,
            output_dir=None,
        )

        m_collector._initialize_trackable_layers(model)
        s_collector._initialize_trackable_layers(model)

        m_collector._collect_data()
        s_collector._collect_data()

        self._load_collector_dict(summarize_statistics(m_collector, s_collector))

        return self

    def load_stats(self, stats: dict) -> DataWorker:
        """
        Loads statistics data into the DataWorker.

        Args:
            stats: A dictionary containing the statistics data.

        Returns:
            The updated DataWorker instance.
        """

        self._load_collector_dict(stats)

        return self

    def load_collector(self, collector_dump_path: str = './collector_output/collector_data_summary.pt') -> DataWorker:
        """
        Loads a DataWorker from a collector dump file.

        Args:
            collector_dump_path: Path to the collector dump file.

        Returns:
            The updated DataWorker instance.
        """

        self._load_collector_dict(torch.load(collector_dump_path))

        return self

    def get_weights_data(self) -> dict[str, torch.Tensor]:
        """
        Returns the extracted weights data.

        Returns:
            A dictionary where keys are parameter names and values are the corresponding weights as numpy arrays.
        """
        assert self._weights_data is not None, "Weights data has not been extracted yet. Call load_model() or load_collector() first."

        return self._weights_data

    def get_grad_data(self) -> dict[str, torch.Tensor]:
        """
        Returns the extracted gradients data.

        Returns:
            A dictionary where keys are parameter names and values are the corresponding gradients as numpy arrays.
        """
        assert self._grad_data is not None, "Gradients data has not been extracted yet. Call load_model() or load_collector() first."

        return self._grad_data

    def get_weights_statistics(self) -> dict[str, torch.Tensor]:
        """
        Returns the weights statistics.

        Returns:
            A dictionary where keys are parameter names and values are dictionaries with statistics for each parameter.
        """
        assert self._weights_statistics is not None, "Weights statistics have not been computed yet. Call load_model() or load_collector() first."

        return self._weights_statistics

    def get_masks_changes(self) -> list[torch.Tensor]:
        """
        Returns the masks changes.

        Returns:
            A list of tensors representing the changes in masks.
        """
        assert self._masks_changes is not None, "Masks changes have not been computed yet. Call load_model() or load_collector() first."

        return self._masks_changes

    def get_masks_statistics(self) -> list[dict[str, torch.Tensor]]:
        """
        Returns the masks statistics.

        Returns:
            A list of dictionaries where each dictionary contains statistics for each mask.
        """
        assert self._masks_statistics is not None, "Masks statistics have not been computed yet. Call load_model() or load_collector() first."

        return self._masks_statistics

    def get_collectors_info(self) -> dict:
        """
        Returns the information about the collectors used.

        Returns:
            A dictionary containing information about the collectors.
        """
        assert self._collectors_info is not None, "Collectors info has not been computed yet. Call load_model() or load_collector() first."

        return self._collectors_info
