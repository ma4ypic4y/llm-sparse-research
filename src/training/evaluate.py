import os
import math

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from ..hooks import summarize_statistics
from ..visualization import Visualizer, DataWorker

class PerplexityEvaluator:
    def __init__(self, eval_steps: int = 1):
        self._collected_metrics = []
        self.eval_steps = eval_steps

    def evaluate(self, eval_preds):
        logits, labels = eval_preds.predictions, eval_preds.label_ids

        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = labels.view(-1)
        loss = F.cross_entropy(torch.tensor(shift_logits), torch.tensor(shift_labels), ignore_index=-100).item()

        perplexity = math.exp(loss) if loss < 300 else float("inf")
        self._collected_metrics.append(perplexity)

        return {"perplexity": perplexity}

    def get_collected_metrics(self) -> list:
        return self._collected_metrics

    def reset_collected_metrics(self):
        self._collected_metrics = []

class Summarize:
    def __init__(self, path: str, s_collector, m_collector, p_evaluator):
        self.s_collector = s_collector
        self.m_collector = m_collector
        self.p_evaluator = p_evaluator

        self.path = path

        self.stats = None

    def make_summarization(self, file_name: str | None):
        try:
            self.stats = summarize_statistics(self.m_collector, self.s_collector)
        except Exception as e:
            print(f"Error during data collection: {e}")
            return

        if self.path is not None and file_name is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            torch.save(self.stats, f'{self.path}/{file_name}')

    def make_plots(self):
        if self.stats is None:
            print("Nothing to visualize: make summarization first")
            return

        if self.p_evaluator is not None:
            try:
                eval_metrics = self.p_evaluator.get_collected_metrics()
                plt.plot([i * self.p_evaluator.eval_steps for i in range(len(eval_metrics))], eval_metrics)
                plt.title("Perplexity")
                plt.xlabel("Step")
                plt.ylabel("Perplexity value")
                plt.gca().set_yscale('log')
                plt.savefig(f"{self.path}/perplexity.png", dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error during data visualization: {e}")

        worker = DataWorker().load_stats(self.stats)
        visualizer = Visualizer(worker)

        if self.s_collector is not None:
            try:
                visualizer.visualize_weights_statistics(sort_by='depth', slice_direction='layers', slice_position=-1)
                plt.savefig(f"{self.path}/zero-dead-prune-end.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_weights_statistics(sort_by='depth', slice_direction='layers', slice_position=0)
                plt.savefig(f"{self.path}/zero-dead-prune-begin.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_weights_distribution(weights_transform='abs_log', sort_by='layer_type', plot_type='violine', slice_direction='layers', slice_position=-1)
                plt.savefig(f"{self.path}/weights-prune-end.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_weights_distribution(weights_transform='abs_log', sort_by='layer_type', plot_type='violine', slice_direction='layers', slice_position=0)
                plt.savefig(f"{self.path}/weights-prune-begin.png", dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error during data visualization: {e}")

        if self.m_collector is not None:
            try:
                visualizer.visualize_masks_summary_statistics(sort_by='depth', slice_direction='layers')
                plt.savefig(f"{self.path}/masks-summary-layers.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_masks_summary_statistics(sort_by='depth', slice_direction='steps')
                plt.savefig(f"{self.path}/masks-summary-steps.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_masks_flick_distribution(weights_transform='norm', sort_by='depth', plot_type='violine')
                plt.savefig(f"{self.path}/masks-flick-distribution.png", dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error during data visualization: {e}")

def infer_model(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    outputs = model.generate(**inputs, max_length=500, top_k=200, temperature=0.8).cpu()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

