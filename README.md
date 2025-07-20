# Sparse Weights Training

Dynamic weight sparsification during GPT2 training on Tiny Shakespeare with magnitude-based pruning and comprehensive metrics analysis.

## Quick Start

```bash
uv venv -p python3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python runner.py --config config.yaml
```

Run experiments (for different sparsity levels):
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

## Project Structure

```
sparse_weights/
├── runner.py                   # Main training script
├── config.yaml                 # Training configuration
├── requirements.txt            # Dependencies
├── analyze_results.py          # Post-training analysis with plots
├── check_sparsity.py           # Quick sparsity checker
├── src/
│   ├── auto_pruning.py         # Adaptive pruning schedule calculation
│   ├── pruning.py              # WeightPruner implementation
│   ├── metrics.py              # Comprehensive sparsity metrics
│   ├── sparsity_utils.py       # Analysis utilities
│   ├── training.py             # Training functions
│   ├── data.py                 # Tiny Shakespeare loader
│   └── utils.py                # Config and logging
└── exp/                        # Output directory (model + metrics)
```

## Configuration Parameters

**Pruning (Adaptive):**
- `target_sparsity: 0.5` - Final sparsity level (50%)
- `warmup_ratio: 0.15` - Warmup phase (15% of steps, no pruning)
- `final_prune_ratio: 0.85` - Complete pruning by 85% of training
- `prune_applications: 10` - Number of pruning steps

**Training:**
- `epochs: 10` - Training epochs (auto-adjusts pruning schedule)
- `batch_size: 16` - Batch size (auto-adjusts pruning schedule)
- `lr: 0.001` - Learning rate

## Pruning Strategy

**Magnitude-based L1 unstructured pruning:**
1. **Warmup** (0-15%): Normal training, no pruning
2. **Active pruning** (15-85%): Gradual sparsity increase every N steps
3. **Finalization** (85-100%): Fixed sparsity, continued training

**Auto-adaptation:** Pruning schedule recalculates based on `epochs × batches_per_epoch`.

## WandB Metrics

**Core:**
- `sparsity/overall` - Prunable weights sparsity
- `pruning/sparsity` - Applied sparsity level
- `weight_revival/revival_rate` - Fraction of zero weights that become non-zero

**Analysis:**
- `layer_sparsity/*` - Per-layer sparsity (76 layers)
- `dead_neurons/*` - Neurons with all-zero connections
- `effective_rank/*` - Matrix rank degradation
- `zero_grad_norm/*` - Gradients of zero weights

**Trends:**
- `trends/sparsity_slope` - Sparsity growth rate
- `trends/avg_revival_rate` - Weight revival dynamics

## Analysis Tools (not checked, full AI generated code)

```bash
# Quick sparsity check
python check_sparsity.py --model_dir exp

# Detailed analysis with plots
python analyze_results.py --output_dir exp --wandb_run_id RUN_ID
```

## Key Research Metrics

**Weight Revival Rate** - Measures irreversible weight death (core hypothesis)
**Dead Neurons Ratio** - Tracks neuron degradation
**Effective Rank** - Quantifies model expressivity loss