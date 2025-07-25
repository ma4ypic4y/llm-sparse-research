Run experiments (for different sparsity levels):
```bash
chmod +x run_experiments.sh
./run_experiments.sh
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
