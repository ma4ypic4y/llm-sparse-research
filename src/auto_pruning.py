"""
Adaptive pruning with automatic parameter calculation
"""

import logging
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader


def calculate_pruning_schedule(
    total_steps: int,
    warmup_ratio: float = 0.15,
    final_prune_ratio: float = 0.85,
    prune_applications: int = 10
) -> Tuple[int, int, int]:
    """
    Calculate pruning parameters based on total training steps

    Args:
        total_steps: Total training steps
        warmup_ratio: Fraction of steps for warmup (0.15 = 15%)
        final_prune_ratio: Fraction of steps to reach target sparsity (0.85 = 85%)
        prune_applications: Number of pruning applications

    Returns:
        Tuple (warmup_steps, final_prune_step, prune_freq)
    """

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    final_prune_step = int(total_steps * final_prune_ratio)

    pruning_steps = final_prune_step - warmup_steps

    if prune_applications > 0:
        prune_freq = max(1, pruning_steps // prune_applications)
    else:
        prune_freq = pruning_steps // 10

    return warmup_steps, final_prune_step, prune_freq


def auto_configure_pruning(config: Dict[str, Any], train_loader: DataLoader) -> Dict[str, Any]:
    """
    Automatically configure pruning parameters based on training setup

    Args:
        config: Experiment configuration
        train_loader: Training data loader

    Returns:
        Updated configuration with calculated pruning parameters
    """
    logger = logging.getLogger('sparse_weights.auto_pruning')

    training_config = config.get('training', {})
    epochs = training_config.get('epochs', 10)

    batches_per_epoch = len(train_loader)
    total_steps = epochs * batches_per_epoch

    logger.info(f"Training setup: {epochs} epochs, {batches_per_epoch} batches/epoch, {total_steps} total steps")

    pruning_config = config.get('pruning', {})

    # Get adaptive parameters or use defaults
    warmup_ratio = pruning_config.get('warmup_ratio', 0.15)
    final_prune_ratio = pruning_config.get('final_prune_ratio', 0.85)
    prune_applications = pruning_config.get('prune_applications', 10)

    # Calculate absolute values
    warmup_steps, final_prune_step, prune_freq = calculate_pruning_schedule(
        total_steps, warmup_ratio, final_prune_ratio, prune_applications
    )

    logger.info(f"Auto-calculated pruning: warmup={warmup_steps}, final={final_prune_step}, freq={prune_freq}")

    # Update configuration
    updated_config = config.copy()
    updated_pruning = updated_config.get('pruning', {}).copy()

    updated_pruning.update({
        'warmup_steps': warmup_steps,
        'final_prune_step': final_prune_step,
        'prune_freq': prune_freq,
        '_auto_calculated': True,
        '_total_steps': total_steps,
        '_batches_per_epoch': batches_per_epoch,
    })

    updated_config['pruning'] = updated_pruning
    return updated_config


def print_pruning_schedule(config: Dict[str, Any]):
    """Print calculated pruning schedule"""
    logger = logging.getLogger('sparse_weights.auto_pruning')

    pruning = config.get('pruning', {})
    training = config.get('training', {})

    if not pruning.get('_auto_calculated'):
        logger.info("Using manual pruning configuration")
        return

    epochs = training.get('epochs', 0)
    batches_per_epoch = pruning.get('_batches_per_epoch', 0)
    total_steps = pruning.get('_total_steps', 0)

    warmup_steps = pruning.get('warmup_steps', 0)
    final_prune_step = pruning.get('final_prune_step', 0)
    prune_freq = pruning.get('prune_freq', 0)
    target_sparsity = pruning.get('target_sparsity', 0)

    logger.info("="*60)
    logger.info("AUTO-CALCULATED PRUNING SCHEDULE")
    logger.info("="*60)
    logger.info(f"Training: {epochs} epochs, {batches_per_epoch} batches/epoch, {total_steps} total steps")
    logger.info(f"Warmup: steps 1-{warmup_steps} ({warmup_steps/total_steps:.1%})")
    logger.info(f"Active pruning: steps {warmup_steps+1}-{final_prune_step}")
    logger.info(f"Pruning frequency: every {prune_freq} steps")
    logger.info(f"Target sparsity: {target_sparsity:.1%}")

    # Show schedule preview
    applications = max(1, (final_prune_step - warmup_steps) // prune_freq)
    sparsity_increment = target_sparsity / applications if applications > 0 else 0

    logger.info("Schedule preview:")
    current_sparsity = 0
    for i in range(min(3, applications)):
        step = warmup_steps + (i + 1) * prune_freq
        current_sparsity += sparsity_increment
        epoch_approx = step / batches_per_epoch if batches_per_epoch > 0 else 0
        logger.info(f"  Step {step} (~epoch {epoch_approx:.1f}): sparsity → {current_sparsity:.1%}")

    if applications > 3:
        logger.info(f"  ... ({applications - 3} more applications)")
        logger.info(f"  Step {final_prune_step}: sparsity → {target_sparsity:.1%}")

    logger.info("="*60)


def validate_pruning_config(config: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate pruning configuration

    Returns:
        Tuple (is_valid, warnings_list)
    """
    warnings = []
    pruning = config.get('pruning', {})

    warmup_steps = pruning.get('warmup_steps', 0)
    final_prune_step = pruning.get('final_prune_step', 0)
    prune_freq = pruning.get('prune_freq', 1)
    total_steps = pruning.get('_total_steps', 0)

    if warmup_steps >= final_prune_step:
        warnings.append("warmup_steps >= final_prune_step")

    if final_prune_step > total_steps:
        warnings.append("final_prune_step > total_steps")

    # Check ratios
    if total_steps > 0:
        warmup_ratio = warmup_steps / total_steps
        if warmup_ratio < 0.05:
            warnings.append("Very short warmup (<5% of steps)")
        elif warmup_ratio > 0.3:
            warnings.append("Very long warmup (>30% of steps)")

        prune_ratio = final_prune_step / total_steps
        if prune_ratio > 0.95:
            warnings.append("Pruning until very end (>95% of steps)")

    # Check number of applications
    if final_prune_step > warmup_steps:
        applications = (final_prune_step - warmup_steps) // prune_freq
        if applications < 3:
            warnings.append("Too few pruning applications (<3)")
        elif applications > 50:
            warnings.append("Too many pruning applications (>50)")

    return len(warnings) == 0, warnings