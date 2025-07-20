#!/bin/bash

# Sparsity Experiments Runner
# Runs sequential experiments with different target sparsity levels

set -e  # Exit on any error

# Array of sparsity levels to test
SPARSITY_LEVELS=(0.20 0.35 0.50 0.65 0.80 0.90)

# Base config file
BASE_CONFIG="config.yaml"
TEMP_CONFIG="config_temp.yaml"

echo "🚀 Starting sparsity experiments..."
echo "Target sparsity levels: ${SPARSITY_LEVELS[@]}"
echo "======================================"

# Check if base config exists
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "❌ Error: Base config file '$BASE_CONFIG' not found!"
    exit 1
fi

# Function to update sparsity in config
update_sparsity() {
    local sparsity=$1
    local temp_config=$2

    # Create temp config with updated sparsity
    sed "s/target_sparsity: [0-9.]\+/target_sparsity: $sparsity/" "$BASE_CONFIG" > "$temp_config"

    echo "📝 Updated config with target_sparsity: $sparsity"
}

# Main experiment loop
for sparsity in "${SPARSITY_LEVELS[@]}"; do
    echo ""
    echo "🎯 Starting experiment with ${sparsity} sparsity (${sparsity/0./}%)"
    echo "================================================="

    # Update config for current sparsity level
    update_sparsity "$sparsity" "$TEMP_CONFIG"

    # Run training
    echo "▶️  Running training..."
    python runner.py --config "$TEMP_CONFIG"

    if [[ $? -eq 0 ]]; then
        echo "✅ Experiment with ${sparsity} sparsity completed successfully"
    else
        echo "❌ Experiment with ${sparsity} sparsity failed!"
        break
    fi

    echo "⏱️  Waiting 10 seconds before next experiment..."
    sleep 10
done

# Cleanup
if [[ -f "$TEMP_CONFIG" ]]; then
    rm "$TEMP_CONFIG"
    echo "🧹 Cleaned up temporary config file"
fi

echo ""
echo "🎉 All experiments completed!"
echo "Check WandB dashboard for results comparison"