#!/bin/bash

set -e  # Exit on any error

SPARSITY_LEVELS=(0.20 0.35 0.50 0.65 0.80 0.90)

BASE_CONFIG="config.yaml"
TEMP_CONFIG="config_temp.yaml"

echo "Target sparsity levels: ${SPARSITY_LEVELS[@]}"
echo "======================================"

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "❌ Error: Base config file '$BASE_CONFIG' not found!"
    exit 1
fi

update_sparsity() {
    local sparsity=$1
    local temp_config=$2

    sed "s/target_sparsity: [0-9.]\+/target_sparsity: $sparsity/" "$BASE_CONFIG" > "$temp_config"

    echo "📝 Updated config with target_sparsity: $sparsity"
}

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
done

# Cleanup
if [[ -f "$TEMP_CONFIG" ]]; then
    rm "$TEMP_CONFIG"
fi

echo ""
echo "🎉 All experiments completed!"