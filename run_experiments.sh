#!/bin/bash

set -e  # Exit on any error

MODES=("none" "masked-activations-layer" "masked-activations" "masked-weights")
SPARSITY_LEVELS=(0.20 0.35 0.50 0.65 0.80 0.90)

BASE_CONFIG="config.yaml"
TEMP_CONFIG="config_temp.yaml"
TEMP_INTER_CONFIG="config_temp_inter.yaml"

echo "Target sparsity levels: ${SPARSITY_LEVELS[@]}"
echo "======================================"

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "❌ Error: Base config file '$BASE_CONFIG' not found!"
    exit 1
fi

update_mode() {
    local mode=$1
    local base_config=$2
    local temp_config=$3

    sed "s/mode: [-'a-z]\+/mode: '$mode'/" "$base_config" > "$temp_config"

    echo "📝 Updated config with mode: $mode"
}

update_sparsity() {
    local sparsity=$1
    local base_config=$2
    local temp_config=$3

    sed "s/target_sparsity: [0-9.]\+/target_sparsity: $sparsity/" "$base_config" > "$temp_config"

    echo "📝 Updated config with target_sparsity: $sparsity"
}


for mode in "${MODES[@]}"; do
    # Update config for current mode
    update_mode "$mode" "$BASE_CONFIG" "$TEMP_INTER_CONFIG"
    for sparsity in "${SPARSITY_LEVELS[@]}"; do
        echo ""
        echo "🎯 Starting experiment with ${sparsity} sparsity (${sparsity/0./}%) and mode ${mode} (for none sparsity = 0.0)"
        echo "================================================="

        # Update config for current sparsity level
        update_sparsity "$sparsity" "$TEMP_INTER_CONFIG" "$TEMP_CONFIG"

        # Run training
        echo "▶️  Running training..."
        python runner.py --config "$TEMP_CONFIG"

        if [[ $? -eq 0 ]]; then
            echo "✅ Experiment with ${sparsity} sparsity completed successfully"
        else
            echo "❌ Experiment with ${sparsity} sparsity failed!"
            break
        fi

        if [[ "$mode" == "none" ]]; then
            echo "🔁 Skipping sparsity loop for mode 'none'"
            break
        fi
    done
done

# Cleanup
if [[ -f "$TEMP_INTER_CONFIG" ]]; then
    rm "$TEMP_INTER_CONFIG"
fi
if [[ -f "$TEMP_CONFIG" ]]; then
    rm "$TEMP_CONFIG"
fi

echo ""
echo "🎉 All experiments completed!"