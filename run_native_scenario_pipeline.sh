#!/usr/bin/env bash
set -euo pipefail

SCENARIOS=(carla_wet_001 carla_rain_001 carla_fog_001 carla_night_001)
MODELS=(segformer_b0 segformer_b2 segformer_b5 mask2former)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/rishan/workspace/conda-envs/carla-segformer
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PROJECT_DIR/hf-cache"
export HF_HUB_CACHE="$PROJECT_DIR/hf-cache/hub"

for scenario in "${SCENARIOS[@]}"; do
  test "$(find "outputs/sequences/$scenario" -maxdepth 1 -name 'frame_*.png' | wc -l)" -eq 100
  test "$(find "outputs/sequences/$scenario/ground_truth" -maxdepth 1 -name 'frame_*.png' | wc -l)" -eq 100

  python run_segformer_sequence.py --input-sequence "$scenario" --output-sequence "${scenario}_segformer_b0" --checkpoint nvidia/segformer-b0-finetuned-cityscapes-1024-1024
  python run_segformer_sequence.py --input-sequence "$scenario" --output-sequence "${scenario}_segformer_b2" --checkpoint nvidia/segformer-b2-finetuned-cityscapes-1024-1024
  python run_segformer_sequence.py --input-sequence "$scenario" --output-sequence "${scenario}_segformer_b5" --checkpoint nvidia/segformer-b5-finetuned-cityscapes-1024-1024
  python run_mask2former_sequence.py --input-sequence "$scenario" --output-sequence "${scenario}_mask2former"

  for model in "${MODELS[@]}"; do
    output="${scenario}_${model}"
    python analyze_temporal_uncertainty.py --sequence "$output" --msp-threshold 0.50 --low-pixel-fraction-threshold 0.10 --persistence-frames 3
    python evaluate_ground_truth.py --sequence "$output" --ground-truth-sequence "$scenario"
    python evaluate_frozen_thresholds.py --sequence "$output" --ground-truth-sequence "$scenario" --threshold-sequence "eval_val_001_${model}"
    for mode in full confidence_only no_persistence no_corridor; do
      python run_abstention_policy.py --sequence "$output" --threshold-sequence "eval_val_001_${model}" --mode "$mode"
      python run_action_policy.py --sequence "$output" --mode "$mode"
    done
  done
done

python audit_experiment.py
