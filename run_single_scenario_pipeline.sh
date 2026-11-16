#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:?Usage: ./run_single_scenario_pipeline.sh SCENARIO [GROUND_TRUTH_SEQUENCE]}"
GROUND_TRUTH="${2:-$SCENARIO}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/rishan/workspace/conda-envs/carla-segformer
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PROJECT_DIR/hf-cache"
export HF_HUB_CACHE="$PROJECT_DIR/hf-cache/hub"

test "$(find "outputs/sequences/$SCENARIO" -maxdepth 1 -name 'frame_*.png' | wc -l)" -eq 100
test "$(find "outputs/sequences/$GROUND_TRUTH/ground_truth" -maxdepth 1 -name 'frame_*.png' | wc -l)" -eq 100

python run_segformer_sequence.py --input-sequence "$SCENARIO" --output-sequence "${SCENARIO}_segformer_b0" --checkpoint nvidia/segformer-b0-finetuned-cityscapes-1024-1024
python run_segformer_sequence.py --input-sequence "$SCENARIO" --output-sequence "${SCENARIO}_segformer_b2" --checkpoint nvidia/segformer-b2-finetuned-cityscapes-1024-1024
python run_segformer_sequence.py --input-sequence "$SCENARIO" --output-sequence "${SCENARIO}_segformer_b5" --checkpoint nvidia/segformer-b5-finetuned-cityscapes-1024-1024
python run_mask2former_sequence.py --input-sequence "$SCENARIO" --output-sequence "${SCENARIO}_mask2former"

for model in segformer_b0 segformer_b2 segformer_b5 mask2former; do
  output="${SCENARIO}_${model}"
  python analyze_temporal_uncertainty.py --sequence "$output" --msp-threshold 0.50 --low-pixel-fraction-threshold 0.10 --persistence-frames 3
  python evaluate_ground_truth.py --sequence "$output" --ground-truth-sequence "$GROUND_TRUTH"
  python evaluate_frozen_thresholds.py --sequence "$output" --ground-truth-sequence "$GROUND_TRUTH" --threshold-sequence "eval_val_001_${model}"
  for mode in full confidence_only no_persistence no_corridor; do
    python run_abstention_policy.py --sequence "$output" --threshold-sequence "eval_val_001_${model}" --mode "$mode"
    python run_action_policy.py --sequence "$output" --mode "$mode"
  done
done

python evaluate_event_response.py --sequence "${SCENARIO}_segformer_b0" --disturbance-start 30 --disturbance-end 59 --msp-threshold 0.50 --low-pixel-fraction-threshold 0.10 --persistence-frames 3
python evaluate_event_response.py --sequence "${SCENARIO}_segformer_b2" --disturbance-start 30 --disturbance-end 59 --msp-threshold 0.50 --low-pixel-fraction-threshold 0.10 --persistence-frames 3
python evaluate_event_response.py --sequence "${SCENARIO}_segformer_b5" --disturbance-start 30 --disturbance-end 59 --msp-threshold 0.50 --low-pixel-fraction-threshold 0.10 --persistence-frames 3
python evaluate_event_response.py --sequence "${SCENARIO}_mask2former" --disturbance-start 30 --disturbance-end 59 --msp-threshold 0.50 --low-pixel-fraction-threshold 0.10 --persistence-frames 3
