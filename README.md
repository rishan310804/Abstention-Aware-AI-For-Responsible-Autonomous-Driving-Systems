# Abstention-Aware AI for Responsible Autonomous Driving Systems

This repository contains the reproducible policy and evaluation code for an
uncertainty-aware autonomous-driving action policy in CARLA.

## What is implemented

- CARLA RGB and semantic-ground-truth capture
- SegFormer B0/B2/B5 and Mask2Former inference
- Confidence, entropy, instability, and temporal uncertainty analysis
- Validation calibration and frozen test thresholds
- Four-state action policy: Normal, Caution, Abstention, and Escalation
- Latched fallback braking and physical CARLA control
- Collision, stopping-time, stopping-distance, throttle, brake, and steering metrics
- Physical occlusion, unknown-object, and independent-debris scenarios
- Third-person replay visualization

## Repository scope

This GitHub repository intentionally excludes CARLA binaries, model caches,
raw logits, and bulk generated outputs. Those artifacts are large and should
remain on the experiment server. The scripts regenerate the required outputs.

## Environment

The experiments used CARLA 0.9.16, Python 3.10, and a workspace Conda
environment. Set the CARLA Python API path and model cache according to the
server installation before running the scripts.

For isolated CARLA execution, use a separate GPU and RPC port, for example:

```bash
export CUDA_VISIBLE_DEVICES=2
./CarlaUE4.sh -RenderOffScreen -opengl -graphicsadapter=2 \
  -quality-level=Low -carla-rpc-port=2200
```

## Action-policy replay

```bash
python replay_action_policy_in_carla.py \
  --sequence physical_debris_001_segformer_b0 \
  --scenario-sequence physical_debris_001 \
  --mode full --port 2200 --tm-port 8260
```

The policy uses nominal driving while evidence is reliable, speed-limited
caution, latched controlled stopping after abstention, and supervisory
escalation for severe persistent uncertainty.

## Important result

In the independent physical-debris comparison, the ungated nominal controller
recorded collision-frame events, while the uncertainty-aware SegFormer B0
policy detected uncertainty, latched fallback braking, and reached standstill
without a collision in the tested replay.

## Reports and visualizations

- `rishan_work_summary_report.md`
- `experiment_audit.json`
- `actual-carla-replay-standalone.html`
- `outputs/visualizations/third_person_policy_replay.gif` (generated locally)

