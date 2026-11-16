# Curated experimental results

This folder contains the compact evidence package for the project. It excludes
raw RGB frames, logits, uncertainty tensors, model caches, and CARLA binaries.

## Included

- Physical occlusion action-policy executions for all four models
- Physical unknown-object action-policy executions for all four models
- Independent physical-debris comparison for SegFormer B0
- Ungated nominal baseline for the independent debris scenario
- Third-person CARLA policy replay GIF
- Compact model/scenario comparison table

Each action-execution JSON contains policy states, effective actions, collision
events, stopping time, stopping distance, throttle, brake, and steering data.

The independent-debris comparison recorded collision-frame events for the
ungated nominal controller and zero collision events for the uncertainty-aware
SegFormer B0 policy, which latched fallback braking.
