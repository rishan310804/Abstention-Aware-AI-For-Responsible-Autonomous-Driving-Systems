# CARLA-Based Perception and Uncertainty Evaluation
## Individual Work Summary Report

**Contributor:** Rishan  
**Project:** Abstention-Aware Autonomous Driving / Responsible AI  
**Execution environment:** Server workspace `/NAS_DISK/BS2026/rishan/carla-segformer-project`  
**Report scope:** CARLA setup, multi-model semantic segmentation, uncertainty analysis, calibration, ground-truth evaluation, disturbance testing, and policy-evidence generation.

## 1. Contribution summary

I implemented and executed the CARLA-based perception and uncertainty-evaluation pipeline required for the abstention-aware driving project. The work covers the complete path from simulator setup and sequence capture to multi-model inference, temporal uncertainty analysis, threshold calibration, ground-truth evaluation, stress testing, and policy summaries.

This report describes the work actually implemented and verified in the CARLA project.

## 2. Computing environment

- CARLA version: **0.9.16**
- CARLA map used: `Carla/Maps/Town10HD_Opt`
- Python: **3.10.22**
- Environment: `/home/rishan/workspace/conda-envs/carla-segformer`
- Environment storage: workspace-backed Conda prefix
- GPU used by this work: **GPU 0, NVIDIA A100-PCIE-40GB**
- GPU isolation: `CUDA_VISIBLE_DEVICES=0`
- Other server GPUs were not used by this pipeline.
- Sequence resolution: **640 × 360**
- Sequence timestep: **0.1 seconds** (10 Hz)
- Hugging Face cache: project workspace, not the home directory.

## 3. CARLA and sequence-generation work

CARLA was installed and tested on the server. The client successfully connected to CARLA 0.9.16 and verified the active map.

The capture pipeline was implemented/used with:

- deterministic seeds;
- configurable Traffic Manager ports;
- selectable spawn points;
- configurable weather presets;
- RGB camera capture;
- CARLA semantic ground truth capture;
- frame metadata containing CARLA frame number, timestamp, vehicle pose, map, weather, seed, and spawn index.

Each completed evaluation sequence contains 100 frames. The ground-truth definition is recorded as raw CARLA semantic tag IDs in the red channel.

### Native CARLA scenarios captured

| Scenario | Weather | Frames | Ground truth | Seed | Spawn index |
|---|---|---:|---:|---:|---:|
| `carla_wet_001` | wet | 100 | 100 | 789 | 15 |
| `carla_rain_001` | rain | 100 | 100 | 890 | 20 |
| `carla_fog_001` | fog | 100 | 100 | 901 | 25 |
| `carla_night_001` | night | 100 | 100 | 912 | 30 |

### Evaluation splits captured

| Split | RGB frames | Ground-truth frames |
|---|---:|---:|
| `eval_normal_001` | 100 | 100 |
| `eval_val_001` | 100 | 100 |
| `eval_test_001` | 100 | 100 |

## 4. Multi-model inference

Four pretrained Cityscapes semantic-segmentation models were executed over the CARLA sequences:

1. SegFormer B0: `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`
2. SegFormer B2: `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`
3. SegFormer B5: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`
4. Mask2Former: `facebook/mask2former-swin-small-cityscapes-semantic`

For each processed frame, the pipeline stores model logits/semantic scores, prediction images, uncertainty tensors, and metadata. The SegFormer models produce 19-class logits. Mask2Former semantic scores are composed and normalized before calculating pixel confidence.

### Mask2Former corrections

Two issues were identified and resolved:

- The published checkpoint lacked the expected final Swin LayerNorm weights. The runner replaced that missing final LayerNorm with `Identity`, and recorded the adjustment in metadata.
- The first confidence calculation treated Mask2Former composed scores incorrectly as logits. The calculation was corrected to use normalized semantic scores, producing meaningful MSP values.

## 5. Temporal uncertainty analysis

The temporal-analysis pipeline was implemented and executed over model output sequences. It computes:

- maximum softmax probability (MSP);
- maximum raw logit;
- normalized entropy;
- low-confidence pixel fraction;
- frame-to-frame prediction instability;
- driving-corridor statistics;
- persistent low-confidence runs;
- temporal uncertainty summaries and CSV files.

The current driving-corridor proxy is the lower 60% of the image and the central 50% of the image width. This is an image-region proxy, not a full camera-homography or TTC model.

### Example baseline and blur results

| Model/sequence | Mean corridor MSP | Mean corridor instability |
|---|---:|---:|
| SegFormer B0, `normal_001` | 0.9805 | 0.0188 |
| SegFormer B0, `blur_001` | 0.7141 | 0.2868 |
| SegFormer B2, `transient_blur_001` | 0.8541 | 0.1314 |
| SegFormer B5, `transient_blur_001` | 0.8648 | 0.1105 |
| Mask2Former, `transient_blur_001` | 0.8283 | 0.1325 |
| SegFormer B2, `transient_noise_001` | 0.9811 | 0.0203 |
| SegFormer B5, `transient_noise_001` | 0.9899 | 0.0194 |
| Mask2Former, `transient_noise_001` | 0.9753 | 0.0244 |

These results demonstrate that blur produced a much stronger uncertainty response than the tested noise and exposure disturbances in these sequences.

## 6. Ground-truth evaluation

The ground-truth evaluation compares predicted semantic labels against CARLA semantic labels and reports:

- mean intersection-over-union (mIoU);
- pixel accuracy;
- driving-corridor accuracy;
- expected calibration error (ECE).

### Validation split results

| Model | mIoU | Pixel accuracy | Corridor accuracy | ECE |
|---|---:|---:|---:|---:|
| SegFormer B0 | 0.1831 | 0.9081 | 0.9780 | 0.0352 |
| SegFormer B2 | 0.2283 | 0.9134 | 0.9716 | 0.0375 |
| SegFormer B5 | 0.2494 | 0.9356 | 0.9835 | 0.0412 |
| Mask2Former | 0.1941 | 0.9140 | 0.9794 | 0.0415 |

### Test split results

| Model | mIoU | Pixel accuracy | Corridor accuracy | ECE |
|---|---:|---:|---:|---:|
| SegFormer B0 | 0.1991 | 0.8854 | 0.9584 | 0.0419 |
| SegFormer B2 | 0.2065 | 0.8864 | 0.9615 | 0.0588 |
| SegFormer B5 | 0.2928 | 0.9099 | 0.9665 | 0.0596 |
| Mask2Former | 0.2217 | 0.8867 | 0.9602 | 0.0688 |

SegFormer B5 achieved the highest mIoU on both validation and test among the evaluated models.

## 7. Confidence-threshold calibration

Thresholds were selected on `eval_val_001` with a target selective error of 5%. The thresholds were then frozen and evaluated on the test split.

| Model | Full-image threshold | Full-image validation coverage | Corridor threshold | Corridor validation coverage |
|---|---:|---:|---:|---:|
| SegFormer B0 | 0.76 | 90.31% | 0.50 | 98.75% |
| SegFormer B2 | 0.75 | 91.75% | 0.50 | 99.27% |
| SegFormer B5 | 0.70 | 96.88% | 0.50 | 99.75% |
| Mask2Former | 0.77 | 91.73% | 0.50 | 99.51% |

### Frozen-threshold test results

| Model | Full-image coverage | Full-image abstention | Full-image selective risk | Corridor selective risk |
|---|---:|---:|---:|---:|
| SegFormer B0 | 87.04% | 12.96% | 6.19% | 3.10% |
| SegFormer B2 | 90.70% | 9.30% | 7.27% | 2.95% |
| SegFormer B5 | 95.79% | 4.21% | 7.11% | 3.10% |
| Mask2Former | 91.81% | 8.19% | 7.63% | 3.45% |

The test evaluation used thresholds calibrated on validation data rather than retuning on the test data.

## 8. Disturbance and stress testing

### Synthetic disturbances

The following synthetic sequences were generated from normal CARLA frames:

- static blur;
- static noise;
- bright exposure;
- dark exposure;
- transient blur, frames 30–59;
- transient noise, frames 30–59;
- transient dark exposure, frames 30–59.

### Additional stress scenarios

Two additional 100-frame scenarios were created from the validation sequence, with ground truth copied from the source sequence:

- `occlusion_001`: a visual occlusion applied during frames 30–59;
- `unknown_object_001`: a visually unfamiliar object placed in the driving corridor during frames 30–59.

All four models were processed for both stress scenarios, including temporal analysis and ground-truth evaluation.

### Native weather results

The following table reports mIoU for the four native weather scenarios:

| Scenario | B0 | B2 | B5 | Mask2Former |
|---|---:|---:|---:|---:|
| Wet | 0.2253 | 0.2654 | 0.2741 | 0.2586 |
| Rain | 0.1932 | 0.2402 | 0.2736 | 0.2146 |
| Fog | 0.1531 | 0.1708 | 0.2150 | 0.1494 |
| Night | 0.2161 | 0.2793 | 0.2802 | 0.2189 |

Fog was the most difficult native condition overall. SegFormer B5 gave the strongest mIoU in every listed native-weather scenario.

## 9. Policy-evidence and ablation work

The policy runner was implemented with four modes:

- `full`: confidence, persistence, instability, and corridor evidence;
- `confidence_only`: confidence evidence only;
- `no_persistence`: removes multi-frame persistence;
- `no_corridor`: evaluates the full image rather than the corridor region.

Each policy run generates:

- frame-level policy decisions CSV;
- state-count summary JSON;
- supervisor queue JSON.

The project audit records **164 policy summaries** and **17 event-response reports**. This demonstrates that the policy analysis infrastructure was executed across the completed scenario/model combinations.

The current policy is a decision-evidence prototype. It does not yet constitute a full closed-loop CARLA vehicle controller with throttle, brake, steering, jerk, collision, or physical stopping-distance measurements.

An explicit no-RL action-policy layer has now been added in `run_action_policy.py`.
It translates each policy state into an auditable control intent:

- `Normal` -> nominal drive;
- `Caution` -> nominal controller with a speed limit;
- `Abstention` -> controlled stop through the fallback controller;
- `Escalation` -> emergency stop and supervisor request.

The generated action plans contain target speed, brake command, control mode,
state transitions, and supervisor requirements. They are currently offline
action plans derived from saved uncertainty evidence. They must not be reported
as physical CARLA control execution until connected to a live vehicle-control
loop and evaluated using collision, stopping-distance, and reaction-time
measurements.

## 10. Reproducibility and audit artifacts

The following project scripts support reproduction:

- `capture_sequence.py`
- `create_disturbance_sequence.py`
- `run_segformer_sequence.py`
- `run_mask2former_sequence.py`
- `analyze_temporal_uncertainty.py`
- `evaluate_ground_truth.py`
- `calibrate_thresholds.py`
- `evaluate_frozen_thresholds.py`
- `evaluate_event_response.py`
- `run_abstention_policy.py`
- `run_single_scenario_pipeline.sh`
- `run_native_scenario_pipeline.sh`
- `audit_experiment.py`

The physical stress-scenario extension is supported by:

- `capture_physical_stress_scenario.py`
- `replay_action_policy_in_carla.py`

These scripts capture and recreate physically rendered CARLA stress props,
rather than applying only post-capture image edits.

An independent physical-debris comparison was also implemented. In the
matched CARLA route, the ungated nominal controller recorded 82 collision
events from frame 18 onward, while the uncertainty-aware B0 policy latched
fallback braking, stopped in 0.2 seconds over 0.49 m, and recorded zero
collision events. This is an action-level safety comparison, not merely a
segmentation-confidence result.

The final completeness record is stored in:

`/NAS_DISK/BS2026/rishan/carla-segformer-project/experiment_audit.json`

All project Python files and shell pipelines were syntax-checked after implementation.

## 11. Scope boundary and remaining integration work

The completed work in this report is the CARLA perception and uncertainty-evaluation block. The following items remain part of the broader project integration plan:

- dataset availability investigation;
- literature review and project-level planning;
- feature-space Mahalanobis/k-NN OOD detection;
- full homography/TTC/object-level cost modelling;
- physical CARLA fallback actuation;
- cryptographic audit-chain implementation;
- HUD/dashboard generation;
- full ten-scenario closed-loop benchmark;
- final team report and presentation integration.

## 12. Final contribution statement

I established and validated a reproducible CARLA-based benchmark for comparing four semantic-segmentation models under nominal, adverse-weather, synthetic-corruption, transient-disturbance, occlusion, and unknown-object conditions. The pipeline produces model predictions, uncertainty measures, temporal reliability indicators, calibrated thresholds, ground-truth metrics, event-response metrics, and policy-state evidence suitable for integration into the team’s broader abstention-aware driving framework.
