#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/rishan/workspace/conda-envs/carla-segformer
cd "$(dirname "${BASH_SOURCE[0]}")"

python capture_sequence.py --frames 100 --sequence carla_wet_001 --tm-port 8050 --seed 789 --spawn-index 15 --weather-preset wet --with-ground-truth
python capture_sequence.py --frames 100 --sequence carla_rain_001 --tm-port 8060 --seed 890 --spawn-index 20 --weather-preset rain --with-ground-truth
python capture_sequence.py --frames 100 --sequence carla_fog_001 --tm-port 8070 --seed 901 --spawn-index 25 --weather-preset fog --with-ground-truth
python capture_sequence.py --frames 100 --sequence carla_night_001 --tm-port 8080 --seed 912 --spawn-index 30 --weather-preset night --with-ground-truth
