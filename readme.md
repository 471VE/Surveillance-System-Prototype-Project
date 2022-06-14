This is an unfinished project based on DeepSORT and MOT challenge. As of now, I am just organizing these two repositories to fit my purposes.

By default, it uses 2 cores.

Examples:
```
python HOTA_evaluation/run_mot_challenge.py

python deep_sort_app.py --sequence_dir=./MOT_custom/KITTI-17 --detection_file=./resources/detections/MOT_custom/KITTI-17.npy --min_confidence=0.3 --nn_budget=100

python tools/generate_detections.py --model=resources/networks/mars-small128.pb --mot_dir=./MOT_custom --output_dir=./resources/detections/MOT_custom

python evaluate_motchallenge.py --mot_dir=./MOT_custom --detection_dir=./resources/detections/MOT_custom

python deep_sort_app.py --sequence_dir=./MOT_custom/KITTI-17 --detection_file=./resources/detections/MOT_custom/KITTI-17.npy --min_confidence=0.3 --nn_budget=100

python real_time_single.py --sequence_dir=./MOT_custom/KITTI-17 --min_confidence=0.3 --nn_budget=100

python real_time_all.py --mot_dir=./MOT_custom
```