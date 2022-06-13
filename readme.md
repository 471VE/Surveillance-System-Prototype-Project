This is an unfinished project based on DeepSORT and MOT challenge. As of now, I am just organizing these two repositories to fit my purposes.

By default, it uses 2 cores.

Examples:
```
python deep_sort_app.py --sequence_dir=./MOT16/train/MOT16-09 --detection_file=./resources/detections/MOT16_POI_train/MOT16-09.npy --min_confidence=0.3 --nn_budget=100 --display=True

python evaluation/scripts/run_mot_challenge.py --TRACKERS_TO_EVAL test_tracker
```