This is an unfinished project based on DeepSORT and MOT challenge. As of now, I am just organizing these two repositories to fit my purposes.

By default, it uses 2 cores.

Examples:
```
python deep_sort_app.py --sequence_dir=./MOT16/test/MOT16-06 --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100 --display=True

python TrackEval/scripts/run_mot_challenge.py --TRACKERS_TO_EVAL test_tracker
```