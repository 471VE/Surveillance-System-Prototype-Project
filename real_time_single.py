import argparse
import cv2
import numpy as np
import os
from time import time

from deep_sort.application_util import preprocessing, visualization
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection

import tools.generate_detections as gd


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


def create_detections(detections, min_height=0):    
    detection_list = []
    for row in detections:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def generate_frame_features(encoder, rows, bgr_image):
    features = encoder(bgr_image, rows[:, 2:6].copy())
    detections_out = [np.r_[(row, feature)] for row, feature
                      in zip(rows, features)]    
    return detections_out


def gather_sequence_info(sequence_dir):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    min_frame_idx = min(image_filenames.keys())
    max_frame_idx = max(image_filenames.keys())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms
    }
    return seq_info


detection_dict = {
    0: {"model": "None",
        "config": "None"},
    1: {"model": "resources/detection/nanodet/model_weights/nanodet-plus-m_416.pth",
        "config": "resources/detection/nanodet/config/nanodet-plus-m_416.yml"},
    2: {"model": "resources/detection/nanodet/model_weights/nanodet-plus-m-1.5x_416.pth",
        "config": "resources/detection/nanodet/config/nanodet-plus-m-1.5x_416.yml"},
}


def generate_detections(detection_mode, detector, image, frame_idx, min_confidence):
    if not detection_mode:
        mask = detector["frame_indices"] == frame_idx
        return detector["detections_in"][mask]
    return detector.inference(image, min_confidence)


def run_app(detection_mode, encoder, sequence_dir, output_file, min_confidence,
    nms_max_overlap, min_detection_height, max_cosine_distance,
    nn_budget, display):
    
    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    
    detector = None
    if not detection_mode:    
        detections_in = np.loadtxt(os.path.join(sequence_dir, "det/det.txt"), delimiter=',')
        frame_indices = detections_in[:, 0].astype(int)
        detector = {"detections_in": detections_in, "frame_indices": frame_indices}
    else:
        import torch
        
        from resources.detection.nanodet.nanodet.util import cfg, load_config
        from resources.detection.nanodet.predictor import Predictor
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        load_config(cfg, detection_dict[detection_mode]["config"])
        detector = Predictor(cfg, detection_dict[detection_mode]["model"])

    def frame_callback(vis, frame_idx):
        begin_time = time()
        
        image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        
        rows = generate_detections(detection_mode, detector, image, frame_idx, min_confidence)            
        detections = generate_frame_features(encoder, rows, image)

        # Load image and generate detections.
        detections = create_detections(detections, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            
        # Update visualization.
        if display:
            vis.set_image(image.copy())
            # vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            vis.draw_fps(time() - begin_time)

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    
        
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument("--model", default="resources/feature_generation/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="base_deepsort")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.35, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()    
     
          
def main():
    args = parse_args()
    
    detection_choices = {
        0: "Detections provided by MOT benchmarks",
        1: "NanoDet-Plus-m",
        2: "NanoDet-Plus-m-1.5x",
    }
    detection_model_prompt = (
        "\nChoose detection model:\n" +
        "".join([f"{key}. {detection_choices[key]}.\n" for key in detection_choices])
    )
    detection_mode = int(input(detection_model_prompt))
    if detection_mode not in (0, 1, 2):
        raise Exception("Unsupported detection model. Exiting...")
    print(f"Choosing option {detection_mode} - {detection_choices[detection_mode]}...\n")
    
    output_dir = f"results/{args.output_dir}/data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.basename(args.sequence_dir)}.txt")

    encoder = gd.create_box_encoder(args.model, batch_size=32)
    
    run_app(detection_mode, encoder, args.sequence_dir, output_file, args.min_confidence,
        args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance,
        args.nn_budget, args.display)
    
if __name__ == "__main__":
    main()