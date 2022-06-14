import argparse
import os
import real_time_single

from deep_sort_app import bool_string
from tools.generate_detections import create_box_encoder


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument("--model", default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="base_deepsort")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.0, type=float)
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
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = f"results/{args.output_dir}/data"
    os.makedirs(output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)
    
    encoder = create_box_encoder(args.model, batch_size=32)
    
    for sequence in sequences:
        sequence_dir = os.path.join(args.mot_dir, sequence)
        output_file = os.path.join(output_dir, f"{sequence}.txt")
        print(output_file)
        
        if not os.path.isdir(sequence_dir):
            continue
        
        print("Running sequence %s" % sequence)
        real_time_single.run_app(encoder, sequence_dir, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance,
            args.nn_budget, args.display)