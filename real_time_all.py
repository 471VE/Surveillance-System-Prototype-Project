import argparse
import os
import real_time_single

from real_time_single import bool_string, initial_setup


def parse_args(args=None):
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")

    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
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
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    if not args:
        return parser.parse_args() 
    return parser.parse_args(args) 

def main(args):
    detection_mode, encoder, output_dir = initial_setup()
    sequences = os.listdir(args.mot_dir)
    
    for sequence in sequences:
        sequence_dir = os.path.join(args.mot_dir, sequence)
        output_file = os.path.join(output_dir, f"{sequence}.txt")        
        if not os.path.isdir(sequence_dir):
            continue
        
        print("Running sequence %s" % sequence)
        real_time_single.run_app(detection_mode, encoder, sequence_dir, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance,
            args.nn_budget, args.display)
    
def run_all_videos(path='./MOT_custom', display='True'):
    main(parse_args([f'--mot_dir={path}',
                     f'--display={display}']))
    
if __name__ == "__main__":
    main(parse_args())