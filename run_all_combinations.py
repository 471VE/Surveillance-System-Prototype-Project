import os
from tqdm.auto import tqdm

import utils.feature_extractor_deepsort as fed
import utils.feature_extractor as fe
from utils.model_info import detection_choices, extractor_choices

import real_time_single
from real_time_all import parse_args

def initial_setup_from_modes(detection_mode, extractor_mode):
    if detection_mode not in range(len(detection_choices)):
        raise Exception("Unsupported detection model. Exiting...")
    
    if extractor_mode not in range(len(extractor_choices)):
        raise Exception("Unsupported detection model. Exiting...")
    
    if extractor_mode == 0:
        encoder = fed.create_box_encoder(extractor_choices[extractor_mode]["model_path"], batch_size=32)
    elif extractor_mode in range(1, len(extractor_choices)):
        encoder = fe.create_feature_extractor(extractor_choices[extractor_mode]["model_name"],
                                              extractor_choices[extractor_mode]["model_path"])
    else:
        raise Exception("Something went wrong when loading ReID model.")
    
    output_dir = f"results/{detection_choices[detection_mode]['short_name']}_{extractor_choices[extractor_mode]['short_name']}/data"
    os.makedirs(output_dir, exist_ok=True)

    return encoder, output_dir

def main(args):
    for detection_choice in tqdm(range(len(detection_choices))):
        for extractor_choice in tqdm(range(len(extractor_choices))):
            
            encoder, output_dir = initial_setup_from_modes(detection_choice, extractor_choice)
            print(f'{detection_choices[detection_choice]["description"]} - {extractor_choices[extractor_choice]["description"]}')
            
            sequences = os.listdir(args.mot_dir)            
            for sequence in sequences:
                sequence_dir = os.path.join(args.mot_dir, sequence)
                output_file = os.path.join(output_dir, f"{sequence}.txt")        
                if not os.path.isdir(sequence_dir):
                    continue
                
                print("Running sequence %s" % sequence)
                real_time_single.run_app(detection_choice, encoder, sequence_dir, output_file, args.min_confidence,
                    args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance,
                    args.nn_budget, args.display)
                
if __name__ == "__main__":
    main(parse_args([f'--mot_dir=./MOT_custom',
                     f'--display=False']))