from asyncio import gather
import cv2
import os
from real_time_single import gather_sequence_info
from tqdm.auto import tqdm

sequences = os.listdir("./MOT_custom")
for sequence in tqdm(sequences):
    sequence_dir = os.path.join("./MOT_custom", sequence)
    if not os.path.isdir(sequence_dir):
        continue
    seq_info = gather_sequence_info(sequence_dir)
    frame = cv2.imread(seq_info["image_filenames"][1])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(seq_info["video_filename"], fourcc, seq_info["framerate"], (width,height))

    for frame_idx in tqdm(range(seq_info["min_frame_idx"], seq_info["max_frame_idx"])):
        video.write(cv2.imread(seq_info["image_filenames"][frame_idx]))

    cv2.destroyAllWindows()
    video.release()