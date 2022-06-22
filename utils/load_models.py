import numpy as np
import os


def load_detector(detection_mode, detection_choices, sequence_dir, min_conf):
    config = detection_choices[detection_mode]["config"]
    weights = detection_choices[detection_mode]["model"]
    
    if not detection_mode:
        class Predictor():
            def __init__(self, sequence_dir, min_conf):
                self.detections_in = np.loadtxt(os.path.join(sequence_dir, "det/det.txt"), delimiter=',')
                self.frame_indices = self.detections_in[:, 0].astype(int)
                self.frame = 1
                self.min_conf = min_conf
                
            def inference(self, img):
                mask = self.frame_indices == self.frame
                results = self.detections_in[mask]
                results = results[results[:, 4] > self.min_conf]
                self.frame += 1
                return results
                
        detector = Predictor(sequence_dir, min_conf)
        return detector
        
    elif detection_mode in (1, 2):
        import torch
        
        from resources.detection.nanodet.nanodet.util import cfg, load_config
        from resources.detection.nanodet.predictor import Predictor
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        load_config(cfg, config)
        detector = Predictor(cfg, weights, min_conf)
        return detector
        
    elif detection_mode in (3, 4):
        import torch
        
        from resources.detection.yolox.yolox.exp.build import get_exp_by_file
        from resources.detection.yolox.predictor import Predictor
        
        exp = get_exp_by_file(config)        
        exp.test_size = (640, 640)
        exp.test_conf = min_conf

        model = exp.get_model()
        ckpt = torch.load(weights, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        detector = Predictor(model, exp)
        return detector
    
    else:
        raise Exception("Something went wrong when loading detector.")