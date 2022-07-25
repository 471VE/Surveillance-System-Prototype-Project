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
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        detector = Predictor(model, exp)
        return detector
    
    elif detection_mode in (5, 6):
        # TODO: Add masks as output
        
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from torch.cuda import is_available
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config))

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.DEVICE = 'cuda' if is_available() else 'cpu'
        cfg.MODEL.WEIGHTS = weights
        
        class DetectronPredictor():
            def __init__(self, cfg, min_conf):
                self.predictor = DefaultPredictor(cfg)
                self.min_conf = min_conf
            
            def inference(self, img):
                outputs = self.predictor(img)
                outputs['instances'] = outputs['instances'][outputs['instances'].pred_classes == 0]
                outputs['instances'] = outputs['instances'][outputs['instances'].scores > self.min_conf]
                bboxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
                bboxes[:, 2] -= bboxes[:, 0]
                bboxes[:, 3] -= bboxes[:, 1]
                scores = outputs["instances"].scores.numpy()
                h, _ = bboxes.shape
                outputs = np.c_[-np.ones((h, 2)), bboxes, scores, -np.ones((h, 3))]
                return outputs
        
        detector = DetectronPredictor(cfg, min_conf)
        return detector
        
    else:
        raise Exception("Something went wrong when loading detector.")