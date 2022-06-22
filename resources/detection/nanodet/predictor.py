import numpy as np
import torch

from .nanodet.data.batch_process import stack_batch_img
from .nanodet.data.collate import naive_collate
from .nanodet.data.transform import Pipeline
from .nanodet.model.arch import build_model
from .nanodet.util import cfg, load_model_weight


class Predictor(object):
    def __init__(self, cfg, model_path, min_conf):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.min_conf = min_conf

    def inference(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        results = [results[0][label] for label in results[0] if cfg.class_names[label] == "person"][0]
        results = np.array([person for person in results if person[4] > self.min_conf])
        results[:, 2] -= results[:, 0]
        results[:, 3] -= results[:, 1]
        h, _ = results.shape
        results = np.c_[-np.ones((h, 2)), results, -np.ones((h, 3))]
        return results
