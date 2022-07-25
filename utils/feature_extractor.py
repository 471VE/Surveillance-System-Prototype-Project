from torchreid.utils import FeatureExtractor
from torch.cuda import is_available
import numpy as np

class CustomFeatureExtractor():
    def __init__(self, model_name, model_path):
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device="cuda" if is_available() else "cpu"
        )
    
    def __call__(self, image, bboxes):
        bboxes = np.array(bboxes)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        bboxes = bboxes.astype(int)
        
        bboxes[:, :2] = np.maximum(0, bboxes[:, :2])
        bboxes[:, 2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bboxes[:, 2:])

        image_patches = [image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] for bbox in bboxes]
        return self.extractor(image_patches).cpu().numpy()
    
def create_feature_extractor(model_name, model_path):
    extractor = CustomFeatureExtractor(model_name, model_path)
    
    def encoder(image, boxes):
        return extractor(image, boxes)
    
    return encoder