detection_choices = {
    0: {"model": "None",
        "config": "None",
        "description": "Detections provided by MOT benchmarks",
        "short_name": "MOT_detections"},
    
    1: {"model": "weights/nanodet-plus-m_416.pth",
        "config": "resources/detection/nanodet/config/nanodet-plus-m_416.yml",
        "description": "NanoDet-Plus-m",
        "short_name": "nanodet_plus_m"},
    
    2: {"model": "weights/nanodet-plus-m-1.5x_416.pth",
        "config": "resources/detection/nanodet/config/nanodet-plus-m-1.5x_416.yml",
        "description": "NanoDet-Plus-m-1.5x",
        "short_name": "nanodet_plus_m_1.5x"},
    
    3: {"model": "weights/yolox_tiny.pth",
        "config": "resources/detection/yolox/exps/yolox_tiny.py",
        "description": "YOLOX-tiny",
        "short_name": "yolox_tiny"},
    
    4: {"model": "weights/yolox_l.pth",
        "config": "resources/detection/yolox/exps/yolox_l.py",
        "description": "YOLOX-l",
        "short_name": "yolox_l"},
    
    5: {"model": "weights/mask_rcnn.pkl",
        "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "description": "Mask R-CNN (R50-FPN)",
        "short_name": "mask_rcnn"},
    
    6: {"model": "weights/cascade_mask_rcnn.pkl",
        "config": "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
        "description": "Cascade R-CNN (R50-FPN)",
        "short_name": "cascade_mask_rcnn"}
}

extractor_choices = {
    0: {"model_path": "weights/mars-small128.pb",
        "model_name": None,
        "description": "Deafult DeepSORT feature extractor",
        "short_name": "default_deepsort"},
    
    1: {"model_path": "weights/shufflenet.pth.tar",
        "model_name": "shufflenet",
        "description": "ShuffleNet",
        "short_name": "shufflenet"},
    
    2: {"model_path": "weights/mlfn.pth.tar",
        "model_name": "mlfn",
        "description": "Multilayered Feedforward Neural Network",
        "short_name": "mlfn"},
    
    3: {"model_path": "weights/mobilenetv2.pth",
        "model_name": "mobilenetv2_x1_0",
        "description": "MobileNetV2",
        "short_name": "mobilenetv2"},
    
    4: {"model_path": "weights/osnet.pth",
        "model_name": "osnet_x1_0",
        "description": "OSNet",
        "short_name": "osnet"},
    
    5: {"model_path": "weights/osnet_ain.pth",
        "model_name": "osnet_ain_x1_0",
        "description": "OSNet-AIN",
        "short_name": "osnet_ain"},
    
    6: {"model_path": "weights/osnet_ibn.pth",
        "model_name": "osnet_ibn_x1_0",
        "description": "OSNet-IBN",
        "short_name": "osnet_ibn"}
}