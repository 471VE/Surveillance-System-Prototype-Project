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
        "short_name": "yolox_l"}
}