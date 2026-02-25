import os
import numpy as np
from ultralytics import YOLO

_weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'detector', 'best.pt')
_model = YOLO(_weights_path)

def detect_teeth(image: np.ndarray) -> list:
    """
    Детекция зубов на КТ-снимке
    
    Args:
        image: Изображение в формате numpy array (RGB)
        
    Returns:
        Список bounding box'ов в формате [x_min, y_min, x_max, y_max, confidence]
    """
    results = _model(image, verbose=False)
    bboxes = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf.cpu().numpy()[0])
            bboxes.append([x1, y1, x2, y2, conf])
    
    return bboxes