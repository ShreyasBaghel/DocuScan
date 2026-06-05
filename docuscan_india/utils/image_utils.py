import cv2
import numpy as np
from PIL import Image

def to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL Image."""
    if cv_img is None:
        raise ValueError("Image is None")
    if len(cv_img.shape) == 2:  # Grayscale
        return Image.fromarray(cv_img)
    # BGR to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR image."""
    if pil_img is None:
        raise ValueError("Image is None")
    rgb_arr = np.array(pil_img)
    if len(rgb_arr.shape) == 2:  # Grayscale
        return rgb_arr
    # RGB to BGR
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

def resize_keep_aspect(cv_img: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """Resize image to fit within max_width x max_height while keeping aspect ratio."""
    if cv_img is None:
        return None
    h, w = cv_img.shape[:2]
    
    # Calculate scale factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    # Only downscale if image is larger than target
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return cv_img
