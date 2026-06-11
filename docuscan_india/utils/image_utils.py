import numpy as np
from PIL import Image

def to_pil(img) -> Image.Image:
    """Convert numpy image array to PIL Image, or return PIL Image directly."""
    if img is None:
        raise ValueError("Image is None")
    if isinstance(img, Image.Image):
        return img
    # If BGR numpy array from OpenCV legacy code
    if len(img.shape) == 2:  # Grayscale
        return Image.fromarray(img)
    # RGB convert
    return Image.fromarray(img)

def to_cv(img) -> np.ndarray:
    """Convert PIL Image to numpy array (RGB), or return numpy array directly."""
    if img is None:
        raise ValueError("Image is None")
    if isinstance(img, Image.Image):
        return np.array(img)
    return img

def resize_keep_aspect(pil_img: Image.Image, max_width: int = 800, max_height: int = 600) -> Image.Image:
    """Resize PIL Image to fit within max_width x max_height while keeping aspect ratio."""
    if pil_img is None:
        return None
    
    w, h = pil_img.size
    
    # Calculate scale factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    # Only downscale if image is larger than target
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return pil_img
