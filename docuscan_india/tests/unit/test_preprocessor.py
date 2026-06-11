import pytest
from PIL import Image, ImageDraw
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr.preprocessor import Preprocessor

def test_preprocessor_initialization():
    config = {
        'deskew': True,
        'denoise': False,
        'clahe': True,
        'binarise': False
    }
    p = Preprocessor(config)
    assert p.config['deskew'] is True
    assert p.config['denoise'] is False
    assert p.config['clahe'] is True
    assert p.config['binarise'] is False

def test_denoise():
    p = Preprocessor()
    # Create random noisy RGB image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    denoised = p.denoise(dummy_img)
    assert isinstance(denoised, Image.Image)
    assert denoised.size == (100, 100)

def test_clahe():
    p = Preprocessor()
    # Create simple low contrast gray image
    dummy_img = Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 128)
    enhanced = p.apply_clahe(dummy_img)
    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == (100, 100)

def test_binarise():
    p = Preprocessor()
    dummy_img = Image.fromarray(np.random.randint(50, 200, (100, 100), dtype=np.uint8))
    binarized = p.binarise(dummy_img)
    assert isinstance(binarized, Image.Image)
    assert binarized.size == (100, 100)
    # Check that output is strictly binary (0 or 255)
    arr = np.array(binarized)
    unique_vals = set(np.unique(arr))
    assert unique_vals.issubset({0, 255})

def test_deskew():
    p = Preprocessor()
    # Generate a horizontal white bar on black background and rotate it using Pillow
    img = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 45, 90, 55], fill=255)
    
    # Rotate by 10 degrees to simulate skew
    skewed = img.rotate(10, expand=False)
    
    # Run deskew
    deskewed = p.deskew(skewed)
    assert isinstance(deskewed, Image.Image)
    assert deskewed.size[0] >= 100

