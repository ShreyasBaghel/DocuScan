import pytest
import numpy as np
import cv2
import os
import sys

# Ensure docuscan_india is in python path
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
    # Create random noisy grayscale image
    dummy_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    denoised = p.denoise(dummy_img)
    assert denoised.shape == (100, 100)

def test_clahe():
    p = Preprocessor()
    # Create simple low contrast gray image
    dummy_img = np.ones((100, 100), dtype=np.uint8) * 128
    enhanced = p.apply_clahe(dummy_img)
    assert enhanced.shape == (100, 100)

def test_binarise():
    p = Preprocessor()
    dummy_img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    binarized = p.binarise(dummy_img)
    assert binarized.shape == (100, 100)
    # Check that output is strictly binary (0 or 255)
    unique_vals = set(np.unique(binarized))
    assert unique_vals.issubset({0, 255})

def test_deskew():
    p = Preprocessor()
    # Generate a horizontal white bar on black background
    img = np.zeros((100, 100), dtype=np.uint8)
    img[45:55, 10:90] = 255
    
    # Rotate by 10 degrees to simulate skew
    M = cv2.getRotationMatrix2D((50, 50), 10, 1.0)
    skewed = cv2.warpAffine(img, M, (100, 100))
    
    # Run deskew (should return non-empty array)
    deskewed = p.deskew(skewed)
    assert deskewed.shape == (100, 100)
