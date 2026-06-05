import cv2
import numpy as np
from typing import Dict, Any

class Preprocessor:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the preprocessor with configuration settings.
        Expected format:
        {
            'deskew': True,
            'denoise': True,
            'clahe': True,
            'binarise': True
        }
        """
        if config is None:
            self.config = {
                'deskew': True,
                'denoise': True,
                'clahe': True,
                'binarise': True
            }
        else:
            self.config = config

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Runs the full preprocessing pipeline on the input image.
        Returns the preprocessed image.
        """
        processed = img.copy()

        # Step 1: Deskew (Hough-line/minAreaRect rotation correction)
        if self.config.get('deskew', True):
            processed = self.deskew(processed)

        # Convert to grayscale for remaining operations if not already
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed.copy()

        # Step 2: Denoise
        if self.config.get('denoise', True):
            gray = self.denoise(gray)

        # Step 3: Contrast Enhancement (CLAHE)
        if self.config.get('clahe', True):
            gray = self.apply_clahe(gray)

        # Step 4: Binarisation
        if self.config.get('binarise', True):
            gray = self.binarise(gray)

        return gray

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Detects skew angle and rotates the image to straighten it.
        We restrict rotation to a max of 15 degrees to prevent turning cards upside down.
        """
        # Convert to gray
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply threshold to isolate text
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Find coordinates of all text pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return img

        # Get minimum area bounding box
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Normalize the angle (varies across OpenCV versions)
        # We want the rotation angle to be close to 0
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is negligible or too large (indicating potential false detection), skip rotation
        if abs(angle) < 0.5 or abs(angle) > 15.0:
            return img

        # Rotate the image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Applies median and Gaussian blur to reduce print/scanning noise."""
        # Median blur to remove salt and pepper noise
        denoised = cv2.medianBlur(gray, 3)
        # Subtle Gaussian blur to smooth edges
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        return denoised

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def binarise(self, gray: np.ndarray) -> np.ndarray:
        """Applies adaptive binarisation for high-contrast text."""
        return cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
