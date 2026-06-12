import cv2
import numpy as np
from PIL import Image

class TamperFeatures:
    @staticmethod
    def calculate_blur_score(pil_img: Image.Image) -> float:
        """
        Calculates the Laplacian variance of the image.
        Lower values indicate more blur.
        """
        try:
            gray = np.array(pil_img.convert("L"))
            # Calculate the Laplacian variance
            val = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(val)
        except Exception:
            return 0.0

    @staticmethod
    def calculate_noise_score(pil_img: Image.Image) -> float:
        """
        Calculates the standard deviation of pixel intensities.
        """
        try:
            gray = np.array(pil_img.convert("L"))
            return float(np.std(gray))
        except Exception:
            return 0.0

    @staticmethod
    def calculate_edge_density(pil_img: Image.Image) -> float:
        """
        Calculates the Canny edge density (fraction of edge pixels in the image).
        """
        try:
            gray = np.array(pil_img.convert("L"))
            # Dynamic thresholding based on median
            med = np.median(gray)
            lower = int(max(0, 0.7 * med))
            upper = int(min(255, 1.3 * med))
            edges = cv2.Canny(gray, lower, upper)
            
            total_pixels = edges.size
            if total_pixels == 0:
                return 0.0
            
            edge_pixels = np.sum(edges > 0)
            return float(edge_pixels / total_pixels)
        except Exception:
            return 0.0
