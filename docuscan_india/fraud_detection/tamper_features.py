import numpy as np
from PIL import Image
from scipy.ndimage import laplace, sobel, label

class TamperFeatures:
    @staticmethod
    def calculate_blur_score(pil_img: Image.Image) -> float:
        """
        Calculates the Laplacian variance of the image.
        Lower values indicate more blur.
        """
        try:
            gray = np.array(pil_img.convert("L"))
            # Calculate the Laplacian variance using SciPy
            val = laplace(gray.astype(np.float64)).var()
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
    def _canny_scipy(img: np.ndarray, low: float, high: float) -> np.ndarray:
        """
        Vectorized SciPy/NumPy implementation of Canny edge detector.
        """
        img_float = img.astype(np.float64)
        dx = sobel(img_float, axis=1) # horizontal
        dy = sobel(img_float, axis=0) # vertical
        
        mag = np.abs(dx) + np.abs(dy)
        
        # Calculate angles in [0, 180)
        angle = np.arctan2(dy, dx) * 180.0 / np.pi
        angle = np.where(angle < 0, angle + 180, angle)
        
        padded_mag = np.pad(mag, 1, mode='constant', constant_values=0)
        
        mask0 = (angle < 22.5) | (angle >= 157.5)
        mask45 = (angle >= 22.5) & (angle < 67.5)
        mask90 = (angle >= 67.5) & (angle < 112.5)
        mask135 = (angle >= 112.5) & (angle < 157.5)
        
        keep = np.zeros_like(mag, dtype=bool)
        c = mag
        
        w_val = padded_mag[1:-1, 0:-2]
        e_val = padded_mag[1:-1, 2:]
        keep |= mask0 & (c >= w_val) & (c >= e_val)
        
        sw_val = padded_mag[2:, 0:-2]
        ne_val = padded_mag[0:-2, 2:]
        keep |= mask45 & (c >= sw_val) & (c >= ne_val)
        
        n_val = padded_mag[0:-2, 1:-1]
        s_val = padded_mag[2:, 1:-1]
        keep |= mask90 & (c >= n_val) & (c >= s_val)
        
        nw_val = padded_mag[0:-2, 0:-2]
        se_val = padded_mag[2:, 2:]
        keep |= mask135 & (c >= nw_val) & (c >= se_val)
        
        nms = np.where(keep, mag, 0.0)
        
        strong = nms >= high
        weak = nms >= low
        
        structure = np.ones((3, 3), dtype=bool)
        labeled, num_features = label(weak, structure=structure)
        
        strong_labels = np.unique(labeled[strong])
        
        # Fast boolean lookup table mapping
        lut = np.zeros(num_features + 1, dtype=bool)
        lut[strong_labels] = True
        edge_mask = lut[labeled]
        
        res = np.zeros_like(img, dtype=np.uint8)
        res[edge_mask] = 255
        return res

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
            
            edges = TamperFeatures._canny_scipy(gray, lower, upper)
            
            total_pixels = edges.size
            if total_pixels == 0:
                return 0.0
            
            edge_pixels = np.sum(edges > 0)
            return float(edge_pixels / total_pixels)
        except Exception:
            return 0.0
