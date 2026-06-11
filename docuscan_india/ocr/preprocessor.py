from PIL import Image, ImageOps, ImageFilter, ImageEnhance
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

    def preprocess(self, img: Image.Image) -> Image.Image:
        """
        Runs the full preprocessing pipeline on the input PIL Image.
        Returns the preprocessed PIL Image.
        """
        processed = img.copy()

        # Step 0: Upscale low-resolution images (Tesseract prefers ~300 DPI / large text)
        w, h = processed.size
        if w < 1800:
            new_w = w * 2
            new_h = h * 2
            processed = processed.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Step 1: Deskew (horizontal projection profile rotation correction)
        if self.config.get('deskew', True):
            processed = self.deskew(processed)

        # Step 2: Denoise
        if self.config.get('denoise', True):
            processed = self.denoise(processed)

        # Step 3: Contrast Enhancement
        if self.config.get('clahe', True):
            processed = self.apply_clahe(processed)

        # Step 4: Binarisation
        if self.config.get('binarise', True):
            processed = self.binarise(processed)

        return processed

    def deskew(self, img: Image.Image) -> Image.Image:
        """
        Detects skew angle and rotates the image to straighten it.
        We restrict rotation to a max of 15 degrees to prevent turning cards upside down.
        """
        # Convert to gray and downscale to 300x300 for high performance projection profile
        small_gray = img.convert("L").resize((300, 300), Image.Resampling.BILINEAR)
        
        # Invert (Tesseract expects black text on white background)
        small_inverted = ImageOps.invert(small_gray)
        binary_arr = np.array(small_inverted) > 127

        best_angle = 0.0
        max_variance = 0.0

        # We test angles between -15 and 15 degrees with 1 degree steps
        angles = np.arange(-15, 16, 1)
        for angle in angles:
            # Rotate using PIL (fill with white in inverted form -> black, value 0)
            rot_img = small_inverted.rotate(float(angle), resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0)
            rot_arr = np.array(rot_img) > 127
            
            # Sum pixels horizontally along each row
            row_sums = np.sum(rot_arr, axis=1)
            # Find variance of row sums
            variance = np.var(row_sums)
            
            if variance > max_variance:
                max_variance = variance
                best_angle = angle

        # If rotation is significant, rotate original image
        if abs(best_angle) >= 0.5:
            # Rotate original image using BICUBIC for high quality
            # Use white background for exposed corners
            return img.rotate(float(best_angle), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255) if img.mode == "RGB" else 255)
            
        return img

    def denoise(self, img: Image.Image) -> Image.Image:
        """Applies median and Gaussian blur filters to reduce noise."""
        gray = img.convert("L")
        # Median filter to remove salt & pepper noise
        median_filtered = gray.filter(ImageFilter.MedianFilter(size=3))
        # Subtle Gaussian blur to smooth edges
        denoised = median_filtered.filter(ImageFilter.GaussianBlur(radius=0.8))
        return denoised

    def apply_clahe(self, img: Image.Image) -> Image.Image:
        """
        Applies a high-performance local contrast normalization filter
        that emulates OpenCV's CLAHE.
        """
        gray = img.convert("L")
        w, h = gray.size
        
        # Calculate optimal radius dynamically based on resolution
        radius = max(30, min(w, h) // 16)
        
        # Convert to float numpy array
        arr = np.array(gray, dtype=float)
        
        # Compute local mean using Gaussian blur
        mean_img = gray.filter(ImageFilter.GaussianBlur(radius))
        mean = np.array(mean_img, dtype=float)
        
        # Compute absolute difference from mean
        diff = arr - mean
        
        # Compute local standard deviation
        # Scale to fit inside standard uint8 image to keep footprint low
        diff_sq_norm = (diff**2) / 255.0
        diff_sq_img = Image.fromarray(diff_sq_norm.astype(np.uint8))
        var_img = diff_sq_img.filter(ImageFilter.GaussianBlur(radius))
        var = np.array(var_img, dtype=float) * 255.0
        
        # Standard deviation (add epsilon to avoid division by zero)
        std = np.sqrt(var) + 1e-5
        
        # Normalize: shift by 128 and scale with a text-gain of 55
        norm = (diff / std) * 55.0 + 128.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        
        return Image.fromarray(norm)

    def binarise(self, img: Image.Image) -> Image.Image:
        """Applies adaptive/local thresholding using a blurred mask."""
        gray = img.convert("L")
        # Create a local threshold mask using a large Gaussian blur (simulates local background)
        w, h = gray.size
        radius = max(15, min(w, h) // 60)
        bg = gray.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Adaptive thresholding: if pixel < bg_pixel - constant, then black (0), else white (255)
        # We implement this in numpy for efficiency
        gray_arr = np.array(gray, dtype=np.int16)
        bg_arr = np.array(bg, dtype=np.int16)
        
        # P_xy < BG_xy - 10 -> 0, else 255
        binary_arr = np.where(gray_arr < (bg_arr - 10), 0, 255).astype(np.uint8)
        return Image.fromarray(binary_arr)
