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
        Applies local contrast enhancement using a custom vectorized CLAHE implementation.
        """
        gray = img.convert("L")
        gray_arr = np.array(gray)
        
        grid_h, grid_w = 8, 8
        h, w = gray_arr.shape
        
        tile_h = h // grid_h
        tile_w = w // grid_w
        
        # Pad image to make it divisible by grid size if necessary
        pad_h = (tile_h - h % tile_h) % tile_h
        pad_w = (tile_w - w % tile_w) % tile_w
        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(gray_arr, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            img_padded = gray_arr
            
        # Calculate histograms and CDFs for each tile
        clip_limit = 2.0
        histograms = np.zeros((grid_h, grid_w, 256), dtype=np.float32)
        for i in range(grid_h):
            for j in range(grid_w):
                tile = img_padded[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
                hist, _ = np.histogram(tile, bins=256, range=(0, 256))
                
                # Clip histogram
                actual_clip = clip_limit * (tile.size / 256.0)
                clipped = np.minimum(hist, actual_clip)
                excess = np.sum(hist) - np.sum(clipped)
                
                # Redistribute excess equally
                redist = excess / 256.0
                hist = clipped + redist
                
                # Calculate CDF
                cdf = hist.cumsum()
                cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0] or 1) * 255.0
                histograms[i, j] = cdf
                
        # Bilinear interpolation
        y_coords = (np.arange(h) - 0.5 * tile_h) / tile_h
        x_coords = (np.arange(w) - 0.5 * tile_w) / tile_w
        
        y_coords = np.clip(y_coords, 0, grid_h - 1)
        x_coords = np.clip(x_coords, 0, grid_w - 1)
        
        y_indices = y_coords.astype(np.int32)
        x_indices = x_coords.astype(np.int32)
        
        y_diff = y_coords - y_indices
        x_diff = x_coords - x_indices
        
        y_indices_next = np.minimum(y_indices + 1, grid_h - 1)
        x_indices_next = np.minimum(x_indices + 1, grid_w - 1)
        
        y_idx = y_indices[:, None]
        y_idx_next = y_indices_next[:, None]
        x_idx = x_indices[None, :]
        x_idx_next = x_indices_next[None, :]
        
        cdf_tl = histograms[y_idx, x_idx, gray_arr]
        cdf_tr = histograms[y_idx, x_idx_next, gray_arr]
        cdf_bl = histograms[y_idx_next, x_idx, gray_arr]
        cdf_br = histograms[y_idx_next, x_idx_next, gray_arr]
        
        wa = (1.0 - y_diff[:, None]) * (1.0 - x_diff[None, :])
        wb = (1.0 - y_diff[:, None]) * x_diff[None, :]
        wc = y_diff[:, None] * (1.0 - x_diff[None, :])
        wd = y_diff[:, None] * x_diff[None, :]
        
        interpolated = wa * cdf_tl + wb * cdf_tr + wc * cdf_bl + wd * cdf_br
        enhanced_arr = np.round(interpolated).astype(np.uint8)
        
        return Image.fromarray(enhanced_arr)

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
