import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional

class ImageLoader:
    @staticmethod
    def load(file_path: str) -> np.ndarray:
        """
        Loads an image from the filesystem.
        Supports JPEG, PNG, TIFF, and PDF (first page).
        Returns a BGR OpenCV image array.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return ImageLoader._load_pdf(file_path)
        else:
            # OpenCV's cv2.imread might fail with non-ASCII characters in path.
            # Use numpy to load it safely.
            try:
                img_array = np.fromfile(file_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image from {file_path}")
                return img
            except Exception as e:
                # Fallback to PIL
                try:
                    pil_img = Image.open(file_path)
                    # Convert to RGB and then BGR
                    rgb_img = pil_img.convert('RGB')
                    return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
                except Exception as ex:
                    raise ValueError(f"Could not load image: {ex}") from e

    @staticmethod
    def _load_pdf(file_path: str) -> np.ndarray:
        """
        Load the first page of a PDF file as an image.
        Uses pymupdf (fitz) if available.
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            if len(doc) == 0:
                raise ValueError("PDF file is empty")
            page = doc[0]
            # Increase resolution (zoom factor = 2) for better OCR
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except ImportError:
            # Fallback warning or attempt PIL
            try:
                from PIL import Image, ImageSequence
                with Image.open(file_path) as doc:
                    # PIL can open some PDFs but requires Ghostscript for rendering
                    # We try to get first frame
                    for frame in ImageSequence.Iterator(doc):
                        rgb_img = frame.convert('RGB')
                        return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
            raise ImportError("PyMuPDF (pymupdf) is required to process PDF files. Please install it.")
