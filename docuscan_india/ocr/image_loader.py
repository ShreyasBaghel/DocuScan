import os
from PIL import Image
from typing import Optional
import io

class ImageLoader:
    @staticmethod
    def load(file_path: str) -> Image.Image:
        """
        Loads an image from the filesystem.
        Supports JPEG, PNG, TIFF, and PDF (first page).
        Returns a PIL Image.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return ImageLoader._load_pdf(file_path)
        else:
            try:
                # Open with PIL
                img = Image.open(file_path)
                img.load()  # Load image data into memory so we can close the file handle
                return img
            except Exception as e:
                raise ValueError(f"Could not load image: {e}")

    @staticmethod
    def _load_pdf(file_path: str) -> Image.Image:
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
            return Image.open(io.BytesIO(img_data))
        except ImportError:
            # Fallback warning or attempt PIL
            try:
                from PIL import ImageSequence
                with Image.open(file_path) as doc:
                    for frame in ImageSequence.Iterator(doc):
                        return frame.convert('RGB')
            except Exception:
                pass
            raise ImportError("PyMuPDF (pymupdf) is required to process PDF files. Please install it.")
