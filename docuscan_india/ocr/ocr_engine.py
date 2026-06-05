import os
import shutil
import pytesseract
import numpy as np
from typing import Dict, Any, Tuple, List
from utils.logger import get_logger

logger = get_logger("ocr_engine")

class OCREngine:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OCR Engine.
        config: Dict containing tesseract and ocr_psm settings.
        """
        self.config = config
        self._setup_tesseract_path()

    def _setup_tesseract_path(self):
        """Locates and configures the Tesseract binary path."""
        tess_config = self.config.get("tesseract", {})
        configured_path = tess_config.get("cmd_path", "")

        # 1. Check configured path
        if configured_path and os.path.exists(configured_path):
            pytesseract.pytesseract.tesseract_cmd = configured_path
            logger.info(f"Tesseract path set from config: {configured_path}")
            return

        # 2. Check standard paths on Windows
        standard_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
        ]
        for path in standard_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract found at standard path: {path}")
                return

        # 3. Check if in system PATH
        path_executable = shutil.which("tesseract")
        if path_executable:
            pytesseract.pytesseract.tesseract_cmd = path_executable
            logger.info(f"Tesseract found in system PATH: {path_executable}")
            return

        logger.warning(
            "Tesseract executable was not found. OCR commands will fail unless "
            "Tesseract is installed and configured in config.yaml."
        )

    def extract(self, img: np.ndarray, doc_type: str = "UNKNOWN") -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Executes OCR on the image.
        Returns:
            - raw_text (str): The full extracted text.
            - confidence (float): Aggregate confidence score from 0.0 to 1.0.
            - word_map (List[Dict]): Word-level details including text, bounding box, and confidence.
        """
        # Determine language and PSM mode
        tess_config = self.config.get("tesseract", {})
        default_lang = tess_config.get("default_lang", "eng+hin")
        
        # Aadhaar cards often contain both English and Hindi. Others default to English.
        lang = default_lang if doc_type == "AADHAAR" else "eng"

        psm_map = self.config.get("ocr_psm", {})
        psm_mode = psm_map.get(doc_type, 3)

        config_str = f"--psm {psm_mode}"

        try:
            # 1. Get raw string text
            raw_text = pytesseract.image_to_string(img, lang=lang, config=config_str)

            # 2. Get word-level detail map for spatial and confidence analysis
            data_dict = pytesseract.image_to_data(img, lang=lang, config=config_str, output_type=pytesseract.Output.DICT)

            word_map = []
            confidences = []

            n_boxes = len(data_dict['text'])
            for i in range(n_boxes):
                text = data_dict['text'][i].strip()
                conf = float(data_dict['conf'][i])
                
                # Filter out structural elements (Tesseract returns conf -1 for blocks/paragraphs/lines)
                if conf != -1:
                    confidences.append(conf)
                    
                    if text:  # Ignore empty spaces
                        word_map.append({
                            'text': text,
                            'left': int(data_dict['left'][i]),
                            'top': int(data_dict['top'][i]),
                            'width': int(data_dict['width'][i]),
                            'height': int(data_dict['height'][i]),
                            'conf': conf / 100.0
                        })

            # Calculate average confidence
            agg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

            return raw_text, agg_confidence, word_map

        except Exception as e:
            logger.error(f"OCR Extraction failed: {e}")
            # If tesseract is missing, we return empty structure to allow testing the pipeline
            return "", 0.0, []
