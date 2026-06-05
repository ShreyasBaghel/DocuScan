import numpy as np
from typing import List
from utils.document_packet import FraudSignal

class ELADetector:
    def detect(self, img: np.ndarray) -> List[FraudSignal]:
        """
        Phase 2 Stub for Error Level Analysis (ELA).
        ELA detects differences in compression ratios across the image, highlighting
        areas that may have been edited or resaved.
        """
        # Phase 2 Implementation logic placeholder:
        # cv2.imwrite("temp_ela.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        # temp = cv2.imread("temp_ela.jpg")
        # diff = cv2.absdiff(img, temp)
        return []
