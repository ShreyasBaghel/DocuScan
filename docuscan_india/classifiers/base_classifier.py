from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from utils.document_packet import DocumentType

class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        """
        Analyzes the text and layout to classify the document type.
        Returns:
            - DocumentType: The classified document type.
            - confidence (float): Confidence score from 0.0 to 1.0.
        """
        pass
